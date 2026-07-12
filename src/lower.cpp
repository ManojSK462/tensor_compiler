#include "tc/lower.hpp"
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <stdexcept>

LoopProgram lower_naive(const Graph& g) {
    LoopProgram prog;
    auto& reg = OpRegistry::instance();

    std::unordered_map<TensorId, BufferId> tensor_to_buf;

    for (TensorId tid : g.graph_inputs) {
        const TensorInfo& ti = g.tensors[tid];
        BufferId bid = prog.add_buffer(ti.shape, ti.dtype, BufferRole::External,
                                       "t" + std::to_string(tid));
        tensor_to_buf[tid] = bid;
    }

    auto is_graph_output = [&](TensorId tid) {
        for (auto go : g.graph_outputs) if (go == tid) return true;
        return false;
    };

    for (const Node& node : g.nodes) {
        const OpSpec& spec = reg.get(node.kind);

        std::vector<BufferId> in_bufs;
        for (TensorId tid : node.inputs)
            in_bufs.push_back(tensor_to_buf.at(tid));

        if (node.outputs.size() != 1)
            throw std::runtime_error("lower_naive: only single-output nodes supported");

        TensorId out_tid  = node.outputs[0];
        const TensorInfo& oti = g.tensors[out_tid];
        BufferRole role = is_graph_output(out_tid) ? BufferRole::Output : BufferRole::Intermediate;
        BufferId out_bid = prog.add_buffer(oti.shape, oti.dtype, role,
                                            "t" + std::to_string(out_tid));
        tensor_to_buf[out_tid] = out_bid;

        LoopNest kernel = spec.lower_naive(in_bufs, out_bid, node.attrs, prog);
        kernel.id   = static_cast<KernelId>(prog.kernels.size());
        kernel.name = "kernel_" + std::to_string(kernel.id);
        prog.kernels.push_back(std::move(kernel));
    }

    return prog;
}

static std::string flat_idx_fused(const Shape& shape, const std::vector<std::string>& vars) {
    if (shape.rank() == 0) return "0";
    std::string expr;
    for (int64_t i = 0; i < shape.rank(); ++i) {
        int64_t stride = 1;
        for (int64_t j = i + 1; j < shape.rank(); ++j) stride *= shape.dims[j];
        if (!expr.empty()) expr += " + ";
        expr += vars[i] + " * " + std::to_string(stride);
    }
    return expr;
}

static std::string bcast_idx_fused(const Shape& in_shape, const Shape& out_shape,
                                    const std::vector<std::string>& out_vars) {
    int64_t offset = out_shape.rank() - in_shape.rank();
    std::vector<std::string> in_vars(in_shape.rank());
    for (int64_t i = 0; i < in_shape.rank(); ++i)
        in_vars[i] = (in_shape.dims[i] == 1) ? "0" : out_vars[i + offset];
    return flat_idx_fused(in_shape, in_vars);
}

LoopProgram lower_fused(const Graph& g) {
    LoopProgram prog;
    auto& reg = OpRegistry::instance();

    std::unordered_map<TensorId, BufferId> tensor_to_buf;

    for (TensorId tid : g.graph_inputs) {
        const TensorInfo& ti = g.tensors[tid];
        BufferId bid = prog.add_buffer(ti.shape, ti.dtype, BufferRole::External,
                                       "t" + std::to_string(tid));
        tensor_to_buf[tid] = bid;
    }

    auto is_graph_output = [&](TensorId tid) {
        for (auto go : g.graph_outputs) if (go == tid) return true;
        return false;
    };

    std::vector<GroupId> group_order;
    std::unordered_set<GroupId> seen;
    for (const Node& node : g.nodes) {
        if (!seen.count(node.group)) {
            group_order.push_back(node.group);
            seen.insert(node.group);
        }
    }

    std::unordered_map<GroupId, std::vector<const Node*>> group_nodes;
    for (const Node& node : g.nodes)
        group_nodes[node.group].push_back(&node);

    for (GroupId gid : group_order) {
        const auto& members = group_nodes[gid];

        bool all_ew = true;
        for (const Node* n : members)
            if (n->cls != OpClass::Elementwise) { all_ew = false; break; }

        if (all_ew) {
            std::unordered_set<TensorId> produced;
            for (const Node* n : members)
                for (TensorId t : n->outputs)
                    produced.insert(t);

            std::unordered_set<NodeId> member_set;
            for (const Node* n : members)
                member_set.insert(n->id);

            TensorId group_out_tid = kInvalidTensor;
            for (const Node* n : members) {
                for (TensorId t : n->outputs) {
                    if (is_graph_output(t)) { group_out_tid = t; break; }
                    for (NodeId cons : g.consumers_of(t))
                        if (!member_set.count(cons)) { group_out_tid = t; break; }
                    if (group_out_tid != kInvalidTensor) break;
                }
                if (group_out_tid != kInvalidTensor) break;
            }
            if (group_out_tid == kInvalidTensor)
                group_out_tid = members.back()->outputs[0];

            const TensorInfo& out_ti = g.tensors[group_out_tid];
            BufferRole role = is_graph_output(group_out_tid) ? BufferRole::Output
                                                              : BufferRole::Intermediate;
            BufferId out_bid = prog.add_buffer(out_ti.shape, out_ti.dtype, role,
                                               "t" + std::to_string(group_out_tid));
            tensor_to_buf[group_out_tid] = out_bid;

            const Shape& out_shape = out_ti.shape;
            std::vector<std::string> out_vars;
            LoopNest nest;
            for (int64_t i = 0; i < out_shape.rank(); ++i) {
                std::string v = "i" + std::to_string(i);
                out_vars.push_back(v);
                LoopTag tag = (i == 0) ? LoopTag::Parallel : LoopTag::Serial;
                nest.loops.push_back({v, out_shape.dims[i], tag});
            }

            std::unordered_set<TensorId> seen_ext;
            for (const Node* n : members)
                for (TensorId t : n->inputs)
                    if (!produced.count(t) && !seen_ext.count(t)) {
                        nest.reads.push_back(tensor_to_buf.at(t));
                        seen_ext.insert(t);
                    }
            nest.writes = {out_bid};

            std::string cont(4 * (1 + static_cast<int>(out_shape.rank())), ' ');
            std::string out_idx = flat_idx_fused(out_shape, out_vars);
            std::ostringstream body;
            bool first_line = true;

            for (const Node* n : members) {
                const OpSpec& spec = reg.get(n->kind);

                std::vector<std::string> input_exprs;
                for (TensorId t : n->inputs) {
                    if (!produced.count(t)) {
                        BufferId bid = tensor_to_buf.at(t);
                        const Buffer& buf = prog.buffers[bid];
                        std::string idx = bcast_idx_fused(buf.shape, out_shape, out_vars);
                        input_exprs.push_back(buf.name + "[" + idx + "]");
                    } else {
                        input_exprs.push_back("_t" + std::to_string(t));
                    }
                }

                std::string expr = spec.scalar_expr(input_exprs);
                TensorId out_t = n->outputs[0];

                if (!first_line) body << "\n" << cont;
                first_line = false;

                if (out_t == group_out_tid)
                    body << prog.buffers[out_bid].name << "[" << out_idx << "] = " << expr << ";";
                else
                    body << "float _t" << out_t << " = " << expr << ";";
            }

            nest.body = body.str();
            nest.id   = static_cast<KernelId>(prog.kernels.size());
            nest.name = "kernel_" + std::to_string(nest.id);
            prog.kernels.push_back(std::move(nest));

        } else {
            for (const Node* n : members) {
                const OpSpec& spec = reg.get(n->kind);

                std::vector<BufferId> in_bufs;
                for (TensorId tid : n->inputs)
                    in_bufs.push_back(tensor_to_buf.at(tid));

                TensorId out_tid = n->outputs[0];
                const TensorInfo& oti = g.tensors[out_tid];
                BufferRole role = is_graph_output(out_tid) ? BufferRole::Output
                                                           : BufferRole::Intermediate;
                BufferId out_bid = prog.add_buffer(oti.shape, oti.dtype, role,
                                                   "t" + std::to_string(out_tid));
                tensor_to_buf[out_tid] = out_bid;

                LoopNest kernel = spec.lower_naive(in_bufs, out_bid, n->attrs, prog);
                kernel.id   = static_cast<KernelId>(prog.kernels.size());
                kernel.name = "kernel_" + std::to_string(kernel.id);
                prog.kernels.push_back(std::move(kernel));
            }
        }
    }

    return prog;
}
