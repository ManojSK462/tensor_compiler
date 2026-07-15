#include "tc/lower.hpp"
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <algorithm>

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

static TensorId find_group_output(const std::vector<const Node*>& members,
                                   const std::unordered_set<NodeId>& member_set,
                                   const Graph& g,
                                   const std::function<bool(TensorId)>& is_graph_output) {
    TensorId result = kInvalidTensor;
    for (const Node* n : members) {
        for (TensorId t : n->outputs) {
            if (is_graph_output(t)) { result = t; break; }
            for (NodeId cons : g.consumers_of(t))
                if (!member_set.count(cons)) { result = t; break; }
            if (result != kInvalidTensor) break;
        }
        if (result != kInvalidTensor) break;
    }
    if (result == kInvalidTensor) result = members.back()->outputs[0];
    return result;
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
        bool has_contraction = false;
        for (const Node* n : members) {
            if (n->cls != OpClass::Elementwise) all_ew = false;
            if (n->cls == OpClass::Contraction) has_contraction = true;
        }

        std::unordered_set<NodeId> member_set;
        for (const Node* n : members) member_set.insert(n->id);

        if (all_ew) {
            std::unordered_set<TensorId> produced;
            for (const Node* n : members)
                for (TensorId t : n->outputs)
                    produced.insert(t);

            TensorId group_out_tid = find_group_output(members, member_set, g, is_graph_output);

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

        } else if (has_contraction) {
            const Node* anchor = nullptr;
            std::vector<const Node*> epi_nodes;
            for (const Node* n : members) {
                if (n->cls == OpClass::Contraction) anchor = n;
                else if (n->cls == OpClass::Elementwise) epi_nodes.push_back(n);
            }

            BufferId A_bid = tensor_to_buf.at(anchor->inputs[0]);
            BufferId B_bid = tensor_to_buf.at(anchor->inputs[1]);

            int64_t M_dim = anchor->attrs.trans_a ? prog.buffers[A_bid].shape.dims[1]
                                                   : prog.buffers[A_bid].shape.dims[0];
            int64_t K_dim = anchor->attrs.trans_a ? prog.buffers[A_bid].shape.dims[0]
                                                   : prog.buffers[A_bid].shape.dims[1];
            int64_t N_dim = anchor->attrs.trans_b ? prog.buffers[B_bid].shape.dims[0]
                                                   : prog.buffers[B_bid].shape.dims[1];
            int64_t KT    = std::min(K_dim, (int64_t)256);

            std::string A_name = prog.buffers[A_bid].name;
            std::string B_name = prog.buffers[B_bid].name;

            TensorId mm_out_tid = anchor->outputs[0];

            std::unordered_set<TensorId> produced_in_group;
            produced_in_group.insert(mm_out_tid);
            for (const Node* n : epi_nodes)
                for (TensorId t : n->outputs)
                    produced_in_group.insert(t);

            TensorId group_out_tid = find_group_output(members, member_set, g, is_graph_output);

            const TensorInfo& out_ti = g.tensors[group_out_tid];
            BufferRole role = is_graph_output(group_out_tid) ? BufferRole::Output
                                                             : BufferRole::Intermediate;
            BufferId out_bid = prog.add_buffer(out_ti.shape, out_ti.dtype, role,
                                               "t" + std::to_string(group_out_tid));
            tensor_to_buf[group_out_tid] = out_bid;

            LoopNest nest;
            nest.loops = {{"i", M_dim, LoopTag::Parallel}, {"j", N_dim, LoopTag::Serial}};

            std::unordered_set<TensorId> seen_ext;
            for (TensorId t : anchor->inputs)
                if (!seen_ext.count(t)) { nest.reads.push_back(tensor_to_buf.at(t)); seen_ext.insert(t); }
            for (const Node* n : epi_nodes)
                for (TensorId t : n->inputs)
                    if (!produced_in_group.count(t) && !seen_ext.count(t)) {
                        nest.reads.push_back(tensor_to_buf.at(t));
                        seen_ext.insert(t);
                    }
            nest.writes = {out_bid};

            std::string a_idx = anchor->attrs.trans_a
                ? "k * " + std::to_string(M_dim) + " + i"
                : "i * " + std::to_string(K_dim) + " + k";
            std::string b_idx = anchor->attrs.trans_b
                ? "j * " + std::to_string(K_dim) + " + k"
                : "k * " + std::to_string(N_dim) + " + j";

            std::ostringstream body;
            body << "float acc = 0.f;\n";
            body << "        for (int64_t ko = 0; ko < " << K_dim << "; ko += " << KT << ") {\n";
            body << "            int64_t ke = ko + " << KT << " < " << K_dim
                 << " ? ko + " << KT << " : " << K_dim << ";\n";
            body << "            for (int64_t k = ko; k < ke; ++k)\n";
            body << "                acc += " << A_name << "[" << a_idx << "] * "
                 << B_name << "[" << b_idx << "];\n";
            body << "        }";

            std::unordered_set<TensorId> epi_produced;
            epi_produced.insert(mm_out_tid);
            std::vector<std::string> ij_vars = {"i", "j"};
            Shape out_shape = out_ti.shape;

            for (const Node* n : epi_nodes) {
                const OpSpec& spec = reg.get(n->kind);
                std::vector<std::string> input_exprs;
                for (TensorId t : n->inputs) {
                    if (epi_produced.count(t)) {
                        input_exprs.push_back(t == mm_out_tid ? "acc" : "_ep" + std::to_string(t));
                    } else {
                        BufferId bid = tensor_to_buf.at(t);
                        const Buffer& buf = prog.buffers[bid];
                        std::string idx = bcast_idx_fused(buf.shape, out_shape, ij_vars);
                        input_exprs.push_back(buf.name + "[" + idx + "]");
                    }
                }
                std::string expr = spec.scalar_expr(input_exprs);
                TensorId out_t = n->outputs[0];
                epi_produced.insert(out_t);
                body << "\n        ";
                if (out_t == group_out_tid)
                    body << prog.buffers[out_bid].name << "[i * " << N_dim << " + j] = " << expr << ";";
                else
                    body << "float _ep" << out_t << " = " << expr << ";";
            }

            if (epi_nodes.empty())
                body << "\n        " << prog.buffers[out_bid].name << "[i * " << N_dim << " + j] = acc;";

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
