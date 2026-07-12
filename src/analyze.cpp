#include "tc/analyze.hpp"
#include <unordered_map>
#include <unordered_set>

static int64_t tensor_bytes(const TensorInfo& ti) {
    return ti.shape.numel() * 4;
}

TrafficStats analyze_savings(const Graph& g) {
    int64_t unfused = 0;
    for (const Node& node : g.nodes) {
        for (TensorId t : node.inputs)
            unfused += tensor_bytes(g.tensors[t]);
        for (TensorId t : node.outputs)
            unfused += tensor_bytes(g.tensors[t]);
    }

    std::unordered_map<GroupId, std::vector<NodeId>> groups;
    for (const Node& node : g.nodes)
        groups[node.group].push_back(node.id);

    int64_t fused = 0;
    for (auto& [gid, members] : groups) {
        std::unordered_set<TensorId> produced;
        for (NodeId nid : members)
            for (TensorId t : g.nodes[nid].outputs)
                produced.insert(t);

        std::unordered_set<NodeId> member_set(members.begin(), members.end());

        std::unordered_set<TensorId> ext_reads;
        for (NodeId nid : members)
            for (TensorId t : g.nodes[nid].inputs)
                if (!produced.count(t))
                    ext_reads.insert(t);

        std::unordered_set<TensorId> ext_writes;
        for (NodeId nid : members) {
            for (TensorId t : g.nodes[nid].outputs) {
                bool is_go = false;
                for (TensorId go : g.graph_outputs)
                    if (go == t) { is_go = true; break; }
                if (is_go) { ext_writes.insert(t); continue; }
                for (NodeId cons : g.consumers_of(t))
                    if (!member_set.count(cons)) { ext_writes.insert(t); break; }
            }
        }

        for (TensorId t : ext_reads)
            fused += tensor_bytes(g.tensors[t]);
        for (TensorId t : ext_writes)
            fused += tensor_bytes(g.tensors[t]);
    }

    return {unfused, fused, unfused - fused};
}
