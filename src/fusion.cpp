#include "tc/fusion.hpp"
#include <unordered_map>
#include <vector>

void fuse(Graph& g) {
    for (auto& node : g.nodes)
        node.group = node.id;

    std::unordered_map<GroupId, std::vector<NodeId>> groups;
    for (const auto& node : g.nodes)
        groups[node.id].push_back(node.id);

    for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
        if (g.nodes[ni].cls != OpClass::Elementwise) continue;

        for (TensorId in_tid : g.nodes[ni].inputs) {
            NodeId prod_id = g.producer_of(in_tid);
            if (prod_id == kInvalidNode) continue;

            if (g.nodes[prod_id].cls != OpClass::Elementwise) continue;
            if (g.consumers_of(in_tid).size() != 1) continue;

            TensorId prod_out = g.nodes[prod_id].outputs[0];
            TensorId node_out = g.nodes[ni].outputs[0];
            if (g.tensors[prod_out].shape != g.tensors[node_out].shape) continue;

            GroupId old_gid = g.nodes[prod_id].group;
            GroupId new_gid = g.nodes[ni].group;
            if (old_gid == new_gid) continue;

            auto& old_members = groups[old_gid];
            for (NodeId nid : old_members)
                g.nodes[nid].group = new_gid;
            auto& new_members = groups[new_gid];
            new_members.insert(new_members.end(), old_members.begin(), old_members.end());
            groups.erase(old_gid);
        }
    }
}
