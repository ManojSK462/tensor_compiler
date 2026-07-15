#include "tc/fusion.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>

static bool group_has_anchor(const std::unordered_map<GroupId, std::vector<NodeId>>& groups,
                              const Graph& g, GroupId gid) {
    auto it = groups.find(gid);
    if (it == groups.end()) return false;
    for (NodeId nid : it->second)
        if (g.nodes[nid].cls == OpClass::Contraction || g.nodes[nid].cls == OpClass::Reduction)
            return true;
    return false;
}

static void merge_into(GroupId old_gid, GroupId new_gid,
                        std::unordered_map<GroupId, std::vector<NodeId>>& groups,
                        Graph& g) {
    auto& old_m = groups[old_gid];
    for (NodeId nid : old_m) g.nodes[nid].group = new_gid;
    auto& new_m = groups[new_gid];
    new_m.insert(new_m.end(), old_m.begin(), old_m.end());
    groups.erase(old_gid);
}

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

            GroupId prod_gid = g.nodes[prod_id].group;
            if (!group_has_anchor(groups, g, prod_gid)) continue;
            if (g.consumers_of(in_tid).size() != 1) continue;

            NodeId anchor_nid = kInvalidNode;
            for (NodeId mid : groups[prod_gid])
                if (g.nodes[mid].cls == OpClass::Contraction || g.nodes[mid].cls == OpClass::Reduction)
                    { anchor_nid = mid; break; }

            TensorId anchor_out = g.nodes[anchor_nid].outputs[0];
            if (g.tensors[g.nodes[ni].outputs[0]].shape != g.tensors[anchor_out].shape) continue;

            GroupId old_gid = g.nodes[ni].group;
            if (old_gid == prod_gid) continue;

            merge_into(old_gid, prod_gid, groups, g);
        }
    }

    for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
        if (g.nodes[ni].cls != OpClass::Elementwise) continue;

        for (TensorId in_tid : g.nodes[ni].inputs) {
            NodeId prod_id = g.producer_of(in_tid);
            if (prod_id == kInvalidNode) continue;

            if (g.nodes[prod_id].cls != OpClass::Elementwise) continue;

            GroupId prod_gid = g.nodes[prod_id].group;
            if (group_has_anchor(groups, g, prod_gid)) continue;
            if (g.consumers_of(in_tid).size() != 1) continue;

            TensorId prod_out = g.nodes[prod_id].outputs[0];
            if (g.tensors[prod_out].shape != g.tensors[g.nodes[ni].outputs[0]].shape) continue;

            GroupId old_gid = prod_gid;
            GroupId new_gid = g.nodes[ni].group;
            if (old_gid == new_gid) continue;

            merge_into(old_gid, new_gid, groups, g);
        }
    }
}
