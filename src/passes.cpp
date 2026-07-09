#include "tc/passes.hpp"
#include <unordered_set>
#include <stdexcept>

void infer_shapes(Graph& g) {
    auto& reg = OpRegistry::instance();
    for (auto& node : g.nodes) {
        const OpSpec& spec = reg.get(node.kind);
        std::vector<Shape> in_shapes;
        in_shapes.reserve(node.inputs.size());
        for (auto t : node.inputs)
            in_shapes.push_back(g.tensors[t].shape);
        Shape inferred = spec.infer_shape(in_shapes, node.attrs);
        for (auto out : node.outputs)
            g.tensors[out].shape = inferred;
    }
}

void validate(const Graph& g) {
    std::unordered_set<TensorId> defined;
    for (auto t : g.graph_inputs) defined.insert(t);

    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& n = g.nodes[i];

        if (n.id != static_cast<NodeId>(i))
            throw std::runtime_error(
                "validate: node id " + std::to_string(n.id) +
                " != position " + std::to_string(i));

        for (auto t : n.inputs) {
            if (!defined.count(t))
                throw std::runtime_error(
                    "validate: node " + std::to_string(n.id) +
                    " reads undefined tensor t" + std::to_string(t));
        }

        for (auto t : n.outputs) {
            if (defined.count(t))
                throw std::runtime_error(
                    "validate: tensor t" + std::to_string(t) + " has two producers");
            defined.insert(t);
        }
    }

    for (auto t : g.graph_outputs) {
        if (!defined.count(t))
            throw std::runtime_error(
                "validate: graph output t" + std::to_string(t) + " is undefined");
    }

    for (const auto& ti : g.tensors) {
        for (auto d : ti.shape.dims) {
            if (d <= 0)
                throw std::runtime_error(
                    "validate: tensor t" + std::to_string(ti.id) +
                    " has non-positive dim " + std::to_string(d));
        }
    }
}
