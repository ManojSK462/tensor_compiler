#include "tc/printer.hpp"

void print_text(const Graph& g, std::ostream& out) {
    out << "Graph {\n";

    out << "  inputs:";
    for (auto t : g.graph_inputs) {
        const auto& ti = g.tensors[t];
        out << "  t" << t << " " << shape_to_str(ti.shape) << ":" << dtype_to_str(ti.dtype);
    }
    out << "\n";

    out << "  nodes:\n";
    for (const auto& n : g.nodes) {
        out << "    n" << n.id << "  " << opkind_to_str(n.kind) << "(";
        for (size_t i = 0; i < n.inputs.size(); ++i) {
            if (i) out << ", ";
            out << "t" << n.inputs[i];
        }
        out << ") -> ";
        for (size_t i = 0; i < n.outputs.size(); ++i) {
            if (i) out << ", ";
            auto t = n.outputs[i];
            out << "t" << t << shape_to_str(g.tensors[t].shape);
        }
        out << "  [" << opclass_to_str(n.cls) << "]\n";
    }

    out << "  outputs:";
    for (auto t : g.graph_outputs)
        out << "  t" << t << shape_to_str(g.tensors[t].shape) << ":" << dtype_to_str(g.tensors[t].dtype);
    out << "\n}\n";
}

void print_dot(const Graph& g, std::ostream& out) {
    auto is_input  = [&](TensorId t) {
        for (auto i : g.graph_inputs)  if (i == t) return true; return false;
    };
    auto is_output = [&](TensorId t) {
        for (auto o : g.graph_outputs) if (o == t) return true; return false;
    };

    out << "digraph G {\n";
    out << "  rankdir=TB;\n";
    out << "  node [fontname=\"Helvetica\" fontsize=11];\n";

    for (const auto& ti : g.tensors) {
        const char* fill = "\"#ffffff\"";
        if (is_input(ti.id))  fill = "\"#c8e6c9\"";
        if (is_output(ti.id)) fill = "\"#ffcdd2\"";
        out << "  t" << ti.id
            << " [label=\"t" << ti.id << "\\n" << shape_to_str(ti.shape) << "\""
            << " shape=rect style=filled fillcolor=" << fill << "];\n";
    }

    for (const auto& n : g.nodes) {
        out << "  n" << n.id
            << " [label=\"" << opkind_to_str(n.kind)
            << "\" shape=ellipse style=filled fillcolor=\"#bbdefb\"];\n";
        for (auto t : n.inputs)  out << "  t" << t << " -> n" << n.id << ";\n";
        for (auto t : n.outputs) out << "  n" << n.id << " -> t" << t << ";\n";
    }

    out << "}\n";
}
