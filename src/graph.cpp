#include "tc/graph.hpp"
#include <stdexcept>

TensorId Graph::new_tensor(Shape shape, DType dtype) {
    TensorId id = static_cast<TensorId>(tensors.size());
    tensors.push_back({id, std::move(shape), dtype, Layout::RowMajorContiguous});
    return id;
}

NodeId Graph::new_node(OpKind kind, OpClass cls,
                       std::vector<TensorId> ins,
                       std::vector<TensorId> outs,
                       Attributes attrs) {
    NodeId id = static_cast<NodeId>(nodes.size());
    nodes.push_back({id, kind, cls, ins, outs, std::move(attrs), kNoGroup});
    for (auto out : outs) producer_map_[out] = id;
    for (auto in  : ins)  consumer_map_[in].push_back(id);
    return id;
}

TensorId Graph::add_op(OpKind kind, std::vector<TensorId> inputs, Attributes attrs) {
    const OpSpec& spec = OpRegistry::instance().get(kind);
    std::vector<Shape> in_shapes;
    in_shapes.reserve(inputs.size());
    for (auto t : inputs) {
        if (t >= static_cast<TensorId>(tensors.size()))
            throw std::runtime_error("add_op: invalid TensorId " + std::to_string(t));
        in_shapes.push_back(tensors[t].shape);
    }
    Shape out_shape = spec.infer_shape(in_shapes, attrs);
    TensorId out    = new_tensor(out_shape, DType::F32);
    new_node(kind, spec.cls, inputs, {out}, attrs);
    return out;
}

NodeId Graph::producer_of(TensorId t) const {
    auto it = producer_map_.find(t);
    return (it != producer_map_.end()) ? it->second : kInvalidNode;
}

std::vector<NodeId> Graph::consumers_of(TensorId t) const {
    auto it = consumer_map_.find(t);
    return (it != consumer_map_.end()) ? it->second : std::vector<NodeId>{};
}

TensorId Graph::input(Shape shape, DType dtype) {
    TensorId id = new_tensor(std::move(shape), dtype);
    graph_inputs.push_back(id);
    return id;
}

void Graph::mark_output(TensorId t) {
    if (t >= static_cast<TensorId>(tensors.size()))
        throw std::runtime_error("mark_output: invalid TensorId " + std::to_string(t));
    graph_outputs.push_back(t);
}

TensorId Graph::matmul(TensorId a, TensorId b, bool trans_a, bool trans_b) {
    Attributes attrs;
    attrs.trans_a = trans_a;
    attrs.trans_b = trans_b;
    return add_op(OpKind::MatMul, {a, b}, attrs);
}

TensorId Graph::add(TensorId a, TensorId b) { return add_op(OpKind::Add, {a, b}); }
TensorId Graph::mul(TensorId a, TensorId b) { return add_op(OpKind::Mul, {a, b}); }
TensorId Graph::sub(TensorId a, TensorId b) { return add_op(OpKind::Sub, {a, b}); }

TensorId Graph::relu(TensorId x)    { return add_op(OpKind::Relu,    {x}); }
TensorId Graph::sigmoid(TensorId x) { return add_op(OpKind::Sigmoid, {x}); }
TensorId Graph::tanh_(TensorId x)   { return add_op(OpKind::Tanh,    {x}); }
TensorId Graph::gelu(TensorId x)    { return add_op(OpKind::Gelu,    {x}); }
TensorId Graph::exp_(TensorId x)    { return add_op(OpKind::Exp,     {x}); }

TensorId Graph::sum(TensorId x, int axis) {
    Attributes attrs; attrs.axis = axis;
    return add_op(OpKind::Sum, {x}, attrs);
}

TensorId Graph::max(TensorId x, int axis) {
    Attributes attrs; attrs.axis = axis;
    return add_op(OpKind::Max, {x}, attrs);
}

TensorId Graph::mean(TensorId x, int axis) {
    Attributes attrs; attrs.axis = axis;
    return add_op(OpKind::Mean, {x}, attrs);
}

TensorId Graph::softmax(TensorId x, int axis) {
    Attributes attrs; attrs.axis = axis;
    return add_op(OpKind::Softmax, {x}, attrs);
}

TensorId Graph::reshape(TensorId x, Shape target) {
    Attributes attrs;
    attrs.target_shape = std::move(target);
    return add_op(OpKind::Reshape, {x}, attrs);
}

TensorId Graph::transpose(TensorId x, std::vector<int64_t> perm) {
    Attributes attrs;
    if (perm.empty()) {
        const Shape& s = tensors[x].shape;
        attrs.perm.resize(s.rank());
        for (int64_t i = 0; i < s.rank(); ++i)
            attrs.perm[i] = s.rank() - 1 - i;
    } else {
        attrs.perm = std::move(perm);
    }
    return add_op(OpKind::Transpose, {x}, attrs);
}

TensorId Graph::broadcast(TensorId x, Shape target) {
    Attributes attrs;
    attrs.target_shape = std::move(target);
    return add_op(OpKind::Broadcast, {x}, attrs);
}
