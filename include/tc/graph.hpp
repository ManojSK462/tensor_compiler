#pragma once
#include "ops.hpp"
#include <vector>
#include <unordered_map>

using NodeId  = uint32_t;
using GroupId = uint32_t;
static constexpr NodeId  kInvalidNode = UINT32_MAX;
static constexpr GroupId kNoGroup     = UINT32_MAX;

struct Node {
    NodeId                id;
    OpKind                kind;
    OpClass               cls;
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    Attributes            attrs;
    GroupId               group = kNoGroup;
};

struct Graph {
    std::vector<Node>       nodes;
    std::vector<TensorInfo> tensors;
    std::vector<TensorId>   graph_inputs;
    std::vector<TensorId>   graph_outputs;

    TensorId input(Shape shape, DType dtype = DType::F32);

    TensorId matmul(TensorId a, TensorId b, bool trans_a = false, bool trans_b = false);

    TensorId add(TensorId a, TensorId b);
    TensorId mul(TensorId a, TensorId b);
    TensorId sub(TensorId a, TensorId b);

    TensorId relu(TensorId x);
    TensorId sigmoid(TensorId x);
    TensorId tanh_(TensorId x);
    TensorId gelu(TensorId x);
    TensorId exp_(TensorId x);

    TensorId sum(TensorId x, int axis = -1);
    TensorId max(TensorId x, int axis = -1);
    TensorId mean(TensorId x, int axis = -1);
    TensorId softmax(TensorId x, int axis = -1);

    TensorId reshape(TensorId x, Shape target);
    TensorId transpose(TensorId x, std::vector<int64_t> perm = {});
    TensorId broadcast(TensorId x, Shape target);

    void mark_output(TensorId t);

    NodeId              producer_of(TensorId t) const;
    std::vector<NodeId> consumers_of(TensorId t) const;

    TensorId add_op(OpKind kind, std::vector<TensorId> inputs, Attributes attrs = {});

private:
    std::unordered_map<uint32_t, NodeId>              producer_map_;
    std::unordered_map<uint32_t, std::vector<NodeId>> consumer_map_;

    TensorId new_tensor(Shape shape, DType dtype = DType::F32);
    NodeId   new_node(OpKind kind, OpClass cls,
                      std::vector<TensorId> ins,
                      std::vector<TensorId> outs,
                      Attributes attrs);
};
