#pragma once
#include "types.hpp"
#include "loop_ir.hpp"
#include <functional>
#include <unordered_map>

enum class OpClass {
    Elementwise,
    Reduction,
    Contraction,
    View,
    Opaque
};

enum class OpKind {
    MatMul,
    Add, Mul, Sub,
    Relu, Sigmoid, Tanh, Gelu, Exp,
    Sum, Max, Mean, Softmax,
    Reshape, Transpose, Broadcast,
    Unknown
};

struct Attributes {
    int                  axis    = -1;
    bool                 trans_a = false;
    bool                 trans_b = false;
    Shape                target_shape;
    std::vector<int64_t> perm;
};

using ShapeFn      = std::function<Shape(const std::vector<Shape>&, const Attributes&)>;
using ScalarExprFn = std::function<std::string(const std::vector<std::string>&)>;
using LowerFn      = std::function<LoopNest(const std::vector<BufferId>&, BufferId,
                                             const Attributes&, const LoopProgram&)>;

struct OpSpec {
    OpKind        kind;
    OpClass       cls;
    int           arity;
    ShapeFn       infer_shape;
    ScalarExprFn  scalar_expr;
    LowerFn       lower_naive;
};

class OpRegistry {
public:
    static OpRegistry& instance();
    void            register_op(OpSpec spec);
    const OpSpec&   get(OpKind kind) const;
    bool            has(OpKind kind) const;

private:
    std::unordered_map<int, OpSpec> specs_;
    OpRegistry() = default;
};

void        register_all_ops();
const char* opkind_to_str(OpKind k);
const char* opclass_to_str(OpClass c);
