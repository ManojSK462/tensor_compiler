#include "tc/ops.hpp"
#include <stdexcept>
#include <algorithm>

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    int64_t rank = std::max(a.rank(), b.rank());
    std::vector<int64_t> result(rank);
    for (int64_t i = 0; i < rank; ++i) {
        int64_t ai = a.rank() - rank + i;
        int64_t bi = b.rank() - rank + i;
        int64_t da = (ai >= 0) ? a.dims[ai] : 1;
        int64_t db = (bi >= 0) ? b.dims[bi] : 1;
        if      (da == db) result[i] = da;
        else if (da == 1)  result[i] = db;
        else if (db == 1)  result[i] = da;
        else throw std::runtime_error(
            "broadcast_shapes: incompatible dims " +
            std::to_string(da) + " vs " + std::to_string(db));
    }
    return Shape(result);
}

std::string dtype_to_str(DType d) {
    switch (d) { case DType::F32: return "f32"; }
    return "?";
}

std::string shape_to_str(const Shape& s) {
    if (s.dims.empty()) return "[]";
    std::string r = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) {
        if (i) r += ",";
        r += std::to_string(s.dims[i]);
    }
    return r + "]";
}

const char* opkind_to_str(OpKind k) {
    switch (k) {
        case OpKind::MatMul:    return "matmul";
        case OpKind::Add:       return "add";
        case OpKind::Mul:       return "mul";
        case OpKind::Sub:       return "sub";
        case OpKind::Relu:      return "relu";
        case OpKind::Sigmoid:   return "sigmoid";
        case OpKind::Tanh:      return "tanh";
        case OpKind::Gelu:      return "gelu";
        case OpKind::Exp:       return "exp";
        case OpKind::Sum:       return "sum";
        case OpKind::Max:       return "max";
        case OpKind::Mean:      return "mean";
        case OpKind::Softmax:   return "softmax";
        case OpKind::Reshape:   return "reshape";
        case OpKind::Transpose: return "transpose";
        case OpKind::Broadcast: return "broadcast";
        case OpKind::Unknown:   return "unknown";
    }
    return "?";
}

const char* opclass_to_str(OpClass c) {
    switch (c) {
        case OpClass::Elementwise:  return "elementwise";
        case OpClass::Reduction:    return "reduction";
        case OpClass::Contraction:  return "contraction";
        case OpClass::View:         return "view";
        case OpClass::Opaque:       return "opaque";
    }
    return "?";
}

OpRegistry& OpRegistry::instance() {
    static OpRegistry inst;
    return inst;
}

void OpRegistry::register_op(OpSpec spec) {
    specs_.emplace(static_cast<int>(spec.kind), std::move(spec));
}

const OpSpec& OpRegistry::get(OpKind kind) const {
    auto it = specs_.find(static_cast<int>(kind));
    if (it == specs_.end())
        throw std::runtime_error(
            std::string("unregistered op: ") + opkind_to_str(kind));
    return it->second;
}

bool OpRegistry::has(OpKind kind) const {
    return specs_.count(static_cast<int>(kind)) > 0;
}

static int64_t normalize_axis(int axis, int64_t rank) {
    if (axis < 0) axis += static_cast<int>(rank);
    if (axis < 0 || axis >= static_cast<int>(rank))
        throw std::runtime_error("axis out of range");
    return static_cast<int64_t>(axis);
}

static ShapeFn reduction_shape_fn() {
    return [](const std::vector<Shape>& in, const Attributes& a) -> Shape {
        int64_t ax = normalize_axis(a.axis, in[0].rank());
        std::vector<int64_t> out;
        for (int64_t i = 0; i < in[0].rank(); ++i)
            if (i != ax) out.push_back(in[0].dims[i]);
        return Shape(out);
    };
}

static ShapeFn same_shape_fn() {
    return [](const std::vector<Shape>& in, const Attributes&) { return in[0]; };
}

void register_all_ops() {
    auto& reg = OpRegistry::instance();

    reg.register_op({
        OpKind::MatMul, OpClass::Contraction, 2,
        [](const std::vector<Shape>& in, const Attributes& a) -> Shape {
            if (in[0].rank() != 2 || in[1].rank() != 2)
                throw std::runtime_error("matmul: inputs must be 2-D");
            int64_t M  = a.trans_a ? in[0].dims[1] : in[0].dims[0];
            int64_t Ka = a.trans_a ? in[0].dims[0] : in[0].dims[1];
            int64_t Kb = a.trans_b ? in[1].dims[1] : in[1].dims[0];
            int64_t N  = a.trans_b ? in[1].dims[0] : in[1].dims[1];
            if (Ka != Kb)
                throw std::runtime_error(
                    "matmul: inner dim mismatch (" +
                    std::to_string(Ka) + " vs " + std::to_string(Kb) + ")");
            return Shape({M, N});
        },
        nullptr
    });

    reg.register_op({
        OpKind::Add, OpClass::Elementwise, 2,
        [](const std::vector<Shape>& in, const Attributes&) {
            return broadcast_shapes(in[0], in[1]);
        },
        [](const std::vector<std::string>& v) { return "(" + v[0] + " + " + v[1] + ")"; }
    });

    reg.register_op({
        OpKind::Mul, OpClass::Elementwise, 2,
        [](const std::vector<Shape>& in, const Attributes&) {
            return broadcast_shapes(in[0], in[1]);
        },
        [](const std::vector<std::string>& v) { return "(" + v[0] + " * " + v[1] + ")"; }
    });

    reg.register_op({
        OpKind::Sub, OpClass::Elementwise, 2,
        [](const std::vector<Shape>& in, const Attributes&) {
            return broadcast_shapes(in[0], in[1]);
        },
        [](const std::vector<std::string>& v) { return "(" + v[0] + " - " + v[1] + ")"; }
    });

    reg.register_op({
        OpKind::Relu, OpClass::Elementwise, 1,
        same_shape_fn(),
        [](const std::vector<std::string>& v) {
            return "(" + v[0] + " > 0.f ? " + v[0] + " : 0.f)";
        }
    });

    reg.register_op({
        OpKind::Sigmoid, OpClass::Elementwise, 1,
        same_shape_fn(),
        [](const std::vector<std::string>& v) {
            return "(1.f / (1.f + std::exp(-" + v[0] + ")))";
        }
    });

    reg.register_op({
        OpKind::Tanh, OpClass::Elementwise, 1,
        same_shape_fn(),
        [](const std::vector<std::string>& v) { return "std::tanh(" + v[0] + ")"; }
    });

    reg.register_op({
        OpKind::Gelu, OpClass::Elementwise, 1,
        same_shape_fn(),
        [](const std::vector<std::string>& v) {
            return "(" + v[0] + " * 0.5f * (1.f + std::erf(" +
                   v[0] + " * 0.7071067811865476f)))";
        }
    });

    reg.register_op({
        OpKind::Exp, OpClass::Elementwise, 1,
        same_shape_fn(),
        [](const std::vector<std::string>& v) { return "std::exp(" + v[0] + ")"; }
    });

    reg.register_op({ OpKind::Sum,     OpClass::Reduction, 1, reduction_shape_fn(), nullptr });
    reg.register_op({ OpKind::Max,     OpClass::Reduction, 1, reduction_shape_fn(), nullptr });
    reg.register_op({ OpKind::Mean,    OpClass::Reduction, 1, reduction_shape_fn(), nullptr });
    reg.register_op({ OpKind::Softmax, OpClass::Reduction, 1, same_shape_fn(),      nullptr });

    reg.register_op({
        OpKind::Reshape, OpClass::View, 1,
        [](const std::vector<Shape>& in, const Attributes& a) -> Shape {
            if (in[0].numel() != a.target_shape.numel())
                throw std::runtime_error(
                    "reshape: numel mismatch (" +
                    std::to_string(in[0].numel()) + " vs " +
                    std::to_string(a.target_shape.numel()) + ")");
            return a.target_shape;
        },
        nullptr
    });

    reg.register_op({
        OpKind::Transpose, OpClass::View, 1,
        [](const std::vector<Shape>& in, const Attributes& a) -> Shape {
            if (a.perm.size() != static_cast<size_t>(in[0].rank()))
                throw std::runtime_error("transpose: perm length != input rank");
            std::vector<int64_t> out(in[0].rank());
            for (int64_t i = 0; i < in[0].rank(); ++i)
                out[i] = in[0].dims[a.perm[i]];
            return Shape(out);
        },
        nullptr
    });

    reg.register_op({
        OpKind::Broadcast, OpClass::View, 1,
        [](const std::vector<Shape>& in, const Attributes& a) -> Shape {
            broadcast_shapes(in[0], a.target_shape);
            return a.target_shape;
        },
        nullptr
    });

    reg.register_op({
        OpKind::Unknown, OpClass::Opaque, 0,
        [](const std::vector<Shape>&, const Attributes&) -> Shape {
            throw std::runtime_error("Unknown op: cannot infer shape");
        },
        nullptr
    });
}
