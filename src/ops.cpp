#include "tc/ops.hpp"
#include <stdexcept>
#include <algorithm>
#include <sstream>

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
        throw std::runtime_error(std::string("unregistered op: ") + opkind_to_str(kind));
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

static std::string flat_idx(const Shape& shape, const std::vector<std::string>& vars) {
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

static std::string bcast_flat_idx(const Shape& in_shape, const Shape& out_shape,
                                   const std::vector<std::string>& out_vars) {
    int64_t offset = out_shape.rank() - in_shape.rank();
    std::vector<std::string> in_vars(in_shape.rank());
    for (int64_t i = 0; i < in_shape.rank(); ++i)
        in_vars[i] = (in_shape.dims[i] == 1) ? "0" : out_vars[i + offset];
    return flat_idx(in_shape, in_vars);
}

static LowerFn elementwise_lower(ScalarExprFn expr_fn, int arity) {
    return [expr_fn, arity](const std::vector<BufferId>& ins, BufferId out,
                             const Attributes&, const LoopProgram& prog) -> LoopNest {
        LoopNest nest;
        nest.reads  = ins;
        nest.writes = {out};

        const Buffer& ob = prog.buffers[out];
        int64_t rank = ob.shape.rank();

        std::vector<std::string> vars;
        for (int64_t i = 0; i < rank; ++i) {
            std::string v = "i" + std::to_string(i);
            vars.push_back(v);
            LoopTag tag = (i == 0) ? LoopTag::Parallel : LoopTag::Serial;
            nest.loops.push_back({v, ob.shape.dims[i], tag});
        }

        std::string out_idx = flat_idx(ob.shape, vars);
        std::vector<std::string> input_exprs;
        for (int a = 0; a < arity; ++a) {
            const Buffer& ib = prog.buffers[ins[a]];
            std::string idx = bcast_flat_idx(ib.shape, ob.shape, vars);
            input_exprs.push_back(ib.name + "[" + idx + "]");
        }

        nest.body = ob.name + "[" + out_idx + "] = " + expr_fn(input_exprs) + ";";
        return nest;
    };
}

static LowerFn matmul_lower() {
    return [](const std::vector<BufferId>& ins, BufferId out,
              const Attributes& a, const LoopProgram& prog) -> LoopNest {
        LoopNest nest;
        nest.reads  = ins;
        nest.writes = {out};

        const Buffer& A = prog.buffers[ins[0]];
        const Buffer& B = prog.buffers[ins[1]];
        const Buffer& C = prog.buffers[out];

        int64_t M = C.shape.dims[0];
        int64_t N = C.shape.dims[1];
        int64_t K = a.trans_a ? A.shape.dims[0] : A.shape.dims[1];

        nest.loops = {{"i", M, LoopTag::Parallel}, {"j", N, LoopTag::Serial}};

        std::string a_idx = a.trans_a
            ? "k * " + std::to_string(M) + " + i"
            : "i * " + std::to_string(K) + " + k";
        std::string b_idx = a.trans_b
            ? "j * " + std::to_string(K) + " + k"
            : "k * " + std::to_string(N) + " + j";

        nest.body =
            "float acc = 0.f;\n"
            "        for (int64_t k = 0; k < " + std::to_string(K) + "; ++k)\n"
            "            acc += " + A.name + "[" + a_idx + "] * " +
                                    B.name + "[" + b_idx + "];\n"
            "        " + C.name + "[i * " + std::to_string(N) + " + j] = acc;";
        return nest;
    };
}

using CombineFn = std::function<std::string(const std::string&, const std::string&)>;

static LowerFn reduction_lower(const std::string& init, CombineFn combine, bool normalize) {
    return [init, combine, normalize](const std::vector<BufferId>& ins, BufferId out,
                                      const Attributes& a, const LoopProgram& prog) -> LoopNest {
        LoopNest nest;
        nest.reads  = ins;
        nest.writes = {out};

        const Buffer& ib = prog.buffers[ins[0]];
        const Buffer& ob = prog.buffers[out];
        int64_t ax       = normalize_axis(a.axis, ib.shape.rank());
        int64_t red_ext  = ib.shape.dims[ax];

        std::vector<std::string> outer_vars;
        for (int64_t i = 0; i < ib.shape.rank(); ++i) {
            if (i == ax) continue;
            std::string v = "i" + std::to_string(i);
            outer_vars.push_back(v);
            LoopTag tag = outer_vars.size() == 1 ? LoopTag::Parallel : LoopTag::Serial;
            nest.loops.push_back({v, ib.shape.dims[i], tag});
        }

        std::vector<std::string> in_vars;
        int oi = 0;
        for (int64_t i = 0; i < ib.shape.rank(); ++i)
            in_vars.push_back(i == ax ? "r" : outer_vars[oi++]);

        std::string in_idx  = flat_idx(ib.shape, in_vars);
        std::string out_idx = flat_idx(ob.shape, outer_vars);
        std::string val_expr = ib.name + "[" + in_idx + "]";
        std::string norm_suffix = normalize
            ? " / " + std::to_string(red_ext) + ".f"
            : "";

        nest.body =
            "float acc = " + init + ";\n"
            "        for (int64_t r = 0; r < " + std::to_string(red_ext) + "; ++r)\n"
            "            acc = " + combine("acc", val_expr) + ";\n"
            "        " + ob.name + "[" + out_idx + "] = acc" + norm_suffix + ";";
        return nest;
    };
}

static LowerFn softmax_lower() {
    return [](const std::vector<BufferId>& ins, BufferId out,
              const Attributes& a, const LoopProgram& prog) -> LoopNest {
        LoopNest nest;
        nest.reads  = ins;
        nest.writes = {out};

        const Buffer& ib = prog.buffers[ins[0]];
        const Buffer& ob = prog.buffers[out];
        int64_t ax       = normalize_axis(a.axis, ib.shape.rank());
        int64_t red_ext  = ib.shape.dims[ax];
        std::string N    = std::to_string(red_ext);

        std::vector<std::string> outer_vars;
        for (int64_t i = 0; i < ib.shape.rank(); ++i) {
            if (i == ax) continue;
            std::string v = "i" + std::to_string(i);
            outer_vars.push_back(v);
            LoopTag tag = outer_vars.size() == 1 ? LoopTag::Parallel : LoopTag::Serial;
            nest.loops.push_back({v, ib.shape.dims[i], tag});
        }

        auto make_idx = [&](const std::string& rv) {
            std::vector<std::string> vars;
            int oi = 0;
            for (int64_t i = 0; i < ib.shape.rank(); ++i)
                vars.push_back(i == ax ? rv : outer_vars[oi++]);
            return flat_idx(ib.shape, vars);
        };

        std::string src = ib.name;
        std::string dst = ob.name;

        nest.body =
            "float mx = " + src + "[" + make_idx("0") + "];\n"
            "        for (int64_t r = 1; r < " + N + "; ++r) { float v = " +
                src + "[" + make_idx("r") + "]; if (v > mx) mx = v; }\n"
            "        float sm = 0.f;\n"
            "        for (int64_t r = 0; r < " + N + "; ++r) {\n"
            "            float e = std::exp(" + src + "[" + make_idx("r") + "] - mx);\n"
            "            " + dst + "[" + make_idx("r") + "] = e; sm += e;\n"
            "        }\n"
            "        for (int64_t r = 0; r < " + N + "; ++r)\n"
            "            " + dst + "[" + make_idx("r") + "] /= sm;";
        return nest;
    };
}

static LowerFn view_copy_lower() {
    return [](const std::vector<BufferId>& ins, BufferId out,
              const Attributes&, const LoopProgram& prog) -> LoopNest {
        LoopNest nest;
        nest.reads  = ins;
        nest.writes = {out};
        int64_t n = prog.buffers[out].shape.numel();
        nest.loops.push_back({"i", n, LoopTag::Parallel});
        nest.body = prog.buffers[out].name + "[i] = " + prog.buffers[ins[0]].name + "[i];";
        return nest;
    };
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
                throw std::runtime_error("matmul: inner dim mismatch (" +
                    std::to_string(Ka) + " vs " + std::to_string(Kb) + ")");
            return Shape({M, N});
        },
        nullptr,
        matmul_lower()
    });

    auto bin_lower = [](auto expr) {
        return elementwise_lower(
            [expr](const std::vector<std::string>& v){ return expr(v[0], v[1]); }, 2);
    };

    reg.register_op({ OpKind::Add, OpClass::Elementwise, 2,
        [](const std::vector<Shape>& in, const Attributes&){ return broadcast_shapes(in[0],in[1]); },
        [](const std::vector<std::string>& v){ return "(" + v[0] + " + " + v[1] + ")"; },
        bin_lower([](const std::string& a, const std::string& b){ return "(" + a + " + " + b + ")"; })
    });

    reg.register_op({ OpKind::Mul, OpClass::Elementwise, 2,
        [](const std::vector<Shape>& in, const Attributes&){ return broadcast_shapes(in[0],in[1]); },
        [](const std::vector<std::string>& v){ return "(" + v[0] + " * " + v[1] + ")"; },
        bin_lower([](const std::string& a, const std::string& b){ return "(" + a + " * " + b + ")"; })
    });

    reg.register_op({ OpKind::Sub, OpClass::Elementwise, 2,
        [](const std::vector<Shape>& in, const Attributes&){ return broadcast_shapes(in[0],in[1]); },
        [](const std::vector<std::string>& v){ return "(" + v[0] + " - " + v[1] + ")"; },
        bin_lower([](const std::string& a, const std::string& b){ return "(" + a + " - " + b + ")"; })
    });

    auto un = [](ScalarExprFn f){
        return std::make_pair(f, elementwise_lower(
            [f](const std::vector<std::string>& v){ return f({v[0]}); }, 1));
    };

    auto relu_expr    = [](const std::vector<std::string>& v){ return "(" + v[0] + " > 0.f ? " + v[0] + " : 0.f)"; };
    auto sigmoid_expr = [](const std::vector<std::string>& v){ return "(1.f / (1.f + std::exp(-" + v[0] + ")))"; };
    auto tanh_expr    = [](const std::vector<std::string>& v){ return "std::tanh(" + v[0] + ")"; };
    auto gelu_expr    = [](const std::vector<std::string>& v){
        return "(" + v[0] + " * 0.5f * (1.f + std::erf(" + v[0] + " * 0.7071067811865476f)))";
    };
    auto exp_expr     = [](const std::vector<std::string>& v){ return "std::exp(" + v[0] + ")"; };

    reg.register_op({ OpKind::Relu,    OpClass::Elementwise, 1, same_shape_fn(), relu_expr,
                      elementwise_lower(relu_expr,    1) });
    reg.register_op({ OpKind::Sigmoid, OpClass::Elementwise, 1, same_shape_fn(), sigmoid_expr,
                      elementwise_lower(sigmoid_expr, 1) });
    reg.register_op({ OpKind::Tanh,    OpClass::Elementwise, 1, same_shape_fn(), tanh_expr,
                      elementwise_lower(tanh_expr,    1) });
    reg.register_op({ OpKind::Gelu,    OpClass::Elementwise, 1, same_shape_fn(), gelu_expr,
                      elementwise_lower(gelu_expr,    1) });
    reg.register_op({ OpKind::Exp,     OpClass::Elementwise, 1, same_shape_fn(), exp_expr,
                      elementwise_lower(exp_expr,     1) });

    CombineFn sum_fn = [](const std::string& a, const std::string& b){ return a + " + " + b; };
    CombineFn max_fn = [](const std::string& a, const std::string& b){ return "(" + a + " > " + b + " ? " + a + " : " + b + ")"; };

    reg.register_op({ OpKind::Sum,     OpClass::Reduction, 1, reduction_shape_fn(), nullptr,
                      reduction_lower("0.f",    sum_fn, false) });
    reg.register_op({ OpKind::Max,     OpClass::Reduction, 1, reduction_shape_fn(), nullptr,
                      reduction_lower("-1e38f", max_fn, false) });
    reg.register_op({ OpKind::Mean,    OpClass::Reduction, 1, reduction_shape_fn(), nullptr,
                      reduction_lower("0.f",    sum_fn, true)  });
    reg.register_op({ OpKind::Softmax, OpClass::Reduction, 1, same_shape_fn(),      nullptr,
                      softmax_lower() });

    reg.register_op({ OpKind::Reshape, OpClass::View, 1,
        [](const std::vector<Shape>& in, const Attributes& a) -> Shape {
            if (in[0].numel() != a.target_shape.numel())
                throw std::runtime_error("reshape: numel mismatch");
            return a.target_shape;
        }, nullptr, view_copy_lower() });

    reg.register_op({ OpKind::Transpose, OpClass::View, 1,
        [](const std::vector<Shape>& in, const Attributes& a) -> Shape {
            if (a.perm.size() != static_cast<size_t>(in[0].rank()))
                throw std::runtime_error("transpose: perm length != input rank");
            std::vector<int64_t> out(in[0].rank());
            for (int64_t i = 0; i < in[0].rank(); ++i) out[i] = in[0].dims[a.perm[i]];
            return Shape(out);
        }, nullptr, view_copy_lower() });

    reg.register_op({ OpKind::Broadcast, OpClass::View, 1,
        [](const std::vector<Shape>& in, const Attributes& a) -> Shape {
            broadcast_shapes(in[0], a.target_shape);
            return a.target_shape;
        }, nullptr, view_copy_lower() });

    reg.register_op({ OpKind::Unknown, OpClass::Opaque, 0,
        [](const std::vector<Shape>&, const Attributes&) -> Shape {
            throw std::runtime_error("Unknown op: cannot infer shape");
        }, nullptr, nullptr });
}
