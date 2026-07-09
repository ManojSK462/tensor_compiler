# Tensor Compiler — Implementation Notes

---

## M0 — Graph IR  ✅

### What was built

The foundational data model and tooling: types, op registry, graph IR, builder API, two passes (shape inference + validate), and two printers (text + DOT).

### File layout

```
include/tc/
  types.hpp    — DType, Layout, Shape, TensorId, TensorInfo
  ops.hpp      — OpClass, OpKind, Attributes, OpSpec, OpRegistry
  graph.hpp    — NodeId, GroupId, Node, Graph (with builder API)
  passes.hpp   — infer_shapes, validate
  printer.hpp  — print_text, print_dot

src/
  ops.cpp      — OpRegistry impl, all 15 op registrations, utility fns
  graph.cpp    — Graph builder methods + adjacency maps
  passes.cpp   — infer_shapes and validate
  printer.cpp  — text printer, Graphviz DOT printer

tests/
  test_m0.cpp  — 7 tests: MLP layer, elementwise chain, softmax,
                  reshape/transpose, 2-layer MLP, reductions, DOT file

CMakeLists.txt — C++17 static lib (tc) + test_m0 executable
```

### How to build and run

```bat
:: Requires VS 2019 Build Tools (cl.exe + bundled cmake)
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S K:\tensor_compiler -B K:\tensor_compiler\build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake --build K:\tensor_compiler\build
K:\tensor_compiler\build\test_m0.exe
```

### Key design choices

**TensorId = vector index.**
`Graph::tensors` is a plain `std::vector<TensorInfo>`. `TensorId` is the index.
Tensors are never removed, so IDs are stable. Same for `NodeId` → `Graph::nodes`.

**Builder infers shapes inline.**
Every `add_op(kind, inputs, attrs)` call immediately runs `OpSpec::infer_shape`,
creates the output tensor with the computed shape, and appends the node.
`infer_shapes()` as a pass re-verifies consistency — it's a foundation for
loading graphs from external sources (ONNX) where shapes may be absent at build time.

**Op registry is a Meyers singleton.**
Call `register_all_ops()` once before any graph is built. After that the registry
is read-only.

**`OpSpec::scalar_expr` stores the per-element C++ expression as a closure.**
Example: `Relu` → `"(v > 0.f ? v : 0.f)"`. These strings are not used in M0 —
they will be composed and emitted verbatim into generated kernel bodies at M2+.

**`Attributes` is a flat struct, not a map.**
Covers all op-specific parameters: `axis`, `trans_a/b`, `target_shape`, `perm`.
Simple and fast for v1. Not general but sufficient.

**Full NumPy broadcast rules.**
`broadcast_shapes(a, b)`: right-align dims (prepend 1s to shorter shape), then
per-dim: equal → keep, one is 1 → take the other, else throw. Broadcast dims
become stride-0 indices in emitted kernels (M2+).

**`tanh_` / `exp_` naming.**
Trailing underscore avoids `std::tanh`/`std::exp` name collision in translation
units that include `<cmath>`.

**`validate()` checks four invariants:**
1. Node id matches its position in `nodes` (topological order is maintained).
2. Every node's input tensors are defined before the node (SSA / topo order).
3. Every non-input tensor has exactly one producer (SSA single-definition).
4. Every tensor dim is strictly positive.

### What M0 demonstrates

```
Graph g;
auto x = g.input({8, 32});
auto W = g.input({32, 64});
auto b = g.input({64});
auto y = g.relu(g.add(g.matmul(x, W), b));
g.mark_output(y);
infer_shapes(g);   // verifies shapes
validate(g);       // asserts invariants
print_text(g, std::cout);          // human-readable
print_dot(g, std::ofstream("out.dot"));  // Graphviz
```

Output (text):
```
Graph {
  inputs:  t0 [8,32]:f32  t1 [32,64]:f32  t2 [64]:f32
  nodes:
    n0  matmul(t0, t1) -> t3[8,64]  [contraction]
    n1  add(t3, t2) -> t4[8,64]  [elementwise]
    n2  relu(t4) -> t5[8,64]  [elementwise]
  outputs:  t5[8,64]:f32
}
```

DOT output can be rendered with `dot -Tpng out.dot -o out.png`.

### Deviations from design doc

- `LowerFn` fields (`lower_naive`, `lower_optimized`) are **not on `OpSpec` yet** —
  they require Loop IR types which don't exist until M1.
- `softmax` is a single `Reduction` node at the graph IR level; its two-pass
  numerically-stable lowering is an M4 concern.
- `Group` field on `Node` is present but always `kNoGroup` — filled by M2's fusion pass.

---

*Next: M1 — Naive end-to-end. Define Loop IR, add `lower_naive` to OpSpec,
emit C++/OpenMP source for each op, compile + run + verify harness.*
