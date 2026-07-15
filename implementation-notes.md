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
  test_m0.cpp  — 7 tests
```

### Key design choices

- Builder infers shapes inline. `infer_shapes` pass re-verifies.
- `TensorId` = vector index. Stable, O(1) lookup. Never removed.
- `Attributes` is a flat struct covering all op-specific fields.
- Full NumPy broadcast rules in `broadcast_shapes()`.
- `OpRegistry` is a Meyers singleton. Call `register_all_ops()` once before any graph.
- `tanh_` / `exp_` use trailing underscore to avoid `std::` name collision.

### Deviations from design doc

- `LowerFn` not on `OpSpec` in M0 — added at M1 when Loop IR is defined.
- `Group` on `Node` always `kNoGroup` — filled by M2 fusion pass.

---

## M1 — Naive end-to-end  ✅

### What was built

Loop IR, naive lowering for all 15 ops, C++/OpenMP codegen, compile+run+verify harness. A small MLP compiles, runs, and matches a reference implementation within fp tolerance.

### New files

```
include/tc/
  loop_ir.hpp  — Buffer, Loop, LoopNest, LoopProgram
  lower.hpp    — lower_naive(Graph) -> LoopProgram
  codegen.hpp  — emit_cpp(LoopProgram) -> std::string
  runtime.hpp  — compile_and_run(src, prog, inputs) -> RunResult

src/
  loop_ir.cpp  — LoopProgram::add_buffer
  lower.cpp    — walks graph nodes, dispatches lower_naive per op
  codegen.cpp  — emits kernel functions + standalone main()
  runtime.cpp  — writes to generated/, compiles via bat+cl.exe, runs

tests/
  test_m1.cpp  — matmul→add→relu and elementwise chain, verified vs reference

generated/     — runtime writes tc_gen.cpp, tc_gen.exe, compile.bat, run.bat, *.bin here
```

### How to build and run

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S K:\tensor_compiler -B K:\tensor_compiler\build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake --build K:\tensor_compiler\build
K:\tensor_compiler\build\test_m1.exe
```

### How the pipeline works end-to-end

```
Graph  →  lower_naive()  →  LoopProgram  →  emit_cpp()  →  C++ source string
                                                                    ↓
                                                        written to generated/tc_gen.cpp
                                                                    ↓
                                                        compile.bat (vcvars64 + cl.exe /O2)
                                                                    ↓
                                                        tc_gen.exe (standalone binary)
                                                                    ↓
                                               run.bat (passes binary input files, captures output)
                                                                    ↓
                                               read back output .bin → compare vs reference
```

### Loop IR design

`LoopNest` has:
- `loops`: ordered list of `Loop {var, extent, tag}` — only the outer loops. Tags: Serial, Parallel, Vectorize, Reduce.
- `body`: a string of C++ that goes inside all the loops verbatim.

For matmul, the reduction `k` loop is embedded in `body` (not in `loops`). This is intentional for M1 — at M3 the matmul lowering will be restructured for tiling.

### LowerFn signature

```cpp
using LowerFn = std::function<LoopNest(
    const std::vector<BufferId>& ins,
    BufferId out,
    const Attributes& attrs,
    const LoopProgram& prog)>;
```

The LowerFn reads buffer shapes from `prog.buffers[id]` to construct index expressions. Kernel id and name are assigned by `lower.cpp` after the fn returns (not inside the fn).

### Key lowering decisions

- **Elementwise**: one loop per output dimension, outer dim tagged Parallel. Broadcast handled by `bcast_flat_idx` — maps input dims to output loop vars, substituting "0" for broadcast dims (stride-0 semantics).
- **Matmul**: loops [i(Parallel), j(Serial)], k loop embedded in body. Shapes taken from output buffer.
- **Reductions (sum/max/mean)**: outer dims parallel, reduction loop in body. Mean divides by axis size in the final assignment.
- **Softmax**: two-pass (max for numerical stability, then exp+normalize) in body; outer dims parallel.
- **View ops (reshape/transpose/broadcast)**: flat element-wise copy. Correct but suboptimal — at M2+ these are folded into index expressions instead.

### Codegen structure

`emit_cpp` generates a standalone C++ binary with:
- One `static void kernel_N(const float* read_bufs..., float* write_bufs...)` per LoopNest.
- `main(argc, argv)`: reads external buffers from binary files (argv[1..N]), runs kernels, writes output buffers to binary files (argv[N+1..]).

### Runtime (Windows-specific)

- Generated files go to `K:\tensor_compiler\generated\` (not `%TEMP%` — avoids sandbox path issues).
- `compile.bat`: calls `vcvars64.bat` then `cl.exe /O2`.
- `run.bat`: invokes the compiled exe with quoted binary file paths (bat file avoids cmd.exe quoting quirks from `system()` with a leading `"`).
- After compilation, checks exe exists before proceeding.

### Correctness test

`test_mlp_layer` computes reference output manually in C++ (triple loop for matmul, then add+relu) and compares element-wise against compiled output within `1e-4` tolerance. Both match on the same random inputs.

### Deviations from design doc

- Codegen produces a standalone exe (file-based I/O), not a `.so` + `dlopen`. On Windows, `LoadLibrary`/`GetProcAddress` is the equivalent but the exe approach is simpler for M1.
- View ops lower to a flat copy rather than being folded into index expressions — folding comes at M2 when we handle fusion and can inline views into consumer kernels.
- No OpenMP at compile time yet (`/O2` only, no `/openmp`). `#pragma omp parallel for` is in the generated source but ignored. Enabled at M3 for benchmarking.

---

---

## M2 — Elementwise fusion  ✅

### What was built

Greedy elementwise fusion pass, fused lowering that composes scalar expressions into a single kernel body, and an analytical memory-traffic savings estimator.

### New files

```
include/tc/
  fusion.hpp   — fuse(Graph&)
  analyze.hpp  — TrafficStats, analyze_savings

src/
  fusion.cpp   — greedy vertical fusion pass
  analyze.cpp  — memory-traffic model
  lower.cpp    — extended with lower_fused(Graph&)

tests/
  test_m2.cpp  — 4 tests
```

### Fusion algorithm

`fuse(Graph&)` runs in topological order:

1. Assign every node its own singleton group (`node.group = node.id`).
2. For each elementwise node N, for each input tensor from an elementwise producer P:
   - Skip if P has class ≠ Elementwise, or the tensor has multiple consumers, or P's output shape ≠ N's output shape.
   - Merge P's entire group into N's group.
3. Non-elementwise nodes (Contraction, Reduction, View) each stay in their own singleton group.

Multi-consumer check (`consumers_of(t).size() != 1`) is the key legality gate. Shape equality is required for the composed iteration space to be well-defined.

### Fused lowering

`lower_fused(Graph&)` processes each group in topological order:

- **All-elementwise group**: emits one `LoopNest`. Loops over the output shape. Body string is built by walking the group's nodes in topological order: external tensors become buffer reads (with broadcast index), internal tensors (produced inside the group) become `float _tN = expr;` locals, and the group output becomes the final buffer write. No intermediate buffer is ever allocated.
- **Non-elementwise singleton group**: delegates to `spec.lower_naive` exactly as `lower_naive` does.

The body indentation: continuation lines use `4 * (1 + rank)` spaces to match the depth `emit_loops` assigns to the innermost loop body, keeping generated code correctly formatted.

### Analytical savings

`analyze_savings(Graph&)` (called after `fuse()`):

```
bytes_unfused = Σ over nodes:   Σ bytes(input tensors) + bytes(output tensor)
bytes_fused   = Σ over groups:  Σ bytes(external reads) + Σ bytes(external writes)
bytes_saved   = bytes_unfused − bytes_fused
```

Internal-to-group tensors contribute to `bytes_unfused` (they'd round-trip in the unfused case) but not to `bytes_fused` (they stay in registers). The difference is the reported saving.

### Test results

| Test | Result |
|---|---|
| `test_elementwise_chain_fuses` | relu+sigmoid fuse to 1 kernel; fused output matches naive |
| `test_multi_consumer_no_fuse` | relu with 2 consumers stays in its own group |
| `test_analyze_savings` | gelu+relu+sigmoid: unfused=3 KB, fused=1 KB, saved=2 KB |
| `test_mlp_partial_fuse` | matmul singleton + add+relu fused; 2 kernels; matches naive |

### Deviations from design doc

- Shape equality (not just broadcast-compatibility) is required for fusion. Cross-shape elementwise fusion (e.g., fusing a [1,N] producer into a [M,N] consumer) is deferred to M3 when epilogue fusion makes it necessary.
- Prologue fusion (folding an elementwise producer into a contraction's load) is deferred to M3+.
- View folding into index expressions is deferred; views still lower as copy kernels.

---

---

## M3 — Tiled matmul + epilogue fusion  ✅

### What was built

Two-step fusion extended to support contraction anchors, tiled matmul kernel with K-loop blocking, and epilogue chains (elementwise nodes consuming matmul output) fused into the same kernel body using register-resident intermediates. A benchmark harness with chrono timing added.

### New / changed files

```
src/
  fusion.cpp   — extended with step 3: epilogue attach onto anchor groups
  lower.cpp    — has_contraction branch in lower_fused; tiled matmul body + epilogue emission
  codegen.cpp  — emit_bench_cpp: same kernels as emit_cpp, timed loop in main()
  runtime.cpp  — compile_and_bench: generates tc_bench.{cpp,exe}, parses TIMING_MS stdout

include/tc/
  codegen.hpp  — added emit_bench_cpp declaration
  runtime.hpp  — added BenchResult struct and compile_and_bench declaration

tests/
  test_m3.cpp  — test_epilogue_fuses, test_two_layer_mlp_fuses, test_benchmark
```

### Extended fusion algorithm

`fuse(Graph&)` now runs three steps:

1. Singleton assignment (same as M2).
2. **Step 3 — Epilogue attach**: for each elementwise node N whose input comes from an anchor group (a group containing a Contraction node), if the tensor has exactly one consumer and N's output shape matches the anchor's output shape, merge N's group into the anchor group.
3. **Step 4 — Elementwise vertical** (same as M2 step 2, but skips anchor groups): pure-elementwise producers fold into their consumer group.

Step 3 runs before step 4 so that epilogue nodes attached to an anchor cannot be further merged by step 4 (guarded by `group_has_anchor()`).

### Tiled matmul + epilogue lowering

The `has_contraction` branch in `lower_fused`:

- Outer loops: `[i(Parallel, M), j(Serial, N)]`.
- K-tiling: `KT = min(K, 256)`. Body contains a `ko`-tiled reduction loop with inner `k` loop, accumulating into `float acc`.
- Epilogue chain: each elementwise node in the group is emitted in topological order after the `ko` loop closes. External inputs use `bcast_idx_fused` against the `{i,j}` iteration space. Internal (group-produced) tensors use the register name: `"acc"` for the matmul output, `"_epN"` (where N is the TensorId) for prior epilogue outputs. The group's final output writes directly to the output buffer (`t5[i*N+j] = expr;`).
- No intermediate buffer is allocated for `acc` or any epilogue value.

Buffer reference fix: A and B buffer names and shapes are copied into local strings/variables before `prog.add_buffer(...)` is called, preventing use-after-reallocation of `std::vector<Buffer>` references.

### Benchmark harness

`emit_bench_cpp` generates a self-contained binary with:
- Same kernel functions as `emit_cpp`.
- `main()`: warmup loop (default 2 reps), timed loop (default 10 reps) using `std::chrono::high_resolution_clock`, sorts times, prints `TIMING_MS: X.XXX` to stdout.
- Timing variables named `_tc0`/`_tc1` (not `t0`/`t1`) to avoid shadowing buffer variables.

`compile_and_bench` in `runtime.cpp` generates `tc_bench.{cpp,exe}` (separate files from `tc_gen.*`), captures stdout via `bench_run.bat > bench_stdout.txt`, parses the `TIMING_MS:` line, and returns `BenchResult{ok, error, outputs, median_ms}`.

### Test results

| Test | Result |
|---|---|
| `test_epilogue_fuses` | matmul+add+relu → 1 fused kernel; output matches naive within 1e-3 |
| `test_two_layer_mlp_fuses` | 2-layer MLP → 2 fused kernels; output matches naive |
| `test_benchmark` (256×256) | unfused: 1793 KB traffic, fused: 769 KB traffic, saved: 1024 KB (57%); naive: 31.5 ms, fused: 53 ms |

The fused kernel is slower on this machine (no OpenMP, small matrix, Windows subprocess overhead). The memory traffic model confirms correct analytical savings. Runtime speedup is expected to be visible on larger shapes with OpenMP enabled (`/openmp` flag).

### Deviations from design doc

- OpenMP not enabled in `cl.exe` invocations (no `/openmp` flag). `#pragma omp parallel for` is present in generated code but ignored. This is why the fused kernel (which has more loop overhead from K-tiling) runs slower than naive.
- K-tiling uses a fixed `KT=min(K,256)`. For K≤256, this produces a single tile (trivial tiling). Actual cache-locality benefit requires K>256.
- Prologue fusion (elementwise producers before matmul) is deferred to M4+.

---

## M4 — Reduction epilogue fusion  ✅

### What was built

Extended the fusion pass and fused lowering to support reduction anchors (Sum, Max, Mean). Elementwise consumers of a reduction now fold into the same kernel via an `acc`-based epilogue, eliminating the intermediate buffer that would otherwise hold the reduction output. Softmax is intentionally excluded from anchor fusion — its three-pass body cannot cleanly inline an epilogue without recomputing exponentials.

### Changed / new files

```
src/
  fusion.cpp  — is_anchor() helper; excludes Softmax from anchor detection
  lower.cpp   — has_reduction flag + new else-if branch in lower_fused

tests/
  test_m4.cpp — 5 tests: sum+relu, mean+mul, max+add, softmax correctness, benchmark
```

### Fusion extension

`is_anchor(node)` returns true for `Contraction` or `Reduction && kind != Softmax`. Softmax groups remain singletons; their consumers, if any, stay as separate groups.

### Reduction epilogue lowering

The `has_reduction` branch in `lower_fused`:

- Outer loops: all non-reduced dimensions of the input (first dim Parallel).
- Body: `float acc = init; for (r=0; r<K; ++r) acc = update(acc, in[...]);`
  - Sum/Mean: `acc + in[...]`, then `acc /= K` for Mean.
  - Max: `(acc > in[...] ? acc : in[...])`.
- Epilogue: same register-resident chain pattern as the contraction branch.
- Buffer names and shapes copied before `prog.add_buffer` to prevent dangling-reference UB.

### Test results

| Test | Result |
|---|---|
| `test_sum_relu_fuses` | sum+relu → 1 kernel, matches naive |
| `test_mean_mul_fuses` | mean+scale → 1 kernel, matches naive |
| `test_max_add_fuses` | max+bias → 1 kernel, matches naive |
| `test_softmax_correctness` | each row sums to 1 within 1e-4 |
| `test_reduction_benchmark` (512×1024 sum+relu) | 4 KB saved; 0.64 ms naive → 0.54 ms fused, 1.19× speedup |

### Deviations from design doc

- Softmax epilogue fusion deferred — three-pass body would require recomputing exp to inline an epilogue.
- Prologue fusion (elementwise producers before the reduction) deferred to M5+.
