# Tensor Compiler — Project Status

Complete reference for picking up UI and hosting work in a new session.

---

## What this project is

A miniature tensor compiler in C++17 modeled on TVM/TorchInductor. It takes a graph of tensor operations, applies fusion optimizations, lowers to a Loop IR, emits standalone C++ source, compiles it with cl.exe (Windows) or g++ (Linux), runs it, and returns the output tensors. All 6 milestones are complete.

---

## Repository

- **Location**: `K:\tensor_compiler`
- **Git remote**: `https://github.com/ManojSK462/tensor_compiler.git`
- **Branch**: `main`
- **Build**: CMake + NMake, build dir at `K:\tensor_compiler\build`
- **Build command**: `cmake --build K:\tensor_compiler\build`

---

## Full file listing

```
include/tc/
  types.hpp        — DType (F32), Shape, TensorId, TensorInfo
  ops.hpp          — OpClass, OpKind, Attributes, OpSpec, OpRegistry
  graph.hpp        — NodeId, GroupId, Node, Graph (builder API)
  passes.hpp       — infer_shapes(Graph&), validate(Graph&)
  printer.hpp      — print_text, print_dot
  loop_ir.hpp      — Buffer (with slot_id), Loop, LoopNest, LoopProgram
  lower.hpp        — lower_naive(Graph&), lower_fused(Graph&)
  codegen.hpp      — emit_cpp(LoopProgram&), emit_bench_cpp(LoopProgram&, warmup, reps)
  runtime.hpp      — RunResult, BenchResult, compile_and_run, compile_and_bench
  fusion.hpp       — fuse(Graph&)
  analyze.hpp      — TrafficStats, analyze_savings(Graph&)
  buffer_reuse.hpp — assign_buffer_slots(LoopProgram&)

src/
  ops.cpp          — OpRegistry, all 15 op registrations + lower fns
  graph.cpp        — Graph builder methods, adjacency maps
  passes.cpp       — shape inference, validation
  printer.cpp      — text + DOT printers
  loop_ir.cpp      — LoopProgram::add_buffer
  lower.cpp        — lower_naive + lower_fused (all branches)
  codegen.cpp      — emit_cpp, emit_bench_cpp (slot-aware)
  runtime.cpp      — compile_and_run, compile_and_bench (Windows/MSVC)
  fusion.cpp       — fuse() with epilogue attach + EW vertical
  analyze.cpp      — memory traffic model
  buffer_reuse.cpp — liveness + greedy slot assignment

tests/
  test_m0.cpp      — 7 graph IR tests
  test_m1.cpp      — naive pipeline correctness
  test_m2.cpp      — elementwise fusion, 4 tests
  test_m3.cpp      — matmul epilogue fusion + benchmark
  test_m4.cpp      — reduction epilogue fusion + softmax + benchmark
  test_m5.cpp      — buffer reuse + attention + e2e benchmark

generated/         — runtime output dir (gitignored): tc_gen.cpp, tc_gen.exe, *.bat, *.bin

implementation-notes.md   — per-milestone design notes and deviations
hosting.md                — what to change in runtime.cpp for Linux deployment
project-status.md         — this file
```

---

## The two-IR pipeline

```
User builds Graph (builder API)
        ↓
infer_shapes(g) + validate(g)      — shape propagation + sanity checks
        ↓
fuse(g)                            — sets node.group, merges groups
        ↓
lower_fused(g) → LoopProgram       — one LoopNest per group
        ↓
assign_buffer_slots(prog)          — optional: compress intermediate mallocs
        ↓
emit_cpp(prog) → std::string       — standalone C++ source
        ↓
compile_and_run(src, prog, inputs) — writes tc_gen.cpp, bat+cl.exe, captures output
        ↓
RunResult { ok, error, outputs }
```

---

## Graph IR (include/tc/graph.hpp)

Builder API — all methods return a TensorId:

```cpp
Graph g;
auto t0 = g.input({M, K});                          // external input
auto t1 = g.matmul(t0, W, trans_a=false, trans_b=false);
auto t2 = g.add(t1, bias);
auto t3 = g.relu(t2);
auto t4 = g.sum(t3, axis=1);
auto t5 = g.softmax(t3, axis=-1);
g.mark_output(t3);                                  // designates graph output
infer_shapes(g);
validate(g);
```

All 15 ops: matmul, add, mul, sub, relu, sigmoid, tanh_, gelu, exp_, sum, max, mean, softmax, reshape, transpose, broadcast.

OpClass values: Elementwise, Reduction, Contraction, View, Opaque.

Node.group is kNoGroup until fuse() is called.

---

## Fusion pass (src/fusion.cpp)

`fuse(Graph& g)` runs two steps:

**Step 1 — Singleton init**: every node gets group = node.id.

**Step 2 — Epilogue attach** (for contraction AND non-softmax reduction anchors):
For each elementwise node N, if its input comes from an anchor group AND has exactly one consumer AND shape matches anchor output → merge N's group into the anchor group.

**Step 3 — EW vertical** (pure elementwise only, skips anchor groups):
For each elementwise node N, if its elementwise producer P has one consumer AND same shape → merge P's group into N's group.

`is_anchor(node)` = Contraction OR (Reduction AND kind != Softmax).

Softmax is intentionally NOT an anchor (its 3-pass body can't inline an epilogue).

---

## Fused lowering (src/lower.cpp)

`lower_fused(Graph& g)` iterates groups in topological order. For each group:

**all_ew branch** (all Elementwise):
- One loop nest over the output shape
- Body: inline scalar expr chain, register-resident intermediates as `float _tN = expr;`
- No intermediate buffer allocated

**has_contraction branch** (Contraction + optional Elementwise epilogue):
- Outer loops: [i(Parallel, M), j(Serial, N)]
- Body: K-tiled reduction (`KT = min(K, 256)`), then epilogue chain using `acc` and `_epN` temporaries
- Key fix: A_name/B_name copied before `prog.add_buffer()` to avoid dangling-reference UB

**has_reduction branch** (Sum/Max/Mean + optional Elementwise epilogue):
- Outer loops: all non-reduced dims (first Parallel)
- Body: `float acc = init; for r: acc = update(acc, in[...]);` then same epilogue chain
- Mean adds `acc /= N;` before epilogue

**else branch** (Softmax singleton, View, unknown):
- One call to `spec.lower_naive()` per member node
- Allocates intermediate buffers normally

---

## Buffer reuse (src/buffer_reuse.cpp)

`assign_buffer_slots(LoopProgram& prog)`:
1. Compute liveness `[first_write_kernel, last_read_kernel]` for each Intermediate buffer
2. Sort by first_write, greedy interval coloring
3. Assign `buffer.slot_id` (int32_t, default -1 = no slot)

`emit_cpp` / `emit_bench_cpp` check slot_id:
- If any intermediate has slot_id >= 0: emit `float* _slot_N = malloc(max_elems);` per slot, then `float* tN = _slot_K;` assignments
- Free slots at the end, not individual buffer pointers

For a linear N-group pipeline, max 2 slots needed regardless of depth.

---

## Codegen (src/codegen.cpp)

`emit_cpp(prog)` → standalone C++ binary:
- `static void kernel_N(const float* reads..., float* writes...)` per LoopNest
- `main(argc, argv)`: reads inputs from binary files, runs kernels, writes outputs to binary files
- Slot-aware malloc (see above)

`emit_bench_cpp(prog, warmup=2, reps=10)` → same kernels, but main() wraps calls in chrono timing loop, prints `TIMING_MS: X.XXX` to stdout.

Timing variables named `_tc0`/`_tc1` (not `t0`/`t1`) to avoid shadowing buffer args.

---

## Runtime (src/runtime.cpp)

**Windows-specific** — uses vcvars64.bat + cl.exe + .bat files.

`compile_and_run(src, prog, inputs)`:
- Writes `generated/tc_gen.cpp`, `generated/compile.bat`, `generated/run.bat`
- Compile: `cl.exe /O2 /openmp /nologo`
- Run: bat passes binary input/output paths as argv
- Returns `RunResult { ok, error, outputs }`

`compile_and_bench(prog, inputs, warmup=2, reps=10)`:
- Same but uses `tc_bench.cpp/.exe`, `bench_compile.bat`, `bench_run.bat`
- Bench binary stdout redirected to `bench_stdout.txt`
- Parses `TIMING_MS:` line from stdout
- Returns `BenchResult { ok, error, outputs, median_ms }`

**OpenMP**: `/openmp` flag added. `#pragma omp parallel for` on the first (outer) loop of every kernel is now respected. 256×256 matmul: naive 3.8 ms (was 31 ms before OpenMP).

---

## How to run tests (Windows)

```powershell
# from K:\tensor_compiler\build\
.\test_m0.exe
.\test_m1.exe
.\test_m2.exe
.\test_m3.exe
.\test_m4.exe
.\test_m5.exe
```

No vcvars64 call needed to run the test exes themselves. vcvars64 is called internally by the generated bat files when compile_and_run fires.

---

## What the UI needs to call

The compiler's public API is entirely in `include/tc/`. A web server wrapping this would:

1. Accept a graph description (JSON or custom DSL) from the UI
2. Build a `Graph` using the builder API
3. Call `infer_shapes`, `validate`, `fuse`, `lower_fused`, `assign_buffer_slots`
4. Call `emit_cpp` to get the source string (for display in UI)
5. Call `compile_and_run` or `compile_and_bench` with user-provided input tensors
6. Return outputs/timing to the UI

The compiler itself is a static library (`tc.lib` on Windows). A thin web server (e.g. a Python Flask app calling a C++ subprocess, or a C++ HTTP server like crow/pistache) would link against `tc` and expose endpoints.

---

## Linux hosting changes (summary — see hosting.md for full detail)

Only `src/runtime.cpp` needs changing:
- Replace bat files with shell scripts
- Replace `cl.exe /O2 /openmp` with `g++ -O2 -fopenmp -std=c++17`
- Replace `K:\tensor_compiler\generated\` with `/var/tc/generated/`
- Add `chmod()` calls for the shell scripts

Everything else compiles unchanged on Linux with `cmake -DCMAKE_CXX_COMPILER=g++`.

---

## Known limitations / deferred work

- Softmax epilogue fusion not implemented (3-pass body complexity)
- Prologue fusion (EW before matmul/reduction) not implemented
- K-tiling only helps when K > 256 (KT = min(K, 256))
- No batched / multi-head attention
- View ops (reshape/transpose/broadcast) still lower as copy kernels (not folded into index expressions)
- No autotuning of tile sizes
- Static shapes only — dynamic shapes not supported
