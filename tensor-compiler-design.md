


# Tensor Compiler — Design Document

**Scope of this document:** the compiler core only. Hosting, UI, and the visualization playground are deliberately out of scope here and covered separately later. This document is detailed enough to implement from, but stays at the design level (data structures, algorithms, invariants, rationale). The implementation is yours to write.

**Keep an implementation journal.** While building, record the implementation in a separate document (e.g. `implementation-notes.md`) — what you actually built per milestone, where you deviated from this design and why, bugs and fixes, benchmark numbers, and any decision you resolved from §12. This design doc stays the *intended* design; the journal is the *as-built* record. Keeping them separate means this document remains a clean spec, and the journal becomes both your debugging trail and the raw material for the project's README and your interview talking points.

---

## 0. Design principles (read first)

Four principles govern every decision below. When a later choice seems arbitrary, it traces back to one of these.

1. **Correctness is universal; optimization is selective.** The compiler must produce *correct* code for any graph it accepts. It must produce *fast* code only for the op classes we choose to optimize. Everything else falls back to a naive-but-correct lowering. This single principle is what keeps the project finite.

2. **Fast path / slow path.** Recognized patterns (elementwise chains, matmul epilogues, one reduction) get fusion, tiling, and good codegen. Unrecognized ops get one naive kernel each. This is exactly how TorchInductor (fallback to eager/ATen) and TVM (default schedules) actually work.

3. **Generalize over op *classes*, not individual ops.** You never write per-op fusion logic. You write per-*class* logic (elementwise, reduction, contraction, view). Adding a new activation is then a one-line op registration, not new compiler code.

4. **The unfused lowering is the correctness oracle.** Because every op has a naive lowering, the unoptimized pipeline is a reference implementation. The fused pipeline is correct iff its output matches the unfused output within floating-point tolerance. This gives you a free, total correctness test and shapes the testing strategy.

---

## 1. Goals and non-goals

### Goals
- Accept a neural network expressed as a computation graph of tensor ops.
- Lower it through progressive IR levels to executable code.
- Optimize via **operator fusion** (the flagship), plus tiling and parallelization.
- Emit and run code, and **measure** fused-vs-unfused performance.
- Report **analytical memory-traffic savings** computed directly from the IR.
- Generalize cleanly over op classes with a correct fallback for anything unrecognized.

### Non-goals (for the core)
- Training / autograd. **Inference only.** (Backward pass is a large, separate concern.)
- Competing with cuBLAS/TVM on absolute performance. The bar is "beats naive unfused execution, measurably, and demonstrates the mechanisms."
- Full op coverage. Coverage means *correctness* via fallback, not a tuned path per op.
- Dynamic shapes in v1 (static shapes only; dynamic is a stretch).
- Distribution / multi-GPU / quantization (all stretch or out of scope).

---

## 2. Language and dependency choices

| Decision | Choice | Rationale | Alternative |
|---|---|---|---|
| Core language | C++17 or C++20 | The project exists to demonstrate deep C++ systems ability; the IR, passes, and codegen are the signal | Python (common for compilers, but forfeits the C++ signal) |
| Codegen target (v1) | C++ + OpenMP source text, compiled by the system compiler at `-O3 -fopenmp` | Keeps you in one language, sidesteps machine-code encoding, still exercises tiling/vectorization/parallelism | Direct LLVM IR (more setup), raw x86 (encoding grind, low concept) |
| Codegen target (stretch) | CUDA C++ or Triton for GPU | This is where fusion's memory-bandwidth win is largest and most 2026-relevant | — |
| Reference for correctness | The compiler's own **unfused path** (primary) and optionally NumPy via a thin Python harness | Unfused path is always available and self-consistent | NumPy-only (requires a binding layer) |
| Build of emitted code | Shell out to `g++`/`clang++`, produce a shared object, `dlopen` it, or produce a standalone executable | Simple, portable, no JIT infrastructure needed in v1 | In-process JIT (stretch) |

**Recommendation:** C++ core, C++/OpenMP codegen, unfused-path oracle. Add a small NumPy cross-check only if you want a second independent oracle; it needs pybind11 and is optional.

---

## 3. Architecture overview

The canonical three-stage design (frontend capture → middle-end optimization → backend codegen), realized as **progressive lowering** through IR levels of descending abstraction.

```
 User graph (builder API)
        │
        ▼
 ┌─────────────────┐
 │   GRAPH IR      │  dataflow DAG of tensor ops; no loops, no memory
 │  (high level)   │  ← shape inference, canonicalization, FUSION
 └─────────────────┘
        │  lowering
        ▼
 ┌─────────────────┐
 │   LOOP IR       │  each fusion group → explicit loop nest over buffers
 │  (mid level)    │  ← tiling, loop ordering, vectorize/parallelize
 └─────────────────┘
        │  codegen
        ▼
 ┌─────────────────┐
 │  C++/OpenMP     │  source text: one function per kernel + a driver
 │  (target)       │
 └─────────────────┘
        │  system compiler + runtime
        ▼
   executable / .so  → run, time, verify
```

Two IR levels is the right amount for this project: a **graph IR** where fusion and graph rewrites live (the whole computation is visible as a DAG), and a **loop IR** where scheduling lives (loops, buffers, and memory become explicit). TVM makes the same split (Relay/graph and TIR/loop). Do not try to optimize on the AST-equivalent or codegen straight off the graph IR; the two levels exist because fusion reasons about *ops* and scheduling reasons about *loops*, and those are different shapes.

---

## 4. Core data model

### 4.1 Types and tensors

```
enum class DType { F32 };            // v1: f32 only. Add F16/I32 later.

struct Shape {
    std::vector<int64_t> dims;       // static, known at compile time in v1
    int64_t rank() const;
    int64_t numel() const;           // product of dims
};

using TensorId = uint32_t;           // stable id, indexes into the graph's tensor table

struct TensorInfo {
    TensorId   id;
    Shape      shape;
    DType      dtype = DType::F32;
    Layout     layout = Layout::RowMajorContiguous;  // v1: one layout
    // producer/consumers are stored in the graph, not here (keep tensor a value type)
};
```

**Layout note:** v1 is row-major contiguous everywhere. Layout optimization (choosing NHWC vs NCHW, inserting transposes lazily) is a real graph-level optimization but a stretch. Keep the `Layout` field so the design admits it later without a rewrite.

### 4.2 Op classes and ops

The **class** is the load-bearing abstraction. Fusion and lowering dispatch on class, not op.

```
enum class OpClass {
    Elementwise,   // unary/binary, 1 output elem ← 1 input elem(s): relu, add, mul, gelu, ...
    Reduction,     // collapse an axis: sum, max, mean, softmax's inner reduce
    Contraction,   // matmul family: matmul, batched matmul, (conv via im2col later)
    View,          // metadata-only reshape/transpose/slice/broadcast (often zero-compute)
    Opaque         // anything with no special handling → fallback naive lowering
};

enum class OpKind { Relu, Add, Mul, Sub, Sigmoid, Tanh, Gelu, Exp,
                    MatMul, Sum, Max, Mean, Softmax,
                    Reshape, Transpose, Broadcast,
                    /* ... */ Unknown };
```

Each `OpKind` registers, in an **op registry**, everything the compiler needs to treat it generically:

```
struct OpSpec {
    OpKind   kind;
    OpClass  cls;
    int      arity;                                   // number of tensor inputs
    // shape rule: given input shapes + attrs, produce output shape(s)
    ShapeFn  infer_shape;
    // scalar semantics for elementwise ops: the per-element expression,
    // emitted into fused kernel bodies. e.g. Relu -> "max({0}, 0.f)"
    ScalarExprFn scalar_expr;                         // null for non-elementwise
    // naive lowering: produce a correct standalone loop nest (the fallback + oracle)
    LowerFn  lower_naive;
    // optional optimized lowering hook (matmul tiling, reduction, ...)
    LowerFn  lower_optimized;                         // null → use naive
};
```

Registering a new elementwise op = one `OpSpec` with a `scalar_expr` and a shape rule (elementwise shape rule is shared). That is the concrete payoff of principle 3.

### 4.3 Graph IR

```
struct Node {
    NodeId              id;
    OpKind              kind;
    OpClass             cls;                  // cached from OpSpec for fast dispatch
    std::vector<TensorId> inputs;             // producer edges (tensor ids)
    std::vector<TensorId> outputs;            // usually 1
    Attributes          attrs;                // axis (reduction), transpose flags (matmul), etc.
    // fusion bookkeeping (filled by the fusion pass):
    GroupId             group = kNoGroup;
};

struct Graph {
    std::vector<Node>          nodes;         // kept in topological order (invariant)
    std::vector<TensorInfo>    tensors;       // tensor table, indexed by TensorId
    std::vector<TensorId>      graph_inputs;  // external inputs (params, activations in)
    std::vector<TensorId>      graph_outputs; // external outputs
    // adjacency derived on demand or maintained incrementally:
    //   producer_of[TensorId]  -> NodeId
    //   consumers_of[TensorId] -> small vector<NodeId>
};
```

**Invariants (assert these; they catch most bugs):**
- Nodes are topologically ordered: every input tensor is produced by an earlier node or is a graph input.
- Every non-input tensor has exactly one producer node (SSA-like: single definition).
- Every tensor's `Shape`/`DType` is filled after shape inference and never changes afterward.
- The graph is acyclic.

**Builder API** (the frontend for v1 — you construct graphs in code):

```
Graph g;
auto x = g.input({M, K});
auto W = g.input({K, N});
auto b = g.input({N});
auto t = g.matmul(x, W);          // returns TensorId of the output
auto u = g.add(t, b);             // broadcast add
auto y = g.relu(u);
g.mark_output(y);
```

Each builder method: looks up the `OpSpec`, runs `infer_shape` to create the output `TensorInfo`, appends a `Node`, returns the output `TensorId`. ONNX / torch.fx ingestion is a *stretch* frontend that emits into this same builder; design the builder as the single construction chokepoint so alternative frontends are additive.

### 4.4 Loop IR (tensor IR)

Produced by lowering. Represents computation as explicit loop nests over explicit buffers. This is where memory becomes real.

```
struct Buffer {
    BufferId id;
    Shape    shape;
    DType    dtype;
    enum { External, Intermediate, Output } role;   // drives allocation + traffic accounting
};

// A loop nest computing one kernel (one fusion group, or one fallback op)
struct LoopNest {
    KernelId              id;
    std::vector<Loop>     loops;      // ordered outer→inner; each has var, extent, tag
    std::vector<BufferId> reads;      // external buffers read
    std::vector<BufferId> writes;     // external buffers written
    Stmt                  body;       // expression tree over induction vars + buffer accesses
};

struct Loop {
    std::string var;                  // "i", "j", "k", or tiled "io","ii"
    int64_t     extent;
    enum { Serial, Parallel, Vectorize, Reduce } tag;
    // tiling produces split loops with a tile-size attribute
};
```

The `body` for a **fused elementwise group** is the composed scalar expression: e.g. for `relu(add(matmul_result, b))` the epilogue body is `max(acc + b[j], 0.f)` evaluated with `acc`/`b[j]` **in registers**, stored to the output once. That register-residency of intermediates is fusion physically happening. Nothing in the loop IR writes the intermediate to a buffer.

---

## 5. Pass pipeline

Ordered. Each pass is a pure function `Graph → Graph` (or `Graph → LoopProgram` for lowering), which keeps them testable in isolation.

```
build → infer_shapes → validate → canonicalize → fuse
      → lower → schedule → codegen → (compile → run/verify)
                                   ↘ analyze_savings (off the fused graph IR)
```

### 5.1 Shape & dtype inference
Walk nodes in topological order. For each, call `OpSpec::infer_shape(input_shapes, attrs)` to fill the output `TensorInfo`. Fail loudly on mismatch (e.g. matmul inner dims unequal, non-broadcastable elementwise). After this pass every tensor has a concrete static shape — the rest of the compiler assumes this.

Shape rules per class:
- **Elementwise:** broadcast the input shapes (NumPy broadcasting rules); output = broadcasted shape.
- **Contraction (matmul):** `[M,K] x [K,N] → [M,N]` (plus batch dims, plus transpose attrs).
- **Reduction:** remove or keep-dim the reduced axis per attr.
- **View:** per-op (reshape validates numel equality; transpose permutes dims).

### 5.2 Validate
Assert all invariants from §4.3. This pass is cheap insurance; run it after every mutating pass in debug builds.

### 5.3 Canonicalize (light)
Small, safe graph rewrites that make fusion more effective. Keep this minimal in v1:
- **Constant folding:** ops with all-constant inputs computed at compile time.
- **Dead-op elimination:** nodes whose outputs reach no graph output are removed (post-order reachability from `graph_outputs`).
- **Identity/algebraic simplification (optional):** `add(x, 0)`, `mul(x, 1)` removed.

Do **not** over-invest here; it's not the signal. Fusion is.

### 5.4 Fusion (the flagship pass)

**Concept it implements:** merge adjacent ops that can share a kernel so intermediate tensors stay in registers instead of round-tripping to memory. The win is largest for memory-bound ops (elementwise, reductions) and for epilogues riding on a compute-bound anchor (matmul).

**Output:** a partition of the graph's nodes into **fusion groups**. Each group becomes exactly one kernel. Every node ends in exactly one group (a singleton group = an unfused op, including all `Opaque` fallbacks).

**Fusion rules, per class pairing:**

| Producer | Consumer | Rule |
|---|---|---|
| Elementwise | Elementwise | Fuse (vertical fusion) if legal (see below). Composes scalar expressions. |
| Contraction / Reduction | Elementwise | **Epilogue fusion:** the elementwise consumer(s) fold into the anchor kernel, applied before the store. |
| Elementwise | Contraction / Reduction | **Prologue fusion** (stretch): producer folds into the anchor's load. Skippable in v1. |
| Reduction | Reduction | Do **not** fuse across the reduction boundary in v1. Each reduction is a group boundary. |
| View | anything | Views are metadata-only where possible: fold a transpose/reshape into the consumer's index expression rather than emitting a copy. If not foldable, it's a real copy kernel. |
| anything | (multi-consumer) | See legality. |

**Legality conditions (must all hold to fuse a producer into a consumer group):**
1. **Shape compatibility** — the producer's output participates elementwise-compatibly in the consumer (same iteration space, modulo broadcast).
2. **Single consumer, or safe duplication** — if the producer's output feeds *multiple* consumers, fusing it into one forces a choice: either (a) leave a group boundary and materialize the intermediate (safe default in v1), or (b) duplicate the producer's compute into each consumer group (recompute-to-save-memory; a real technique, but gate it behind a cheap-op check — only duplicate elementwise, never a matmul).
3. **No cycles** — fusing must not create a dependency cycle among groups. (Fusing A→B is illegal if there's another path A→…→B through a *different* group, because the merged group would depend on itself. Check via the group DAG.)
4. **Class rule allows it** — per the table above.

**Algorithm (greedy, anchor-oriented — concrete enough to implement):**

```
1. Assign every node its own singleton group.
2. Identify anchors = Contraction and Reduction nodes.
3. Epilogue attach:
   for each anchor A (in topological order):
     repeatedly, for each elementwise consumer C of A's group's output
     that is single-consumer and legal:
        merge C's group into A's group.
        (keep extending down the elementwise chain)
4. Elementwise vertical fusion (for elementwise not already attached to an anchor):
   traverse nodes in topological order;
   for each elementwise node N with a single-consumer, legal elementwise producer P
   that is not an anchor group:
        merge P's group into N's group (or N into P — pick one direction and be consistent).
5. Views: fold into neighbor index expressions where legal; else leave as copy groups.
6. Everything still in a singleton group = its own kernel (this includes all Opaque fallbacks).
7. Verify the resulting group DAG is acyclic (legality #3) — if a merge created a cycle,
   it was illegal; the check in step-legality should have prevented it, assert here.
```

This greedy scheme is intentionally simple and *defensible*: you can explain exactly why each merge happened. Cost-model-driven fusion (search over partitions) is a stretch; name it as future work in interviews rather than implementing it.

**Interview line this pass buys you:** "Fusion is a graph-IR partitioning pass: greedy epilogue-attach onto compute-bound anchors plus vertical fusion of elementwise chains, gated by shape, single-consumer, and acyclicity legality. Multi-consumer intermediates either force a boundary or duplicate cheap elementwise compute."

### 5.5 Lowering (Graph IR → Loop IR)

Per fusion group, dispatch on the group's *anchor class*:

- **Elementwise group** → one loop nest over the output shape. The body is the composed scalar expression built by substituting each op's `scalar_expr`, threading intermediates as SSA registers (C++ locals), never as buffers. Broadcasts become index arithmetic (stride-0 on broadcast dims).
- **Contraction (+ epilogue) group** → a **tiled matmul** loop nest (see scheduling). The epilogue expression is applied to each output element `acc` before the store: `C[i][j] = epilogue(acc)`, where `epilogue` is the composed elementwise chain (e.g. `relu(acc + b[j])`).
- **Reduction (+ pro/epilogue) group** → a loop nest with an inner reduction loop tagged `Reduce`, an accumulator register, optional fused elementwise on the loaded inputs (prologue) and on the reduced result (epilogue). Softmax = max-reduce, then exp/sum (a second pass), then divide — design it as the standard two-pass numerically-stable form.
- **Opaque / fallback group** → call `OpSpec::lower_naive`. One correct standalone loop nest, no fusion. **This path must exist for every op** and is the correctness oracle.

Buffer allocation: group-external inputs/outputs become `External`/`Output` buffers; anything fully internal to a group never becomes a buffer at all (that's the point). Cross-group intermediates become `Intermediate` buffers with liveness-based reuse (a simple linear-scan over the group DAG's topological order; stretch to make it tight).

### 5.6 Schedule (Loop IR optimization)

Keep v1 schedules **fixed and hand-written per class**, not searched:

- **Elementwise:** collapse the nest to 1D over `numel`, tag outer `Parallel` (`#pragma omp parallel for`), inner `Vectorize` (`#pragma omp simd`).
- **Matmul:** classic cache **tiling** — split M, N, K into tiles (`Mt, Nt, Kt`), order loops `(io, jo, ko, ii, ji, ki)` for locality, parallelize the outer `io` (or `io,jo`), vectorize the innermost `ji`. Fixed tile sizes chosen for a typical L1/L2 (e.g. 64/64/256) with a comment explaining the cache reasoning. Autotuning tile sizes is a labeled stretch goal.
- **Reduction:** parallelize across the non-reduced axes; keep the reduction loop serial with a register accumulator; vectorize where the reduction is associative and the axis allows.

Tiling is worth doing well because "I tiled the matmul for cache locality and here's the before/after" is a strong, concrete result and directly relevant to storage/perf roles.

### 5.7 Codegen (Loop IR → C++/OpenMP source)

Emit, as text:
- One `void kernel_<id>(const float* in0, ..., float* out, /*dims*/)` per `LoopNest`, with the loops, pragmas, and body rendered from the loop IR.
- A `run(const Inputs&, Outputs&)` driver that allocates intermediate buffers, calls kernels in group-topological order, and wires buffers.
- A tiny generated `main`/benchmark entry (or expose `run` from a `.so`).

Then shell out: `g++ -O3 -march=native -fopenmp generated.cpp -shared -fPIC -o kernel.so`, `dlopen`, resolve `run`, execute.

**Two codegen modes from the same loop IR**, which you need for the demo:
- **Fused mode:** emit the fused kernels as designed.
- **Unfused mode:** emit every op as its own naive kernel (skip the fusion pass, or emit each singleton). This is both the correctness oracle *and* the performance baseline for the savings demo.

### 5.8 Analyze savings (off the fused graph IR — no execution needed)

Pure analysis producing the "money shot" number.

**Memory-traffic model:**
```
bytes(T)            = T.shape.numel() * sizeof(dtype)
traffic_unfused(G)  = Σ over nodes n:  Σ bytes(inputs(n)) + bytes(output(n))
traffic_fused(G)    = Σ over groups g: Σ bytes(external_inputs(g)) + bytes(external_outputs(g))
saved_bytes         = traffic_unfused − traffic_fused
```
Intermediates internal to a group contribute to `traffic_unfused` (they'd round-trip) but *not* to `traffic_fused` (they stay in registers) — that difference is the reported saving. For memory-bound regions, estimated speedup ≈ `traffic_unfused / traffic_fused`. Label it clearly as an **analytical estimate**; the measured wall-clock number (from §5.7 fused-vs-unfused runs) is the separate, stronger demo you add once codegen executes.

---

## 6. The fallback path (correctness for arbitrary graphs)

This is the answer to "what if a graph has an op we didn't optimize."

- Every `OpKind` — including `Unknown`/`Opaque` — has a `lower_naive` producing a correct standalone kernel.
- The fusion pass only ever *replaces a group with an equivalent fused kernel*; an unrecognized op simply never joins a group and lowers naively.
- Therefore the compiler is **correct on any graph it can shape-infer**, and fast on the recognized classes. Correctness is total; optimization is partial. This is the production pattern (Inductor's eager fallback), stated as a guarantee rather than a gap.

For a genuinely unknown op you can't even shape-infer, fail at the `infer_shapes` pass with a clear error — that's an accept/reject boundary, not a miscompile.

---

## 7. Runtime and benchmark harness

- **Correctness harness:** for a test graph, run **unfused** and **fused** modes on the same random inputs; assert max abs/rel error within fp tolerance (e.g. `1e-4` for f32, looser after many fused ops). Optionally cross-check unfused against NumPy for a second independent oracle.
- **Performance harness:** time `run()` for unfused vs fused over N iterations (warm up first, report median, pin threads). Emit a table: op-group, unfused ms, fused ms, speedup, plus the analytical saved-bytes from §5.8.
- **Determinism:** fixed seeds; note that fp reduction order changes results slightly (that's why tolerance, not equality).

---

## 8. Testing strategy

Layered, cheap-to-expensive:
1. **Shape inference tests** — per op, correct output shapes and correct rejection of bad shapes.
2. **Pass unit tests** — canonicalize removes dead ops; fusion groups a known chain the expected way; illegal fusions are refused (multi-consumer, cycle).
3. **Numerical correctness** — fused == unfused within tolerance across a suite of graphs (elementwise chain, matmul+bias+relu, softmax, a small MLP, a transformer block). This is principle 4 doing the heavy lifting.
4. **Perf regression (optional)** — fused speedup stays above a floor on the flagship graph.

---

## 9. Phased build plan (each milestone is independently demoable)

| Milestone | Deliverable | Demo |
|---|---|---|
| **M0 — Graph IR** | Types, tensor table, `Node`/`Graph`, builder API, op registry, shape inference, validate, a text/DOT printer | Build a graph in code, print it as text and as a DOT graph |
| **M1 — Naive end-to-end** | Naive lowering for all v1 ops, C++/OpenMP codegen (unfused mode), compile+run+verify harness | A small MLP compiles and runs correctly; this is the oracle + fallback |
| **M2 — Elementwise fusion** | Fusion pass (elementwise vertical only), fused lowering, analytical savings | An elementwise chain fuses to one kernel; report saved bytes; fused==unfused |
| **M3 — Matmul + epilogue (flagship)** | Tiled matmul schedule, epilogue fusion onto the anchor | `matmul→bias→relu` fused, benchmarked vs unfused, **measurably faster** |
| **M4 — Reduction** | Softmax or layernorm with reduction-boundary + pro/epilogue fusion | A transformer-ish block with softmax compiles, fuses, verifies |
| **M5 — Real net + measured savings** | Buffer reuse, benchmark polish, run a real small model end-to-end | "Compiled a real MLP/transformer block, here's the fused-vs-unfused wall-clock and the memory saved" |
| **Stretch** | GPU (CUDA/Triton) codegen; ONNX/torch.fx frontend; tile autotuning; prologue fusion; layout opt; the visualization UI (separate doc) | — |

Build order is strict: **M1 before M2** (you need the oracle and the baseline before you can trust or measure fusion). Do not start fusion before naive execution works end-to-end.

---

## 10. Key design decisions and their rationale (interview cheat-sheet)

- **Two IRs, not one.** Fusion reasons about ops (graph IR); scheduling reasons about loops and memory (loop IR). Different concerns, different representations, progressive lowering between them.
- **Op *classes* drive everything.** Fusion legality and lowering dispatch on class, so the op set scales without new compiler code.
- **Fast path / slow path.** Optimize recognized classes; naive-lower everything else. Correctness universal, optimization selective — the production pattern.
- **Unfused path is the oracle.** Free, total correctness test; also the perf baseline. This is why M1 precedes M2.
- **Greedy anchor-oriented fusion.** Epilogue-attach onto compute-bound anchors + vertical elementwise fusion, gated by shape/single-consumer/acyclicity. Simple and fully explainable; cost-model search is named as future work.
- **Register-resident intermediates.** Fusion's whole win is that internal intermediates never become buffers — visible directly in the loop IR (they're C++ locals, not `Buffer`s).
- **Analytical + measured savings.** Traffic model gives an early, execution-free number; wall-clock gives the strong later number. Both labeled honestly.
- **Static shapes v1.** Dynamic shapes need guards/specialization (à la TorchDynamo size hints) — deferred deliberately, not overlooked.

---

## 11. Reference works (grounding, in reading order)

Links verified where noted. "Paper" links go to the primary source; "code" links go to the reference implementation you should skim to sanity-check the designs in this document against a real compiler.

### 11.1 Papers (reading order)

1. **Halide** — Ragan-Kelley et al., PLDI 2013. Separation of *algorithm* from *schedule*, the idea the schedule layer (§5.6) is built on.
   - Project + paper: https://halide-lang.org/
2. **TVM** — Chen et al., OSDI 2018. The canonical three-stage end-to-end tensor compiler; graph IR (Relay) + loop IR (TIR). Your project is a small TVM.
   - Paper (arXiv): https://arxiv.org/abs/1802.04799
   - Paper (USENIX OSDI '18): https://www.usenix.org/conference/osdi18/presentation/chen
   - Relay graph-IR paper: https://arxiv.org/abs/1904.08368
3. **PyTorch 2 / TorchInductor / TorchDynamo** — Ansel et al., ASPLOS 2024. The modern production realization: bytecode graph capture, loop-level IR, Triton/C++ codegen, eager fallback. The "2026" reference and the source of the fallback pattern (§6).
   - Paper (PDF): https://docs.pytorch.org/assets/pytorch2-2.pdf
4. **XLA** — Google / OpenXLA. Whole-program fusion + layout optimization; HLO IR.
   - Docs: https://openxla.org/xla
5. **MLIR** — Lattner et al., 2020/2021. Multi-level IR infrastructure / progressive lowering; the framework modern stacks build on. Know the concept even if you don't build on it.
   - Paper (arXiv): https://arxiv.org/abs/2002.11054
   - Docs: https://mlir.llvm.org/
6. **Triton** — Tillet, Kung, Cox, MAPL 2019. Tiled GPU-kernel language; the codegen target of Inductor and the natural stretch backend (§5.7).
   - Paper (ACM DOI): https://doi.org/10.1145/3315508.3329973
   - Docs: https://triton-lang.org/

### 11.2 Source code (skim to validate this doc's designs)

- **TVM** — https://github.com/apache/tvm  (Relay = graph IR, TIR = loop IR; compare against §4)
- **PyTorch / TorchInductor** — https://github.com/pytorch/pytorch  (Inductor lives under `torch/_inductor`; its loop-level IR and fallback are the closest match to §5–6)
- **Triton** — https://github.com/triton-lang/triton
- **OpenXLA / XLA** — https://github.com/openxla/xla
- **MLIR (in LLVM)** — https://github.com/llvm/llvm-project  (the `mlir/` subtree)
- **IREE** (MLIR-based end-to-end) — https://github.com/iree-org/iree

### 11.3 Source honesty

The three-stage structure, Relay/TIR split, XLA fusion, Inductor's capture-and-fallback, and Triton-as-target are grounded in the papers/pages above (verified this session). The classical mechanisms (fusion, tiling, IR levels, single-definition tensors) are standard compiler knowledge from training, not re-derived from a fetch. **The concrete design in this document — the exact struct layouts, the greedy anchor-oriented fusion algorithm, the legality conditions, the memory-traffic formula, and the milestone plan — is original synthesis for this project, not lifted from any one paper or repo.** It is a sound, buildable starting design, not a validated one. Before committing to the struct designs, skim TVM's Relay/TIR and Inductor's IR (links above) and correct anything here that contradicts a real implementation; when they disagree with this doc, trust them.

Implementing the *ideas* in these works in your own code is standard practice and carries no copyright issue; the code and prose you write are yours. Do not copy source verbatim from the repos above — read them to understand, then implement independently.

---

## 12. Open questions to resolve before/while coding

1. Exact v1 op set (proposal: `matmul`, `add`, `mul`, `sub`, `relu`, `sigmoid`, `gelu`, `exp`, `sum`, `max`, `softmax`, `reshape`, `transpose`, `broadcast`). Confirm and freeze.
2. Broadcasting semantics — adopt NumPy rules exactly, or a restricted subset for v1?
3. Multi-consumer policy default — materialize (boundary) vs duplicate-cheap-elementwise. Proposal: materialize in v1, add duplication in M3+.
4. Buffer reuse aggressiveness in v1 — none (allocate per intermediate) vs simple liveness reuse. Proposal: none in M1–M4, add in M5.
5. Second oracle — add the NumPy cross-check (pybind11) or rely solely on the unfused path? Proposal: unfused-only first.
