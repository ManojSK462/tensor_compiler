#include "tc/types.hpp"
#include "tc/ops.hpp"
#include "tc/graph.hpp"
#include "tc/passes.hpp"
#include "tc/fusion.hpp"
#include "tc/analyze.hpp"
#include "tc/lower.hpp"
#include "tc/codegen.hpp"
#include "tc/runtime.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

static bool near(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

static std::vector<float> rand_vec(int64_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static void test_sum_relu_fuses() {
    const int64_t M = 8, N = 16;

    Graph g;
    auto x = g.input({M, N});
    g.mark_output(g.relu(g.sum(x, 1)));
    infer_shapes(g);
    validate(g);

    fuse(g);

    assert(g.nodes[0].group == g.nodes[1].group);

    LoopProgram prog_fused = lower_fused(g);
    assert(prog_fused.kernels.size() == 1);

    LoopProgram prog_naive = lower_naive(g);

    auto xd = rand_vec(M * N, 1);

    RunResult fr = compile_and_run(emit_cpp(prog_fused), prog_fused, {xd});
    RunResult nr = compile_and_run(emit_cpp(prog_naive), prog_naive, {xd});

    assert(fr.ok && nr.ok);
    assert(fr.outputs[0].size() == static_cast<size_t>(M));
    for (int64_t i = 0; i < M; ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    std::cout << "test_sum_relu_fuses: PASS (sum+relu -> 1 kernel, matches naive)\n";
}

static void test_mean_mul_fuses() {
    const int64_t M = 6, N = 12;

    Graph g;
    auto x     = g.input({M, N});
    auto scale = g.input({M});
    g.mark_output(g.mul(g.mean(x, 1), scale));
    infer_shapes(g);
    validate(g);

    fuse(g);

    assert(g.nodes[0].group == g.nodes[1].group);

    LoopProgram prog_fused = lower_fused(g);
    assert(prog_fused.kernels.size() == 1);

    LoopProgram prog_naive = lower_naive(g);

    auto xd  = rand_vec(M * N, 2);
    auto sd  = rand_vec(M,     3);

    RunResult fr = compile_and_run(emit_cpp(prog_fused), prog_fused, {xd, sd});
    RunResult nr = compile_and_run(emit_cpp(prog_naive), prog_naive, {xd, sd});

    assert(fr.ok && nr.ok);
    for (int64_t i = 0; i < M; ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    std::cout << "test_mean_mul_fuses: PASS (mean+mul -> 1 kernel, matches naive)\n";
}

static void test_max_add_fuses() {
    const int64_t M = 5, N = 20;

    Graph g;
    auto x    = g.input({M, N});
    auto bias = g.input({M});
    g.mark_output(g.add(g.max(x, 1), bias));
    infer_shapes(g);
    validate(g);

    fuse(g);

    assert(g.nodes[0].group == g.nodes[1].group);

    LoopProgram prog_fused = lower_fused(g);
    assert(prog_fused.kernels.size() == 1);

    LoopProgram prog_naive = lower_naive(g);

    auto xd = rand_vec(M * N, 4);
    auto bd = rand_vec(M,     5);

    RunResult fr = compile_and_run(emit_cpp(prog_fused), prog_fused, {xd, bd});
    RunResult nr = compile_and_run(emit_cpp(prog_naive), prog_naive, {xd, bd});

    assert(fr.ok && nr.ok);
    for (int64_t i = 0; i < M; ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    std::cout << "test_max_add_fuses: PASS (max+add -> 1 kernel, matches naive)\n";
}

static void test_softmax_correctness() {
    const int64_t M = 4, N = 16;

    Graph g;
    auto x = g.input({M, N});
    g.mark_output(g.softmax(x, 1));
    infer_shapes(g);
    validate(g);

    fuse(g);
    LoopProgram prog = lower_fused(g);

    auto xd = rand_vec(M * N, 6);
    RunResult r = compile_and_run(emit_cpp(prog), prog, {xd});
    assert(r.ok);

    for (int64_t i = 0; i < M; ++i) {
        float s = 0.f;
        for (int64_t j = 0; j < N; ++j) s += r.outputs[0][i * N + j];
        assert(near(s, 1.f, 1e-4f));
    }

    std::cout << "test_softmax_correctness: PASS (each row sums to 1)\n";
}

static void test_reduction_benchmark() {
    const int64_t M = 512, N = 1024;

    Graph g;
    auto x = g.input({M, N});
    g.mark_output(g.relu(g.sum(x, 1)));
    infer_shapes(g);
    validate(g);

    TrafficStats s = analyze_savings(g);
    fuse(g);

    LoopProgram prog_fused = lower_fused(g);
    LoopProgram prog_naive = lower_naive(g);

    auto xd = rand_vec(M * N, 7);

    BenchResult fr = compile_and_bench(prog_fused, {xd});
    BenchResult nr = compile_and_bench(prog_naive, {xd});

    assert(fr.ok && nr.ok);
    for (int64_t i = 0; i < M; ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    double speedup = nr.median_ms / fr.median_ms;

    std::cout << "test_reduction_benchmark: PASS\n";
    std::cout << "  unfused traffic: " << s.bytes_unfused / 1024 << " KB\n";
    std::cout << "  fused traffic:   " << s.bytes_fused   / 1024 << " KB\n";
    std::cout << "  saved:           " << s.bytes_saved   / 1024 << " KB\n";
    std::cout << "  naive:  " << nr.median_ms << " ms\n";
    std::cout << "  fused:  " << fr.median_ms << " ms\n";
    std::cout << "  speedup: " << speedup << "x\n";
}

int main() {
    register_all_ops();

    test_sum_relu_fuses();
    test_mean_mul_fuses();
    test_max_add_fuses();
    test_softmax_correctness();
    test_reduction_benchmark();

    std::cout << "\nM4: all tests passed.\n";
    return 0;
}
