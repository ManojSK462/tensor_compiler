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

static void test_epilogue_fuses() {
    const int64_t M = 8, K = 16, N = 32;

    Graph g;
    auto x = g.input({M, K});
    auto W = g.input({K, N});
    auto b = g.input({N});
    g.mark_output(g.relu(g.add(g.matmul(x, W), b)));
    infer_shapes(g);
    validate(g);

    fuse(g);

    assert(g.nodes[0].group == g.nodes[1].group);
    assert(g.nodes[1].group == g.nodes[2].group);

    LoopProgram prog_fused = lower_fused(g);
    assert(prog_fused.kernels.size() == 1);

    LoopProgram prog_naive = lower_naive(g);

    auto xd = rand_vec(M * K, 1);
    auto wd = rand_vec(K * N, 2);
    auto bd = rand_vec(N,     3);

    RunResult fr = compile_and_run(emit_cpp(prog_fused), prog_fused, {xd, wd, bd});
    RunResult nr = compile_and_run(emit_cpp(prog_naive), prog_naive, {xd, wd, bd});

    assert(fr.ok && nr.ok);
    assert(fr.outputs[0].size() == static_cast<size_t>(M * N));
    for (int64_t i = 0; i < M * N; ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    std::cout << "test_epilogue_fuses: PASS"
              << " (matmul+add+relu -> 1 fused kernel, matches naive)\n";
}

static void test_two_layer_mlp_fuses() {
    const int64_t B = 4, D0 = 8, D1 = 16, D2 = 8;

    Graph g;
    auto x  = g.input({B, D0});
    auto W1 = g.input({D0, D1});
    auto b1 = g.input({D1});
    auto W2 = g.input({D1, D2});
    auto b2 = g.input({D2});

    auto h = g.relu(g.add(g.matmul(x, W1), b1));
    g.mark_output(g.relu(g.add(g.matmul(h, W2), b2)));
    infer_shapes(g);
    validate(g);

    fuse(g);

    LoopProgram prog_fused = lower_fused(g);
    LoopProgram prog_naive = lower_naive(g);
    assert(prog_fused.kernels.size() == 2);

    auto xd  = rand_vec(B * D0, 1);
    auto w1d = rand_vec(D0 * D1, 2);
    auto b1d = rand_vec(D1,      3);
    auto w2d = rand_vec(D1 * D2, 4);
    auto b2d = rand_vec(D2,      5);

    RunResult fr = compile_and_run(emit_cpp(prog_fused), prog_fused, {xd, w1d, b1d, w2d, b2d});
    RunResult nr = compile_and_run(emit_cpp(prog_naive), prog_naive, {xd, w1d, b1d, w2d, b2d});

    assert(fr.ok && nr.ok);
    for (int64_t i = 0; i < B * D2; ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    std::cout << "test_two_layer_mlp_fuses: PASS (2 fused kernels, matches naive)\n";
}

static void test_benchmark() {
    const int64_t M = 256, K = 256, N = 256;

    Graph g;
    auto x = g.input({M, K});
    auto W = g.input({K, N});
    auto b = g.input({N});
    g.mark_output(g.relu(g.add(g.matmul(x, W), b)));
    infer_shapes(g);
    validate(g);

    TrafficStats s = analyze_savings(g);
    fuse(g);

    LoopProgram prog_fused = lower_fused(g);
    LoopProgram prog_naive = lower_naive(g);

    auto xd = rand_vec(M * K, 1);
    auto wd = rand_vec(K * N, 2);
    auto bd = rand_vec(N,     3);

    std::vector<std::vector<float>> inputs = {xd, wd, bd};

    BenchResult fused_br = compile_and_bench(prog_fused, inputs);
    BenchResult naive_br = compile_and_bench(prog_naive, inputs);

    assert(fused_br.ok && naive_br.ok);
    for (int64_t i = 0; i < M * N; ++i)
        assert(near(fused_br.outputs[0][i], naive_br.outputs[0][i]));

    double speedup = naive_br.median_ms / fused_br.median_ms;

    std::cout << "test_benchmark: PASS\n";
    std::cout << "  unfused traffic: " << s.bytes_unfused / 1024 << " KB\n";
    std::cout << "  fused traffic:   " << s.bytes_fused   / 1024 << " KB\n";
    std::cout << "  saved:           " << s.bytes_saved   / 1024 << " KB\n";
    std::cout << "  naive:  " << naive_br.median_ms << " ms\n";
    std::cout << "  fused:  " << fused_br.median_ms << " ms\n";
    std::cout << "  speedup: " << speedup << "x\n";
}

int main() {
    register_all_ops();

    test_epilogue_fuses();
    test_two_layer_mlp_fuses();
    test_benchmark();

    std::cout << "\nM3: all tests passed.\n";
    return 0;
}
