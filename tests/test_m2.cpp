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

static bool near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

static std::vector<float> rand_vec(int64_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static void test_elementwise_chain_fuses() {
    const int64_t M = 8, N = 16;

    Graph g;
    auto x = g.input({M, N});
    g.mark_output(g.sigmoid(g.relu(x)));
    infer_shapes(g);
    validate(g);

    fuse(g);

    assert(g.nodes[0].group == g.nodes[1].group);

    LoopProgram prog_fused = lower_fused(g);
    assert(prog_fused.kernels.size() == 1);

    LoopProgram prog_naive = lower_naive(g);

    auto x_data = rand_vec(M * N, 1);
    RunResult fr = compile_and_run(emit_cpp(prog_fused), prog_fused, {x_data});
    RunResult nr = compile_and_run(emit_cpp(prog_naive), prog_naive, {x_data});

    assert(fr.ok && nr.ok);
    assert(fr.outputs[0].size() == static_cast<size_t>(M * N));
    for (int64_t i = 0; i < M * N; ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    std::cout << "test_elementwise_chain_fuses: PASS (1 kernel for relu+sigmoid)\n";
}

static void test_multi_consumer_no_fuse() {
    const int64_t M = 8, N = 16;

    Graph g;
    auto x  = g.input({M, N});
    auto r  = g.relu(x);
    auto y1 = g.sigmoid(r);
    auto y2 = g.gelu(r);
    g.mark_output(y1);
    g.mark_output(y2);
    infer_shapes(g);
    validate(g);

    fuse(g);

    assert(g.nodes[0].group != g.nodes[1].group);
    assert(g.nodes[0].group != g.nodes[2].group);

    std::cout << "test_multi_consumer_no_fuse: PASS (relu not fused with multi-consumer)\n";
}

static void test_analyze_savings() {
    const int64_t M = 8, N = 16;

    Graph g;
    auto x = g.input({M, N});
    g.mark_output(g.sigmoid(g.relu(g.gelu(x))));
    infer_shapes(g);
    validate(g);
    fuse(g);

    TrafficStats s = analyze_savings(g);

    int64_t elem = M * N * 4;
    assert(s.bytes_unfused == 6 * elem);
    assert(s.bytes_fused   == 2 * elem);
    assert(s.bytes_saved   == 4 * elem);

    std::cout << "test_analyze_savings: PASS"
              << "  unfused=" << s.bytes_unfused / 1024 << " KB"
              << "  fused="   << s.bytes_fused   / 1024 << " KB"
              << "  saved="   << s.bytes_saved    / 1024 << " KB\n";
}

static void test_mlp_partial_fuse() {
    const int64_t M = 4, K = 8, N = 16;

    Graph g;
    auto x = g.input({M, K});
    auto W = g.input({K, N});
    auto b = g.input({N});
    g.mark_output(g.relu(g.add(g.matmul(x, W), b)));
    infer_shapes(g);
    validate(g);

    fuse(g);

    assert(g.nodes[0].group != g.nodes[1].group);
    assert(g.nodes[1].group == g.nodes[2].group);

    LoopProgram prog_fused = lower_fused(g);
    assert(prog_fused.kernels.size() == 2);

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

    std::cout << "test_mlp_partial_fuse: PASS"
              << " (matmul singleton + add+relu fused, 2 kernels, matches naive)\n";
}

int main() {
    register_all_ops();

    test_elementwise_chain_fuses();
    test_multi_consumer_no_fuse();
    test_analyze_savings();
    test_mlp_partial_fuse();

    std::cout << "\nM2: all tests passed.\n";
    return 0;
}
