#include "tc/types.hpp"
#include "tc/ops.hpp"
#include "tc/graph.hpp"
#include "tc/passes.hpp"
#include "tc/fusion.hpp"
#include "tc/analyze.hpp"
#include "tc/lower.hpp"
#include "tc/codegen.hpp"
#include "tc/runtime.hpp"
#include "tc/buffer_reuse.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <unordered_set>

static bool near(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

static std::vector<float> rand_vec(int64_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static void test_buffer_reuse() {
    const int64_t M = 4, H = 8, D = 4;

    Graph g;
    auto x  = g.input({M, H});
    auto W1 = g.input({H, H});  auto b1 = g.input({H});
    auto W2 = g.input({H, H});  auto b2 = g.input({H});
    auto W3 = g.input({H, H});  auto b3 = g.input({H});
    auto W4 = g.input({H, D});

    auto h1 = g.relu(g.add(g.matmul(x,  W1), b1));
    auto h2 = g.relu(g.add(g.matmul(h1, W2), b2));
    auto h3 = g.relu(g.add(g.matmul(h2, W3), b3));
    g.mark_output(g.matmul(h3, W4));

    infer_shapes(g);
    validate(g);
    fuse(g);

    LoopProgram prog = lower_fused(g);
    assert(prog.kernels.size() == 4);

    assign_buffer_slots(prog);

    std::unordered_set<int32_t> slots_used;
    int int_count = 0;
    for (const Buffer& b : prog.buffers) {
        if (b.role != BufferRole::Intermediate) continue;
        ++int_count;
        assert(b.slot_id >= 0);
        slots_used.insert(b.slot_id);
    }
    assert(int_count == 3);
    assert(slots_used.size() == 2);

    LoopProgram prog_ref = lower_fused(g);

    auto inputs = std::vector<std::vector<float>>{
        rand_vec(M * H, 1), rand_vec(H * H, 2), rand_vec(H,     3),
        rand_vec(H * H, 4), rand_vec(H,     5),
        rand_vec(H * H, 6), rand_vec(H,     7),
        rand_vec(H * D, 8)
    };

    RunResult fr = compile_and_run(emit_cpp(prog),     prog,     inputs);
    RunResult nr = compile_and_run(emit_cpp(prog_ref), prog_ref, inputs);

    assert(fr.ok && nr.ok);
    for (size_t i = 0; i < nr.outputs[0].size(); ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    std::cout << "test_buffer_reuse: PASS"
              << " (3 intermediates -> 2 slots, output matches)\n";
}

static void test_attention() {
    const int64_t T = 8, D = 16;

    Graph g;
    auto Q     = g.input({T, D});
    auto K     = g.input({T, D});
    auto V     = g.input({T, D});
    auto scale = g.input({1});

    auto scores = g.matmul(Q, K, false, true);
    auto scaled = g.mul(scores, scale);
    auto attn   = g.softmax(scaled, 1);
    g.mark_output(g.matmul(attn, V));

    infer_shapes(g);
    validate(g);
    fuse(g);

    LoopProgram prog_fused = lower_fused(g);
    LoopProgram prog_naive = lower_naive(g);

    auto Qd = rand_vec(T * D, 1);
    auto Kd = rand_vec(T * D, 2);
    auto Vd = rand_vec(T * D, 3);
    std::vector<float> scale_d = {1.f / std::sqrt(static_cast<float>(D))};

    RunResult fr = compile_and_run(emit_cpp(prog_fused), prog_fused, {Qd, Kd, Vd, scale_d});
    RunResult nr = compile_and_run(emit_cpp(prog_naive), prog_naive, {Qd, Kd, Vd, scale_d});

    assert(fr.ok && nr.ok);
    for (size_t i = 0; i < nr.outputs[0].size(); ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    std::cout << "test_attention: PASS"
              << " (Q@K^T/sqrt(D)->softmax->@V, fused matches naive)\n";
}

static void test_e2e_benchmark() {
    const int64_t M = 256, D_in = 256, D_h = 128, D_out = 64;

    Graph g;
    auto x  = g.input({M, D_in});
    auto W1 = g.input({D_in, D_h});  auto b1 = g.input({D_h});
    auto W2 = g.input({D_h, D_out}); auto b2 = g.input({D_out});

    auto h = g.relu(g.add(g.matmul(x, W1), b1));
    g.mark_output(g.relu(g.add(g.matmul(h, W2), b2)));

    infer_shapes(g);
    validate(g);
    fuse(g);

    LoopProgram prog_fused = lower_fused(g);
    assign_buffer_slots(prog_fused);

    LoopProgram prog_naive = lower_naive(g);

    auto inputs = std::vector<std::vector<float>>{
        rand_vec(M * D_in,   1),
        rand_vec(D_in * D_h, 2), rand_vec(D_h,      3),
        rand_vec(D_h * D_out,4), rand_vec(D_out,     5)
    };

    BenchResult fr = compile_and_bench(prog_fused, inputs);
    BenchResult nr = compile_and_bench(prog_naive, inputs);

    assert(fr.ok && nr.ok);
    for (size_t i = 0; i < nr.outputs[0].size(); ++i)
        assert(near(fr.outputs[0][i], nr.outputs[0][i]));

    double speedup = nr.median_ms / fr.median_ms;

    std::cout << "test_e2e_benchmark: PASS\n";
    std::cout << "  naive:   " << nr.median_ms << " ms\n";
    std::cout << "  fused:   " << fr.median_ms << " ms\n";
    std::cout << "  speedup: " << speedup << "x\n";
}

int main() {
    register_all_ops();

    test_buffer_reuse();
    test_attention();
    test_e2e_benchmark();

    std::cout << "\nM5: all tests passed.\n";
    return 0;
}
