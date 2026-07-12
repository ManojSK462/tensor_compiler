#include "tc/types.hpp"
#include "tc/ops.hpp"
#include "tc/graph.hpp"
#include "tc/passes.hpp"
#include "tc/lower.hpp"
#include "tc/codegen.hpp"
#include "tc/runtime.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <fstream>

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

static std::vector<float> ref_matmul_add_relu(
    int64_t M, int64_t K, int64_t N,
    const std::vector<float>& x,
    const std::vector<float>& W,
    const std::vector<float>& b)
{
    std::vector<float> out(M * N, 0.f);
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int64_t k = 0; k < K; ++k)
                acc += x[i*K+k] * W[k*N+j];
            out[i*N+j] = std::max(0.f, acc + b[j]);
        }
    return out;
}

static void test_mlp_layer() {
    const int64_t M = 4, K = 8, N = 16;

    Graph g;
    auto x = g.input({M, K});
    auto W = g.input({K, N});
    auto b = g.input({N});
    g.mark_output(g.relu(g.add(g.matmul(x, W), b)));

    infer_shapes(g);
    validate(g);

    LoopProgram prog = lower_naive(g);
    std::string src  = emit_cpp(prog);

    std::ofstream dump("test_m1_gen.cpp");
    dump << src;
    std::cout << "Generated source written to test_m1_gen.cpp\n";

    auto x_data = rand_vec(M*K, 1);
    auto W_data = rand_vec(K*N, 2);
    auto b_data = rand_vec(N,   3);

    RunResult res = compile_and_run(src, prog, {x_data, W_data, b_data});

    if (!res.ok) {
        std::cerr << "FAIL: " << res.error << "\n";
        std::exit(1);
    }

    std::vector<float> ref = ref_matmul_add_relu(M, K, N, x_data, W_data, b_data);

    assert(res.outputs.size() == 1);
    assert(res.outputs[0].size() == static_cast<size_t>(M*N));

    for (int64_t i = 0; i < M*N; ++i) {
        if (!near(res.outputs[0][i], ref[i])) {
            std::cerr << "MISMATCH at [" << i << "]: got " << res.outputs[0][i]
                      << " expected " << ref[i] << "\n";
            std::exit(1);
        }
    }

    std::cout << "test_mlp_layer: PASS (M=" << M << " K=" << K << " N=" << N << ")\n";
}

static void test_elementwise_chain() {
    const int64_t M = 8, N = 16;

    Graph g;
    auto x = g.input({M, N});
    auto y = g.sigmoid(g.relu(x));
    g.mark_output(y);

    infer_shapes(g);
    validate(g);

    LoopProgram prog = lower_naive(g);
    std::string src  = emit_cpp(prog);

    auto x_data = rand_vec(M*N, 7);
    RunResult res = compile_and_run(src, prog, {x_data});

    if (!res.ok) { std::cerr << "FAIL: " << res.error << "\n"; std::exit(1); }

    for (int64_t i = 0; i < M*N; ++i) {
        float relu_x = std::max(0.f, x_data[i]);
        float ref    = 1.f / (1.f + std::exp(-relu_x));
        if (!near(res.outputs[0][i], ref)) {
            std::cerr << "MISMATCH at [" << i << "]\n";
            std::exit(1);
        }
    }

    std::cout << "test_elementwise_chain: PASS\n";
}

int main() {
    register_all_ops();

    test_mlp_layer();
    test_elementwise_chain();

    std::cout << "\nM1: all tests passed.\n";
    return 0;
}
