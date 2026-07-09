#include "tc/types.hpp"
#include "tc/ops.hpp"
#include "tc/graph.hpp"
#include "tc/passes.hpp"
#include "tc/printer.hpp"
#include <iostream>
#include <cassert>
#include <fstream>

static void run(const char* label, Graph& g) {
    infer_shapes(g);
    validate(g);
    std::cout << "\n=== " << label << " ===\n";
    print_text(g, std::cout);
}

static void test_mlp_layer() {
    Graph g;
    auto x = g.input({4, 8});
    auto W = g.input({8, 16});
    auto b = g.input({16});
    auto t = g.matmul(x, W);
    auto u = g.add(t, b);
    auto y = g.relu(u);
    g.mark_output(y);

    run("MLP layer: matmul -> add -> relu", g);

    assert(g.tensors[t].shape == Shape({4, 16}));
    assert(g.tensors[u].shape == Shape({4, 16}));
    assert(g.tensors[y].shape == Shape({4, 16}));
}

static void test_elementwise_chain() {
    Graph g;
    auto x = g.input({32, 64});
    auto a = g.relu(x);
    auto b = g.sigmoid(a);
    auto c = g.mul(b, x);
    auto d = g.gelu(c);
    g.mark_output(d);

    run("Elementwise chain: relu -> sigmoid -> mul -> gelu", g);

    assert(g.tensors[d].shape == Shape({32, 64}));
    assert(g.nodes.size() == 4);
}

static void test_softmax() {
    Graph g;
    auto x = g.input({8, 32});
    auto y = g.softmax(x, -1);
    g.mark_output(y);

    run("Softmax over last axis", g);

    assert(g.tensors[y].shape == Shape({8, 32}));
}

static void test_reshape_transpose() {
    Graph g;
    auto x = g.input({4, 8});
    auto r = g.reshape(x, Shape({2, 2, 8}));
    auto t = g.transpose(r, {0, 2, 1});
    g.mark_output(t);

    run("Reshape [4,8]->[2,2,8], Transpose->[2,8,2]", g);

    assert(g.tensors[r].shape == Shape({2, 2, 8}));
    assert(g.tensors[t].shape == Shape({2, 8, 2}));
}

static void test_two_layer_mlp() {
    Graph g;
    auto x  = g.input({8, 32});
    auto W1 = g.input({32, 64});
    auto b1 = g.input({64});
    auto W2 = g.input({64, 16});
    auto b2 = g.input({16});

    auto h1 = g.relu(g.add(g.matmul(x, W1), b1));
    auto h2 = g.relu(g.add(g.matmul(h1, W2), b2));
    g.mark_output(h2);

    run("Two-layer MLP", g);

    assert(g.tensors[h1].shape == Shape({8, 64}));
    assert(g.tensors[h2].shape == Shape({8, 16}));
    assert(g.nodes.size() == 6);
}

static void test_reductions() {
    Graph g;
    auto x  = g.input({4, 8, 16});
    auto s  = g.sum(x,  -1);
    auto mx = g.max(x,   1);
    auto mn = g.mean(x,  0);
    g.mark_output(s);
    g.mark_output(mx);
    g.mark_output(mn);

    run("Reductions: sum/max/mean on 3-D tensor", g);

    assert(g.tensors[s].shape  == Shape({4, 8}));
    assert(g.tensors[mx].shape == Shape({4, 16}));
    assert(g.tensors[mn].shape == Shape({8, 16}));
}

static void test_dot_output() {
    Graph g;
    auto x = g.input({4, 8});
    auto W = g.input({8, 16});
    auto b = g.input({16});
    g.mark_output(g.relu(g.add(g.matmul(x, W), b)));

    infer_shapes(g);
    validate(g);

    std::ofstream f("test_m0_mlp.dot");
    print_dot(g, f);
    std::cout << "\nDOT -> test_m0_mlp.dot  (dot -Tpng test_m0_mlp.dot -o out.png)\n";
}

int main() {
    register_all_ops();

    test_mlp_layer();
    test_elementwise_chain();
    test_softmax();
    test_reshape_transpose();
    test_two_layer_mlp();
    test_reductions();
    test_dot_output();

    std::cout << "\nM0: all tests passed.\n";
    return 0;
}
