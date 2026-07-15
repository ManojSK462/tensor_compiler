#pragma once
#include "loop_ir.hpp"
#include <string>
#include <vector>

struct RunResult {
    bool                            ok;
    std::string                     error;
    std::vector<std::vector<float>> outputs;
};

struct BenchResult {
    bool                            ok;
    std::string                     error;
    std::vector<std::vector<float>> outputs;
    double                          median_ms;
};

RunResult compile_and_run(const std::string& src,
                          const LoopProgram& prog,
                          const std::vector<std::vector<float>>& inputs);

BenchResult compile_and_bench(const LoopProgram& prog,
                               const std::vector<std::vector<float>>& inputs,
                               int warmup = 2, int reps = 10);
