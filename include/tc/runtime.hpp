#pragma once
#include "loop_ir.hpp"
#include <string>
#include <vector>

struct RunResult {
    bool                            ok;
    std::string                     error;
    std::vector<std::vector<float>> outputs;
};

RunResult compile_and_run(const std::string& src,
                          const LoopProgram& prog,
                          const std::vector<std::vector<float>>& inputs);
