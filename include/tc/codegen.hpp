#pragma once
#include "loop_ir.hpp"
#include <string>

std::string emit_cpp(const LoopProgram& prog);
std::string emit_bench_cpp(const LoopProgram& prog, int warmup = 2, int reps = 10);
