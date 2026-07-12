#pragma once
#include "graph.hpp"
#include "loop_ir.hpp"

LoopProgram lower_naive(const Graph& g);
LoopProgram lower_fused(const Graph& g);
