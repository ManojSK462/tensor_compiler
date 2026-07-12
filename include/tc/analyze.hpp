#pragma once
#include "graph.hpp"
#include <cstdint>

struct TrafficStats {
    int64_t bytes_unfused;
    int64_t bytes_fused;
    int64_t bytes_saved;
};

TrafficStats analyze_savings(const Graph& g);
