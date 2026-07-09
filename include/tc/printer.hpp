#pragma once
#include "graph.hpp"
#include <ostream>

void print_text(const Graph& g, std::ostream& out);
void print_dot(const Graph& g, std::ostream& out);
