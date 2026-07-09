cmake_minimum_required(VERSION 3.16)
project(tensor_compiler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Core library ──────────────────────────────────────────────────────────────
add_library(tc STATIC
    src/ops.cpp
    src/graph.cpp
    src/passes.cpp
    src/printer.cpp
)
target_include_directories(tc PUBLIC include)

# ── M0 test ───────────────────────────────────────────────────────────────────
add_executable(test_m0 tests/test_m0.cpp)
target_link_libraries(test_m0 PRIVATE tc)
