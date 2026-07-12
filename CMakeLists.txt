cmake_minimum_required(VERSION 3.16)
project(tensor_compiler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(tc STATIC
    src/ops.cpp
    src/graph.cpp
    src/passes.cpp
    src/printer.cpp
    src/loop_ir.cpp
    src/lower.cpp
    src/codegen.cpp
    src/runtime.cpp
    src/fusion.cpp
    src/analyze.cpp
)
target_include_directories(tc PUBLIC include)

add_executable(test_m0 tests/test_m0.cpp)
target_link_libraries(test_m0 PRIVATE tc)

add_executable(test_m1 tests/test_m1.cpp)
target_link_libraries(test_m1 PRIVATE tc)

add_executable(test_m2 tests/test_m2.cpp)
target_link_libraries(test_m2 PRIVATE tc)
