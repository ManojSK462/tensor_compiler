#pragma once
#include "types.hpp"
#include <string>
#include <vector>
#include <memory>

using BufferId = uint32_t;
using KernelId = uint32_t;
static constexpr BufferId kInvalidBuffer = UINT32_MAX;

enum class BufferRole { External, Intermediate, Output };

struct Buffer {
    BufferId    id;
    Shape       shape;
    DType       dtype;
    BufferRole  role;
    std::string name;
    int32_t     slot_id = -1;
};

enum class LoopTag { Serial, Parallel, Vectorize, Reduce };

struct Loop {
    std::string var;
    int64_t     extent;
    LoopTag     tag;
};

struct LoopNest {
    KernelId            id;
    std::string         name;
    std::vector<Loop>   loops;
    std::vector<BufferId> reads;
    std::vector<BufferId> writes;
    std::string         body;
};

struct LoopProgram {
    std::vector<Buffer>   buffers;
    std::vector<LoopNest> kernels;

    BufferId add_buffer(Shape shape, DType dtype, BufferRole role, std::string name);
};
