#include "tc/loop_ir.hpp"

BufferId LoopProgram::add_buffer(Shape shape, DType dtype, BufferRole role, std::string name) {
    BufferId id = static_cast<BufferId>(buffers.size());
    buffers.push_back({id, std::move(shape), dtype, role, std::move(name)});
    return id;
}
