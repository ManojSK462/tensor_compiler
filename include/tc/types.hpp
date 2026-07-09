#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

enum class DType  { F32 };
enum class Layout { RowMajorContiguous };

struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}
    Shape(std::initializer_list<int64_t> il) : dims(il) {}

    int64_t rank()  const { return static_cast<int64_t>(dims.size()); }
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }

    bool operator==(const Shape& o) const { return dims == o.dims; }
    bool operator!=(const Shape& o) const { return dims != o.dims; }
};

using TensorId = uint32_t;
static constexpr TensorId kInvalidTensor = UINT32_MAX;

struct TensorInfo {
    TensorId id     = kInvalidTensor;
    Shape    shape;
    DType    dtype  = DType::F32;
    Layout   layout = Layout::RowMajorContiguous;
};

Shape       broadcast_shapes(const Shape& a, const Shape& b);
std::string dtype_to_str(DType d);
std::string shape_to_str(const Shape& s);
