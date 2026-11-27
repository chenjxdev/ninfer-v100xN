#include "ninfer/ops/l2_clip.h"

#include "ops/launcher/l2_clip.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {

void l2_clip(const Tensor& x, const Tensor* scale, float threshold, Tensor& out,
             cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        (scale != nullptr && scale->dtype != DType::BF16)) {
        throw std::invalid_argument("l2_clip: x/scale/out must be BF16");
    }
    if (!std::isfinite(threshold) || threshold <= 0.0F) {
        throw std::invalid_argument("l2_clip: threshold must be positive and finite");
    }
    for (int d = 0; d < 4; ++d) {
        if (x.ne[d] != out.ne[d]) { throw std::invalid_argument("l2_clip: x/out shapes must match"); }
    }
    if (!x.is_contiguous() || !out.is_contiguous() ||
        (scale != nullptr && !scale->is_contiguous())) {
        throw std::invalid_argument("l2_clip: x/scale/out must be contiguous");
    }
    if (x.ne[0] <= 0) { throw std::invalid_argument("l2_clip: ne[0] must be positive"); }
    const std::int64_t n = x.numel();
    if (n == 0) { return; }
    if (x.data == nullptr || out.data == nullptr || (scale != nullptr && scale->data == nullptr)) {
        throw std::invalid_argument("l2_clip: x/scale/out data must be non-null");
    }
    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("l2_clip: row count exceeds CUDA grid limit");
    }
    if (scale != nullptr) {
        if (scale->ne[0] != x.ne[0] || scale->numel() % x.ne[0] != 0) {
            throw std::invalid_argument("l2_clip: scale row width must equal x.ne[0]");
        }
        const std::int64_t scale_rows = scale->numel() / x.ne[0];
        if (scale_rows == 0 || rows % scale_rows != 0) {
            throw std::invalid_argument("l2_clip: scale rows must divide and repeat over x rows");
        }
    }
    detail::l2_clip_launch(x, scale, threshold, out, stream);
}

} // namespace ninfer::ops
