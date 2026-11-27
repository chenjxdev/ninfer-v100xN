#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void l2_clip_kernel(const __nv_bfloat16* x, const __nv_bfloat16* scale,
                               __nv_bfloat16* out, std::int32_t width, std::int64_t rows,
                               std::int64_t scale_rows, float threshold) {
    constexpr int kWarps = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * kWarps + warp;
    if (row >= rows) { return; }
    const std::int64_t base = row * width;
    const std::int64_t scale_base = (scale == nullptr ? 0 : row % scale_rows) * width;
    float sum = 0.0F;
    for (std::int32_t d = lane; d < width; d += 32) {
        float value = __bfloat162float(x[base + d]);
        if (scale != nullptr) { value *= __bfloat162float(scale[scale_base + d]); }
        sum += value * value;
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    sum = __shfl_sync(0xffffffffU, sum, 0);
    const float factor = fminf(1.0F, threshold / (sqrtf(sum) + 1.0e-6F));
    for (std::int32_t d = lane; d < width; d += 32) {
        float value = __bfloat162float(x[base + d]);
        if (scale != nullptr) { value *= __bfloat162float(scale[scale_base + d]); }
        out[base + d] = __float2bfloat16_rn(value * factor);
    }
}

} // namespace ninfer::ops
