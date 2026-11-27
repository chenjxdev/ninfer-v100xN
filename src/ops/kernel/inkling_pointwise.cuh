#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void inkling_softcap_kernel(const __nv_bfloat16* x, float cap, __nv_bfloat16* out,
                                       std::int64_t count) {
    const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        const float value = __bfloat162float(x[index]);
        out[index] = __float2bfloat16_rn(cap * tanhf(value / cap));
    }
}

__global__ void inkling_log_position_scale_kernel(
    const __nv_bfloat16* x, const std::int32_t* positions, std::uint32_t floor, float alpha,
    __nv_bfloat16* out, std::int32_t rows_per_token, std::int32_t tokens) {
    const std::int64_t count = static_cast<std::int64_t>(rows_per_token) * tokens;
    const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (index >= count) { return; }
    const std::int32_t token = static_cast<std::int32_t>(index / rows_per_token);
    const float position = static_cast<float>(positions[token]) + 1.0F;
    const float tau = 1.0F + alpha * logf(fmaxf(1.0F, position / floor));
    out[index] = __float2bfloat16_rn(__bfloat162float(x[index]) * tau);
}

} // namespace ninfer::ops
