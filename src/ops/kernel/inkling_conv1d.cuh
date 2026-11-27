#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void inkling_conv1d_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                      const __nv_bfloat16* dynamic, const __nv_bfloat16* state_in,
                                      __nv_bfloat16* state_out, __nv_bfloat16* out,
                                      std::int32_t channels, std::int32_t tokens,
                                      std::int32_t heads, bool weight_transposed,
                                      __nv_bfloat16* history) {
    const std::int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) { return; }
    float window[3] = {
        __bfloat162float(state_in[c]),
        __bfloat162float(state_in[channels + c]),
        __bfloat162float(state_in[2 * channels + c]),
    };
    if (history != nullptr) {
        history[c] = __float2bfloat16_rn(window[0]);
        history[channels + c] = __float2bfloat16_rn(window[1]);
        history[2 * channels + c] = __float2bfloat16_rn(window[2]);
    }
    const float output_scale = rsqrtf(dynamic == nullptr ? 5.0F : 9.0F);
    const std::int32_t channels_per_head = dynamic == nullptr ? 0 : channels / heads;
    const std::int32_t head = dynamic == nullptr ? 0 : c / channels_per_head;
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t index = t * channels + c;
        const float current = __bfloat162float(x[index]);
        const float values[4] = {window[0], window[1], window[2], current};
        float sum = 0.0F;
        for (std::int32_t tap = 0; tap < 4; ++tap) {
            const std::int32_t weight_index = weight_transposed ? tap * channels + c : c * 4 + tap;
            float coefficient = __bfloat162float(weight[weight_index]);
            if (dynamic != nullptr) {
                coefficient += __bfloat162float(dynamic[(4 * head + tap) * tokens + t]);
            }
            sum += coefficient * values[tap];
        }
        out[index] = __float2bfloat16_rn((current + sum) * output_scale);
        window[0] = window[1];
        window[1] = window[2];
        window[2] = current;
        if (history != nullptr) {
            __nv_bfloat16* snapshot = history + (t + 1) * 3 * channels;
            snapshot[c] = __float2bfloat16_rn(window[0]);
            snapshot[channels + c] = __float2bfloat16_rn(window[1]);
            snapshot[2 * channels + c] = __float2bfloat16_rn(window[2]);
        }
    }
    state_out[c] = __float2bfloat16_rn(window[0]);
    state_out[channels + c] = __float2bfloat16_rn(window[1]);
    state_out[2 * channels + c] = __float2bfloat16_rn(window[2]);
}

} // namespace ninfer::ops
