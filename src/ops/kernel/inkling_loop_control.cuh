#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void inkling_loop_control_normalize_kernel(
    const __nv_bfloat16* cur, const __nv_bfloat16* message, float* normalized,
    std::int32_t hidden) {
    constexpr int kBlock = 256;
    __shared__ float reduction[kBlock];
    const std::int32_t token = blockIdx.x;
    const std::int32_t tid = threadIdx.x;
    const std::int32_t input_base = token * hidden;
    for (std::int32_t part = 0; part < 2; ++part) {
        const __nv_bfloat16* input = part == 0 ? cur : message;
        float partial = 0.0F;
        for (std::int32_t h = tid; h < hidden; h += kBlock) {
            const float value = __bfloat162float(input[input_base + h]);
            partial += value * value;
        }
        reduction[tid] = partial;
        __syncthreads();
        for (int offset = kBlock / 2; offset != 0; offset >>= 1) {
            if (tid < offset) { reduction[tid] += reduction[tid + offset]; }
            __syncthreads();
        }
        const float inv_rms = rsqrtf(reduction[0] / hidden + 1.0e-6F);
        for (std::int32_t h = tid; h < hidden; h += kBlock) {
            normalized[token * 2 * hidden + part * hidden + h] =
                __bfloat162float(input[input_base + h]) * inv_rms;
        }
        __syncthreads();
    }
}

__global__ void inkling_loop_control_gate_kernel(
    const float* normalized, const __nv_bfloat16* down, float* gate,
    std::int32_t hidden, std::int32_t rank) {
    constexpr int kWarps = 8;
    const std::int32_t token = blockIdx.x;
    const std::int32_t r = blockIdx.y * kWarps + (threadIdx.x >> 5);
    if (r >= rank) { return; }
    const std::int32_t lane = threadIdx.x & 31;
    const float* input = normalized + token * 2 * hidden;
    float value = 0.0F;
    for (std::int32_t i = lane; i < 2 * hidden; i += 32) {
        value += __bfloat162float(down[r * 2 * hidden + i]) * input[i];
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (lane == 0) { gate[token * rank + r] = value; }
}

__global__ void inkling_loop_control_output_kernel(
    const __nv_bfloat16* cur, const __nv_bfloat16* message, const __nv_bfloat16* gain,
    const __nv_bfloat16* forget, const float* gate, __nv_bfloat16* out,
    std::int32_t hidden, std::int32_t rank) {
    constexpr int kWarps = 8;
    const std::int32_t token = blockIdx.x;
    const std::int32_t h = blockIdx.y * kWarps + (threadIdx.x >> 5);
    if (h >= hidden) { return; }
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t input_base = token * hidden;
    float gain_value = 0.0F;
    float forget_value = 0.0F;
    for (std::int32_t r = lane; r < rank; r += 32) {
        const float gate_value = gate[token * rank + r];
        gain_value += __bfloat162float(gain[h * rank + r]) * gate_value;
        forget_value += __bfloat162float(forget[h * rank + r]) * gate_value;
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        gain_value += __shfl_down_sync(0xffffffffU, gain_value, offset);
        forget_value += __shfl_down_sync(0xffffffffU, forget_value, offset);
    }
    if (lane == 0) {
        gain_value = tanhf(gain_value);
        forget_value = tanhf(forget_value);
        const float current = __bfloat162float(cur[input_base + h]);
        const float incoming = __bfloat162float(message[input_base + h]);
        out[input_base + h] =
            __float2bfloat16_rn((1.0F - forget_value) * current + gain_value * incoming);
    }
}

} // namespace ninfer::ops
