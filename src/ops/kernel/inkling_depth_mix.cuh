#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void inkling_depth_mix_kernel(const __nv_bfloat16* states, const float* queries,
                                         __nv_bfloat16* out, std::int32_t hidden,
                                         std::int32_t depth, std::int32_t tokens,
                                         std::int32_t query_count) {
    constexpr int kBlock = 256;
    extern __shared__ float shared[];
    float* normalized = shared;
    float* reduction = normalized + hidden * depth;
    float* scores = reduction + kBlock;
    float* probabilities = scores + depth;
    const std::int32_t token = blockIdx.x;
    const std::int32_t tid = threadIdx.x;
    const std::int32_t state_base = token * hidden * depth;

    for (std::int32_t state = 0; state < depth; ++state) {
        float partial = 0.0F;
        for (std::int32_t h = tid; h < hidden; h += kBlock) {
            const float value = __bfloat162float(states[state_base + state * hidden + h]);
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
            normalized[state * hidden + h] =
                __bfloat162float(states[state_base + state * hidden + h]) * inv_rms;
        }
        __syncthreads();
    }

    const float score_scale = rsqrtf(static_cast<float>(hidden));
    for (std::int32_t query = 0; query < query_count; ++query) {
        for (std::int32_t state = 0; state < depth; ++state) {
            float partial = 0.0F;
            for (std::int32_t h = tid; h < hidden; h += kBlock) {
                partial += normalized[state * hidden + h] * queries[query * hidden + h];
            }
            reduction[tid] = partial;
            __syncthreads();
            for (int offset = kBlock / 2; offset != 0; offset >>= 1) {
                if (tid < offset) { reduction[tid] += reduction[tid + offset]; }
                __syncthreads();
            }
            if (tid == 0) { scores[state] = reduction[0] * score_scale; }
            __syncthreads();
        }

        if (tid == 0) {
            float maximum = scores[0];
            for (std::int32_t state = 1; state < depth; ++state) {
                maximum = fmaxf(maximum, scores[state]);
            }
            float denominator = 0.0F;
            for (std::int32_t state = 0; state < depth; ++state) {
                probabilities[state] = expf(scores[state] - maximum);
                denominator += probabilities[state];
            }
            for (std::int32_t state = 0; state < depth; ++state) {
                probabilities[state] /= denominator;
            }
        }
        __syncthreads();

        const std::int32_t output_base = (query + query_count * token) * hidden;
        for (std::int32_t h = tid; h < hidden; h += kBlock) {
            float value = 0.0F;
            for (std::int32_t state = 0; state < depth; ++state) {
                value += normalized[state * hidden + h] * probabilities[state];
            }
            out[output_base + h] = __float2bfloat16_rn(value);
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops
