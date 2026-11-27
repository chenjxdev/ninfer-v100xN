#pragma once

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void inkling_router_probs_kernel(const __nv_bfloat16* x, const __nv_bfloat16* gate,
                                            float* probabilities, std::int32_t hidden,
                                            std::int32_t experts) {
    constexpr int kWarps = 8;
    const std::int32_t token = blockIdx.x;
    const std::int32_t expert = blockIdx.y * kWarps + (threadIdx.x >> 5);
    if (expert >= experts) { return; }
    const std::int32_t lane = threadIdx.x & 31;
    const __nv_bfloat16* input = x + token * hidden;
    float logit = 0.0F;
    for (std::int32_t h = lane; h < hidden; h += 32) {
        logit += __bfloat162float(gate[expert * hidden + h]) * __bfloat162float(input[h]);
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        logit += __shfl_down_sync(0xffffffffU, logit, offset);
    }
    if (lane == 0) { probabilities[token * experts + expert] = 1.0F / (1.0F + expf(-logit)); }
}

__global__ void inkling_router_select_kernel(const float* probabilities,
                                             const float* selection_bias, float weight_scale,
                                             std::int32_t* selected, float* weights,
                                             std::int32_t experts, std::int32_t top_k) {
    __shared__ float scores[256];
    const std::int32_t token = blockIdx.x;
    for (std::int32_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        scores[expert] = probabilities[token * experts + expert] + selection_bias[expert];
    }
    __syncthreads();
    if (threadIdx.x >= 32) { return; }
    const std::int32_t lane = threadIdx.x;
    const std::int32_t output_base = token * top_k;
    float denominator = 1.0e-20F;
    for (std::int32_t k = 0; k < top_k; ++k) {
        float best_score = -CUDART_INF_F;
        std::int32_t best = experts;
        for (std::int32_t expert = lane; expert < experts; expert += 32) {
            const float score = scores[expert];
            if (score > best_score || (score == best_score && expert < best)) {
                best_score = score;
                best = expert;
            }
        }
        for (int offset = 16; offset != 0; offset >>= 1) {
            const float other_score = __shfl_down_sync(0xffffffffU, best_score, offset);
            const std::int32_t other = __shfl_down_sync(0xffffffffU, best, offset);
            if (other_score > best_score || (other_score == best_score && other < best)) {
                best_score = other_score;
                best = other;
            }
        }
        if (lane == 0) {
            selected[output_base + k] = best;
            const float probability = probabilities[token * experts + best];
            weights[output_base + k] = probability;
            denominator += probability;
            scores[best] = -CUDART_INF_F;
        }
        __syncwarp();
    }
    if (lane == 0) {
        for (std::int32_t k = 0; k < top_k; ++k) {
            weights[output_base + k] = weights[output_base + k] / denominator * weight_scale;
        }
    }
}

} // namespace ninfer::ops
