#pragma once

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ float inkling_attention_score(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* r,
    const __nv_bfloat16* relative_bias, const __nv_bfloat16* sink_logits, std::int32_t key,
    std::int32_t query, std::int32_t head, std::int32_t heads, std::int32_t kv_heads,
    std::int32_t tokens, std::int32_t keys, std::int32_t relative_length, std::int32_t extent,
    float scale, std::uint32_t sliding_window) {
    const std::int32_t query_position = keys - tokens + query;
    const std::int32_t distance = query_position - key;
    if (key < 0 || key >= keys || distance < 0 ||
        (sliding_window != 0 && static_cast<std::uint32_t>(distance) >= sliding_window)) {
        return -CUDART_INF_F;
    }
    const std::int32_t q_base = (query * heads + head) * 128;
    const std::int32_t k_base = key * kv_heads * 128;
    float score = 0.0F;
    for (std::int32_t d = 0; d < 128; ++d) {
        score += __bfloat162float(q[q_base + d]) * __bfloat162float(k[k_base + d]);
    }
    score *= scale;
    if (distance < extent) {
        const std::int32_t r_base = (query * heads + head) * relative_length;
        for (std::int32_t j = 0; j < relative_length; ++j) {
            score += __bfloat162float(r[r_base + j]) *
                     __bfloat162float(relative_bias[distance * relative_length + j]);
        }
    }
    score -= __bfloat162float(sink_logits[query * heads + head]);
    return score;
}

__global__ void inkling_attention_dense_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* r, const __nv_bfloat16* relative_bias, const __nv_bfloat16* sink_logits,
    __nv_bfloat16* out, std::int32_t heads, std::int32_t kv_heads, std::int32_t tokens,
    std::int32_t keys, std::int32_t relative_length, std::int32_t extent, float scale,
    std::uint32_t sliding_window) {
    constexpr int kBlock = 128;
    constexpr int kTile = 256;
    __shared__ float scores[kTile];
    __shared__ float reduction[kBlock];
    __shared__ float running_max;
    __shared__ float running_denominator;
    const std::int32_t query = blockIdx.x;
    const std::int32_t head = blockIdx.y;
    const std::int32_t kv_head = head / (heads / kv_heads);
    const std::int32_t tid = threadIdx.x;
    const __nv_bfloat16* head_k = k + kv_head * 128;
    const __nv_bfloat16* head_v = v + kv_head * 128;
    float accumulator = 0.0F;
    if (tid == 0) {
        running_max = 0.0F; // zero-value sink
        running_denominator = 1.0F;
    }
    __syncthreads();

    for (std::int32_t tile = 0; tile < keys; tile += kTile) {
        scores[tid] = inkling_attention_score(q, head_k, r, relative_bias, sink_logits,
            tile + tid, query, head, heads, kv_heads, tokens, keys, relative_length, extent, scale,
            sliding_window);
        scores[tid + kBlock] = inkling_attention_score(q, head_k, r, relative_bias, sink_logits,
            tile + tid + kBlock, query, head, heads, kv_heads, tokens, keys, relative_length, extent,
            scale, sliding_window);
        reduction[tid] = fmaxf(scores[tid], scores[tid + kBlock]);
        __syncthreads();
        for (int offset = kBlock / 2; offset != 0; offset >>= 1) {
            if (tid < offset) { reduction[tid] = fmaxf(reduction[tid], reduction[tid + offset]); }
            __syncthreads();
        }
        const float new_max = fmaxf(running_max, reduction[0]);
        const float correction = expf(running_max - new_max);
        __syncthreads();
        reduction[tid] = expf(scores[tid] - new_max) + expf(scores[tid + kBlock] - new_max);
        __syncthreads();
        for (int offset = kBlock / 2; offset != 0; offset >>= 1) {
            if (tid < offset) { reduction[tid] += reduction[tid + offset]; }
            __syncthreads();
        }
        if (tid == 0) {
            running_denominator = running_denominator * correction + reduction[0];
            running_max = new_max;
        }
        accumulator *= correction;
        for (int j = 0; j < kTile; ++j) {
            const std::int32_t key = tile + j;
            if (key < keys) {
                accumulator += expf(scores[j] - new_max) *
                               __bfloat162float(head_v[key * kv_heads * 128 + tid]);
            }
        }
        __syncthreads();
    }
    const std::int32_t output_index = (query * heads + head) * 128 + tid;
    out[output_index] = __float2bfloat16_rn(accumulator / running_denominator);
}

} // namespace ninfer::ops
