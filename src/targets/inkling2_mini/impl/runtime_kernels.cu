#include "targets/inkling2_mini/impl/runtime_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::targets::inkling2_mini::detail {
namespace {

__global__ void pack_depth_state_kernel(const __nv_bfloat16* source, __nv_bfloat16* states,
                                        std::int32_t hidden, std::int32_t tokens,
                                        std::int32_t depth, std::int32_t state_index) {
    const std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int32_t count = hidden * tokens;
    if (index >= count) { return; }
    const std::int32_t token = index / hidden;
    const std::int32_t h = index - token * hidden;
    states[(token * depth + state_index) * hidden + h] = source[index];
}

struct DepthSources {
    const __nv_bfloat16* values[5];
};

__global__ void mix_depth_states_kernel(DepthSources sources, const float* queries,
                                        __nv_bfloat16* out, std::int32_t hidden,
                                        std::int32_t depth, std::int32_t query_count) {
    constexpr int kBlock = 256;
    extern __shared__ float shared[];
    float* normalized = shared;
    float* reduction = normalized + hidden * depth;
    float* scores = reduction + kBlock;
    float* probabilities = scores + depth;
    const std::int32_t token = blockIdx.x;
    const std::int32_t tid = threadIdx.x;
    for (std::int32_t state = 0; state < depth; ++state) {
        const __nv_bfloat16* source = sources.values[state] + token * hidden;
        float partial = 0.0F;
        for (std::int32_t h = tid; h < hidden; h += kBlock) {
            const float value = __bfloat162float(source[h]);
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
            normalized[state * hidden + h] = __bfloat162float(source[h]) * inv_rms;
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
        const std::int32_t output_base = (query * gridDim.x + token) * hidden;
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

__device__ __forceinline__ float inkling_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void mix_depth_states_warp_kernel(DepthSources sources, const float* queries,
                                             __nv_bfloat16* out, std::int32_t hidden,
                                             std::int32_t depth, std::int32_t query_count,
                                             std::int32_t tokens) {
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t item = blockIdx.x * 8 + warp;
    const std::int32_t query = item % query_count;
    const std::int32_t token = item / query_count;
    if (token >= tokens) { return; }
    float inverse_rms[5];
    float scores[5];
    for (std::int32_t state = 0; state < depth; ++state) {
        const __nv_bfloat16* source = sources.values[state] + token * hidden;
        float sum = 0.0F;
        for (std::int32_t h = lane; h < hidden; h += 32) {
            const float value = __bfloat162float(source[h]);
            sum += value * value;
        }
        sum = inkling_warp_sum(sum);
        inverse_rms[state] = rsqrtf(__shfl_sync(0xffffffffU, sum, 0) / hidden + 1.0e-6F);
    }
    const float score_scale = rsqrtf(static_cast<float>(hidden));
    for (std::int32_t state = 0; state < depth; ++state) {
        const __nv_bfloat16* source = sources.values[state] + token * hidden;
        float sum = 0.0F;
        for (std::int32_t h = lane; h < hidden; h += 32) {
            sum += __bfloat162float(source[h]) * inverse_rms[state] * queries[query * hidden + h];
        }
        sum = inkling_warp_sum(sum);
        scores[state] = __shfl_sync(0xffffffffU, sum, 0) * score_scale;
    }
    float maximum = scores[0];
    for (std::int32_t state = 1; state < depth; ++state) { maximum = fmaxf(maximum, scores[state]); }
    float probabilities[5];
    float denominator = 0.0F;
    for (std::int32_t state = 0; state < depth; ++state) {
        probabilities[state] = expf(scores[state] - maximum);
        denominator += probabilities[state];
    }
    const std::int32_t output_base = (query * tokens + token) * hidden;
    for (std::int32_t h = lane; h < hidden; h += 32) {
        float value = 0.0F;
        for (std::int32_t state = 0; state < depth; ++state) {
            value += __bfloat162float(sources.values[state][token * hidden + h]) *
                     inverse_rms[state] * probabilities[state] / denominator;
        }
        out[output_base + h] = __float2bfloat16_rn(value);
    }
}

__global__ void extract_depth_output_kernel(const __nv_bfloat16* mixed, __nv_bfloat16* out,
                                            std::int32_t hidden, std::int32_t queries,
                                            std::int32_t tokens, std::int32_t query_index) {
    const std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int32_t count = hidden * tokens;
    if (index >= count) { return; }
    const std::int32_t token = index / hidden;
    const std::int32_t h = index - token * hidden;
    out[index] = mixed[(token * queries + query_index) * hidden + h];
}

__global__ void pack_loop_query_kernel(const float* q, const float* kv, float* out,
                                       std::int32_t hidden, std::int32_t pair) {
    const std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int32_t count = 4 * hidden;
    if (index >= count) { return; }
    const std::int32_t message = index / hidden;
    const std::int32_t h = index - message * hidden;
    const std::int32_t row = 2 * pair + (message == 2 || message == 3 ? 1 : 0);
    const float* source = message == 1 || message == 2 ? kv : q;
    out[index] = source[row * hidden + h];
}

__global__ void append_kv_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                 __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                                 std::int32_t channels, std::int32_t tokens,
                                 std::int32_t cache_tokens, std::int32_t position) {
    const std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int32_t count = channels * tokens;
    if (index >= count) { return; }
    const std::int32_t token = index / channels;
    const std::int32_t channel = index - token * channels;
    const std::int32_t destination = (position + token) * channels + channel;
    (void)cache_tokens;
    k_cache[destination] = k[index];
    v_cache[destination] = v[index];
}

__global__ void append_kv_ring_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                      __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                                      std::int32_t channels, std::int32_t tokens,
                                      std::int32_t capacity, std::int32_t position) {
    const std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int32_t count = channels * tokens;
    if (index >= count) { return; }
    const std::int32_t token = index / channels;
    const std::int32_t channel = index - token * channels;
    const std::int32_t destination = ((position + token) % capacity) * channels + channel;
    k_cache[destination] = k[index];
    v_cache[destination] = v[index];
}

__global__ void gather_kv_ring_kernel(const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                      __nv_bfloat16* k, __nv_bfloat16* v,
                                      std::int32_t channels, std::int32_t tokens,
                                      std::int32_t capacity, std::int32_t first_position) {
    const std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int32_t count = channels * tokens;
    if (index >= count) { return; }
    const std::int32_t token = index / channels;
    const std::int32_t channel = index - token * channels;
    const std::int32_t source = ((first_position + token) % capacity) * channels + channel;
    k[index] = k_cache[source];
    v[index] = v_cache[source];
}

__global__ void quantize_kv_int8_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                        std::int8_t* k_codes, float* k_scales,
                                        std::int8_t* v_codes, float* v_scales,
                                        std::int32_t channels, std::int32_t tokens,
                                        std::int32_t position) {
    __shared__ float reduction[128];
    __shared__ float k_scale, v_scale;
    const std::int32_t head = blockIdx.y;
    const std::int32_t token = blockIdx.x;
    const std::int32_t dim = threadIdx.x;
    const std::int32_t offset = token * channels + head * 128 + dim;
    reduction[dim] = fabsf(__bfloat162float(k[offset]));
    __syncthreads();
    for (int delta = 64; delta; delta /= 2) {
        if (dim < delta) { reduction[dim] = fmaxf(reduction[dim], reduction[dim + delta]); }
        __syncthreads();
    }
    if (dim == 0) {
        const float maximum = reduction[0];
        k_scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
        k_scales[(position + token) * gridDim.y + head] = k_scale;
    }
    __syncthreads();
    k_codes[(position + token) * channels + head * 128 + dim] =
        static_cast<std::int8_t>(__float2int_rn(__bfloat162float(k[offset]) / k_scale));

    reduction[dim] = fabsf(__bfloat162float(v[offset]));
    __syncthreads();
    for (int delta = 64; delta; delta /= 2) {
        if (dim < delta) { reduction[dim] = fmaxf(reduction[dim], reduction[dim + delta]); }
        __syncthreads();
    }
    if (dim == 0) {
        const float maximum = reduction[0];
        v_scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
        v_scales[(position + token) * gridDim.y + head] = v_scale;
    }
    __syncthreads();
    v_codes[(position + token) * channels + head * 128 + dim] =
        static_cast<std::int8_t>(__float2int_rn(__bfloat162float(v[offset]) / v_scale));
}

__device__ __forceinline__ float attention_score_int8(
    const __nv_bfloat16* q, const std::int8_t* k, const float* k_scales,
    const __nv_bfloat16* r, const __nv_bfloat16* relative_bias,
    const __nv_bfloat16* sink_logits, std::int32_t key, std::int32_t query, std::int32_t head,
    std::int32_t heads, std::int32_t kv_heads, std::int32_t tokens, std::int32_t keys,
    std::int32_t relative_length, std::int32_t extent, float scale) {
    const std::int32_t query_position = keys - tokens + query;
    const std::int32_t distance = query_position - key;
    if (key < 0 || key >= keys || distance < 0) { return -CUDART_INF_F; }
    const std::int32_t q_base = (query * heads + head) * 128;
    const std::int32_t k_base = key * kv_heads * 128;
    const std::int32_t kv_head = head / (heads / kv_heads);
    float dot = 0.0F;
    for (std::int32_t d = 0; d < 128; ++d) {
        dot += __bfloat162float(q[q_base + d]) * static_cast<float>(k[k_base + d]);
    }
    float score = dot * k_scales[key * kv_heads + kv_head] * scale;
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

__global__ void inkling_attention_int8_kernel(
    const __nv_bfloat16* q, const std::int8_t* k, const float* k_scales,
    const std::int8_t* v, const float* v_scales, const __nv_bfloat16* r,
    const __nv_bfloat16* relative_bias, const __nv_bfloat16* sink_logits,
    __nv_bfloat16* out, std::int32_t heads, std::int32_t kv_heads, std::int32_t tokens,
    std::int32_t keys, std::int32_t relative_length, std::int32_t extent, float scale) {
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
    const __nv_bfloat16* head_q = q;
    float accumulator = 0.0F;
    if (tid == 0) {
        running_max = 0.0F;
        running_denominator = 1.0F;
    }
    __syncthreads();
    for (std::int32_t tile = 0; tile < keys; tile += kTile) {
        scores[tid] = attention_score_int8(head_q, k + kv_head * 128, k_scales, r,
            relative_bias, sink_logits, tile + tid, query, head, heads, kv_heads, tokens, keys,
            relative_length, extent, scale);
        scores[tid + kBlock] = attention_score_int8(head_q, k + kv_head * 128, k_scales, r,
            relative_bias, sink_logits, tile + tid + kBlock, query, head, heads, kv_heads, tokens,
            keys, relative_length, extent, scale);
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
                    static_cast<float>(v[(key * kv_heads + kv_head) * 128 + tid]) *
                    v_scales[key * kv_heads + kv_head];
            }
        }
        __syncthreads();
    }
    out[(query * heads + head) * 128 + tid] =
        __float2bfloat16_rn(accumulator / running_denominator);
}

__global__ void bf16_gemv_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                 __nv_bfloat16* out, std::int32_t rows, std::int32_t columns) {
    constexpr int kWarps = 8;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t row = blockIdx.x * kWarps + warp;
    if (row >= rows) { return; }
    const __nv_bfloat16* source = weight + row * columns;
    float sum = 0.0F;
    if ((columns & 7) == 0) {
        const auto* source4 = reinterpret_cast<const uint4*>(source);
        const auto* x4 = reinterpret_cast<const uint4*>(x);
        const std::int32_t packs = columns / 8;
        for (std::int32_t pack = lane; pack < packs; pack += 32) {
            const uint4 w = source4[pack];
            const uint4 a = x4[pack];
            const std::uint32_t ww[4] = {w.x, w.y, w.z, w.w};
            const std::uint32_t aa[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const float2 weight_pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ww[part]));
                const float2 activation_pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&aa[part]));
                sum += weight_pair.x * activation_pair.x + weight_pair.y * activation_pair.y;
            }
        }
    } else if ((columns & 1) == 0) {
        const auto* source2 = reinterpret_cast<const __nv_bfloat162*>(source);
        const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
        const std::int32_t pairs = columns / 2;
        for (std::int32_t pair = lane; pair < pairs; pair += 32) {
            const float2 w = __bfloat1622float2(source2[pair]);
            const float2 a = __bfloat1622float2(x2[pair]);
            sum += w.x * a.x + w.y * a.y;
        }
    } else {
        for (std::int32_t column = lane; column < columns; column += 32) {
            sum += __bfloat162float(source[column]) * __bfloat162float(x[column]);
        }
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0) { out[row] = __float2bfloat16_rn(sum); }
}

template <int Tokens>
__global__ void bf16_gemv_multi_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                       __nv_bfloat16* out, std::int32_t rows,
                                       std::int32_t columns) {
    constexpr int kWarps = 8;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t row = blockIdx.x * kWarps + warp;
    if (row >= rows) { return; }
    const __nv_bfloat16* source = weight + row * columns;
    float sums[Tokens];
#pragma unroll
    for (int token = 0; token < Tokens; ++token) { sums[token] = 0.0F; }
    if ((columns & 7) == 0) {
        const auto* source4 = reinterpret_cast<const uint4*>(source);
        const auto* x4 = reinterpret_cast<const uint4*>(x);
        const std::int32_t packs = columns / 8;
        for (std::int32_t pack = lane; pack < packs; pack += 32) {
            const uint4 w = source4[pack];
            const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
            float weights[8];
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const float2 values = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(&words[part]));
                weights[2 * part] = values.x;
                weights[2 * part + 1] = values.y;
            }
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                const uint4 a = x4[token * packs + pack];
                const std::uint32_t activations[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    const float2 values = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&activations[part]));
                    sums[token] += weights[2 * part] * values.x +
                                   weights[2 * part + 1] * values.y;
                }
            }
        }
    } else {
        for (std::int32_t column = lane; column < columns; column += 32) {
            const float w = __bfloat162float(source[column]);
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                sums[token] += w * __bfloat162float(x[token * columns + column]);
            }
        }
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        float sum = sums[token];
        for (int offset = 16; offset != 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) { out[token * rows + row] = __float2bfloat16_rn(sum); }
    }
}

struct BF16GemvGroup {
    BF16GemvItem items[8];
};

template <int Tokens>
__global__ void bf16_gemv_group_kernel(BF16GemvGroup group, std::int32_t total_rows,
                                       std::int32_t columns) {
    constexpr int kWarps = 8;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t global_row = blockIdx.x * kWarps + warp;
    if (global_row >= total_rows) { return; }
    int item_index = 0;
    while (item_index < 7 && global_row >= group.items[item_index].row_begin +
           group.items[item_index].rows) { ++item_index; }
    const BF16GemvItem& item = group.items[item_index];
    const std::int32_t row = global_row - item.row_begin;
    const auto* x = static_cast<const __nv_bfloat16*>(item.input);
    const auto* source = static_cast<const __nv_bfloat16*>(item.weight) + row * columns;
    auto* out = static_cast<__nv_bfloat16*>(item.output);
    float sums[Tokens];
#pragma unroll
    for (int token = 0; token < Tokens; ++token) { sums[token] = 0.0F; }
    const auto* source4 = reinterpret_cast<const uint4*>(source);
    const auto* x4 = reinterpret_cast<const uint4*>(x);
    const std::int32_t packs = columns / 8;
    for (std::int32_t pack = lane; pack < packs; pack += 32) {
        const uint4 w = source4[pack];
        const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
        float weights[8];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const float2 values = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&words[part]));
            weights[2 * part] = values.x;
            weights[2 * part + 1] = values.y;
        }
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const uint4 a = x4[token * packs + pack];
            const std::uint32_t activations[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const float2 values = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(&activations[part]));
                sums[token] += weights[2 * part] * values.x + weights[2 * part + 1] * values.y;
            }
        }
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        float sum = sums[token];
        for (int offset = 16; offset != 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) { out[token * item.rows + row] = __float2bfloat16_rn(sum); }
    }
}

template <int Tokens>
__global__ void bf16_gemv_split_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                       __nv_bfloat16* out, std::int32_t rows,
                                       std::int32_t columns) {
    constexpr int kWarps = 8;
    __shared__ float partial[kWarps][Tokens];
    const std::int32_t row = blockIdx.x;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t warp = threadIdx.x >> 5;
    const __nv_bfloat16* source = weight + row * columns;
    float sums[Tokens];
#pragma unroll
    for (int token = 0; token < Tokens; ++token) { sums[token] = 0.0F; }
    if ((columns & 7) == 0) {
        const auto* source4 = reinterpret_cast<const uint4*>(source);
        const auto* x4 = reinterpret_cast<const uint4*>(x);
        const std::int32_t packs = columns / 8;
        const std::int32_t begin = warp * packs / kWarps;
        const std::int32_t end = (warp + 1) * packs / kWarps;
        for (std::int32_t pack = begin + lane; pack < end; pack += 32) {
            const uint4 w = source4[pack];
            const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
            float weights[8];
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const float2 values = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(&words[part]));
                weights[2 * part] = values.x;
                weights[2 * part + 1] = values.y;
            }
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                const uint4 a = x4[token * packs + pack];
                const std::uint32_t activations[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    const float2 values = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&activations[part]));
                    sums[token] += weights[2 * part] * values.x +
                                   weights[2 * part + 1] * values.y;
                }
            }
        }
    } else {
        const std::int32_t begin = warp * columns / kWarps;
        const std::int32_t end = (warp + 1) * columns / kWarps;
        for (std::int32_t column = begin + lane; column < end; column += 32) {
            const float w = __bfloat162float(source[column]);
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                sums[token] += w * __bfloat162float(x[token * columns + column]);
            }
        }
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        float sum = sums[token];
        for (int offset = 16; offset != 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) { partial[warp][token] = sum; }
    }
    __syncthreads();
    if (warp == 0 && lane < Tokens) {
        float sum = 0.0F;
#pragma unroll
        for (int part = 0; part < kWarps; ++part) { sum += partial[part][lane]; }
        out[lane * rows + row] = __float2bfloat16_rn(sum);
    }
}

template <int Tokens>
void launch_bf16_gemv(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                      __nv_bfloat16* out, std::int32_t rows, std::int32_t columns,
                      int blocks, cudaStream_t stream) {
    if constexpr (Tokens == 1) {
        bf16_gemv_kernel<<<blocks, 256, 0, stream>>>(x, weight, out, rows, columns);
    } else {
        bf16_gemv_multi_kernel<Tokens><<<blocks, 256, 0, stream>>>(x, weight, out, rows, columns);
    }
}

__global__ void fill_positions_kernel(std::int32_t* positions, std::int32_t count,
                                      std::int32_t first) {
    const std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) { positions[index] = first + index; }
}

int blocks_for(std::int64_t count) {
    return static_cast<int>(ops::div_up(count, static_cast<std::int64_t>(256)));
}

} // namespace

void pack_depth_state(const Tensor& source, Tensor& states, std::int32_t state_index,
                      cudaStream_t stream) {
    const std::int32_t hidden = source.ne[0];
    const std::int32_t tokens = source.ne[1];
    pack_depth_state_kernel<<<blocks_for(source.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source.data), static_cast<__nv_bfloat16*>(states.data),
        hidden, tokens, states.ne[1], state_index);
    CUDA_CHECK(cudaGetLastError());
}

void mix_depth_states(const std::array<const void*, 5>& sources, std::int32_t depth,
                      const Tensor& query, Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    DepthSources device_sources{};
    for (int index = 0; index < depth; ++index) {
        device_sources.values[index] = static_cast<const __nv_bfloat16*>(sources[index]);
    }
    static const bool use_warp_prefill = std::getenv("NINFER_INKLING_PREFILL_WARP_MIX") != nullptr;
    if (use_warp_prefill && out.ne[1] > 8 && depth > 0 && depth <= 5) {
        const int blocks = static_cast<int>(ops::div_up(
            static_cast<std::int64_t>(out.ne[1]) * query.ne[1], static_cast<std::int64_t>(8)));
        mix_depth_states_warp_kernel<<<blocks, kBlock, 0, stream>>>(device_sources,
            static_cast<const float*>(query.data), static_cast<__nv_bfloat16*>(out.data),
            query.ne[0], depth, query.ne[1], out.ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::size_t shared_bytes =
        static_cast<std::size_t>(query.ne[0] * depth + kBlock + 2 * depth) * sizeof(float);
    if (depth <= 0 || depth > 5 || shared_bytes > 48 * 1024) {
        throw std::invalid_argument("Inkling direct depth-mix geometry is invalid");
    }
    mix_depth_states_kernel<<<out.ne[1], kBlock, shared_bytes, stream>>>(device_sources,
        static_cast<const float*>(query.data), static_cast<__nv_bfloat16*>(out.data),
        query.ne[0], depth, query.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void extract_depth_output(const Tensor& mixed, std::int32_t query_index, Tensor& out,
                          cudaStream_t stream) {
    extract_depth_output_kernel<<<blocks_for(out.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(mixed.data), static_cast<__nv_bfloat16*>(out.data),
        out.ne[0], mixed.ne[1], out.ne[1], query_index);
    CUDA_CHECK(cudaGetLastError());
}

void pack_loop_query(const Tensor& q, const Tensor& kv, std::int32_t core_index, Tensor& out,
                     cudaStream_t stream) {
    pack_loop_query_kernel<<<blocks_for(out.numel()), 256, 0, stream>>>(
        static_cast<const float*>(q.data), static_cast<const float*>(kv.data),
        static_cast<float*>(out.data), out.ne[0], core_index);
    CUDA_CHECK(cudaGetLastError());
}

void append_kv(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache,
               std::int32_t position, cudaStream_t stream) {
    append_kv_kernel<<<blocks_for(k.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<__nv_bfloat16*>(k_cache.data), static_cast<__nv_bfloat16*>(v_cache.data),
        k.ne[0], k.ne[1], k_cache.ne[1], position);
    CUDA_CHECK(cudaGetLastError());
}

void append_kv_ring(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache,
                    std::int32_t position, std::int32_t capacity, cudaStream_t stream) {
    append_kv_ring_kernel<<<blocks_for(k.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<__nv_bfloat16*>(k_cache.data), static_cast<__nv_bfloat16*>(v_cache.data),
        k.ne[0], k.ne[1], capacity, position);
    CUDA_CHECK(cudaGetLastError());
}

void gather_kv_ring(const Tensor& k_cache, const Tensor& v_cache, std::int32_t first_position,
                    std::int32_t count, std::int32_t capacity, Tensor& k, Tensor& v,
                    cudaStream_t stream) {
    gather_kv_ring_kernel<<<blocks_for(k.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k_cache.data), static_cast<const __nv_bfloat16*>(v_cache.data),
        static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(v.data),
        k.ne[0], count, capacity, first_position);
    CUDA_CHECK(cudaGetLastError());
}

void append_kv_int8(const Tensor& k, const Tensor& v, Tensor& k_codes, Tensor& k_scales,
                    Tensor& v_codes, Tensor& v_scales, std::int32_t position,
                    cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(k.ne[1]),
                    static_cast<unsigned int>(k_scales.ne[0]));
    quantize_kv_int8_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<std::int8_t*>(k_codes.data), static_cast<float*>(k_scales.data),
        static_cast<std::int8_t*>(v_codes.data), static_cast<float*>(v_scales.data),
        k.ne[0], k.ne[1], position);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_attention_int8(const Tensor& q, const Tensor& k_codes, const Tensor& k_scales,
                            const Tensor& v_codes, const Tensor& v_scales, const Tensor& r,
                            const Tensor& relative_bias, const Tensor& sink_logits, float scale,
                            Tensor& out, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(q.ne[2]), static_cast<unsigned int>(q.ne[1]));
    inkling_attention_int8_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const std::int8_t*>(k_codes.data),
        static_cast<const float*>(k_scales.data), static_cast<const std::int8_t*>(v_codes.data),
        static_cast<const float*>(v_scales.data), static_cast<const __nv_bfloat16*>(r.data),
        static_cast<const __nv_bfloat16*>(relative_bias.data),
        static_cast<const __nv_bfloat16*>(sink_logits.data), static_cast<__nv_bfloat16*>(out.data),
        q.ne[1], k_scales.ne[0], q.ne[2], k_codes.ne[1], r.ne[0], relative_bias.ne[1], scale);
    CUDA_CHECK(cudaGetLastError());
}

void fill_positions(Tensor& positions, std::int32_t first, cudaStream_t stream) {
    fill_positions_kernel<<<blocks_for(positions.numel()), 256, 0, stream>>>(
        static_cast<std::int32_t*>(positions.data), positions.ne[0], first);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kWarps = 8;
    const int blocks = static_cast<int>(ops::div_up(static_cast<std::int64_t>(weight.n),
                                                    static_cast<std::int64_t>(kWarps)));
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const auto* source = static_cast<const __nv_bfloat16*>(weight.qdata);
    auto* output = static_cast<__nv_bfloat16*>(out.data);
    switch (x.ne[1]) {
    case 1:
        launch_bf16_gemv<1>(input, source, output, weight.n, weight.k, blocks, stream);
        break;
    case 2:
        launch_bf16_gemv<2>(input, source, output, weight.n, weight.k, blocks, stream);
        break;
    case 3:
        launch_bf16_gemv<3>(input, source, output, weight.n, weight.k, blocks, stream);
        break;
    case 4:
        launch_bf16_gemv<4>(input, source, output, weight.n, weight.k, blocks, stream);
        break;
    case 5:
        launch_bf16_gemv<5>(input, source, output, weight.n, weight.k, blocks, stream);
        break;
    case 6:
        launch_bf16_gemv<6>(input, source, output, weight.n, weight.k, blocks, stream);
        break;
    case 7:
        launch_bf16_gemv<7>(input, source, output, weight.n, weight.k, blocks, stream);
        break;
    case 8:
        launch_bf16_gemv<8>(input, source, output, weight.n, weight.k, blocks, stream);
        break;
    default:
        throw std::invalid_argument("Inkling BF16 GEMV supports one through eight tokens");
    }
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gemv_group(const std::array<BF16GemvItem, 8>& items, std::int32_t count,
                     std::int32_t columns, std::int32_t tokens, cudaStream_t stream) {
    if (count <= 0 || count > 8 || columns <= 0 || (columns & 7) != 0 || tokens <= 0) {
        throw std::invalid_argument("Inkling grouped BF16 GEMV geometry is invalid");
    }
    if (tokens > 8) {
        for (std::int32_t begin = 0; begin < tokens; begin += 8) {
            auto chunk = items;
            for (int index = 0; index < count; ++index) {
                chunk[index].input = static_cast<const std::uint16_t*>(items[index].input) +
                                     static_cast<std::size_t>(begin) * columns;
                chunk[index].output = static_cast<std::uint16_t*>(items[index].output) +
                                      static_cast<std::size_t>(begin) * items[index].rows;
            }
            bf16_gemv_group(chunk, count, columns, std::min(8, tokens - begin), stream);
        }
        return;
    }
    BF16GemvGroup group{};
    std::int32_t total_rows = 0;
    for (int index = 0; index < count; ++index) {
        group.items[index] = items[index];
        group.items[index].row_begin = total_rows;
        total_rows += items[index].rows;
    }
    const int blocks = static_cast<int>(ops::div_up(static_cast<std::int64_t>(total_rows),
                                                    static_cast<std::int64_t>(8)));
    switch (tokens) {
    case 1: bf16_gemv_group_kernel<1><<<blocks, 256, 0, stream>>>(group, total_rows, columns); break;
    case 2: bf16_gemv_group_kernel<2><<<blocks, 256, 0, stream>>>(group, total_rows, columns); break;
    case 3: bf16_gemv_group_kernel<3><<<blocks, 256, 0, stream>>>(group, total_rows, columns); break;
    case 4: bf16_gemv_group_kernel<4><<<blocks, 256, 0, stream>>>(group, total_rows, columns); break;
    case 5: bf16_gemv_group_kernel<5><<<blocks, 256, 0, stream>>>(group, total_rows, columns); break;
    case 6: bf16_gemv_group_kernel<6><<<blocks, 256, 0, stream>>>(group, total_rows, columns); break;
    case 7: bf16_gemv_group_kernel<7><<<blocks, 256, 0, stream>>>(group, total_rows, columns); break;
    case 8: bf16_gemv_group_kernel<8><<<blocks, 256, 0, stream>>>(group, total_rows, columns); break;
    default: throw std::invalid_argument("Inkling grouped BF16 GEMV supports one through eight tokens");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::inkling2_mini::detail
