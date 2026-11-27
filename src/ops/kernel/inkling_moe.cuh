#pragma once

#include "ops/launcher/inkling_moe.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ float inkling_moe_weight_at(
    const detail::InklingMoeWeightView& weight, std::int32_t row, std::int32_t column) {
    if (weight.mxfp8) {
        const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
        const auto* scales = static_cast<const __nv_bfloat16*>(weight.group_scales);
        return detail::decode_nvfp4_e4m3(codes[row * weight.k + column]) *
               __bfloat162float(scales[row * (weight.k / 32) + column / 32]);
    }
    if (!weight.nvfp4) {
        return __bfloat162float(
            static_cast<const __nv_bfloat16*>(weight.qdata)[row * weight.k + column]);
    }
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    const std::uint8_t packed = codes[row * weight.k / 2 + column / 2];
    const std::uint8_t code = (column & 1) == 0 ? (packed & 0x0fU) : (packed >> 4);
    const std::int32_t group = column / 16;
    const std::int32_t row_tile = row / 128;
    const std::int32_t row_inner = row % 128;
    const std::int32_t scale_tile = group / 4;
    const std::int32_t scale_lane = group % 4;
    const std::int32_t k_tiles = weight.k / 64;
    const std::size_t scale_offset =
        static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512U +
        static_cast<std::size_t>(row_inner % 32) * 16U +
        static_cast<std::size_t>(row_inner / 32) * 4U + scale_lane;
    return detail::decode_nvfp4_e2m1x2(code).x * detail::decode_nvfp4_e4m3(scales[scale_offset]) /
           weight.divisor;
}

__device__ __forceinline__ float inkling_e2m1(std::uint32_t code) {
    const std::uint32_t magnitude = code & 7U;
    const std::uint32_t sign = (code & 8U) << 28;
    if (magnitude == 0) { return __uint_as_float(sign); }
    if (magnitude == 1) { return __uint_as_float(sign | 0x3f000000U); }
    const std::uint32_t exponent = ((magnitude >> 1) - 1 + 127) << 23;
    const std::uint32_t mantissa = (magnitude & 1U) ? 0x00400000U : 0U;
    return __uint_as_float(sign | exponent | mantissa);
}

__device__ __forceinline__ float inkling_e4m3(std::uint8_t storage) {
    const std::uint32_t sign = static_cast<std::uint32_t>(storage & 0x80U) << 24;
    const std::uint32_t exponent = (storage >> 3) & 0x0fU;
    const std::uint32_t mantissa = storage & 7U;
    if (exponent == 0) {
        const float value = static_cast<float>(mantissa) * 0.001953125F;
        return (storage & 0x80U) == 0 ? value : -value;
    }
    if (exponent == 15 && mantissa == 7) { return __uint_as_float(sign | 0x7fc00000U); }
    return __uint_as_float(sign | ((exponent + 120) << 23) | (mantissa << 20));
}

__device__ __forceinline__ float inkling_moe_dot(
    const detail::InklingMoeWeightView& weight, std::int32_t row,
    const __nv_bfloat16* input, std::int32_t lane) {
    if (weight.mxfp8) {
        const auto* codes = static_cast<const std::uint8_t*>(weight.qdata) + row * weight.k;
        const auto* scales = static_cast<const __nv_bfloat16*>(weight.group_scales) + row * (weight.k / 32);
        float sum = 0.0F;
        for (std::int32_t column = lane; column < weight.k; column += 32) {
            const float scale = __bfloat162float(scales[column / 32]);
            sum += inkling_e4m3(codes[column]) * scale * __bfloat162float(input[column]);
        }
        return sum;
    }
    if (!weight.nvfp4) {
        const auto* source = static_cast<const __nv_bfloat16*>(weight.qdata) + row * weight.k;
        float sum = 0.0F;
        for (std::int32_t column = lane; column < weight.k; column += 32) {
            sum += __bfloat162float(source[column]) * __bfloat162float(input[column]);
        }
        return sum;
    }
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata) + row * weight.k / 2;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    const auto* input4 = reinterpret_cast<const uint4*>(input);
    const std::int32_t row_tile = row / 128;
    const std::int32_t row_inner = row % 128;
    const std::int32_t k_tiles = weight.k / 64;
    float sum = 0.0F;
    const std::int32_t groups = weight.k / 16;
    for (std::int32_t group = lane; group < groups; group += 32) {
        const std::int32_t scale_tile = group / 4;
        const std::int32_t scale_lane = group % 4;
        const std::size_t scale_offset =
            static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512U +
            static_cast<std::size_t>(row_inner % 32) * 16U +
            static_cast<std::size_t>(row_inner / 32) * 4U + scale_lane;
        const float scale = inkling_e4m3(scales[scale_offset]);
        const std::uint64_t packed =
            *reinterpret_cast<const std::uint64_t*>(codes + group * 8);
        const uint4 activation0 = input4[group * 2];
        const uint4 activation1 = input4[group * 2 + 1];
        const std::uint32_t activation_words[8] = {
            activation0.x, activation0.y, activation0.z, activation0.w,
            activation1.x, activation1.y, activation1.z, activation1.w};
        float block_sum = 0.0F;
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            const std::uint8_t storage = static_cast<std::uint8_t>(packed >> (byte * 8));
            const float2 value{inkling_e2m1(storage & 0x0fU), inkling_e2m1(storage >> 4)};
            const float2 activation = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&activation_words[byte]));
            block_sum += value.x * activation.x + value.y * activation.y;
        }
        sum += block_sum * scale;
    }
    return sum / weight.divisor;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

template <int Rows>
__device__ __forceinline__ void inkling_moe_dot_rows(
    const detail::InklingMoeWeightView& weight, std::int32_t first_row,
    const __nv_bfloat16* input, std::int32_t lane, const float* e2m1_table,
    float (&sums)[Rows]) {
#pragma unroll
    for (int row = 0; row < Rows; ++row) { sums[row] = 0.0F; }
    if (weight.mxfp8) {
        const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
        const auto* scales = static_cast<const __nv_bfloat16*>(weight.group_scales);
        const auto* input4 = reinterpret_cast<const uint4*>(input);
        const std::int32_t groups = weight.k / 32;
        for (std::int32_t group = lane; group < groups; group += 32) {
            const uint4 activation0 = input4[group * 4];
            const uint4 activation1 = input4[group * 4 + 1];
            const uint4 activation2 = input4[group * 4 + 2];
            const uint4 activation3 = input4[group * 4 + 3];
            const std::uint32_t activation_words[16] = {
                activation0.x, activation0.y, activation0.z, activation0.w,
                activation1.x, activation1.y, activation1.z, activation1.w,
                activation2.x, activation2.y, activation2.z, activation2.w,
                activation3.x, activation3.y, activation3.z, activation3.w};
            float block_sums[Rows];
            float row_scales[Rows];
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                block_sums[row] = 0.0F;
                row_scales[row] = __bfloat162float(scales[(first_row + row) * groups + group]);
            }
#pragma unroll
            for (int pair = 0; pair < 16; ++pair) {
                const float2 activation = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(&activation_words[pair]));
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    const std::uint8_t* row_codes = codes + (first_row + row) * weight.k + group * 32 + pair * 2;
                    block_sums[row] += inkling_e4m3(row_codes[0]) * activation.x +
                                       inkling_e4m3(row_codes[1]) * activation.y;
                }
            }
#pragma unroll
            for (int row = 0; row < Rows; ++row) { sums[row] += block_sums[row] * row_scales[row]; }
        }
    } else if (!weight.nvfp4) {
        const auto* source = static_cast<const __nv_bfloat16*>(weight.qdata);
        for (std::int32_t column = lane; column < weight.k; column += 32) {
            const float activation = __bfloat162float(input[column]);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                sums[row] += __bfloat162float(source[(first_row + row) * weight.k + column]) * activation;
            }
        }
    } else {
        const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
        const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
        const auto* input4 = reinterpret_cast<const uint4*>(input);
        const std::int32_t k_tiles = weight.k / 64;
        const std::int32_t groups = weight.k / 16;
        for (std::int32_t group = lane; group < groups; group += 32) {
            std::uint64_t packed[Rows];
            float row_scales[Rows];
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const std::int32_t logical_row = first_row + row;
                const std::int32_t row_tile = logical_row / 128;
                const std::int32_t row_inner = logical_row % 128;
                const std::size_t scale_offset =
                    static_cast<std::size_t>(row_tile * k_tiles + group / 4) * 512U +
                    static_cast<std::size_t>(row_inner % 32) * 16U +
                    static_cast<std::size_t>(row_inner / 32) * 4U + group % 4;
                packed[row] = *reinterpret_cast<const std::uint64_t*>(
                    codes + logical_row * weight.k / 2 + group * 8);
                row_scales[row] = inkling_e4m3(scales[scale_offset]);
            }
            const uint4 activation0 = input4[group * 2];
            const uint4 activation1 = input4[group * 2 + 1];
            const std::uint32_t activation_words[8] = {
                activation0.x, activation0.y, activation0.z, activation0.w,
                activation1.x, activation1.y, activation1.z, activation1.w};
            float block_sums[Rows];
#pragma unroll
            for (int row = 0; row < Rows; ++row) { block_sums[row] = 0.0F; }
#pragma unroll
            for (int byte = 0; byte < 8; ++byte) {
                const float2 activation = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(&activation_words[byte]));
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    const std::uint8_t storage = static_cast<std::uint8_t>(packed[row] >> (byte * 8));
                    const float2 value{e2m1_table[storage & 0x0fU], e2m1_table[storage >> 4]};
                    block_sums[row] += value.x * activation.x + value.y * activation.y;
                }
            }
#pragma unroll
            for (int row = 0; row < Rows; ++row) { sums[row] += block_sums[row] * row_scales[row]; }
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) { sums[row] /= weight.divisor; }
}

__global__ void inkling_moe_gate_up_fused_kernel(
    const __nv_bfloat16* x, const detail::InklingMoeWeightView* w13,
    const std::int32_t* selected, __nv_bfloat16* activations, std::int32_t hidden,
    std::int32_t intermediate, std::int32_t top_k, float beta, float linear_beta) {
    __shared__ float e2m1_table[16];
    if (threadIdx.x < 16) { e2m1_table[threadIdx.x] = inkling_e2m1(threadIdx.x); }
    __syncthreads();
    constexpr int kWarps = 8;
    const std::int32_t token = blockIdx.x;
    const std::int32_t route = blockIdx.z;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t i = blockIdx.y * kWarps + warp;
    if (i >= intermediate) { return; }
    const std::int32_t expert = selected[token * top_k + route];
    float sums[2];
    inkling_moe_dot_rows<2>(w13[expert], 2 * i, x + token * hidden, lane, e2m1_table, sums);
    float gate = warp_sum(sums[0]);
    float up = warp_sum(sums[1]);
    if (lane == 0) {
        const float gated = beta * tanhf(gate / beta) / (1.0F + expf(-gate));
        const float limited_up = linear_beta > 0.0F ? linear_beta * tanhf(up / linear_beta) : up;
        activations[(token * top_k + route) * intermediate + i] =
            __float2bfloat16_rn(gated * limited_up);
    }
}

__global__ void inkling_moe_down_rows4_kernel(
    const __nv_bfloat16* activations, const detail::InklingMoeWeightView* w2,
    const std::int32_t* selected, const float* route_weights, float* accumulation,
    std::int32_t hidden, std::int32_t intermediate, std::int32_t top_k) {
    __shared__ float e2m1_table[16];
    if (threadIdx.x < 16) { e2m1_table[threadIdx.x] = inkling_e2m1(threadIdx.x); }
    __syncthreads();
    constexpr int kWarps = 8;
    const std::int32_t token = blockIdx.x;
    const std::int32_t route = blockIdx.z;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t h = (blockIdx.y * kWarps + warp) * 4;
    if (h >= hidden) { return; }
    const std::int32_t expert = selected[token * top_k + route];
    float sums[4];
    inkling_moe_dot_rows<4>(w2[expert], h,
        activations + (token * top_k + route) * intermediate, lane, e2m1_table, sums);
    float results[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) { results[row] = warp_sum(sums[row]); }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < 4 && h + row < hidden; ++row) {
            accumulation[(route * gridDim.x + token) * hidden + h + row] =
                route_weights[token * top_k + route] * results[row];
        }
    }
}

// Four-row jobs use the host-free upper bound ceil(pairs/4)+128. Every sorted row retains its
// original flat assignment ID, so downstream route-major reduction is independent of binning order.
__global__ void inkling_moe_bin_kernel(const std::int32_t* selected,
                                       detail::InklingMoePrefillJob* jobs,
                                       std::int32_t* order, std::int32_t* job_count,
                                       std::int32_t pairs, std::int32_t top_k) {
    constexpr int kExperts = 128;
    constexpr int kTile = 4;
    __shared__ std::int32_t counts[kExperts];
    __shared__ std::int32_t offsets[kExperts + 1];
    __shared__ std::int32_t positions[kExperts];
    const std::int32_t tid = threadIdx.x;
    if (tid < kExperts) {
        counts[tid] = 0;
        offsets[tid] = 0;
        positions[tid] = 0;
    }
    __syncthreads();
    for (std::int32_t index = tid; index < pairs; index += kExperts) {
        const std::int32_t expert = selected[index];
        if (expert >= 0 && expert < kExperts) { atomicAdd(&counts[expert], 1); }
    }
    __syncthreads();
    if (tid == 0) {
        offsets[0] = 0;
        for (std::int32_t expert = 0; expert < kExperts; ++expert) {
            offsets[expert + 1] = offsets[expert] + counts[expert];
            positions[expert] = offsets[expert];
        }
    }
    __syncthreads();
    for (std::int32_t index = tid; index < pairs; index += kExperts) {
        const std::int32_t expert = selected[index];
        if (expert >= 0 && expert < kExperts) {
            const std::int32_t position = atomicAdd(&positions[expert], 1);
            order[position] = index;
        }
    }
    __syncthreads();
    if (tid == 0) {
        std::int32_t job_index = 0;
        for (std::int32_t expert = 0; expert < kExperts; ++expert) {
            const std::int32_t first = offsets[expert];
            const std::int32_t count = counts[expert];
            for (std::int32_t begin = 0; begin < count; begin += kTile) {
                jobs[job_index++] = detail::InklingMoePrefillJob{
                    expert, first + begin, min(kTile, count - begin), 0};
            }
        }
        *job_count = job_index;
    }
    (void)top_k;
}

template <int Rows, int Tile>
__device__ __forceinline__ void inkling_moe_prefill_dot(
    const detail::InklingMoeWeightView& weight, std::int32_t first_row,
    const __nv_bfloat16* x, std::int32_t hidden, std::int32_t top_k,
    const std::int32_t* order, std::int32_t first, std::int32_t count,
    std::int32_t lane, const float* e2m1_table, float (&sums)[Rows][Tile]) {
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int token = 0; token < Tile; ++token) { sums[row][token] = 0.0F; }
    }
    if (weight.mxfp8) {
        const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
        const auto* scales = static_cast<const __nv_bfloat16*>(weight.group_scales);
        const std::int32_t groups = weight.k / 32;
        for (std::int32_t group = lane; group < groups; group += 32) {
            float block[Rows][Tile];
            float row_scales[Rows];
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                row_scales[row] = __bfloat162float(scales[(first_row + row) * groups + group]);
#pragma unroll
                for (int token = 0; token < Tile; ++token) { block[row][token] = 0.0F; }
            }
#pragma unroll
            for (int token = 0; token < Tile; ++token) {
                if (token >= count) { continue; }
                const std::int32_t source_token = order[first + token] / top_k;
                const auto* input4 = reinterpret_cast<const uint4*>(x + source_token * hidden);
                const uint4 a0 = input4[group * 4];
                const uint4 a1 = input4[group * 4 + 1];
                const uint4 a2 = input4[group * 4 + 2];
                const uint4 a3 = input4[group * 4 + 3];
                const std::uint32_t words[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                    a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
#pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const float2 activation = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&words[pair]));
#pragma unroll
                    for (int row = 0; row < Rows; ++row) {
                        const std::uint8_t* rc = codes + (first_row + row) * weight.k + group * 32 + pair * 2;
                        const float w0 = inkling_e4m3(rc[0]);
                        const float w1 = inkling_e4m3(rc[1]);
                        block[row][token] += w0 * activation.x + w1 * activation.y;
                    }
                }
            }
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
#pragma unroll
                for (int token = 0; token < Tile; ++token) {
                    sums[row][token] += block[row][token] * row_scales[row];
                }
            }
        }
    } else if (!weight.nvfp4) {
        const auto* source = static_cast<const __nv_bfloat16*>(weight.qdata);
        for (std::int32_t column = lane; column < weight.k; column += 32) {
            float values[Rows];
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                values[row] = __bfloat162float(source[(first_row + row) * weight.k + column]);
            }
#pragma unroll
            for (int token = 0; token < Tile; ++token) {
                if (token >= count) { continue; }
                const std::int32_t source_token = order[first + token] / top_k;
                const float activation = __bfloat162float(x[source_token * hidden + column]);
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    sums[row][token] += values[row] * activation;
                }
            }
        }
    } else {
        const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
        const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
        const auto* input4 = reinterpret_cast<const uint4*>(x);
        const std::int32_t k_tiles = weight.k / 64;
        const std::int32_t groups = weight.k / 16;
        for (std::int32_t group = lane; group < groups; group += 32) {
            std::uint64_t packed[Rows];
            float row_scales[Rows];
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const std::int32_t logical_row = first_row + row;
                const std::int32_t row_tile = logical_row / 128;
                const std::int32_t row_inner = logical_row % 128;
                const std::size_t scale_offset =
                    static_cast<std::size_t>(row_tile * k_tiles + group / 4) * 512U +
                    static_cast<std::size_t>(row_inner % 32) * 16U +
                    static_cast<std::size_t>(row_inner / 32) * 4U + group % 4;
                packed[row] = *reinterpret_cast<const std::uint64_t*>(
                    codes + logical_row * weight.k / 2 + group * 8);
                row_scales[row] = inkling_e4m3(scales[scale_offset]);
            }
            float block[Rows][Tile];
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
#pragma unroll
                for (int token = 0; token < Tile; ++token) { block[row][token] = 0.0F; }
            }
#pragma unroll
            for (int token = 0; token < Tile; ++token) {
                if (token >= count) { continue; }
                const std::int32_t source_token = order[first + token] / top_k;
                const uint4 a0 = input4[source_token * hidden / 8 + group * 2];
                const uint4 a1 = input4[source_token * hidden / 8 + group * 2 + 1];
                const std::uint32_t words[8] = {a0.x, a0.y, a0.z, a0.w,
                                                a1.x, a1.y, a1.z, a1.w};
#pragma unroll
                for (int byte = 0; byte < 8; ++byte) {
                    const float2 activation = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&words[byte]));
#pragma unroll
                    for (int row = 0; row < Rows; ++row) {
                        const std::uint8_t storage = static_cast<std::uint8_t>(packed[row] >> (byte * 8));
                        const float2 value{e2m1_table[storage & 0x0fU], e2m1_table[storage >> 4]};
                        block[row][token] += value.x * activation.x + value.y * activation.y;
                    }
                }
            }
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
#pragma unroll
                for (int token = 0; token < Tile; ++token) {
                    sums[row][token] += block[row][token] * row_scales[row];
                }
            }
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int token = 0; token < Tile; ++token) { sums[row][token] /= weight.divisor; }
    }
}

__global__ void inkling_moe_prefill_gate_up_kernel(
    const __nv_bfloat16* x, const detail::InklingMoeWeightView* w13,
    const detail::InklingMoePrefillJob* jobs, const std::int32_t* order,
    const std::int32_t* job_count, __nv_bfloat16* activations, std::int32_t hidden,
    std::int32_t intermediate, std::int32_t top_k, float beta, float linear_beta) {
    __shared__ float e2m1_table[16];
    if (threadIdx.x < 16) { e2m1_table[threadIdx.x] = inkling_e2m1(threadIdx.x); }
    __syncthreads();
    if (blockIdx.x >= *job_count) { return; }
    constexpr int kWarps = 8;
    constexpr int kTile = 4;
    const auto job = jobs[blockIdx.x];
    const std::int32_t i = blockIdx.y * kWarps + (threadIdx.x >> 5);
    if (i >= intermediate) { return; }
    const std::int32_t lane = threadIdx.x & 31;
    float sums[2][kTile];
    inkling_moe_prefill_dot<2, kTile>(w13[job.expert], 2 * i, x, hidden, top_k, order,
        job.first_assignment, job.count, lane, e2m1_table, sums);
    for (int token = 0; token < job.count; ++token) {
        const std::int32_t assignment = order[job.first_assignment + token];
        const std::int32_t source_token = assignment / top_k;
        const std::int32_t route = assignment % top_k;
        const float gate = warp_sum(sums[0][token]);
        const float up = warp_sum(sums[1][token]);
        if (lane == 0) {
            const float gated = beta * tanhf(gate / beta) / (1.0F + expf(-gate));
            const float limited_up = linear_beta > 0.0F ? linear_beta * tanhf(up / linear_beta) : up;
            activations[(source_token * top_k + route) * intermediate + i] =
                __float2bfloat16_rn(gated * limited_up);
        }
    }
}

__global__ void inkling_moe_prefill_down_kernel(
    const __nv_bfloat16* activations, const detail::InklingMoeWeightView* w2,
    const detail::InklingMoePrefillJob* jobs, const std::int32_t* order,
    const std::int32_t* job_count, const float* route_weights, float* accumulation,
    std::int32_t tokens,
    std::int32_t hidden, std::int32_t intermediate, std::int32_t top_k) {
    __shared__ float e2m1_table[16];
    if (threadIdx.x < 16) { e2m1_table[threadIdx.x] = inkling_e2m1(threadIdx.x); }
    __syncthreads();
    if (blockIdx.x >= *job_count) { return; }
    constexpr int kWarps = 8;
    constexpr int kTile = 4;
    const auto job = jobs[blockIdx.x];
    const std::int32_t h = (blockIdx.y * kWarps + (threadIdx.x >> 5)) * 4;
    if (h >= hidden) { return; }
    const std::int32_t lane = threadIdx.x & 31;
    float sums[4][kTile];
    inkling_moe_prefill_dot<4, kTile>(w2[job.expert], h, activations, intermediate, top_k, order,
        job.first_assignment, job.count, lane, e2m1_table, sums);
    for (int token = 0; token < job.count; ++token) {
        const std::int32_t assignment = order[job.first_assignment + token];
        const std::int32_t source_token = assignment / top_k;
        const std::int32_t route = assignment % top_k;
        float results[4];
#pragma unroll
        for (int row = 0; row < 4; ++row) { results[row] = warp_sum(sums[row][token]); }
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < 4 && h + row < hidden; ++row) {
                accumulation[(route * tokens + source_token) * hidden + h + row] =
                    route_weights[assignment] * results[row];
            }
        }
    }
}

__global__ void inkling_moe_gate_up_kernel(
    const __nv_bfloat16* x, const detail::InklingMoeWeightView* w13,
    const std::int32_t* selected, float* projected, std::int32_t hidden,
    std::int32_t intermediate, std::int32_t top_k) {
    constexpr int kWarps = 8;
    const std::int32_t token = blockIdx.x;
    const std::int32_t route = blockIdx.z;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t branch = blockIdx.y * kWarps + warp;
    if (branch >= 2 * intermediate) { return; }
    const std::int32_t expert = selected[token * top_k + route];
    const detail::InklingMoeWeightView weight = w13[expert];
    const __nv_bfloat16* input = x + token * hidden;
    float sum = inkling_moe_dot(weight, branch, input, lane);
    sum = warp_sum(sum);
    if (lane == 0) { projected[(token * top_k + route) * 2 * intermediate + branch] = sum; }
}

__global__ void inkling_moe_situ_kernel(const float* projected, __nv_bfloat16* activations,
                                        std::int32_t intermediate, std::int32_t routes,
                                        float beta, float linear_beta) {
    const std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int32_t count = routes * intermediate;
    if (index >= count) { return; }
    const float gate = projected[2 * index];
    const float up = projected[2 * index + 1];
    const float gated = beta * tanhf(gate / beta) / (1.0F + expf(-gate));
    const float limited_up = linear_beta > 0.0F ? linear_beta * tanhf(up / linear_beta) : up;
    activations[index] = __float2bfloat16_rn(gated * limited_up);
}

__global__ void inkling_moe_down_kernel(
    const __nv_bfloat16* activations, const detail::InklingMoeWeightView* w2,
    const std::int32_t* selected, const float* route_weights, float* accumulation,
    std::int32_t hidden, std::int32_t intermediate, std::int32_t top_k) {
    constexpr int kWarps = 8;
    const std::int32_t token = blockIdx.x;
    const std::int32_t route = blockIdx.z;
    const std::int32_t lane = threadIdx.x & 31;
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t h = blockIdx.y * kWarps + warp;
    if (h >= hidden) { return; }
    const std::int32_t expert = selected[token * top_k + route];
    const detail::InklingMoeWeightView weight = w2[expert];
    const __nv_bfloat16* input = activations + (token * top_k + route) * intermediate;
    float sum = inkling_moe_dot(weight, h, input, lane);
    sum = warp_sum(sum);
    if (lane == 0) {
        atomicAdd(accumulation + token * hidden + h,
                  route_weights[token * top_k + route] * sum);
    }
}

__global__ void inkling_moe_reduce_kernel(const float* accumulation, __nv_bfloat16* out,
                                           std::int64_t count, std::int32_t hidden,
                                           std::int32_t top_k) {
    const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (index >= count) { return; }
    const std::int32_t token = static_cast<std::int32_t>(index / hidden);
    const std::int32_t h = static_cast<std::int32_t>(index % hidden);
    const std::int32_t tokens = static_cast<std::int32_t>(count / hidden);
    float sum = 0.0F;
    for (std::int32_t route = 0; route < top_k; ++route) {
        sum += accumulation[(route * tokens + token) * hidden + h];
    }
    out[index] = __float2bfloat16_rn(sum);
}

} // namespace ninfer::ops
