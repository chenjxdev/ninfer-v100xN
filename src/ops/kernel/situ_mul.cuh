#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSituMulPairsPerThread = 4;

__device__ __forceinline__ float situ_value(float gate, float up, float beta, float linear_beta) {
    const float gated = beta * tanhf(gate / beta) / (1.0F + expf(-gate));
    const float limited_up = linear_beta > 0.0F ? linear_beta * tanhf(up / linear_beta) : up;
    return gated * limited_up;
}

__device__ __forceinline__ __nv_bfloat162 situ_mul_pair(__nv_bfloat162 gate, __nv_bfloat162 up,
                                                        float beta, float linear_beta) {
    return __floats2bfloat162_rn(
        situ_value(__low2float(gate), __low2float(up), beta, linear_beta),
        situ_value(__high2float(gate), __high2float(up), beta, linear_beta));
}

__global__ void situ_mul_scalar_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                       __nv_bfloat16* out, std::int64_t n, float beta,
                                       float linear_beta) {
    const std::int64_t start = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        out[i] = __float2bfloat16_rn(situ_value(__bfloat162float(gate[i]),
                                                __bfloat162float(up[i]), beta, linear_beta));
    }
}

__global__ void situ_mul_strided_input_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, std::int64_t n,
    std::int32_t ne0, std::int32_t ne1, std::int32_t ne2, std::int64_t gnb0, std::int64_t gnb1,
    std::int64_t gnb2, std::int64_t gnb3, std::int64_t unb0, std::int64_t unb1, std::int64_t unb2,
    std::int64_t unb3, float beta, float linear_beta) {
    const std::int64_t start = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto* gate_bytes = reinterpret_cast<const unsigned char*>(gate);
    const auto* up_bytes = reinterpret_cast<const unsigned char*>(up);
    for (std::int64_t i = start; i < n; i += stride) {
        std::int64_t rem = i;
        const auto d0 = static_cast<std::int32_t>(rem % ne0);
        rem /= ne0;
        const auto d1 = static_cast<std::int32_t>(rem % ne1);
        rem /= ne1;
        const auto d2 = static_cast<std::int32_t>(rem % ne2);
        const auto d3 = static_cast<std::int32_t>(rem / ne2);
        const std::int64_t goff = static_cast<std::int64_t>(d0) * gnb0 + d1 * gnb1 + d2 * gnb2 +
                                  d3 * gnb3;
        const std::int64_t uoff = static_cast<std::int64_t>(d0) * unb0 + d1 * unb1 + d2 * unb2 +
                                  d3 * unb3;
        const auto gv = *reinterpret_cast<const __nv_bfloat16*>(gate_bytes + goff);
        const auto uv = *reinterpret_cast<const __nv_bfloat16*>(up_bytes + uoff);
        out[i] = __float2bfloat16_rn(
            situ_value(__bfloat162float(gv), __bfloat162float(uv), beta, linear_beta));
    }
}

__launch_bounds__(256) __global__ void situ_mul_dim0_split_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, std::int32_t rows,
    std::int64_t gate_col_stride, std::int64_t up_col_stride, float beta, float linear_beta) {
    const int col = blockIdx.y;
    const std::int64_t pair0 =
        (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) * kSituMulPairsPerThread;
    const std::int64_t row_pairs = rows / 2;
    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate + col * gate_col_stride);
    const auto* up2 = reinterpret_cast<const __nv_bfloat162*>(up + col * up_col_stride);
    auto* out2 = reinterpret_cast<__nv_bfloat162*>(out + col * static_cast<std::int64_t>(rows));
    for (int p = 0; p < kSituMulPairsPerThread; ++p) {
        const std::int64_t j = pair0 + p;
        if (j < row_pairs) { out2[j] = situ_mul_pair(gate2[j], up2[j], beta, linear_beta); }
    }
}

__launch_bounds__(256) __global__ void situ_mul_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, std::int64_t n,
    float beta, float linear_beta) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSituMulPairsPerThread;
    const std::int64_t n2 = n / 2;
    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    const auto* up2 = reinterpret_cast<const __nv_bfloat162*>(up);
    auto* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    for (std::int64_t j = tid * kSituMulPairsPerThread; j < n2; j += stride) {
        for (int p = 0; p < kSituMulPairsPerThread && j + p < n2; ++p) {
            out2[j + p] = situ_mul_pair(gate2[j + p], up2[j + p], beta, linear_beta);
        }
    }
    if (tid == 0 && (n & 1) != 0) {
        const std::int64_t i = n - 1;
        out[i] = __float2bfloat16_rn(situ_value(__bfloat162float(gate[i]),
                                                __bfloat162float(up[i]), beta, linear_beta));
    }
}

} // namespace ninfer::ops
