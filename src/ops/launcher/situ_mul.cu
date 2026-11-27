#include "ops/launcher/situ_mul.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/situ_mul.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

bool can_use_dim0_split(const Tensor& gate, const Tensor& up, const Tensor& out) {
    if (gate.ne[2] != 1 || gate.ne[3] != 1 || (gate.ne[0] & 1) != 0 ||
        gate.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16)) ||
        up.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16)) ||
        gate.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162)) != 0 ||
        up.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162)) != 0) {
        return false;
    }
    const auto addresses = reinterpret_cast<std::uintptr_t>(gate.data) |
                           reinterpret_cast<std::uintptr_t>(up.data) |
                           reinterpret_cast<std::uintptr_t>(out.data);
    return (addresses & (alignof(__nv_bfloat162) - 1)) == 0;
}

} // namespace

void situ_mul_launch(const Tensor& gate, const Tensor& up, float beta, float linear_beta,
                     Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    constexpr int kBlock = 128;
    if (!gate.is_contiguous() || !up.is_contiguous()) {
        if (can_use_dim0_split(gate, up, out)) {
            const std::int64_t row_pairs = gate.ne[0] / 2;
            constexpr std::int64_t kPairsPerBlock = kBlock * kSituMulPairsPerThread;
            const int grid_x = static_cast<int>(std::max<std::int64_t>(1, div_up(row_pairs, kPairsPerBlock)));
            for_each_token_slice(gate.ne[1], 1, [&](int token_offset, int token_count) {
                const Tensor gate_slice = gate.slice(1, token_offset, token_count);
                const Tensor up_slice = up.slice(1, token_offset, token_count);
                Tensor out_slice = out.slice(1, token_offset, token_count);
                const dim3 grid(static_cast<unsigned int>(grid_x),
                                static_cast<unsigned int>(token_count));
                situ_mul_dim0_split_kernel<<<grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(gate_slice.data),
                    static_cast<const __nv_bfloat16*>(up_slice.data),
                    static_cast<__nv_bfloat16*>(out_slice.data), gate.ne[0],
                    gate.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
                    up.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)), beta, linear_beta);
                CUDA_CHECK(cudaGetLastError());
            });
            return;
        }
        const int grid = static_cast<int>(div_up(n, static_cast<std::int64_t>(kBlock)));
        situ_mul_strided_input_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(up.data),
            static_cast<__nv_bfloat16*>(out.data), n, gate.ne[0], gate.ne[1], gate.ne[2], gate.nb[0],
            gate.nb[1], gate.nb[2], gate.nb[3], up.nb[0], up.nb[1], up.nb[2], up.nb[3], beta,
            linear_beta);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const auto addresses = reinterpret_cast<std::uintptr_t>(gate.data) |
                           reinterpret_cast<std::uintptr_t>(up.data) |
                           reinterpret_cast<std::uintptr_t>(out.data);
    if ((addresses & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int grid = static_cast<int>(div_up(n, static_cast<std::int64_t>(kBlock)));
        situ_mul_scalar_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(up.data),
            static_cast<__nv_bfloat16*>(out.data), n, beta, linear_beta);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kSituMulPairsPerThread;
    const int grid = static_cast<int>(std::max<std::int64_t>(1, div_up(n2, kPairsPerBlock)));
    situ_mul_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data), n, beta,
        linear_beta);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
