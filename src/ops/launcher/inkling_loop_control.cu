#include "ops/launcher/inkling_loop_control.h"

#include "ops/common/math.h"
#include "ops/kernel/inkling_loop_control.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {

void inkling_loop_control_launch(const Tensor& cur, const Tensor& message, const Tensor& down,
                                 const Tensor& gain, const Tensor& forget, Tensor& normalized,
                                 Tensor& gate, Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / 32;
    const std::int32_t hidden = cur.ne[0];
    const std::int32_t rank = down.ne[0];
    inkling_loop_control_normalize_kernel<<<cur.ne[1], kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(cur.data), static_cast<const __nv_bfloat16*>(message.data),
        static_cast<float*>(normalized.data), hidden);
    CUDA_CHECK(cudaGetLastError());
    const dim3 gate_grid(static_cast<unsigned int>(cur.ne[1]),
        static_cast<unsigned int>(div_up(static_cast<std::int64_t>(rank),
                                        static_cast<std::int64_t>(kWarps))));
    inkling_loop_control_gate_kernel<<<gate_grid, kBlock, 0, stream>>>(
        static_cast<const float*>(normalized.data), static_cast<const __nv_bfloat16*>(down.data),
        static_cast<float*>(gate.data), hidden, rank);
    CUDA_CHECK(cudaGetLastError());
    const dim3 output_grid(static_cast<unsigned int>(cur.ne[1]),
        static_cast<unsigned int>(div_up(static_cast<std::int64_t>(hidden),
                                        static_cast<std::int64_t>(kWarps))));
    inkling_loop_control_output_kernel<<<output_grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(cur.data), static_cast<const __nv_bfloat16*>(message.data),
        static_cast<const __nv_bfloat16*>(gain.data), static_cast<const __nv_bfloat16*>(forget.data),
        static_cast<const float*>(gate.data), static_cast<__nv_bfloat16*>(out.data), hidden, rank);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
