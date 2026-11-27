#include "ops/launcher/inkling_pointwise.h"

#include "ops/common/math.h"
#include "ops/kernel/inkling_pointwise.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {

void inkling_softcap_launch(const Tensor& x, float cap, Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int64_t count = x.numel();
    const int blocks = static_cast<int>(div_up(count, static_cast<std::int64_t>(kBlock)));
    inkling_softcap_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), cap, static_cast<__nv_bfloat16*>(out.data), count);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_log_position_scale_launch(const Tensor& x, const Tensor& positions,
                                       std::uint32_t floor, float alpha, Tensor& out,
                                       cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t rows_per_token = x.ne[0] * x.ne[1];
    const std::int64_t count = static_cast<std::int64_t>(rows_per_token) * x.ne[2];
    const int blocks = static_cast<int>(div_up(count, static_cast<std::int64_t>(kBlock)));
    inkling_log_position_scale_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::int32_t*>(positions.data),
        floor, alpha, static_cast<__nv_bfloat16*>(out.data), rows_per_token, x.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
