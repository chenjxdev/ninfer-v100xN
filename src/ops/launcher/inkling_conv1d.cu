#include "ops/launcher/inkling_conv1d.h"

#include "ops/common/math.h"
#include "ops/kernel/inkling_conv1d.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {

void inkling_conv1d_launch(const Tensor& x, const Tensor& weight, const Tensor* dynamic,
                           const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                           cudaStream_t stream, Tensor* state_history) {
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>(
        div_up(static_cast<std::int64_t>(x.ne[0]), static_cast<std::int64_t>(kBlock)));
    const bool transposed = weight.ne[0] == 4;
    const std::int32_t heads = dynamic == nullptr ? 0 : dynamic->ne[0] / 4;
    inkling_conv1d_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        dynamic == nullptr ? nullptr : static_cast<const __nv_bfloat16*>(dynamic->data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), static_cast<__nv_bfloat16*>(out.data),
        x.ne[0], x.ne[1], heads, transposed,
        state_history == nullptr ? nullptr : static_cast<__nv_bfloat16*>(state_history->data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
