#include "ops/launcher/inkling_attention_dense.h"

#include "ops/kernel/inkling_attention_dense.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {

void inkling_attention_dense_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& r, const Tensor& relative_bias,
                                    const Tensor& sink_logits, float scale,
                                    std::uint32_t sliding_window, Tensor& out,
                                    cudaStream_t stream) {
    constexpr int kBlock = 128;
    const dim3 grid(static_cast<unsigned int>(q.ne[2]), static_cast<unsigned int>(q.ne[1]));
    inkling_attention_dense_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const __nv_bfloat16*>(r.data),
        static_cast<const __nv_bfloat16*>(relative_bias.data),
        static_cast<const __nv_bfloat16*>(sink_logits.data), static_cast<__nv_bfloat16*>(out.data),
        q.ne[1], k.ne[1], q.ne[2], k.ne[2], r.ne[0], relative_bias.ne[1], scale, sliding_window);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
