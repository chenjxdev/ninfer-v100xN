#include "ops/launcher/inkling_router.h"

#include "ops/common/math.h"
#include "ops/kernel/inkling_router.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {

void inkling_router_launch(const Tensor& x, const Tensor& gate, const Tensor& selection_bias,
                           float weight_scale, Tensor& probabilities, Tensor& selected,
                           Tensor& weights, cudaStream_t stream) {
    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / 32;
    const dim3 probability_grid(static_cast<unsigned int>(x.ne[1]),
        static_cast<unsigned int>(div_up(static_cast<std::int64_t>(gate.ne[0]),
                                        static_cast<std::int64_t>(kWarps))));
    inkling_router_probs_kernel<<<probability_grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<float*>(probabilities.data), x.ne[0], gate.ne[0]);
    CUDA_CHECK(cudaGetLastError());
    inkling_router_select_kernel<<<x.ne[1], 128, 0, stream>>>(
        static_cast<const float*>(probabilities.data), static_cast<const float*>(selection_bias.data),
        weight_scale, static_cast<std::int32_t*>(selected.data), static_cast<float*>(weights.data),
        gate.ne[0], selected.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
