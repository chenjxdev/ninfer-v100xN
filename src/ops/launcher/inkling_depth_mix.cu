#include "ops/launcher/inkling_depth_mix.h"

#include "ops/kernel/inkling_depth_mix.cuh"
#include "core/device.h"

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {

void inkling_depth_mix_launch(const Tensor& states, const Tensor& queries, Tensor& out,
                              cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t hidden = states.ne[0];
    const std::int32_t depth = states.ne[1];
    const std::size_t shared_bytes =
        static_cast<std::size_t>(hidden * depth + kBlock + 2 * depth) * sizeof(float);
    if (shared_bytes > 48 * 1024) {
        throw std::invalid_argument("inkling_depth_mix: shared-memory geometry exceeds 48 KiB");
    }
    const dim3 grid(static_cast<unsigned int>(states.ne[2]));
    inkling_depth_mix_kernel<<<grid, kBlock, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(states.data), static_cast<const float*>(queries.data),
        static_cast<__nv_bfloat16*>(out.data), hidden, depth, states.ne[2], queries.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
