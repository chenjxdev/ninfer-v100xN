#include "ops/launcher/l2_clip.h"

#include "ops/common/math.h"
#include "ops/kernel/l2_clip.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {

void l2_clip_launch(const Tensor& x, const Tensor* scale, float threshold, Tensor& out,
                    cudaStream_t stream) {
    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / 32;
    const std::int64_t rows = x.numel() / x.ne[0];
    const auto blocks = static_cast<unsigned int>(div_up(rows, static_cast<std::int64_t>(kWarps)));
    const std::int64_t scale_rows = scale == nullptr ? 0 : scale->numel() / x.ne[0];
    l2_clip_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        scale == nullptr ? nullptr : static_cast<const __nv_bfloat16*>(scale->data),
        static_cast<__nv_bfloat16*>(out.data), x.ne[0], rows, scale_rows, threshold);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
