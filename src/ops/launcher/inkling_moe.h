#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

struct InklingMoePrefillJob {
    std::int32_t expert = 0;
    std::int32_t first_assignment = 0;
    std::int32_t count = 0;
    std::int32_t reserved = 0;
};

struct InklingMoeWeightView {
    const void* qdata = nullptr;
    const void* scales = nullptr;
    const void* group_scales = nullptr;
    std::int32_t n = 0;
    std::int32_t k = 0;
    float divisor = 1.0F;
    bool nvfp4 = false;
    bool mxfp8 = false;
};

void inkling_moe_launch(const Tensor& x, std::span<const Weight> w13,
                        std::span<const Weight> w2, const Tensor& selected,
                        const Tensor& route_weights, float beta, float linear_beta,
                        Tensor& projected, Tensor& activations, Tensor& accumulation,
                        DeviceSpan weight_views, bool upload_views, Tensor& out,
                        Tensor& prefill_jobs, Tensor& prefill_order, Tensor& prefill_count,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail
