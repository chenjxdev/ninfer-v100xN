#include "ops/launcher/inkling_moe.h"

#include "ops/common/math.h"
#include "ops/kernel/inkling_moe.cuh"
#include "core/device.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace ninfer::ops::detail {
namespace {

static_assert(sizeof(InklingMoePrefillJob) == 4 * sizeof(std::int32_t));

InklingMoeWeightView view(const Weight& weight) {
    return InklingMoeWeightView{
        .qdata = weight.qdata,
        .scales = weight.scales,
        .n = weight.n,
        .k = weight.k,
        .divisor = weight.qtype == QType::NVFP4 ? weight.weight_scale_divisor : 1.0F,
        .nvfp4 = weight.qtype == QType::NVFP4,
        .mxfp8 = weight.qtype == QType::FP8_E4M3FN_ROW_BF16S,
    };
}

} // namespace

void inkling_moe_launch(const Tensor& x, std::span<const Weight> w13,
                        std::span<const Weight> w2, const Tensor& selected,
                        const Tensor& route_weights, float beta, float linear_beta,
                        Tensor& projected, Tensor& activations, Tensor& accumulation,
                        DeviceSpan weight_views, bool upload_views, Tensor& out,
                        Tensor& prefill_jobs, Tensor& prefill_order, Tensor& prefill_count,
                        cudaStream_t stream) {
    constexpr int kBlock = 256;
    constexpr int kWarps = kBlock / 32;
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t top_k = selected.ne[0];
    const std::int32_t intermediate = w2.front().k;
    if (upload_views) {
        std::vector<InklingMoeWeightView> host_views;
        host_views.reserve(2 * w13.size());
        for (const Weight& weight : w13) { host_views.push_back(view(weight)); }
        for (const Weight& weight : w2) { host_views.push_back(view(weight)); }
        CUDA_CHECK(cudaMemcpyAsync(weight_views.data, host_views.data(),
                                   host_views.size() * sizeof(InklingMoeWeightView),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const auto* device13 = static_cast<const InklingMoeWeightView*>(weight_views.data);
    const auto* device2 = device13 + w13.size();

    if (tokens > 8) {
        // Device-side counting sort: route metadata and the actual job count never return to host.
        inkling_moe_bin_kernel<<<1, 128, 0, stream>>>(
            static_cast<const std::int32_t*>(selected.data),
            static_cast<InklingMoePrefillJob*>(prefill_jobs.data),
            static_cast<std::int32_t*>(prefill_order.data),
            static_cast<std::int32_t*>(prefill_count.data), tokens * top_k, top_k);
        CUDA_CHECK(cudaGetLastError());
        const int max_jobs = (tokens * top_k + 3) / 4 + 128;
        const dim3 gate_grid(static_cast<unsigned int>(max_jobs),
            static_cast<unsigned int>(div_up(static_cast<std::int64_t>(intermediate),
                                            static_cast<std::int64_t>(kWarps))));
        inkling_moe_prefill_gate_up_kernel<<<gate_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), device13,
            static_cast<const InklingMoePrefillJob*>(prefill_jobs.data),
            static_cast<const std::int32_t*>(prefill_order.data),
            static_cast<const std::int32_t*>(prefill_count.data),
            static_cast<__nv_bfloat16*>(activations.data), hidden, intermediate, top_k,
            beta, linear_beta);
        CUDA_CHECK(cudaGetLastError());
        const dim3 down_grid(static_cast<unsigned int>(max_jobs),
            static_cast<unsigned int>(div_up(static_cast<std::int64_t>(hidden),
                                            static_cast<std::int64_t>(kWarps * 4))));
        inkling_moe_prefill_down_kernel<<<down_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(activations.data), device2,
            static_cast<const InklingMoePrefillJob*>(prefill_jobs.data),
            static_cast<const std::int32_t*>(prefill_order.data),
            static_cast<const std::int32_t*>(prefill_count.data),
            static_cast<const float*>(route_weights.data), static_cast<float*>(accumulation.data),
            tokens, hidden, intermediate, top_k);
        CUDA_CHECK(cudaGetLastError());
        const std::int64_t count = out.numel();
        const int blocks = static_cast<int>(div_up(count, static_cast<std::int64_t>(kBlock)));
        inkling_moe_reduce_kernel<<<blocks, kBlock, 0, stream>>>(
            static_cast<const float*>(accumulation.data), static_cast<__nv_bfloat16*>(out.data),
            count, hidden, top_k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const dim3 gate_grid(static_cast<unsigned int>(tokens),
                         static_cast<unsigned int>(div_up(static_cast<std::int64_t>(intermediate),
                                                         static_cast<std::int64_t>(kWarps))),
                         static_cast<unsigned int>(top_k));
    inkling_moe_gate_up_fused_kernel<<<gate_grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), device13,
        static_cast<const std::int32_t*>(selected.data), static_cast<__nv_bfloat16*>(activations.data),
        hidden, intermediate, top_k, beta, linear_beta);
    CUDA_CHECK(cudaGetLastError());
    const dim3 down_grid(static_cast<unsigned int>(tokens),
                         static_cast<unsigned int>(div_up(static_cast<std::int64_t>(hidden),
                                                         static_cast<std::int64_t>(kWarps * 4))),
                         static_cast<unsigned int>(top_k));
    inkling_moe_down_rows4_kernel<<<down_grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activations.data), device2,
        static_cast<const std::int32_t*>(selected.data),
        static_cast<const float*>(route_weights.data), static_cast<float*>(accumulation.data),
        hidden, intermediate, top_k);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t count = out.numel();
    const int blocks = static_cast<int>(div_up(count, static_cast<std::int64_t>(kBlock)));
    inkling_moe_reduce_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const float*>(accumulation.data), static_cast<__nv_bfloat16*>(out.data), count,
        hidden, top_k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
