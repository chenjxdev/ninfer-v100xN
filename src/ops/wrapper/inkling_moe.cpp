#include "ninfer/ops/inkling_moe.h"

#include "core/layout.h"
#include "ops/launcher/inkling_moe.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void validate_weight(const Weight& weight, std::int32_t n, std::int32_t k) {
    if (weight.n != n || weight.k != k || weight.ndim != 2 || weight.shape[0] != n ||
        weight.shape[1] != k) {
        throw std::invalid_argument("inkling_moe: expert weight shape mismatch");
    }
    if (weight.qtype == QType::NVFP4) {
        (void)detail::validate_nvfp4_weight(weight, "inkling_moe");
        return;
    }
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.payload_bytes < static_cast<std::uint64_t>(n) * k * 2) {
        throw std::invalid_argument("inkling_moe: expert must be native NVFP4 or contiguous BF16");
    }
}

} // namespace

std::size_t inkling_moe_workspace_capacity_bytes(std::int32_t hidden,
                                                 std::int32_t intermediate,
                                                 std::int32_t top_k,
                                                 std::int32_t min_tokens,
                                                 std::int32_t max_tokens) {
    if (hidden <= 0 || intermediate <= 0 || top_k <= 0 || min_tokens <= 0 ||
        max_tokens < min_tokens) {
        throw std::invalid_argument("inkling_moe workspace: invalid geometry or token interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {2 * intermediate, top_k * max_tokens});
    (void)layout.alloc(DType::BF16, {intermediate, top_k * max_tokens});
    (void)layout.alloc(DType::FP32, {hidden, max_tokens, top_k});
    if (max_tokens > 8) {
        const std::int32_t max_jobs = (top_k * max_tokens + 3) / 4 + 128;
        (void)layout.alloc(DType::I32, {4, max_jobs});
        (void)layout.alloc(DType::I32, {top_k * max_tokens});
        (void)layout.alloc(DType::I32, {1});
    }
    (void)layout.alloc_bytes(2 * 256 * sizeof(detail::InklingMoeWeightView));
    return layout.peak_bytes(1);
}

void inkling_moe_impl(const Tensor& x, std::span<const Weight> w13, std::span<const Weight> w2,
                      const Tensor& selected, const Tensor& route_weights, float beta,
                      float linear_beta, DeviceSpan weight_views, bool upload_views,
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || selected.dtype != DType::I32 ||
        route_weights.dtype != DType::FP32) {
        throw std::invalid_argument("inkling_moe: invalid input/output dtypes");
    }
    if (!std::isfinite(beta) || beta <= 0.0F || !std::isfinite(linear_beta) || linear_beta < 0.0F ||
        w13.empty() || w13.size() != w2.size()) {
        throw std::invalid_argument("inkling_moe: invalid expert arrays or SiTU parameters");
    }
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t top_k = selected.ne[0];
    if (hidden <= 0 || tokens <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || top_k <= 0 ||
        top_k > 16 || selected.ne[1] != tokens || selected.ne[2] != 1 || selected.ne[3] != 1 ||
        route_weights.ne[0] != top_k || route_weights.ne[1] != tokens ||
        route_weights.ne[2] != 1 || route_weights.ne[3] != 1 || out.ne[0] != hidden ||
        out.ne[1] != tokens || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("inkling_moe: invalid x/selected/route-weight/out shapes");
    }
    if (!x.is_contiguous() || !selected.is_contiguous() || !route_weights.is_contiguous() ||
        !out.is_contiguous() || x.data == nullptr || selected.data == nullptr ||
        route_weights.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("inkling_moe: tensors must be contiguous and non-null");
    }
    const std::int32_t intermediate = w2.front().k;
    if (upload_views) {
        for (std::size_t expert = 0; expert < w13.size(); ++expert) {
            validate_weight(w13[expert], 2 * intermediate, hidden);
            validate_weight(w2[expert], hidden, intermediate);
            if (w13[expert].qtype != w2[expert].qtype) {
                throw std::invalid_argument("inkling_moe: w13/w2 expert formats must match");
            }
        }
    }

    auto scope = workspace.scope();
    Tensor projected = workspace.alloc(DType::FP32, {2 * intermediate, top_k * tokens});
    Tensor activations = workspace.alloc(DType::BF16, {intermediate, top_k * tokens});
    Tensor accumulation = workspace.alloc(DType::FP32, {hidden, tokens, top_k});
    Tensor prefill_jobs, prefill_order;
    // Maximum-chunk tables are preallocated in the existing arena; only the device owns job count.
    Tensor prefill_count;
    if (tokens > 8) {
        const std::int32_t max_jobs = (top_k * tokens + 3) / 4 + 128;
        prefill_jobs = workspace.alloc(DType::I32, {4, max_jobs});
        prefill_order = workspace.alloc(DType::I32, {top_k * tokens});
        prefill_count = workspace.alloc(DType::I32, {1});
    }
    if (upload_views) {
        weight_views = workspace.alloc_bytes(2 * w13.size() * sizeof(detail::InklingMoeWeightView));
    } else if (weight_views.data == nullptr ||
               weight_views.bytes < 2 * w13.size() * sizeof(detail::InklingMoeWeightView)) {
        throw std::invalid_argument("inkling_moe: prepared weight views are missing or too small");
    }
    detail::inkling_moe_launch(x, w13, w2, selected, route_weights, beta, linear_beta, projected,
                               activations, accumulation, weight_views, upload_views, out,
                               prefill_jobs, prefill_order, prefill_count, stream);
}

void inkling_moe(const Tensor& x, std::span<const Weight> w13, std::span<const Weight> w2,
                 const Tensor& selected, const Tensor& route_weights, float beta,
                 float linear_beta, WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    inkling_moe_impl(x, w13, w2, selected, route_weights, beta, linear_beta, {}, true, workspace,
                     out, stream);
}

void inkling_moe_prepared(const Tensor& x, std::span<const Weight> w13,
                          std::span<const Weight> w2, const Tensor& selected,
                          const Tensor& route_weights, float beta, float linear_beta,
                          DeviceSpan weight_views, WorkspaceArena& workspace, Tensor& out,
                          cudaStream_t stream) {
    inkling_moe_impl(x, w13, w2, selected, route_weights, beta, linear_beta, weight_views, false,
                     workspace, out, stream);
}

} // namespace ninfer::ops
