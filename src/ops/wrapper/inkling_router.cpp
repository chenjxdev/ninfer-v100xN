#include "ninfer/ops/inkling_router.h"

#include "ops/launcher/inkling_router.h"
#include "core/layout.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops {

std::size_t inkling_router_workspace_capacity_bytes(std::int32_t experts,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (experts <= 0 || experts > 256 || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("inkling_router workspace: invalid expert count or token interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {experts, max_tokens});
    return layout.peak_bytes(1);
}

void inkling_router(const Tensor& x, const Tensor& gate, const Tensor& selection_bias,
                    float weight_scale, WorkspaceArena& workspace, Tensor& selected,
                    Tensor& weights, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || gate.dtype != DType::BF16 || selection_bias.dtype != DType::FP32 ||
        selected.dtype != DType::I32 || weights.dtype != DType::FP32) {
        throw std::invalid_argument("inkling_router: invalid input/output dtypes");
    }
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t experts = gate.ne[0];
    const std::int32_t top_k = selected.ne[0];
    if (hidden <= 0 || tokens <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || experts <= 0 || experts > 256 ||
        gate.ne[1] != hidden || gate.ne[2] != 1 || gate.ne[3] != 1 || selection_bias.ne[0] != experts ||
        selection_bias.ne[1] != 1 || selection_bias.ne[2] != 1 || selection_bias.ne[3] != 1 ||
        top_k <= 0 || top_k > experts || top_k > 16 || selected.ne[1] != tokens ||
        selected.ne[2] != 1 ||
        selected.ne[3] != 1 || weights.ne[0] != top_k || weights.ne[1] != tokens ||
        weights.ne[2] != 1 || weights.ne[3] != 1 || !std::isfinite(weight_scale)) {
        throw std::invalid_argument("inkling_router: invalid tensor shapes or weight scale");
    }
    if (!x.is_contiguous() || !gate.is_contiguous() || !selection_bias.is_contiguous() ||
        !selected.is_contiguous() || !weights.is_contiguous()) {
        throw std::invalid_argument("inkling_router: all tensors must be contiguous");
    }
    if (x.data == nullptr || gate.data == nullptr || selection_bias.data == nullptr ||
        selected.data == nullptr || weights.data == nullptr) {
        throw std::invalid_argument("inkling_router: tensor data must be non-null");
    }
    auto scope = workspace.scope();
    Tensor probabilities = workspace.alloc(DType::FP32, {experts, tokens});
    detail::inkling_router_launch(x, gate, selection_bias, weight_scale, probabilities, selected,
                                  weights, stream);
}

} // namespace ninfer::ops
