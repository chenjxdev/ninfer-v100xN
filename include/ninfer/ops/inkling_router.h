#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Inkling sigmoid router. x is BF16 [H,T], gate is BF16 [E,H], selection_bias is FP32 [E].
 * selected is I32 [K,T] and weights is FP32 [K,T], sorted by descending biased selection score
 * with lower expert ids winning ties. Bias is used only for selection; normalized route weights use
 * the original sigmoid probabilities and denominator sum(probabilities)+1e-20, then weight_scale.
 */
[[nodiscard]] std::size_t inkling_router_workspace_capacity_bytes(
    std::int32_t experts, std::int32_t min_tokens, std::int32_t max_tokens);

void inkling_router(const Tensor& x, const Tensor& gate, const Tensor& selection_bias,
                    float weight_scale, WorkspaceArena& workspace, Tensor& selected,
                    Tensor& weights, cudaStream_t stream);

} // namespace ninfer::ops
