#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

[[nodiscard]] std::size_t inkling_moe_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t intermediate, std::int32_t top_k,
    std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Computes the selected routed-expert sum for independent per-expert weights. x is BF16 [H,T],
 * selected is I32 [K,T], route_weights is FP32 [K,T], and out is BF16 [H,T]. w13/w2 contain the
 * same expert count and use native interleaved gate/up rows in w13. Each expert may independently
 * use BlockScaleK16M128x4 NVFP4 or contiguous BF16. NVFP4 divisors are applied exactly once.
 */
void inkling_moe(const Tensor& x, std::span<const Weight> w13, std::span<const Weight> w2,
                 const Tensor& selected, const Tensor& route_weights, float beta,
                 float linear_beta, WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

// Prepared form: weight_views contains device-resident [w13 experts | w2 experts] view records.
void inkling_moe_prepared(const Tensor& x, std::span<const Weight> w13,
                          std::span<const Weight> w2, const Tensor& selected,
                          const Tensor& route_weights, float beta, float linear_beta,
                          DeviceSpan weight_views, WorkspaceArena& workspace, Tensor& out,
                          cudaStream_t stream);

} // namespace ninfer::ops
