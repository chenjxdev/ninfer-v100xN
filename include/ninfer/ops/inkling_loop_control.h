#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Applies one Inkling cross-pass controller. cur/message/out are contiguous BF16 [H,T]; down is
 * BF16 [R,2H], and gain/forget are BF16 [H,R]. The controller RMS-normalizes cur and message,
 * projects their concatenation through down, then computes tanh(gain) and tanh(forget):
 *
 *   out = (1 - forget) * cur + gain * message.
 *
 * All projections and gates use FP32 intermediates. R is positive and at most 256.
 */
[[nodiscard]] std::size_t inkling_loop_control_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t rank, std::int32_t min_tokens, std::int32_t max_tokens);

void inkling_loop_control(const Tensor& cur, const Tensor& message, const Tensor& down,
                          const Tensor& gain, const Tensor& forget, WorkspaceArena& workspace,
                          Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
