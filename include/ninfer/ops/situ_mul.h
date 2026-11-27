#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Inkling/Kimi SiTU activation with optional linear-beta limiting:
 *
 *   gated = beta * tanh(gate / beta) * sigmoid(gate)
 *   limited_up = linear_beta > 0 ? linear_beta * tanh(up / linear_beta) : up
 *   out = gated * limited_up.
 *
 * Inputs and output have matching BF16 shapes. Output is contiguous; inputs may be strided views
 * into a fused gate/up tensor. beta is positive and finite, linear_beta is nonnegative and finite.
 * Arithmetic is FP32 with round-to-nearest BF16 output and no persistent state.
 */
void situ_mul(const Tensor& gate, const Tensor& up, float beta, float linear_beta, Tensor& out,
              cudaStream_t stream);

} // namespace ninfer::ops
