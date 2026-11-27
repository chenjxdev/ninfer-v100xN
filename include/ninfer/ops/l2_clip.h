#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Scales and L2-clips each logical row over ne[0]:
 *
 *   y = x * scale (when supplied)
 *   norm = sqrt(sum(y*y)) + 1e-6
 *   out = y * min(1, threshold / norm).
 *
 * x and out are same-shaped contiguous BF16 tensors. scale, when non-null, is a contiguous BF16
 * tensor whose rows cover x.ne[0] and repeat across the higher dimensions. This implements
 * Inkling's Q/K/relative-position clipping without normalizing rows whose norm is below threshold.
 */
void l2_clip(const Tensor& x, const Tensor* scale, float threshold, Tensor& out,
             cudaStream_t stream);

} // namespace ninfer::ops
