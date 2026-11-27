#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Inkling attention over depth. states is contiguous BF16 [H,S,T], queries is contiguous FP32
 * [H,Q], and out is contiguous BF16 [H,Q,T]. Each state is RMS-normalized over H with eps=1e-6.
 * For every query/token, scores are dot(normalized_state, query)/sqrt(H), softmax is over S, and
 * out is the probability-weighted sum of normalized states. S is in [1,5] and Q in [1,4].
 */
void inkling_depth_mix(const Tensor& states, const Tensor& queries, Tensor& out,
                       cudaStream_t stream);

} // namespace ninfer::ops
