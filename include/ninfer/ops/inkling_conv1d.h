#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Inkling causal four-tap short convolution with residual and recurrent state.
 *
 * x/out are contiguous BF16 [C,T], state input/output are BF16 [C,3], and static weight is either
 * contiguous [C,4] or [4,C]. dynamic, when supplied, is contiguous BF16 [4*H,T], with C divisible
 * by H. For each channel c belonging to head h:
 *
 *   y[t] = sum(j=0..3, static[c,j] * u[t-3+j])
 *        + sum(j=0..3, dynamic[4*h+j,t] * u[t-3+j])
 *   out[t] = (x[t] + y[t]) / sqrt((dynamic ? 9 : 5)).
 *
 * The dynamic term is absent for static calls. State output receives the final three inputs and
 * may exactly alias state input. Unlike causal_conv1d_silu this op applies no SiLU.
 * When state_history is non-null it is contiguous BF16 [C,3,N], N >= T+1. Slice zero is the
 * input state and slice t+1 is the state after token t, allowing speculative rollback without
 * replaying the accepted prefix.
 */
void inkling_conv1d(const Tensor& x, const Tensor& weight, const Tensor* dynamic,
                    const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                    cudaStream_t stream, Tensor* state_history = nullptr);

} // namespace ninfer::ops
