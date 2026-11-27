#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void inkling_conv1d_launch(const Tensor& x, const Tensor& weight, const Tensor* dynamic,
                           const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                           cudaStream_t stream, Tensor* state_history);

} // namespace ninfer::ops::detail
