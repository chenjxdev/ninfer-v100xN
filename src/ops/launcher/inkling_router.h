#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void inkling_router_launch(const Tensor& x, const Tensor& gate, const Tensor& selection_bias,
                           float weight_scale, Tensor& probabilities, Tensor& selected,
                           Tensor& weights, cudaStream_t stream);

} // namespace ninfer::ops::detail
