#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void inkling_depth_mix_launch(const Tensor& states, const Tensor& queries, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
