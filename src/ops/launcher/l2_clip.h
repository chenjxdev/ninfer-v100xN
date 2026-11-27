#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void l2_clip_launch(const Tensor& x, const Tensor* scale, float threshold, Tensor& out,
                    cudaStream_t stream);

} // namespace ninfer::ops::detail
