#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void situ_mul_launch(const Tensor& gate, const Tensor& up, float beta, float linear_beta,
                     Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
