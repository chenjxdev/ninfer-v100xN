#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void inkling_loop_control_launch(const Tensor& cur, const Tensor& message, const Tensor& down,
                                 const Tensor& gain, const Tensor& forget, Tensor& normalized,
                                 Tensor& gate, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
