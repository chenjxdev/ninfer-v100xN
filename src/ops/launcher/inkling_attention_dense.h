#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void inkling_attention_dense_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& r, const Tensor& relative_bias,
                                    const Tensor& sink_logits, float scale,
                                    std::uint32_t sliding_window, Tensor& out,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
