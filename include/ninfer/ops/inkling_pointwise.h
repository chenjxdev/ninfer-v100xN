#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// out = cap * tanh(x / cap), for matching contiguous BF16 tensors.
void inkling_softcap(const Tensor& x, float cap, Tensor& out, cudaStream_t stream);

// Multiplies BF16 x [D,H,T] by 1 + alpha*log(max(1,(position+1)/floor)) per token.
void inkling_log_position_scale(const Tensor& x, const Tensor& positions, std::uint32_t floor,
                                float alpha, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
