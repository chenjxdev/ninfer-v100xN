#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void inkling_softcap_launch(const Tensor& x, float cap, Tensor& out, cudaStream_t stream);
void inkling_log_position_scale_launch(const Tensor& x, const Tensor& positions,
                                       std::uint32_t floor, float alpha, Tensor& out,
                                       cudaStream_t stream);

} // namespace ninfer::ops::detail
