#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Single-sequence Inkling relative-position attention over a dense complete K/V history.
 * Q is BF16 [128,Hq,T], K/V are BF16 [128,Hkv,N], R is BF16 [NR,Hq,T], relative_bias is BF16
 * [NR,extent], sink_logits is BF16 [Hq,T], and out is BF16 [128,Hq,T]. Queries have sequential
 * positions N-T..N-1, so the dense history ends with the current chunk.
 *
 * For distance d=query_position-key_position, visible keys satisfy d>=0 and, when
 * sliding_window!=0, d<sliding_window. The learned bias is dot(R, relative_bias[:,d]) for
 * d<extent and zero otherwise. sink_logits is subtracted from every visible score and a zero-value
 * sink logit participates in the denominator. This matches llama.cpp's table-minus-sink convention.
 */
void inkling_attention_dense(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& r,
                             const Tensor& relative_bias, const Tensor& sink_logits,
                             float scale, std::uint32_t sliding_window, Tensor& out,
                             cudaStream_t stream);

} // namespace ninfer::ops
