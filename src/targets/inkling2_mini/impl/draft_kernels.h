#pragma once

#include "core/tensor.h"

#include <array>
#include <cuda_runtime.h>

namespace ninfer::targets::inkling2_mini::detail {

void draft_prepare_embedding(const Tensor& anchor, const Tensor& mask, Tensor& out,
                             cudaStream_t stream);
void draft_mix_rmsnorm(const std::array<Tensor, 4>& features, int token_begin, int tokens,
                       const Tensor& alpha_row, const Tensor& norm, Tensor& out,
                       cudaStream_t stream);
void draft_rope(Tensor& q, Tensor& k, const Tensor& positions, cudaStream_t stream);
void draft_rope_context_k(Tensor& k, const Tensor& positions, cudaStream_t stream);
void draft_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                     const Tensor& context_k, const Tensor& context_v,
                     int context_length, bool local, Tensor& scores, Tensor& out,
                     cudaStream_t stream);
void draft_rnn_inputs(const Tensor& state, const Tensor& previous_token,
                      const Weight& markov_embedding, const Tensor& hidden,
                      const Tensor& f_rnn, Tensor& out, cudaStream_t stream);
void draft_rnn_gates(const Tensor& projection, Tensor& state, Tensor& latent,
                     cudaStream_t stream);
void draft_fold_lm_head(const Weight& lm_head, const Weight& hidden_out,
                        const Tensor& hidden_bias, Weight& folded_weight, Tensor& folded_bias,
                        cudaStream_t stream);

} // namespace ninfer::targets::inkling2_mini::detail
