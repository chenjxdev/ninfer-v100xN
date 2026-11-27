#pragma once

#include "core/tensor.h"

#include <array>
#include <cuda_runtime.h>

namespace ninfer::targets::inkling2_mini::detail {

void pack_depth_state(const Tensor& source, Tensor& states, std::int32_t state_index,
                      cudaStream_t stream);
void mix_depth_states(const std::array<const void*, 5>& sources, std::int32_t depth,
                      const Tensor& query, Tensor& out, cudaStream_t stream);
void extract_depth_output(const Tensor& mixed, std::int32_t query_index, Tensor& out,
                          cudaStream_t stream);
void pack_loop_query(const Tensor& q, const Tensor& kv, std::int32_t core_index, Tensor& out,
                     cudaStream_t stream);
void append_kv(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache,
               std::int32_t position, cudaStream_t stream);
void append_kv_ring(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache,
                    std::int32_t position, std::int32_t capacity, cudaStream_t stream);
void gather_kv_ring(const Tensor& k_cache, const Tensor& v_cache, std::int32_t first_position,
                    std::int32_t count, std::int32_t capacity, Tensor& k, Tensor& v,
                    cudaStream_t stream);
void append_kv_int8(const Tensor& k, const Tensor& v, Tensor& k_codes, Tensor& k_scales,
                    Tensor& v_codes, Tensor& v_scales, std::int32_t position,
                    cudaStream_t stream);
void inkling_attention_int8(const Tensor& q, const Tensor& k_codes, const Tensor& k_scales,
                            const Tensor& v_codes, const Tensor& v_scales, const Tensor& r,
                            const Tensor& relative_bias, const Tensor& sink_logits, float scale,
                            Tensor& out, cudaStream_t stream);
void fill_positions(Tensor& positions, std::int32_t first, cudaStream_t stream);
void bf16_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
struct BF16GemvItem {
    const void* input = nullptr;
    const void* weight = nullptr;
    void* output = nullptr;
    std::int32_t rows = 0;
    std::int32_t row_begin = 0;
};
void bf16_gemv_group(const std::array<BF16GemvItem, 8>& items, std::int32_t count,
                     std::int32_t columns, std::int32_t tokens, cudaStream_t stream);

} // namespace ninfer::targets::inkling2_mini::detail
