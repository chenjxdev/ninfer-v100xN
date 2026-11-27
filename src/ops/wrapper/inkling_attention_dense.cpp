#include "ninfer/ops/inkling_attention_dense.h"

#include "ops/launcher/inkling_attention_dense.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops {

void inkling_attention_dense(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& r,
                             const Tensor& relative_bias, const Tensor& sink_logits,
                             float scale, std::uint32_t sliding_window, Tensor& out,
                             cudaStream_t stream) {
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        r.dtype != DType::BF16 || relative_bias.dtype != DType::BF16 ||
        sink_logits.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("inkling_attention_dense: all tensors must be BF16");
    }
    if (!std::isfinite(scale) || scale <= 0.0F) {
        throw std::invalid_argument("inkling_attention_dense: scale must be positive and finite");
    }
    const std::int32_t heads = q.ne[1];
    const std::int32_t kv_heads = k.ne[1];
    const std::int32_t tokens = q.ne[2];
    const std::int32_t keys = k.ne[2];
    const std::int32_t relative_length = r.ne[0];
    const std::int32_t extent = relative_bias.ne[1];
    if (q.ne[0] != 128 || heads <= 0 || tokens <= 0 || q.ne[3] != 1 || kv_heads <= 0 ||
        heads % kv_heads != 0 || keys < tokens || k.ne[0] != 128 || k.ne[3] != 1 ||
        v.ne[0] != 128 || v.ne[1] != kv_heads || v.ne[2] != keys || v.ne[3] != 1 ||
        relative_length <= 0 || relative_length > 64 || r.ne[1] != heads || r.ne[2] != tokens ||
        r.ne[3] != 1 || extent <= 0 || relative_bias.ne[0] != relative_length ||
        relative_bias.ne[2] != 1 || relative_bias.ne[3] != 1 || sink_logits.ne[0] != heads ||
        sink_logits.ne[1] != tokens || sink_logits.ne[2] != 1 || sink_logits.ne[3] != 1 ||
        out.ne[0] != 128 || out.ne[1] != heads || out.ne[2] != tokens || out.ne[3] != 1) {
        throw std::invalid_argument("inkling_attention_dense: invalid Q/K/V/R/bias/sink/out shapes");
    }
    if (!q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() || !r.is_contiguous() ||
        !relative_bias.is_contiguous() || !sink_logits.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("inkling_attention_dense: all tensors must be contiguous");
    }
    if (q.data == nullptr || k.data == nullptr || v.data == nullptr || r.data == nullptr ||
        relative_bias.data == nullptr || sink_logits.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("inkling_attention_dense: tensor data must be non-null");
    }
    detail::inkling_attention_dense_launch(q, k, v, r, relative_bias, sink_logits, scale,
                                           sliding_window, out, stream);
}

} // namespace ninfer::ops
