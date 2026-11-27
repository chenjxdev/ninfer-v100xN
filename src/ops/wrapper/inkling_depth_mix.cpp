#include "ninfer/ops/inkling_depth_mix.h"

#include "ops/launcher/inkling_depth_mix.h"

#include <stdexcept>

namespace ninfer::ops {

void inkling_depth_mix(const Tensor& states, const Tensor& queries, Tensor& out,
                       cudaStream_t stream) {
    if (states.dtype != DType::BF16 || out.dtype != DType::BF16 || queries.dtype != DType::FP32) {
        throw std::invalid_argument("inkling_depth_mix: states/out must be BF16 and queries FP32");
    }
    const std::int32_t hidden = states.ne[0];
    const std::int32_t depth = states.ne[1];
    const std::int32_t tokens = states.ne[2];
    const std::int32_t query_count = queries.ne[1];
    if (hidden <= 0 || depth <= 0 || depth > 5 || tokens <= 0 || states.ne[3] != 1 ||
        query_count <= 0 || query_count > 4 || queries.ne[0] != hidden || queries.ne[2] != 1 ||
        queries.ne[3] != 1 || out.ne[0] != hidden || out.ne[1] != query_count ||
        out.ne[2] != tokens || out.ne[3] != 1) {
        throw std::invalid_argument("inkling_depth_mix: expected states [H,S<=5,T], queries [H,Q<=4], out [H,Q,T]");
    }
    if (!states.is_contiguous() || !queries.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("inkling_depth_mix: all tensors must be contiguous");
    }
    if (states.data == nullptr || queries.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("inkling_depth_mix: tensor data must be non-null");
    }
    detail::inkling_depth_mix_launch(states, queries, out, stream);
}

} // namespace ninfer::ops
