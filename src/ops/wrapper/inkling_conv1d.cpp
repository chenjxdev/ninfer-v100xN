#include "ninfer/ops/inkling_conv1d.h"

#include "ops/launcher/inkling_conv1d.h"

#include <stdexcept>

namespace ninfer::ops {

void inkling_conv1d(const Tensor& x, const Tensor& weight, const Tensor* dynamic,
                    const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                    cudaStream_t stream, Tensor* state_history) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        conv_state_in.dtype != DType::BF16 || conv_state_out.dtype != DType::BF16 ||
        (dynamic != nullptr && dynamic->dtype != DType::BF16) ||
        (state_history != nullptr && state_history->dtype != DType::BF16)) {
        throw std::invalid_argument("inkling_conv1d: all tensors must be BF16");
    }
    if (x.ne[0] <= 0 || x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        out.ne[0] != x.ne[0] || out.ne[1] != x.ne[1] || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("inkling_conv1d: x/out must have matching nonempty [C,T] shapes");
    }
    const std::int32_t channels = x.ne[0];
    const Tensor* states[] = {&conv_state_in, &conv_state_out};
    for (const Tensor* state : states) {
        if (state->ne[0] != channels || state->ne[1] != 3 || state->ne[2] != 1 ||
            state->ne[3] != 1) {
            throw std::invalid_argument("inkling_conv1d: states must have shape [C,3]");
        }
    }
    if (state_history != nullptr &&
        (state_history->ne[0] != channels || state_history->ne[1] != 3 ||
         state_history->ne[2] < x.ne[1] + 1 || state_history->ne[3] != 1)) {
        throw std::invalid_argument("inkling_conv1d: state_history must have shape [C,3,N>=T+1]");
    }
    const bool transposed = weight.ne[0] == 4 && weight.ne[1] == channels;
    if (!((weight.ne[0] == channels && weight.ne[1] == 4) || transposed) ||
        weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("inkling_conv1d: weight must have shape [C,4] or [4,C]");
    }
    if (dynamic != nullptr) {
        if (dynamic->ne[0] <= 0 || dynamic->ne[0] % 4 != 0 || dynamic->ne[1] != x.ne[1] ||
            dynamic->ne[2] != 1 || dynamic->ne[3] != 1) {
            throw std::invalid_argument("inkling_conv1d: dynamic must have shape [4*H,T]");
        }
        const std::int32_t heads = dynamic->ne[0] / 4;
        if (channels % heads != 0) {
            throw std::invalid_argument("inkling_conv1d: dynamic heads must divide C");
        }
    }
    if (!x.is_contiguous() || !weight.is_contiguous() || !out.is_contiguous() ||
        !conv_state_in.is_contiguous() || !conv_state_out.is_contiguous() ||
        (dynamic != nullptr && !dynamic->is_contiguous()) ||
        (state_history != nullptr && !state_history->is_contiguous())) {
        throw std::invalid_argument("inkling_conv1d: all tensors must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || out.data == nullptr ||
        conv_state_in.data == nullptr || conv_state_out.data == nullptr ||
        (dynamic != nullptr && dynamic->data == nullptr) ||
        (state_history != nullptr && state_history->data == nullptr)) {
        throw std::invalid_argument("inkling_conv1d: tensor data must be non-null");
    }
    detail::inkling_conv1d_launch(x, weight, dynamic, conv_state_in, conv_state_out, out, stream,
                                  state_history);
}

} // namespace ninfer::ops
