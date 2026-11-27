#include "ninfer/ops/inkling_loop_control.h"

#include "ops/launcher/inkling_loop_control.h"
#include "core/layout.h"

#include <stdexcept>

namespace ninfer::ops {

std::size_t inkling_loop_control_workspace_capacity_bytes(std::int32_t hidden,
                                                          std::int32_t rank,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    if (hidden <= 0 || rank <= 0 || rank > 256 || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("inkling_loop_control workspace: invalid rank or token interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {2 * hidden, max_tokens});
    (void)layout.alloc(DType::FP32, {rank, max_tokens});
    return layout.peak_bytes(1);
}

void inkling_loop_control(const Tensor& cur, const Tensor& message, const Tensor& down,
                          const Tensor& gain, const Tensor& forget, WorkspaceArena& workspace,
                          Tensor& out, cudaStream_t stream) {
    if (cur.dtype != DType::BF16 || message.dtype != DType::BF16 || down.dtype != DType::BF16 ||
        gain.dtype != DType::BF16 || forget.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("inkling_loop_control: all tensors must be BF16");
    }
    const std::int32_t hidden = cur.ne[0];
    const std::int32_t tokens = cur.ne[1];
    const std::int32_t rank = down.ne[0];
    if (hidden <= 0 || tokens <= 0 || cur.ne[2] != 1 || cur.ne[3] != 1 || rank <= 0 || rank > 256 ||
        down.ne[1] != 2 * hidden || down.ne[2] != 1 || down.ne[3] != 1 ||
        message.ne[0] != hidden || message.ne[1] != tokens || message.ne[2] != 1 ||
        message.ne[3] != 1 || out.ne[0] != hidden || out.ne[1] != tokens || out.ne[2] != 1 ||
        out.ne[3] != 1 || gain.ne[0] != hidden || gain.ne[1] != rank || gain.ne[2] != 1 ||
        gain.ne[3] != 1 || forget.ne[0] != hidden || forget.ne[1] != rank || forget.ne[2] != 1 ||
        forget.ne[3] != 1) {
        throw std::invalid_argument("inkling_loop_control: invalid cur/message/controller shapes");
    }
    if (!cur.is_contiguous() || !message.is_contiguous() || !down.is_contiguous() ||
        !gain.is_contiguous() || !forget.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("inkling_loop_control: all tensors must be contiguous");
    }
    if (cur.data == nullptr || message.data == nullptr || down.data == nullptr || gain.data == nullptr ||
        forget.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("inkling_loop_control: tensor data must be non-null");
    }
    auto scope = workspace.scope();
    Tensor normalized = workspace.alloc(DType::FP32, {2 * hidden, tokens});
    Tensor gate = workspace.alloc(DType::FP32, {rank, tokens});
    detail::inkling_loop_control_launch(cur, message, down, gain, forget, normalized, gate, out,
                                        stream);
}

} // namespace ninfer::ops
