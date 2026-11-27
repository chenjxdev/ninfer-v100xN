#include "ninfer/ops/situ_mul.h"

#include "ops/launcher/situ_mul.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops {

void situ_mul(const Tensor& gate, const Tensor& up, float beta, float linear_beta, Tensor& out,
              cudaStream_t stream) {
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("situ_mul: gate/up/out must be BF16");
    }
    if (!std::isfinite(beta) || beta <= 0.0F || !std::isfinite(linear_beta) ||
        linear_beta < 0.0F) {
        throw std::invalid_argument("situ_mul: beta must be positive and linear_beta nonnegative");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument("situ_mul: gate/up/out shapes must match");
        }
    }
    if (!out.is_contiguous()) { throw std::invalid_argument("situ_mul: out must be contiguous"); }
    if (out.numel() == 0) { return; }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("situ_mul: gate/up/out data must be non-null");
    }

    detail::situ_mul_launch(gate, up, beta, linear_beta, out, stream);
}

} // namespace ninfer::ops
