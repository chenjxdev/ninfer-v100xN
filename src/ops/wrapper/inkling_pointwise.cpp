#include "ninfer/ops/inkling_pointwise.h"

#include "ops/launcher/inkling_pointwise.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_matching_bf16(const Tensor& x, const Tensor& out, const char* operation) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(operation) + ": x/out must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (x.ne[d] != out.ne[d]) {
            throw std::invalid_argument(std::string(operation) + ": x/out shapes must match");
        }
    }
    if (!x.is_contiguous() || !out.is_contiguous() || x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument(std::string(operation) + ": x/out must be contiguous and non-null");
    }
}

} // namespace

void inkling_softcap(const Tensor& x, float cap, Tensor& out, cudaStream_t stream) {
    require_matching_bf16(x, out, "inkling_softcap");
    if (!std::isfinite(cap) || cap <= 0.0F) {
        throw std::invalid_argument("inkling_softcap: cap must be positive and finite");
    }
    if (x.numel() == 0) { return; }
    detail::inkling_softcap_launch(x, cap, out, stream);
}

void inkling_log_position_scale(const Tensor& x, const Tensor& positions, std::uint32_t floor,
                                float alpha, Tensor& out, cudaStream_t stream) {
    require_matching_bf16(x, out, "inkling_log_position_scale");
    if (x.ne[0] <= 0 || x.ne[1] <= 0 || x.ne[2] <= 0 || x.ne[3] != 1 ||
        positions.dtype != DType::I32 || positions.ne[0] != x.ne[2] || positions.ne[1] != 1 ||
        positions.ne[2] != 1 || positions.ne[3] != 1 || !positions.is_contiguous() ||
        positions.data == nullptr || floor == 0 || !std::isfinite(alpha) || alpha < 0.0F) {
        throw std::invalid_argument("inkling_log_position_scale: invalid x/positions/floor/alpha");
    }
    detail::inkling_log_position_scale_launch(x, positions, floor, alpha, out, stream);
}

} // namespace ninfer::ops
