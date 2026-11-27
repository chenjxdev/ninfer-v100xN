#include "ninfer/ops/situ_mul.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion criterion() {
    return {/*absolute*/ 3.0e-5, /*relative*/ 4.1e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> oracle(const std::vector<float>& gate, const std::vector<float>& up, float beta,
                           float linear_beta) {
    std::vector<double> expected(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i];
        const double u = up[i];
        const double gated = beta * std::tanh(g / beta) / (1.0 + std::exp(-g));
        const double limited_up = linear_beta > 0.0F ? linear_beta * std::tanh(u / linear_beta) : u;
        expected[i] = gated * limited_up;
    }
    return expected;
}

int contiguous_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    constexpr float beta = 4.0F;
    constexpr float linear_beta = 25.0F;
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -30.0F, 30.0F);
    fill_uniform(up, seed + 1, -80.0F, 80.0F);
    round_to_bf16(gate);
    round_to_bf16(up);
    const auto expected = oracle(gate, up, beta, linear_beta);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits = encode_bf16(up);
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());
    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor up_tensor(device_up.data(), DType::BF16, {rows, columns});
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::situ_mul(gate_tensor, up_tensor, beta, linear_beta, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected, criterion());
    failures += verify_exact("situ_mul gate unchanged",
                             from_device<std::uint16_t>(device_gate.data(), count), gate_bits);
    failures += verify_exact("situ_mul up unchanged",
                             from_device<std::uint16_t>(device_up.data(), count), up_bits);
    failures += device_gate.verify_guards("situ_mul gate");
    failures += device_up.verify_guards("situ_mul up");
    failures += device_out.verify_guards("situ_mul out");
    return failures;
}

int fused_strided_case() {
    constexpr std::int32_t rows = 1152;
    constexpr std::int32_t columns = 17;
    constexpr float beta = 4.0F;
    constexpr float linear_beta = 25.0F;
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, 401U, -30.0F, 30.0F);
    fill_uniform(up, 402U, -80.0F, 80.0F);
    round_to_bf16(gate);
    round_to_bf16(up);
    std::vector<float> fused(2 * count);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t source = static_cast<std::size_t>(column) * rows;
        const std::size_t target = static_cast<std::size_t>(column) * 2 * rows;
        std::copy_n(gate.data() + source, rows, fused.data() + target);
        std::copy_n(up.data() + source, rows, fused.data() + target + rows);
    }
    const auto expected = oracle(gate, up, beta, linear_beta);
    const auto fused_bits = encode_bf16(fused);
    GuardedDeviceBuffer device_fused(fused_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_fused.copy_from_host(fused_bits.data(), device_fused.bytes());
    Tensor fused_tensor(device_fused.data(), DType::BF16, {2 * rows, columns});
    Tensor gate_tensor = fused_tensor.slice(0, 0, rows);
    Tensor up_tensor = fused_tensor.slice(0, rows, rows);
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::situ_mul(gate_tensor, up_tensor, beta, linear_beta, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("situ_mul fused strided", from_device_bf16(device_out.data(), count),
                                    expected, criterion());
    failures += verify_exact("situ_mul fused input unchanged",
        from_device<std::uint16_t>(device_fused.data(), fused_bits.size()), fused_bits);
    failures += device_fused.verify_guards("situ_mul fused input");
    failures += device_out.verify_guards("situ_mul fused out");
    return failures;
}

int edge_case() {
    constexpr float beta = 4.0F;
    constexpr float linear_beta = 25.0F;
    std::vector<float> gate{-100.0F, -30.0F, -4.0F, -1.0F, 0.0F, 1.0F, 4.0F, 30.0F, 100.0F};
    std::vector<float> up{100.0F, -80.0F, 25.0F, -1.0F, 0.0F, 1.0F, -25.0F, 80.0F, -100.0F};
    round_to_bf16(gate);
    round_to_bf16(up);
    const auto expected = oracle(gate, up, beta, linear_beta);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits = encode_bf16(up);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());
    Tensor gate_tensor(device_gate.data(), DType::BF16, {static_cast<int>(gate.size())});
    Tensor up_tensor(device_up.data(), DType::BF16, {static_cast<int>(up.size())});
    Tensor out_tensor(device_out.data(), DType::BF16, {static_cast<int>(expected.size())});
    ops::situ_mul(gate_tensor, up_tensor, beta, linear_beta, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("situ_mul edges", from_device_bf16(device_out.data(), expected.size()),
                                    expected, criterion());
    failures += device_out.verify_guards("situ_mul edge out");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += contiguous_case("situ_mul [1152,1]", 1152, 1, 101U);
    failures += contiguous_case("situ_mul [1152,48]", 1152, 48, 102U);
    failures += fused_strided_case();
    failures += edge_case();
    std::cout << (failures ? "FAIL" : "OK") << " situ_mul\n";
    return failures ? 1 : 0;
}
