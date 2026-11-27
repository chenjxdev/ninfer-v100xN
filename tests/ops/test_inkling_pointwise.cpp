#include "ninfer/ops/inkling_pointwise.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

int softcap_case() {
    constexpr float cap = 30.0F;
    std::vector<float> x(200058);
    fill_uniform(x, 101U, -100.0F, 100.0F);
    static constexpr float edges[] = {-100.0F, -30.0F, -1.0F, 0.0F, 1.0F,
                                      30.0F, 100.0F, 3.0F, -3.0F};
    for (std::size_t i = 0; i < 9 && i < x.size(); ++i) { x[i] = edges[i]; }
    round_to_bf16(x);
    std::vector<double> expected(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        expected[i] = cap * std::tanh(static_cast<double>(x[i]) / cap);
    }
    const auto x_bits = encode_bf16(x);
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(x_bits.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    Tensor x_tensor(device_x.data(), DType::BF16, {static_cast<int>(x.size())});
    Tensor out_tensor(device_out.data(), DType::BF16, {static_cast<int>(x.size())});
    ops::inkling_softcap(x_tensor, cap, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("inkling_softcap vocabulary",
                                    from_device_bf16(device_out.data(), x.size()), expected,
                                    {/*absolute*/ 2.0e-5, /*relative*/ 4.1e-3});
    failures += verify_exact("inkling_softcap input unchanged",
                             from_device<std::uint16_t>(device_x.data(), x.size()), x_bits);
    failures += device_x.verify_guards("inkling_softcap input");
    failures += device_out.verify_guards("inkling_softcap output");
    return failures;
}

int log_scale_case(const char* label, int width, int heads, std::uint32_t seed) {
    constexpr std::uint32_t floor = 64;
    constexpr float alpha = 0.5F;
    const std::vector<std::int32_t> positions{0, 1, 63, 64, 65, 127, 1023};
    const int tokens = static_cast<int>(positions.size());
    const std::size_t count = static_cast<std::size_t>(width) * heads * tokens;
    std::vector<float> x(count);
    fill_uniform(x, seed, -4.0F, 4.0F);
    round_to_bf16(x);
    std::vector<double> expected(count);
    const int rows_per_token = width * heads;
    for (std::size_t index = 0; index < count; ++index) {
        const int token = static_cast<int>(index / rows_per_token);
        const double tau = 1.0 + alpha * std::log(std::max(1.0, (positions[token] + 1.0) / floor));
        expected[index] = x[index] * tau;
    }
    const auto x_bits = encode_bf16(x);
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_positions(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_out(x_bits.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_positions.copy_from_host(positions.data(), device_positions.bytes());
    Tensor x_tensor(device_x.data(), DType::BF16, {width, heads, tokens});
    Tensor positions_tensor(device_positions.data(), DType::I32, {tokens});
    Tensor out_tensor(device_out.data(), DType::BF16, {width, heads, tokens});
    ops::inkling_log_position_scale(x_tensor, positions_tensor, floor, alpha, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    {/*absolute*/ 2.0e-5, /*relative*/ 4.1e-3});
    failures += verify_exact("inkling_log_position_scale input unchanged",
                             from_device<std::uint16_t>(device_x.data(), count), x_bits);
    failures += device_x.verify_guards("inkling_log_position_scale input");
    failures += device_positions.verify_guards("inkling_log_position_scale positions");
    failures += device_out.verify_guards("inkling_log_position_scale output");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += softcap_case();
    failures += log_scale_case("inkling_log_position_scale Q", 128, 24, 201U);
    failures += log_scale_case("inkling_log_position_scale R", 16, 24, 301U);
    std::cout << (failures ? "FAIL" : "OK") << " inkling_pointwise\n";
    return failures ? 1 : 0;
}
