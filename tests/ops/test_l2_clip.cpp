#include "ninfer/ops/l2_clip.h"
#include "ops/op_tester.h"

#include <algorithm>
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

std::vector<double> oracle(const std::vector<float>& x, const std::vector<float>& scale,
                           std::int32_t width, std::int32_t scale_rows, float threshold) {
    const std::size_t rows = x.size() / static_cast<std::size_t>(width);
    std::vector<double> expected(x.size());
    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        const std::size_t scale_row = scale.empty() ? 0 : row % static_cast<std::size_t>(scale_rows);
        for (std::int32_t d = 0; d < width; ++d) {
            const double value = x[row * width + d] *
                (scale.empty() ? 1.0 : static_cast<double>(scale[scale_row * width + d]));
            sum += value * value;
        }
        const double factor = std::min(1.0, threshold / (std::sqrt(sum) + 1.0e-6));
        for (std::int32_t d = 0; d < width; ++d) {
            const double value = x[row * width + d] *
                (scale.empty() ? 1.0 : static_cast<double>(scale[scale_row * width + d]));
            expected[row * width + d] = value * factor;
        }
    }
    return expected;
}

int run_case(const char* label, std::int32_t width, std::int32_t heads, std::int32_t tokens,
             bool with_scale, float threshold, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(width) * heads * tokens;
    const std::size_t scale_count = with_scale ? static_cast<std::size_t>(width) * heads : 0;
    std::vector<float> x(count), scale(scale_count);
    fill_uniform(x, seed, -8.0F, 8.0F);
    if (with_scale) { fill_uniform(scale, seed + 1, -2.0F, 2.0F); }
    round_to_bf16(x);
    round_to_bf16(scale);
    const auto expected = oracle(x, scale, width, heads, threshold);
    const auto x_bits = encode_bf16(x);
    const auto scale_bits = encode_bf16(scale);
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_scale(std::max<std::size_t>(scale_count, 1) * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    if (with_scale) { device_scale.copy_from_host(scale_bits.data(), scale_count * sizeof(std::uint16_t)); }
    Tensor x_tensor(device_x.data(), DType::BF16, {width, heads, tokens});
    Tensor scale_tensor(device_scale.data(), DType::BF16, {width, heads});
    Tensor out_tensor(device_out.data(), DType::BF16, {width, heads, tokens});
    ops::l2_clip(x_tensor, with_scale ? &scale_tensor : nullptr, threshold, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    {/*absolute*/ 2.0e-5, /*relative*/ 4.1e-3});
    failures += verify_exact("l2_clip input unchanged", from_device<std::uint16_t>(device_x.data(), count),
                             x_bits);
    failures += device_x.verify_guards("l2_clip input");
    failures += device_scale.verify_guards("l2_clip scale");
    failures += device_out.verify_guards("l2_clip output");
    return failures;
}

int zero_case() {
    constexpr std::int32_t width = 128;
    constexpr std::int32_t rows = 3;
    const std::size_t count = width * rows;
    std::vector<float> x(count, 0.0F), scale(width, 0.0F);
    const auto expected = oracle(x, scale, width, 1, 10.0F);
    const auto x_bits = encode_bf16(x);
    const auto scale_bits = encode_bf16(scale);
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_scale(scale_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(x_bits.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_scale.copy_from_host(scale_bits.data(), device_scale.bytes());
    Tensor x_tensor(device_x.data(), DType::BF16, {width, rows});
    Tensor scale_tensor(device_scale.data(), DType::BF16, {width});
    Tensor out_tensor(device_out.data(), DType::BF16, {width, rows});
    ops::l2_clip(x_tensor, &scale_tensor, 10.0F, out_tensor, nullptr);
    cuda_synchronize();
    return verify_pointwise("l2_clip zero rows", from_device_bf16(device_out.data(), count), expected,
                            {/*absolute*/ 0.0, /*relative*/ 0.0});
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case("l2_clip Q", 128, 24, 9, true, 10.0F, 101U);
    failures += run_case("l2_clip K", 128, 8, 7, false, 10.0F, 201U);
    failures += run_case("l2_clip relative", 16, 24, 5, true, 4.0F, 301U);
    failures += zero_case();
    std::cout << (failures ? "FAIL" : "OK") << " l2_clip\n";
    return failures ? 1 : 0;
}
