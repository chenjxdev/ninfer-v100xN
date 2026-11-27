#include "ninfer/ops/inkling_conv1d.h"
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

struct Oracle {
    std::vector<double> out;
    std::vector<double> state;
};

Oracle oracle(const std::vector<float>& x, const std::vector<float>& weight,
              const std::vector<float>& dynamic, const std::vector<float>& state, int channels,
              int tokens, int heads) {
    Oracle result;
    result.out.resize(x.size());
    result.state.resize(state.size());
    const auto input = [&](int c, int position) -> double {
        if (position < 0) { return state[(3 + position) * channels + c]; }
        return x[position * channels + c];
    };
    const double scale = 1.0 / std::sqrt(dynamic.empty() ? 5.0 : 9.0);
    const int channels_per_head = dynamic.empty() ? 0 : channels / heads;
    for (int c = 0; c < channels; ++c) {
        const int head = dynamic.empty() ? 0 : c / channels_per_head;
        for (int t = 0; t < tokens; ++t) {
            double sum = 0.0;
            for (int tap = 0; tap < 4; ++tap) {
                double coefficient = weight[c * 4 + tap];
                if (!dynamic.empty()) { coefficient += dynamic[(4 * head + tap) * tokens + t]; }
                sum += coefficient * input(c, t - 3 + tap);
            }
            result.out[t * channels + c] = (input(c, t) + sum) * scale;
        }
        for (int lag = 0; lag < 3; ++lag) {
            result.state[lag * channels + c] = input(c, tokens - 3 + lag);
        }
    }
    return result;
}

int run_case(const char* label, int channels, int tokens, int heads, bool use_dynamic,
             bool transposed_weight, bool alias_state, std::uint32_t seed) {
    const std::size_t x_count = static_cast<std::size_t>(channels) * tokens;
    const std::size_t weight_count = static_cast<std::size_t>(channels) * 4;
    const std::size_t state_count = static_cast<std::size_t>(channels) * 3;
    const std::size_t dynamic_count = use_dynamic ? static_cast<std::size_t>(4 * heads) * tokens : 0;
    std::vector<float> x(x_count), weight(weight_count), state(state_count), dynamic(dynamic_count);
    fill_uniform(x, seed, -2.0F, 2.0F);
    fill_uniform(weight, seed + 1, -1.0F, 1.0F);
    fill_uniform(state, seed + 2, -2.0F, 2.0F);
    if (use_dynamic) { fill_uniform(dynamic, seed + 3, -0.5F, 0.5F); }
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(state);
    round_to_bf16(dynamic);
    const auto expected = oracle(x, weight, dynamic, state, channels, tokens, heads);
    std::vector<double> expected_history(state_count * (tokens + 1));
    for (int snapshot = 0; snapshot <= tokens; ++snapshot) {
        for (int lag = 0; lag < 3; ++lag) {
            const int position = snapshot - 3 + lag;
            for (int c = 0; c < channels; ++c) {
                expected_history[(snapshot * 3 + lag) * channels + c] = position < 0
                    ? state[(3 + position) * channels + c] : x[position * channels + c];
            }
        }
    }

    std::vector<float> physical_weight(weight_count);
    for (int c = 0; c < channels; ++c) {
        for (int tap = 0; tap < 4; ++tap) {
            physical_weight[transposed_weight ? tap * channels + c : c * 4 + tap] = weight[c * 4 + tap];
        }
    }
    const auto x_bits = encode_bf16(x);
    const auto weight_bits = encode_bf16(physical_weight);
    const auto state_bits = encode_bf16(state);
    const auto dynamic_bits = encode_bf16(dynamic);
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_state_in(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_state_out(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dynamic(std::max<std::size_t>(dynamic_bits.size(), 1) * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_history(expected_history.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());
    device_state_in.copy_from_host(state_bits.data(), device_state_in.bytes());
    if (use_dynamic) { device_dynamic.copy_from_host(dynamic_bits.data(), dynamic_count * sizeof(std::uint16_t)); }

    Tensor x_tensor(device_x.data(), DType::BF16, {channels, tokens});
    Tensor weight_tensor = transposed_weight
        ? Tensor(device_weight.data(), DType::BF16, {4, channels})
        : Tensor(device_weight.data(), DType::BF16, {channels, 4});
    Tensor dynamic_tensor(device_dynamic.data(), DType::BF16, {4 * heads, tokens});
    Tensor state_in_tensor(device_state_in.data(), DType::BF16, {channels, 3});
    void* state_out_data = alias_state ? device_state_in.data() : device_state_out.data();
    Tensor state_out_tensor(state_out_data, DType::BF16, {channels, 3});
    Tensor out_tensor(device_out.data(), DType::BF16, {channels, tokens});
    Tensor history_tensor(device_history.data(), DType::BF16, {channels, 3, tokens + 1});
    ops::inkling_conv1d(x_tensor, weight_tensor, use_dynamic ? &dynamic_tensor : nullptr,
                        state_in_tensor, state_out_tensor, out_tensor, nullptr, &history_tensor);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), x_count), expected.out,
                                    {/*absolute*/ 2.0e-5, /*relative*/ 4.1e-3});
    failures += verify_pointwise("inkling_conv1d final state",
        from_device_bf16(state_out_data, state_count), expected.state,
        {/*absolute*/ 0.0, /*relative*/ 0.0});
    failures += verify_pointwise("inkling_conv1d rollback history",
        from_device_bf16(device_history.data(), expected_history.size()), expected_history,
        {/*absolute*/ 0.0, /*relative*/ 0.0});
    failures += verify_exact("inkling_conv1d x unchanged",
                             from_device<std::uint16_t>(device_x.data(), x_count), x_bits);
    failures += verify_exact("inkling_conv1d weight unchanged",
                             from_device<std::uint16_t>(device_weight.data(), weight_count), weight_bits);
    if (!alias_state) {
        failures += verify_exact("inkling_conv1d distinct input state unchanged",
            from_device<std::uint16_t>(device_state_in.data(), state_count), state_bits);
        failures += device_state_out.verify_guards("inkling_conv1d output state");
    }
    failures += device_x.verify_guards("inkling_conv1d x");
    failures += device_weight.verify_guards("inkling_conv1d weight");
    failures += device_state_in.verify_guards("inkling_conv1d input state");
    failures += device_dynamic.verify_guards("inkling_conv1d dynamic");
    failures += device_out.verify_guards("inkling_conv1d output");
    failures += device_history.verify_guards("inkling_conv1d history");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case("inkling_conv1d static decode alias", 128, 1, 1, false, false, true, 101U);
    failures += run_case("inkling_conv1d static prefill", 128, 8, 1, false, false, false, 201U);
    failures += run_case("inkling_conv1d dynamic transposed", 64, 5, 8, true, true, false, 301U);
    failures += run_case("inkling_conv1d dynamic long alias", 1024, 17, 8, true, true, true, 401U);
    std::cout << (failures ? "FAIL" : "OK") << " inkling_conv1d\n";
    return failures ? 1 : 0;
}
