#include "ninfer/ops/inkling_loop_control.h"
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

std::vector<double> oracle(const std::vector<float>& cur, const std::vector<float>& message,
                           const std::vector<float>& down, const std::vector<float>& gain,
                           const std::vector<float>& forget, int hidden, int rank, int tokens) {
    std::vector<double> expected(cur.size()), normalized(2 * hidden), gate(rank);
    for (int token = 0; token < tokens; ++token) {
        const int input_base = token * hidden;
        for (int part = 0; part < 2; ++part) {
            const auto& input = part == 0 ? cur : message;
            double sum = 0.0;
            for (int h = 0; h < hidden; ++h) {
                sum += double(input[input_base + h]) * input[input_base + h];
            }
            const double inv_rms = 1.0 / std::sqrt(sum / hidden + 1.0e-6);
            for (int h = 0; h < hidden; ++h) {
                normalized[part * hidden + h] = input[input_base + h] * inv_rms;
            }
        }
        for (int r = 0; r < rank; ++r) {
            double value = 0.0;
            for (int i = 0; i < 2 * hidden; ++i) {
                value += down[r * 2 * hidden + i] * normalized[i];
            }
            gate[r] = value;
        }
        for (int h = 0; h < hidden; ++h) {
            double gain_value = 0.0, forget_value = 0.0;
            for (int r = 0; r < rank; ++r) {
                gain_value += gain[h * rank + r] * gate[r];
                forget_value += forget[h * rank + r] * gate[r];
            }
            gain_value = std::tanh(gain_value);
            forget_value = std::tanh(forget_value);
            expected[input_base + h] = (1.0 - forget_value) * cur[input_base + h] +
                                       gain_value * message[input_base + h];
        }
    }
    return expected;
}

int run_case(const char* label, int hidden, int rank, int tokens, std::uint32_t seed) {
    const std::size_t input_count = static_cast<std::size_t>(hidden) * tokens;
    const std::size_t down_count = static_cast<std::size_t>(rank) * 2 * hidden;
    const std::size_t projection_count = static_cast<std::size_t>(hidden) * rank;
    std::vector<float> cur(input_count), message(input_count), down(down_count),
        gain(projection_count), forget(projection_count);
    fill_uniform(cur, seed, -2.0F, 2.0F);
    fill_uniform(message, seed + 1, -2.0F, 2.0F);
    fill_uniform(down, seed + 2, -0.05F, 0.05F);
    fill_uniform(gain, seed + 3, -0.05F, 0.05F);
    fill_uniform(forget, seed + 4, -0.05F, 0.05F);
    round_to_bf16(cur);
    round_to_bf16(message);
    round_to_bf16(down);
    round_to_bf16(gain);
    round_to_bf16(forget);
    const auto expected = oracle(cur, message, down, gain, forget, hidden, rank, tokens);
    const auto cur_bits = encode_bf16(cur);
    const auto message_bits = encode_bf16(message);
    const auto down_bits = encode_bf16(down);
    const auto gain_bits = encode_bf16(gain);
    const auto forget_bits = encode_bf16(forget);
    GuardedDeviceBuffer device_cur(cur_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_message(message_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_down(down_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_gain(gain_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_forget(forget_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(input_count * sizeof(std::uint16_t));
    device_cur.copy_from_host(cur_bits.data(), device_cur.bytes());
    device_message.copy_from_host(message_bits.data(), device_message.bytes());
    device_down.copy_from_host(down_bits.data(), device_down.bytes());
    device_gain.copy_from_host(gain_bits.data(), device_gain.bytes());
    device_forget.copy_from_host(forget_bits.data(), device_forget.bytes());
    Tensor cur_tensor(device_cur.data(), DType::BF16, {hidden, tokens});
    Tensor message_tensor(device_message.data(), DType::BF16, {hidden, tokens});
    Tensor down_tensor(device_down.data(), DType::BF16, {rank, 2 * hidden});
    Tensor gain_tensor(device_gain.data(), DType::BF16, {hidden, rank});
    Tensor forget_tensor(device_forget.data(), DType::BF16, {hidden, rank});
    Tensor out_tensor(device_out.data(), DType::BF16, {hidden, tokens});
    DeviceArena workspace(ops::inkling_loop_control_workspace_capacity_bytes(hidden, rank, 1, tokens));
    ops::inkling_loop_control(cur_tensor, message_tensor, down_tensor, gain_tensor, forget_tensor,
                              workspace, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), input_count), expected,
                                    {/*absolute*/ 5.0e-5, /*relative*/ 4.6e-3});
    failures += verify_exact("inkling_loop_control cur unchanged",
        from_device<std::uint16_t>(device_cur.data(), input_count), cur_bits);
    failures += verify_exact("inkling_loop_control message unchanged",
        from_device<std::uint16_t>(device_message.data(), input_count), message_bits);
    failures += verify_exact("inkling_loop_control down unchanged",
        from_device<std::uint16_t>(device_down.data(), down_count), down_bits);
    failures += device_cur.verify_guards("inkling_loop_control cur");
    failures += device_message.verify_guards("inkling_loop_control message");
    failures += device_down.verify_guards("inkling_loop_control down");
    failures += device_gain.verify_guards("inkling_loop_control gain");
    failures += device_forget.verify_guards("inkling_loop_control forget");
    failures += device_out.verify_guards("inkling_loop_control output");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case("inkling_loop_control small", 128, 64, 3, 101U);
    failures += run_case("inkling_loop_control full hidden", 2304, 64, 2, 201U);
    std::cout << (failures ? "FAIL" : "OK") << " inkling_loop_control\n";
    return failures ? 1 : 0;
}
