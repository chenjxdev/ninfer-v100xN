#include "ninfer/ops/inkling_router.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
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
    std::vector<int> selected;
    std::vector<double> weights;
};

Oracle oracle(const std::vector<float>& x, const std::vector<float>& gate,
              const std::vector<float>& bias, float weight_scale, int hidden, int experts,
              int tokens, int top_k) {
    Oracle result;
    result.selected.resize(static_cast<std::size_t>(tokens) * top_k);
    result.weights.resize(result.selected.size());
    std::vector<double> probabilities(experts);
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    for (int token = 0; token < tokens; ++token) {
        for (int expert = 0; expert < experts; ++expert) {
            double logit = 0.0;
            for (int h = 0; h < hidden; ++h) {
                logit += gate[expert * hidden + h] * x[token * hidden + h];
            }
            probabilities[expert] = 1.0 / (1.0 + std::exp(-logit));
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const double sa = probabilities[a] + bias[a];
            const double sb = probabilities[b] + bias[b];
            return sa == sb ? a < b : sa > sb;
        });
        double denominator = 1.0e-20;
        for (int k = 0; k < top_k; ++k) { denominator += probabilities[order[k]]; }
        for (int k = 0; k < top_k; ++k) {
            result.selected[token * top_k + k] = order[k];
            result.weights[token * top_k + k] = probabilities[order[k]] / denominator * weight_scale;
        }
    }
    return result;
}

int run_case(const char* label, int hidden, int experts, int tokens, int top_k, bool zero_gate,
             bool zero_bias, std::uint32_t seed) {
    constexpr float weight_scale = 2.5F;
    const std::size_t x_count = static_cast<std::size_t>(hidden) * tokens;
    const std::size_t gate_count = static_cast<std::size_t>(experts) * hidden;
    const std::size_t output_count = static_cast<std::size_t>(tokens) * top_k;
    std::vector<float> x(x_count), gate(gate_count), bias(experts);
    fill_uniform(x, seed, -2.0F, 2.0F);
    if (!zero_gate) { fill_uniform(gate, seed + 1, -0.05F, 0.05F); }
    if (!zero_bias) { fill_uniform(bias, seed + 2, -0.5F, 0.5F); }
    round_to_bf16(x);
    round_to_bf16(gate);
    const auto expected = oracle(x, gate, bias, weight_scale, hidden, experts, tokens, top_k);
    const auto x_bits = encode_bf16(x);
    const auto gate_bits = encode_bf16(gate);
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_bias(bias.size() * sizeof(float));
    GuardedDeviceBuffer device_selected(output_count * sizeof(std::int32_t));
    GuardedDeviceBuffer device_weights(output_count * sizeof(float));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_bias.copy_from_host(bias.data(), device_bias.bytes());
    Tensor x_tensor(device_x.data(), DType::BF16, {hidden, tokens});
    Tensor gate_tensor(device_gate.data(), DType::BF16, {experts, hidden});
    Tensor bias_tensor(device_bias.data(), DType::FP32, {experts});
    Tensor selected_tensor(device_selected.data(), DType::I32, {top_k, tokens});
    Tensor weights_tensor(device_weights.data(), DType::FP32, {top_k, tokens});
    DeviceArena workspace(ops::inkling_router_workspace_capacity_bytes(experts, 1, tokens));
    ops::inkling_router(x_tensor, gate_tensor, bias_tensor, weight_scale, workspace, selected_tensor,
                        weights_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_exact(label, from_device<std::int32_t>(device_selected.data(), output_count),
                                expected.selected);
    const auto actual_weights = from_device<float>(device_weights.data(), output_count);
    const std::vector<double> actual_weights_f64(actual_weights.begin(), actual_weights.end());
    failures += verify_pointwise("inkling_router weights", actual_weights_f64, expected.weights,
                                 {/*absolute*/ 2.0e-6, /*relative*/ 2.0e-4});
    failures += verify_exact("inkling_router x unchanged",
                             from_device<std::uint16_t>(device_x.data(), x_count), x_bits);
    failures += verify_exact("inkling_router gate unchanged",
                             from_device<std::uint16_t>(device_gate.data(), gate_count), gate_bits);
    failures += device_x.verify_guards("inkling_router x");
    failures += device_gate.verify_guards("inkling_router gate");
    failures += device_bias.verify_guards("inkling_router bias");
    failures += device_selected.verify_guards("inkling_router selected");
    failures += device_weights.verify_guards("inkling_router weights");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case("inkling_router random", 128, 128, 3, 8, false, false, 101U);
    failures += run_case("inkling_router full hidden", 2304, 128, 2, 8, false, false, 201U);
    failures += run_case("inkling_router biased selection unbiased weights", 128, 128, 1, 8, true,
                         false, 301U);
    failures += run_case("inkling_router ties", 128, 128, 1, 8, true, true, 401U);
    std::cout << (failures ? "FAIL" : "OK") << " inkling_router\n";
    return failures ? 1 : 0;
}
