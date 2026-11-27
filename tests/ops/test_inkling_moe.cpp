#include "ninfer/ops/inkling_moe.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHidden = 128;
constexpr int kIntermediate = 64;
constexpr int kExperts = 4;
constexpr int kTopK = 2;
int kTokens = 2;
constexpr float kBeta = 4.0F;
constexpr float kLinearBeta = 25.0F;

template <class WeightAt>
std::vector<double> oracle(const std::vector<float>& x, const std::vector<int>& selected,
                           const std::vector<float>& route_weights, WeightAt weight_at) {
    std::vector<double> expected(static_cast<std::size_t>(kHidden) * kTokens, 0.0);
    std::vector<double> activations(kIntermediate);
    for (int token = 0; token < kTokens; ++token) {
        for (int route = 0; route < kTopK; ++route) {
            const int expert = selected[token * kTopK + route];
            for (int i = 0; i < kIntermediate; ++i) {
                double gate = 0.0, up = 0.0;
                for (int h = 0; h < kHidden; ++h) {
                    const double value = x[token * kHidden + h];
                    gate += weight_at(expert, 0, 2 * i, h) * value;
                    up += weight_at(expert, 0, 2 * i + 1, h) * value;
                }
                const double gated = kBeta * std::tanh(gate / kBeta) / (1.0 + std::exp(-gate));
                const double limited_up = kLinearBeta * std::tanh(up / kLinearBeta);
                activations[i] = bf16_to_f32(f32_to_bf16(static_cast<float>(gated * limited_up)));
            }
            for (int h = 0; h < kHidden; ++h) {
                double value = 0.0;
                for (int i = 0; i < kIntermediate; ++i) {
                    value += weight_at(expert, 1, h, i) * activations[i];
                }
                expected[token * kHidden + h] +=
                    route_weights[token * kTopK + route] * value;
            }
        }
    }
    return expected;
}

int execute(const char* label, const std::vector<float>& x, const std::vector<int>& selected,
            const std::vector<float>& route_weights, std::span<const Weight> w13,
            std::span<const Weight> w2, const std::vector<double>& expected,
            PointwiseCriterion criterion) {
    std::vector<std::uint16_t> x_bits(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { x_bits[i] = f32_to_bf16(x[i]); }
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_selected(selected.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_route(route_weights.size() * sizeof(float));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_selected.copy_from_host(selected.data(), device_selected.bytes());
    device_route.copy_from_host(route_weights.data(), device_route.bytes());
    Tensor x_tensor(device_x.data(), DType::BF16, {kHidden, kTokens});
    Tensor selected_tensor(device_selected.data(), DType::I32, {kTopK, kTokens});
    Tensor route_tensor(device_route.data(), DType::FP32, {kTopK, kTokens});
    Tensor out_tensor(device_out.data(), DType::BF16, {kHidden, kTokens});
    DeviceArena workspace(ops::inkling_moe_workspace_capacity_bytes(
        kHidden, kIntermediate, kTopK, 1, kTokens));
    ops::inkling_moe(x_tensor, w13, w2, selected_tensor, route_tensor, kBeta, kLinearBeta,
                     workspace, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), expected.size()),
                                    expected, criterion);
    failures += verify_exact("inkling_moe x unchanged",
                             from_device<std::uint16_t>(device_x.data(), x.size()), x_bits);
    failures += verify_exact("inkling_moe selected unchanged",
                             from_device<std::int32_t>(device_selected.data(), selected.size()), selected);
    failures += device_x.verify_guards("inkling_moe x");
    failures += device_selected.verify_guards("inkling_moe selected");
    failures += device_route.verify_guards("inkling_moe route weights");
    failures += device_out.verify_guards("inkling_moe output");
    return failures;
}

int nvfp4_case(const std::vector<float>& x, const std::vector<int>& selected,
               const std::vector<float>& route_weights) {
    using namespace quantized_weight;
    std::vector<PackedWeight> host13, host2;
    std::vector<DeviceBuffer> device13, device2;
    host13.reserve(kExperts);
    host2.reserve(kExperts);
    device13.reserve(kExperts);
    device2.reserve(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        const float divisor = 1.0F + expert;
        host13.push_back(make_patterned_weight(QType::NVFP4, 2 * kIntermediate, kHidden,
            100U + expert, {.weight_scale_divisor = divisor, .input_scale_divisor = 1.0F}));
        host2.push_back(make_patterned_weight(QType::NVFP4, kHidden, kIntermediate,
            200U + expert, {.weight_scale_divisor = divisor * 1.5F, .input_scale_divisor = 1.0F}));
    }
    std::vector<Weight> views13, views2;
    views13.reserve(kExperts);
    views2.reserve(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        device13.emplace_back(host13[expert].payload.size());
        device2.emplace_back(host2[expert].payload.size());
        device13[expert].copy_from_host(host13[expert].payload.data(), device13[expert].bytes);
        device2[expert].copy_from_host(host2[expert].payload.data(), device2[expert].bytes);
        views13.push_back(host13[expert].device_weight(device13[expert].p));
        views2.push_back(host2[expert].device_weight(device2[expert].p));
    }
    const auto expected = oracle(x, selected, route_weights, [&](int expert, int which, int row, int column) {
        return logical_weight_fp64(which == 0 ? host13[expert] : host2[expert], row, column);
    });
    int failures = execute("inkling_moe NVFP4", x, selected, route_weights, views13, views2,
                           expected, {/*absolute*/ 3.0e-4, /*relative*/ 7.0e-3});
    for (int expert = 0; expert < kExperts; ++expert) {
        std::vector<std::uint8_t> after13(host13[expert].payload.size());
        std::vector<std::uint8_t> after2(host2[expert].payload.size());
        device13[expert].copy_to_host(after13.data(), after13.size());
        device2[expert].copy_to_host(after2.data(), after2.size());
        failures += verify_exact("inkling_moe NVFP4 w13 preserved", after13, host13[expert].payload);
        failures += verify_exact("inkling_moe NVFP4 w2 preserved", after2, host2[expert].payload);
    }
    return failures;
}

int bf16_case(const std::vector<float>& x, const std::vector<int>& selected,
              const std::vector<float>& route_weights) {
    using direct_bf16_weight::DeviceWeight;
    using direct_bf16_weight::make_patterned;
    std::vector<DeviceWeight> weights13, weights2;
    weights13.reserve(kExperts);
    weights2.reserve(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        weights13.emplace_back(make_patterned(2 * kIntermediate, kHidden, 300U + expert));
        weights2.emplace_back(make_patterned(kHidden, kIntermediate, 400U + expert));
    }
    std::vector<Weight> views13, views2;
    for (auto& weight : weights13) { views13.push_back(weight.view()); }
    for (auto& weight : weights2) { views2.push_back(weight.view()); }
    const auto expected = oracle(x, selected, route_weights, [&](int expert, int which, int row, int column) {
        const auto& host = which == 0 ? weights13[expert].host : weights2[expert].host;
        return static_cast<double>(bf16_to_f32(host.bits[row * host.k + column]));
    });
    int failures = execute("inkling_moe BF16", x, selected, route_weights, views13, views2,
                           expected, {/*absolute*/ 2.0e-4, /*relative*/ 6.0e-3});
    for (auto& weight : weights13) { failures += weight.verify_preserved("inkling_moe BF16 w13"); }
    for (auto& weight : weights2) { failures += weight.verify_preserved("inkling_moe BF16 w2"); }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const int token_count : {2, 16}) {
        kTokens = token_count;
        std::vector<float> x(kHidden * kTokens);
        fill_uniform(x, 101U, -1.0F, 1.0F);
        round_to_bf16(x);
        std::vector<int> selected;
        std::vector<float> route_weights;
        selected.reserve(kTokens * kTopK);
        route_weights.reserve(kTokens * kTopK);
        for (int token = 0; token < kTokens; ++token) {
            for (int route = 0; route < kTopK; ++route) {
                selected.push_back((token * kTopK + route) % kExperts);
                route_weights.push_back(0.2F + 0.1F * ((token + route) % 4));
            }
        }
        failures += nvfp4_case(x, selected, route_weights);
        failures += bf16_case(x, selected, route_weights);
    }
    std::cout << (failures ? "FAIL" : "OK") << " inkling_moe\n";
    return failures ? 1 : 0;
}
