#include "ninfer/ops/inkling_depth_mix.h"
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

std::vector<double> oracle(const std::vector<float>& states, const std::vector<float>& queries,
                           int hidden, int depth, int tokens, int query_count) {
    std::vector<double> normalized(states.size());
    for (int token = 0; token < tokens; ++token) {
        for (int state = 0; state < depth; ++state) {
            const int base = (token * depth + state) * hidden;
            double sum = 0.0;
            for (int h = 0; h < hidden; ++h) { sum += double(states[base + h]) * states[base + h]; }
            const double inv_rms = 1.0 / std::sqrt(sum / hidden + 1.0e-6);
            for (int h = 0; h < hidden; ++h) { normalized[base + h] = states[base + h] * inv_rms; }
        }
    }
    std::vector<double> expected(static_cast<std::size_t>(hidden) * query_count * tokens);
    std::vector<double> scores(depth), probabilities(depth);
    for (int token = 0; token < tokens; ++token) {
        for (int query = 0; query < query_count; ++query) {
            double maximum = -INFINITY;
            for (int state = 0; state < depth; ++state) {
                const int base = (token * depth + state) * hidden;
                double score = 0.0;
                for (int h = 0; h < hidden; ++h) {
                    score += normalized[base + h] * queries[query * hidden + h];
                }
                scores[state] = score / std::sqrt(static_cast<double>(hidden));
                maximum = std::max(maximum, scores[state]);
            }
            double denominator = 0.0;
            for (int state = 0; state < depth; ++state) {
                probabilities[state] = std::exp(scores[state] - maximum);
                denominator += probabilities[state];
            }
            for (int state = 0; state < depth; ++state) { probabilities[state] /= denominator; }
            const int output_base = (token * query_count + query) * hidden;
            for (int h = 0; h < hidden; ++h) {
                double value = 0.0;
                for (int state = 0; state < depth; ++state) {
                    value += normalized[(token * depth + state) * hidden + h] * probabilities[state];
                }
                expected[output_base + h] = value;
            }
        }
    }
    return expected;
}

int run_case(const char* label, int hidden, int depth, int tokens, int query_count,
             std::uint32_t seed) {
    const std::size_t state_count = static_cast<std::size_t>(hidden) * depth * tokens;
    const std::size_t query_size = static_cast<std::size_t>(hidden) * query_count;
    const std::size_t output_count = query_size * tokens;
    std::vector<float> states(state_count), queries(query_size);
    fill_uniform(states, seed, -2.0F, 2.0F);
    fill_uniform(queries, seed + 1, -1.0F, 1.0F);
    round_to_bf16(states);
    const auto expected = oracle(states, queries, hidden, depth, tokens, query_count);
    const auto state_bits = encode_bf16(states);
    GuardedDeviceBuffer device_states(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_queries(queries.size() * sizeof(float));
    GuardedDeviceBuffer device_out(output_count * sizeof(std::uint16_t));
    device_states.copy_from_host(state_bits.data(), device_states.bytes());
    device_queries.copy_from_host(queries.data(), device_queries.bytes());
    Tensor states_tensor(device_states.data(), DType::BF16, {hidden, depth, tokens});
    Tensor queries_tensor(device_queries.data(), DType::FP32, {hidden, query_count});
    Tensor out_tensor(device_out.data(), DType::BF16, {hidden, query_count, tokens});
    ops::inkling_depth_mix(states_tensor, queries_tensor, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), output_count), expected,
                                    {/*absolute*/ 3.0e-5, /*relative*/ 4.2e-3});
    failures += verify_exact("inkling_depth_mix states unchanged",
        from_device<std::uint16_t>(device_states.data(), state_count), state_bits);
    failures += verify_exact("inkling_depth_mix queries unchanged",
        from_device<float>(device_queries.data(), query_size), queries);
    failures += device_states.verify_guards("inkling_depth_mix states");
    failures += device_queries.verify_guards("inkling_depth_mix queries");
    failures += device_out.verify_guards("inkling_depth_mix output");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case("inkling_depth_mix loop messages", 128, 5, 3, 4, 101U);
    failures += run_case("inkling_depth_mix full hidden", 2304, 5, 2, 4, 201U);
    failures += run_case("inkling_depth_mix single", 2304, 1, 1, 1, 301U);
    std::cout << (failures ? "FAIL" : "OK") << " inkling_depth_mix\n";
    return failures ? 1 : 0;
}
