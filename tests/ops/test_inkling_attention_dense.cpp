#include "ninfer/ops/inkling_attention_dense.h"
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

std::vector<double> oracle(const std::vector<float>& q, const std::vector<float>& k,
                           const std::vector<float>& v, const std::vector<float>& r,
                           const std::vector<float>& relative_bias,
                           const std::vector<float>& sink_logits, int heads, int kv_heads,
                           int tokens, int keys, int relative_length, int extent, float scale,
                           std::uint32_t sliding_window) {
    std::vector<double> expected(q.size()), scores(keys);
    for (int query = 0; query < tokens; ++query) {
        const int query_position = keys - tokens + query;
        for (int head = 0; head < heads; ++head) {
            const int kv_head = head / (heads / kv_heads);
            double maximum = 0.0;
            for (int key = 0; key < keys; ++key) {
                const int distance = query_position - key;
                if (distance < 0 ||
                    (sliding_window != 0 && static_cast<std::uint32_t>(distance) >= sliding_window)) {
                    scores[key] = -INFINITY;
                    continue;
                }
                double score = 0.0;
                for (int d = 0; d < 128; ++d) {
                    score += q[(query * heads + head) * 128 + d] *
                             k[(key * kv_heads + kv_head) * 128 + d];
                }
                score *= scale;
                if (distance < extent) {
                    for (int j = 0; j < relative_length; ++j) {
                        score += r[(query * heads + head) * relative_length + j] *
                                 relative_bias[distance * relative_length + j];
                    }
                }
                score -= sink_logits[query * heads + head];
                scores[key] = score;
                maximum = std::max(maximum, score);
            }
            double denominator = std::exp(0.0 - maximum);
            std::vector<double> numerator(128, 0.0);
            for (int key = 0; key < keys; ++key) {
                if (!std::isfinite(scores[key])) { continue; }
                const double probability = std::exp(scores[key] - maximum);
                denominator += probability;
                for (int d = 0; d < 128; ++d) {
                    numerator[d] += probability * v[(key * kv_heads + kv_head) * 128 + d];
                }
            }
            for (int d = 0; d < 128; ++d) {
                expected[(query * heads + head) * 128 + d] = numerator[d] / denominator;
            }
        }
    }
    return expected;
}

int run_case(const char* label, int heads, int kv_heads, int relative_length, int extent,
             int keys, int tokens, std::uint32_t sliding_window, std::uint32_t seed) {
    const std::size_t q_count = static_cast<std::size_t>(128) * heads * tokens;
    const std::size_t kv_count = static_cast<std::size_t>(128) * kv_heads * keys;
    const std::size_t r_count = static_cast<std::size_t>(relative_length) * heads * tokens;
    const std::size_t bias_count = static_cast<std::size_t>(relative_length) * extent;
    const std::size_t sink_count = static_cast<std::size_t>(heads) * tokens;
    std::vector<float> q(q_count), k(kv_count), v(kv_count), r(r_count),
        relative_bias(bias_count), sink_logits(sink_count);
    fill_uniform(q, seed, -1.0F, 1.0F);
    fill_uniform(k, seed + 1, -1.0F, 1.0F);
    fill_uniform(v, seed + 2, -1.0F, 1.0F);
    fill_uniform(r, seed + 3, -1.0F, 1.0F);
    fill_uniform(relative_bias, seed + 4, -0.2F, 0.2F);
    fill_uniform(sink_logits, seed + 5, -0.5F, 0.5F);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    round_to_bf16(r);
    round_to_bf16(relative_bias);
    round_to_bf16(sink_logits);
    const float scale = 1.0F / std::sqrt(128.0F);
    const auto expected = oracle(q, k, v, r, relative_bias, sink_logits, heads, kv_heads, tokens,
                                 keys, relative_length, extent, scale, sliding_window);
    const auto q_bits = encode_bf16(q);
    const auto k_bits = encode_bf16(k);
    const auto v_bits = encode_bf16(v);
    const auto r_bits = encode_bf16(r);
    const auto bias_bits = encode_bf16(relative_bias);
    const auto sink_bits = encode_bf16(sink_logits);
    GuardedDeviceBuffer device_q(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_k(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_v(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_r(r_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_bias(bias_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_sink(sink_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(q_count * sizeof(std::uint16_t));
    device_q.copy_from_host(q_bits.data(), device_q.bytes());
    device_k.copy_from_host(k_bits.data(), device_k.bytes());
    device_v.copy_from_host(v_bits.data(), device_v.bytes());
    device_r.copy_from_host(r_bits.data(), device_r.bytes());
    device_bias.copy_from_host(bias_bits.data(), device_bias.bytes());
    device_sink.copy_from_host(sink_bits.data(), device_sink.bytes());
    Tensor q_tensor(device_q.data(), DType::BF16, {128, heads, tokens});
    Tensor k_tensor(device_k.data(), DType::BF16, {128, kv_heads, keys});
    Tensor v_tensor(device_v.data(), DType::BF16, {128, kv_heads, keys});
    Tensor r_tensor(device_r.data(), DType::BF16, {relative_length, heads, tokens});
    Tensor bias_tensor(device_bias.data(), DType::BF16, {relative_length, extent});
    Tensor sink_tensor(device_sink.data(), DType::BF16, {heads, tokens});
    Tensor out_tensor(device_out.data(), DType::BF16, {128, heads, tokens});
    ops::inkling_attention_dense(q_tensor, k_tensor, v_tensor, r_tensor, bias_tensor, sink_tensor,
                                 scale, sliding_window, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), q_count), expected,
                                    {/*absolute*/ 1.5e-4, /*relative*/ 5.2e-3});
    failures += verify_exact("inkling_attention_dense q unchanged",
                             from_device<std::uint16_t>(device_q.data(), q_count), q_bits);
    failures += verify_exact("inkling_attention_dense k unchanged",
                             from_device<std::uint16_t>(device_k.data(), kv_count), k_bits);
    failures += verify_exact("inkling_attention_dense v unchanged",
                             from_device<std::uint16_t>(device_v.data(), kv_count), v_bits);
    failures += device_q.verify_guards("inkling_attention_dense q");
    failures += device_k.verify_guards("inkling_attention_dense k");
    failures += device_v.verify_guards("inkling_attention_dense v");
    failures += device_r.verify_guards("inkling_attention_dense r");
    failures += device_bias.verify_guards("inkling_attention_dense bias");
    failures += device_sink.verify_guards("inkling_attention_dense sink");
    failures += device_out.verify_guards("inkling_attention_dense output");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case("inkling_attention_dense full", 4, 2, 16, 8, 37, 5, 0, 101U);
    failures += run_case("inkling_attention_dense swa", 4, 2, 16, 16, 37, 5, 4, 201U);
    failures += run_case("inkling_attention_dense real full", 24, 8, 16, 1024, 17, 3, 0, 301U);
    failures += run_case("inkling_attention_dense swa1024", 24, 8, 16, 1024, 1030, 3, 1024, 401U);
    std::cout << (failures ? "FAIL" : "OK") << " inkling_attention_dense\n";
    return failures ? 1 : 0;
}
