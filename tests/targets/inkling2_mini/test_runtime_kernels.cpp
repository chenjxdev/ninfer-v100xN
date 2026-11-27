#include "targets/inkling2_mini/impl/runtime_kernels.h"
#include "ninfer/ops/inkling_attention_dense.h"
#include "ninfer/ops/inkling_depth_mix.h"
#include "ops/op_tester.h"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    constexpr int rows = 64;
    constexpr int columns = 1024;
    int failures = 0;
    for (int tokens = 1; tokens <= 8; ++tokens) {
        std::vector<float> x(static_cast<std::size_t>(columns) * tokens);
        std::vector<float> weight(static_cast<std::size_t>(rows) * columns);
        fill_uniform(x, 900U + tokens, -2.0F, 2.0F);
        fill_uniform(weight, 1000U + tokens, -1.0F, 1.0F);
        round_to_bf16(x);
        round_to_bf16(weight);
        std::vector<std::uint16_t> x_bits(x.size()), weight_bits(weight.size());
        for (std::size_t i = 0; i < x.size(); ++i) { x_bits[i] = f32_to_bf16(x[i]); }
        for (std::size_t i = 0; i < weight.size(); ++i) { weight_bits[i] = f32_to_bf16(weight[i]); }
        std::vector<double> expected(x.size() / columns * rows, 0.0);
        expected.resize(static_cast<std::size_t>(rows) * tokens);
        for (int token = 0; token < tokens; ++token) {
            for (int row = 0; row < rows; ++row) {
                double sum = 0.0;
                for (int column = 0; column < columns; ++column) {
                    sum += static_cast<double>(weight[row * columns + column]) *
                           x[token * columns + column];
                }
                expected[token * rows + row] = sum;
            }
        }
        GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer device_weight(weight_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer device_out(expected.size() * sizeof(std::uint16_t));
        device_x.copy_from_host(x_bits.data(), device_x.bytes());
        device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());
        Tensor input(device_x.data(), DType::BF16, {columns, tokens});
        Tensor output(device_out.data(), DType::BF16, {rows, tokens});
        Weight matrix{};
        matrix.payload = device_weight.data();
        matrix.qdata = device_weight.data();
        matrix.payload_bytes = device_weight.bytes();
        matrix.qtype = QType::BF16_CTRL;
        matrix.ndim = 2;
        matrix.n = matrix.shape[0] = matrix.padded_shape[0] = rows;
        matrix.k = matrix.shape[1] = matrix.padded_shape[1] = columns;
        matrix.layout = QuantLayout::Contiguous;
        targets::inkling2_mini::detail::bf16_gemv(input, matrix, output, nullptr);
        cuda_synchronize();
        failures += verify_pointwise("Inkling BF16 multi-token GEMV",
            from_device_bf16(device_out.data(), expected.size()), expected,
            {/*absolute*/ 2.0e-5, /*relative*/ 4.1e-3});
        failures += device_out.verify_guards("Inkling BF16 multi-token GEMV output");

        GuardedDeviceBuffer group_out0(expected.size() / 2 * sizeof(std::uint16_t));
        GuardedDeviceBuffer group_out1(expected.size() / 2 * sizeof(std::uint16_t));
        GuardedDeviceBuffer group_x1(device_x.bytes());
        group_x1.copy_from_host(x_bits.data(), device_x.bytes());
        std::array<targets::inkling2_mini::detail::BF16GemvItem, 8> group_items{};
        group_items[0] = {device_x.data(), device_weight.data(), group_out0.data(), rows / 2, 0};
        group_items[1] = {group_x1.data(),
            static_cast<const std::uint16_t*>(device_weight.data()) + (rows / 2) * columns,
            group_out1.data(), rows / 2, 0};
        targets::inkling2_mini::detail::bf16_gemv_group(group_items, 2, columns, tokens, nullptr);
        cuda_synchronize();
        const auto group0 = from_device_bf16(group_out0.data(), expected.size() / 2);
        const auto group1 = from_device_bf16(group_out1.data(), expected.size() / 2);
        std::vector<double> grouped;
        for (int token = 0; token < tokens; ++token) {
            grouped.insert(grouped.end(), group0.begin() + token * (rows / 2),
                           group0.begin() + (token + 1) * (rows / 2));
            grouped.insert(grouped.end(), group1.begin() + token * (rows / 2),
                           group1.begin() + (token + 1) * (rows / 2));
        }
        failures += verify_pointwise("Inkling grouped BF16 GEMV", grouped, expected,
                                     {/*absolute*/ 2.0e-5, /*relative*/ 4.1e-3});
        failures += group_out0.verify_guards("Inkling grouped GEMV output0");
        failures += group_out1.verify_guards("Inkling grouped GEMV output1");
    }
    {
        constexpr int hidden = 256;
        constexpr int tokens = 16;
        constexpr int depth = 5;
        constexpr int queries = 4;
        std::vector<DeviceBuffer> source_buffers;
        source_buffers.reserve(depth);
        std::array<const void*, depth> source_views{};
        std::vector<std::uint16_t> packed_bits(static_cast<std::size_t>(hidden) * depth * tokens);
        for (int state = 0; state < depth; ++state) {
            std::vector<float> values(static_cast<std::size_t>(hidden) * tokens);
            fill_uniform(values, 1200U + state, -2.0F, 2.0F);
            round_to_bf16(values);
            std::vector<std::uint16_t> bits(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
            source_buffers.emplace_back(bits.size() * sizeof(std::uint16_t));
            source_buffers.back().copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
            source_views[state] = source_buffers.back().p;
            for (int token = 0; token < tokens; ++token) {
                for (int h = 0; h < hidden; ++h) {
                    packed_bits[(token * depth + state) * hidden + h] = bits[token * hidden + h];
                }
            }
        }
        std::vector<float> query_values(static_cast<std::size_t>(hidden) * queries);
        fill_uniform(query_values, 1300U, -1.0F, 1.0F);
        GuardedDeviceBuffer device_packed(packed_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer device_queries(query_values.size() * sizeof(float));
        GuardedDeviceBuffer device_expected(packed_bits.size() / depth * queries * sizeof(std::uint16_t));
        GuardedDeviceBuffer device_direct(packed_bits.size() / depth * queries * sizeof(std::uint16_t));
        device_packed.copy_from_host(packed_bits.data(), device_packed.bytes());
        device_queries.copy_from_host(query_values.data(), device_queries.bytes());
        Tensor states(device_packed.data(), DType::BF16, {hidden, depth, tokens});
        Tensor query(device_queries.data(), DType::FP32, {hidden, queries});
        Tensor expected(device_expected.data(), DType::BF16, {hidden, queries, tokens});
        Tensor direct(device_direct.data(), DType::BF16, {hidden, tokens, queries});
        ops::inkling_depth_mix(states, query, expected, nullptr);
        targets::inkling2_mini::detail::mix_depth_states(source_views, depth, query, direct, nullptr);
        cuda_synchronize();
        const auto expected_values = from_device_bf16(device_expected.data(), expected.numel());
        const auto direct_values = from_device_bf16(device_direct.data(), direct.numel());
        std::vector<double> reordered(expected_values.size());
        for (int query_index = 0; query_index < queries; ++query_index) {
            for (int token = 0; token < tokens; ++token) {
                for (int h = 0; h < hidden; ++h) {
                    reordered[(token * queries + query_index) * hidden + h] =
                        direct_values[(query_index * tokens + token) * hidden + h];
                }
            }
        }
        failures += verify_exact("Inkling direct depth-mix query-major output", reordered,
                                 expected_values);
        failures += device_direct.verify_guards("Inkling direct depth-mix output");

        GuardedDeviceBuffer device_warp(device_direct.bytes());
        Tensor warp(device_warp.data(), DType::BF16, {hidden, tokens, queries});
        setenv("NINFER_INKLING_PREFILL_WARP_MIX", "1", 1);
        targets::inkling2_mini::detail::mix_depth_states(source_views, depth, query, warp, nullptr);
        unsetenv("NINFER_INKLING_PREFILL_WARP_MIX");
        cuda_synchronize();
        const auto warp_values = from_device_bf16(device_warp.data(), warp.numel());
        std::vector<double> warp_reordered(expected_values.size());
        for (int query_index = 0; query_index < queries; ++query_index) {
            for (int token = 0; token < tokens; ++token) {
                for (int h = 0; h < hidden; ++h) {
                    warp_reordered[(token * queries + query_index) * hidden + h] =
                        warp_values[(query_index * tokens + token) * hidden + h];
                }
            }
        }
        failures += verify_pointwise("Inkling warp prefill depth-mix", warp_reordered,
            expected_values, {/*absolute*/ 2.0e-5, /*relative*/ 4.1e-3});
        failures += device_warp.verify_guards("Inkling warp prefill depth-mix output");
    }
    {
        constexpr int channels = 8;
        constexpr int capacity = 16;
        const int chunks[] = {3, 5, 1, 7, 9, 6, 8, 2};
        GuardedDeviceBuffer k_cache(channels * capacity * sizeof(std::uint16_t));
        GuardedDeviceBuffer v_cache(channels * capacity * sizeof(std::uint16_t));
        Tensor k_ring(k_cache.data(), DType::BF16, {channels, capacity});
        Tensor v_ring(v_cache.data(), DType::BF16, {channels, capacity});
        std::vector<std::uint16_t> expected_k(channels * 41), expected_v(channels * 41);
        int position = 0;
        for (const int count : chunks) {
            std::vector<std::uint16_t> k(channels * count), v(channels * count);
            for (int token = 0; token < count; ++token) {
                for (int channel = 0; channel < channels; ++channel) {
                    const int index = token * channels + channel;
                    k[index] = f32_to_bf16(static_cast<float>((position + token) * channels + channel + 1) * 0.25F);
                    v[index] = f32_to_bf16(-static_cast<float>((position + token) * channels + channel + 1) * 0.125F);
                    expected_k[(position + token) * channels + channel] = k[index];
                    expected_v[(position + token) * channels + channel] = v[index];
                }
            }
            GuardedDeviceBuffer device_k(k.size() * sizeof(std::uint16_t));
            GuardedDeviceBuffer device_v(v.size() * sizeof(std::uint16_t));
            device_k.copy_from_host(k.data(), device_k.bytes());
            device_v.copy_from_host(v.data(), device_v.bytes());
            Tensor k_tensor(device_k.data(), DType::BF16, {channels, count});
            Tensor v_tensor(device_v.data(), DType::BF16, {channels, count});
            targets::inkling2_mini::detail::append_kv_ring(k_tensor, v_tensor, k_ring, v_ring,
                                                            position, capacity, nullptr);
            position += count;
        }
        constexpr int first = 26;
        constexpr int count = 15;
        GuardedDeviceBuffer gathered_k(channels * count * sizeof(std::uint16_t));
        GuardedDeviceBuffer gathered_v(channels * count * sizeof(std::uint16_t));
        Tensor k_out(gathered_k.data(), DType::BF16, {channels, count});
        Tensor v_out(gathered_v.data(), DType::BF16, {channels, count});
        targets::inkling2_mini::detail::gather_kv_ring(k_ring, v_ring, first, count, capacity,
                                                        k_out, v_out, nullptr);
        cuda_synchronize();
        const std::vector<std::uint16_t> expected_window_k(expected_k.begin() + first * channels,
                                                            expected_k.end());
        const std::vector<std::uint16_t> expected_window_v(expected_v.begin() + first * channels,
                                                            expected_v.end());
        failures += verify_exact("Inkling ring K gather", from_device<std::uint16_t>(
            gathered_k.data(), channels * count), expected_window_k);
        failures += verify_exact("Inkling ring V gather", from_device<std::uint16_t>(
            gathered_v.data(), channels * count), expected_window_v);
        failures += gathered_k.verify_guards("Inkling ring K output");
        failures += gathered_v.verify_guards("Inkling ring V output");
    }
    {
        constexpr int heads = 2;
        constexpr int tokens = 3;
        constexpr int keys = 8;
        constexpr int relative = 4;
        const auto make_bits = [](std::vector<float>& values, std::uint32_t seed) {
            fill_uniform(values, seed, -1.0F, 1.0F);
            round_to_bf16(values);
            std::vector<std::uint16_t> bits(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
            return bits;
        };
        std::vector<float> q_values(128 * heads * tokens), k_values(128 * keys), v_values(128 * keys),
            r_values(relative * heads * tokens), bias_values(relative * keys), sink_values(heads * tokens);
        const auto q = make_bits(q_values, 1500U), k = make_bits(k_values, 1501U),
            v = make_bits(v_values, 1502U), r = make_bits(r_values, 1503U),
            bias = make_bits(bias_values, 1504U), sinks = make_bits(sink_values, 1505U);
        const auto upload = [](GuardedDeviceBuffer& buffer, const std::vector<std::uint16_t>& data) {
            buffer.copy_from_host(data.data(), data.size() * sizeof(std::uint16_t));
        };
        GuardedDeviceBuffer device_q(q.size() * 2), device_k(k.size() * 2), device_v(v.size() * 2),
            device_r(r.size() * 2), device_bias(bias.size() * 2), device_sinks(sinks.size() * 2),
            device_expected(q.size() * 2), device_actual(q.size() * 2),
            device_k_codes(128 * keys), device_v_codes(128 * keys),
            device_k_scales(keys * sizeof(float)), device_v_scales(keys * sizeof(float));
        upload(device_q, q); upload(device_k, k); upload(device_v, v); upload(device_r, r);
        upload(device_bias, bias); upload(device_sinks, sinks);
        Tensor q_tensor(device_q.data(), DType::BF16, {128, heads, tokens});
        Tensor k_tensor(device_k.data(), DType::BF16, {128, 1, keys});
        Tensor v_tensor(device_v.data(), DType::BF16, {128, 1, keys});
        Tensor r_tensor(device_r.data(), DType::BF16, {relative, heads, tokens});
        Tensor bias_tensor(device_bias.data(), DType::BF16, {relative, keys});
        Tensor sink_tensor(device_sinks.data(), DType::BF16, {heads, tokens});
        Tensor expected(device_expected.data(), DType::BF16, {128, heads, tokens});
        Tensor actual(device_actual.data(), DType::BF16, {128, heads, tokens});
        Tensor k_codes(device_k_codes.data(), DType::I32, {32, keys});
        Tensor v_codes(device_v_codes.data(), DType::I32, {32, keys});
        Tensor k_scales(device_k_scales.data(), DType::FP32, {1, keys});
        Tensor v_scales(device_v_scales.data(), DType::FP32, {1, keys});
        ops::inkling_attention_dense(q_tensor, k_tensor, v_tensor, r_tensor, bias_tensor,
                                     sink_tensor, 0.125F, 0, expected, nullptr);
        targets::inkling2_mini::detail::append_kv_int8(k_tensor.view({128, keys}),
            v_tensor.view({128, keys}), k_codes, k_scales, v_codes, v_scales, 0, nullptr);
        targets::inkling2_mini::detail::inkling_attention_int8(q_tensor, k_codes, k_scales,
            v_codes, v_scales, r_tensor, bias_tensor, sink_tensor, 0.125F, actual, nullptr);
        cuda_synchronize();
        failures += verify_pointwise("Inkling INT8 KV attention",
            from_device_bf16(device_actual.data(), actual.numel()),
            from_device_bf16(device_expected.data(), expected.numel()),
            {/*absolute*/ 0.03, /*relative*/ 0.06});
        failures += device_k_codes.verify_guards("Inkling INT8 K codes");
        failures += device_v_codes.verify_guards("Inkling INT8 V codes");
        failures += device_actual.verify_guards("Inkling INT8 attention output");
    }
    std::cout << (failures ? "FAIL" : "OK") << " inkling2_mini runtime_kernels\n";
    return failures ? 1 : 0;
}
