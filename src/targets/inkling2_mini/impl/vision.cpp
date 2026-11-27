#include "targets/inkling2_mini/impl/vision.h"

#include "artifact/materializer.h"
#include "ninfer/ops/host_vision.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {
namespace {

constexpr int H = 1024;
constexpr int HEADS = 16;
constexpr int HEAD_DIM = 64;
constexpr int PATCH = 20;
constexpr int MERGE = 2;
constexpr int PATCH_VALUES = PATCH * PATCH * 3;
constexpr float kEpsilon = 1.0e-5F;

struct VisionLayerWeights {
    ops::HostLinearWeight q, k, v, output, fc1, fc2;
    std::vector<float> q_bias, k_bias, v_bias, output_bias, fc1_bias, fc2_bias;
    std::vector<float> norm1_weight, norm1_bias, norm2_weight, norm2_bias, scale1, scale2;
};

std::uint16_t read_u16(const std::byte* data) {
    std::uint16_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

void rope_2d(std::vector<float>& qkv, std::span<const std::int32_t> positions, int tokens) {
    for (int token = 0; token < tokens; ++token) {
        for (int plane = 0; plane < 2; ++plane) {
            for (int head = 0; head < HEADS; ++head) {
                for (int pair = 0; pair < HEAD_DIM / 4; ++pair) {
                    const float frequency = std::pow(10000.0F, -static_cast<float>(pair) / (HEAD_DIM / 4));
                    for (int half = 0; half < 2; ++half) {
                        const float angle = positions[static_cast<std::size_t>(half) * tokens + token] * frequency;
                        const float cosine = std::cos(angle), sine = std::sin(angle);
                        const std::size_t offset = static_cast<std::size_t>(token) * 3 * H + plane * H +
                            head * HEAD_DIM + half * (HEAD_DIM / 2) + pair * 2;
                        const float first = qkv[offset], second = qkv[offset + 1];
                        qkv[offset] = ops::host_round_bf16(first * cosine - second * sine);
                        qkv[offset + 1] = ops::host_round_bf16(second * cosine + first * sine);
                    }
                }
            }
        }
    }
}

} // namespace

struct InklingVision::Impl {
    ops::HostLinearWeight patch, proj_in, proj_out;
    std::vector<float> class_embedding, cls_position, position, pre_norm_weight, pre_norm_bias;
    VisionLayerWeights layers[23];

    Impl(artifact::MaterializedArtifact& artifact, const VisionBindingPlan& plan) {
        const auto matrix = [&](artifact::ObjectHandle handle, int rows, int columns) {
            const auto bytes = artifact.take_host_bytes(handle);
            if (bytes.size() != static_cast<std::size_t>(rows) * columns * 2) {
                throw std::invalid_argument("Inkling vision matrix payload size mismatch");
            }
            ops::HostLinearWeight result{rows, columns, std::vector<float>(static_cast<std::size_t>(rows) * columns)};
            for (std::size_t i = 0; i < result.values.size(); ++i) {
                result.values[i] = ops::host_bf16_to_float(read_u16(bytes.data() + i * 2));
            }
            return result;
        };
        const auto vector = [&](artifact::ObjectHandle handle, std::size_t count) {
            const auto bytes = artifact.take_host_bytes(handle);
            if (bytes.size() != count * 2) {
                throw std::invalid_argument("Inkling vision vector payload size mismatch");
            }
            std::vector<float> result(count);
            for (std::size_t i = 0; i < count; ++i) {
                result[i] = ops::host_bf16_to_float(read_u16(bytes.data() + i * 2));
            }
            return result;
        };
        patch = matrix(plan.patch_embedding, H, PATCH_VALUES);
        proj_in = matrix(plan.proj_in, 4096, 4096);
        proj_out = matrix(plan.proj_out, 2304, 4096);
        class_embedding = vector(plan.class_embedding, H);
        cls_position = vector(plan.cls_position_embedding, H);
        position = vector(plan.position_embedding, H * H);
        pre_norm_weight = vector(plan.ln_pre_weight, H);
        pre_norm_bias = vector(plan.ln_pre_bias, H);
        for (int index = 0; index < 23; ++index) {
            const VisionLayerPlan& source = plan.layers[index];
            VisionLayerWeights& out = layers[index];
            out.q = matrix(source.q_weight, H, H);
            out.k = matrix(source.k_weight, H, H);
            out.v = matrix(source.v_weight, H, H);
            out.output = matrix(source.out_weight, H, H);
            out.fc1 = matrix(source.c_fc_weight, 4096, H);
            out.fc2 = matrix(source.c_proj_weight, H, 4096);
            out.q_bias = vector(source.q_bias, H);
            out.k_bias = vector(source.k_bias, H);
            out.v_bias = vector(source.v_bias, H);
            out.output_bias = vector(source.out_bias, H);
            out.fc1_bias = vector(source.c_fc_bias, 4096);
            out.fc2_bias = vector(source.c_proj_bias, H);
            out.norm1_weight = vector(source.ln1_weight, H);
            out.norm1_bias = vector(source.ln1_bias, H);
            out.norm2_weight = vector(source.ln2_weight, H);
            out.norm2_bias = vector(source.ln2_bias, H);
            out.scale1 = vector(source.ls1, H);
            out.scale2 = vector(source.ls2, H);
        }
    }

    InklingVisionResult encode(const media::decode::Image& image) const {
        if (image.width <= 0 || image.height <= 0 ||
            image.rgb.size() != static_cast<std::size_t>(image.width) * image.height * 3) {
            throw std::invalid_argument("Inkling vision requires a nonempty RGB image");
        }
        const int width = (image.width + PATCH * MERGE - 1) / (PATCH * MERGE) * (PATCH * MERGE);
        const int height = (image.height + PATCH * MERGE - 1) / (PATCH * MERGE) * (PATCH * MERGE);
        const int grid_w = width / PATCH;
        const int grid_h = height / PATCH;
        const int patches = grid_w * grid_h;
        std::vector<float> patch_values(static_cast<std::size_t>(patches) * PATCH_VALUES);
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                const int patch_index = gy * grid_w + gx;
                for (int channel = 0; channel < 3; ++channel) {
                    for (int y = 0; y < PATCH; ++y) {
                        for (int x = 0; x < PATCH; ++x) {
                            const int source_x = gx * PATCH + x;
                            const int source_y = gy * PATCH + y;
                            float value = -1.0F;
                            if (source_x < image.width && source_y < image.height) {
                                value = image.rgb[(source_y * image.width + source_x) * 3 + channel];
                            }
                            const std::size_t offset = static_cast<std::size_t>(patch_index) * PATCH_VALUES +
                                y * PATCH * 3 + x * 3 + channel;
                            patch_values[offset] = value / 127.5F - 1.0F;
                        }
                    }
                }
            }
        }
        auto x = ops::host_linear(patch_values, patch);
        patch_values.clear();
        patch_values.shrink_to_fit();

        std::vector<std::int32_t> indices(static_cast<std::size_t>(patches) * 4);
        std::vector<float> weights(indices.size());
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                const int patch = gy * grid_w + gx;
                const float source_x = std::clamp((gx + 0.5F) * 32.0F / grid_w - 0.5F, 0.0F, 31.0F);
                const float source_y = std::clamp((gy + 0.5F) * 32.0F / grid_h - 0.5F, 0.0F, 31.0F);
                const int x0 = static_cast<int>(std::floor(source_x));
                const int y0 = static_cast<int>(std::floor(source_y));
                const int x1 = std::min(x0 + 1, 31);
                const int y1 = std::min(y0 + 1, 31);
                const float fx = source_x - x0;
                const float fy = source_y - y0;
                const int corner = patch * 4;
                indices[corner] = y0 * 32 + x0;
                indices[corner + 1] = y0 * 32 + x1;
                indices[corner + 2] = y1 * 32 + x0;
                indices[corner + 3] = y1 * 32 + x1;
                weights[corner] = (1 - fy) * (1 - fx);
                weights[corner + 1] = (1 - fy) * fx;
                weights[corner + 2] = fy * (1 - fx);
                weights[corner + 3] = fy * fx;
            }
        }
        ops::host_vision_pos_embed(x, position, indices, weights, H);
        std::vector<float> tokens(static_cast<std::size_t>(patches + 1) * H);
        for (int h = 0; h < H; ++h) {
            tokens[h] = ops::host_round_bf16(class_embedding[h] + cls_position[h]);
        }
        std::copy(x.begin(), x.end(), tokens.begin() + H);
        x = ops::host_layer_norm(tokens, pre_norm_weight, pre_norm_bias, kEpsilon);
        tokens.clear();
        tokens.shrink_to_fit();

        const int positions_count = patches + 1;
        std::vector<std::int32_t> rope_positions(static_cast<std::size_t>(positions_count) * 2, 0);
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                const int token = 1 + gy * grid_w + gx;
                rope_positions[token] = gx + 1;
                rope_positions[positions_count + token] = gy + 1;
            }
        }
        const std::vector<std::int32_t> segments{0, positions_count};
        for (const VisionLayerWeights& layer : layers) {
            auto normalized = ops::host_layer_norm(x, layer.norm1_weight, layer.norm1_bias, kEpsilon);
            auto q = ops::host_linear(normalized, layer.q);
            auto k = ops::host_linear(normalized, layer.k);
            auto v = ops::host_linear(normalized, layer.v);
            ops::host_add_bias(q, layer.q_bias);
            ops::host_add_bias(k, layer.k_bias);
            ops::host_add_bias(v, layer.v_bias);
            std::vector<float> qkv(static_cast<std::size_t>(positions_count) * 3 * H);
            for (int token = 0; token < positions_count; ++token) {
                std::copy_n(q.data() + static_cast<std::size_t>(token) * H, H,
                            qkv.data() + static_cast<std::size_t>(token) * 3 * H);
                std::copy_n(k.data() + static_cast<std::size_t>(token) * H, H,
                            qkv.data() + static_cast<std::size_t>(token) * 3 * H + H);
                std::copy_n(v.data() + static_cast<std::size_t>(token) * H, H,
                            qkv.data() + static_cast<std::size_t>(token) * 3 * H + 2 * H);
            }
            rope_2d(qkv, rope_positions, positions_count);
            auto attended = ops::host_vision_attention(qkv, segments, HEADS, HEAD_DIM);
            auto projected = ops::host_linear(attended, layer.output);
            ops::host_add_bias(projected, layer.output_bias);
            for (std::size_t i = 0; i < projected.size(); ++i) {
                projected[i] = ops::host_round_bf16(projected[i] * layer.scale1[i % H]);
            }
            ops::host_residual_add(x, projected);
            normalized = ops::host_layer_norm(x, layer.norm2_weight, layer.norm2_bias, kEpsilon);
            auto up = ops::host_linear(normalized, layer.fc1);
            ops::host_add_bias(up, layer.fc1_bias);
            ops::host_gelu(up, true);
            auto down = ops::host_linear(up, layer.fc2);
            ops::host_add_bias(down, layer.fc2_bias);
            for (std::size_t i = 0; i < down.size(); ++i) {
                down[i] = ops::host_round_bf16(down[i] * layer.scale2[i % H]);
            }
            ops::host_residual_add(x, down);
        }

        const int merged_w = grid_w / MERGE;
        const int merged_h = grid_h / MERGE;
        const int merged_count = merged_w * merged_h;
        std::vector<float> merged(static_cast<std::size_t>(merged_count) * 4 * H);
        for (int my = 0; my < merged_h; ++my) {
            for (int mx = 0; mx < merged_w; ++mx) {
                for (int dy = 0; dy < MERGE; ++dy) {
                    for (int dx = 0; dx < MERGE; ++dx) {
                        const int source_token = 1 + (my * MERGE + dy) * grid_w + mx * MERGE + dx;
                        const int feature = dx + MERGE * dy;
                        const std::size_t destination = (static_cast<std::size_t>(my) * merged_w + mx) * 4 * H + feature * H;
                        std::copy_n(x.data() + static_cast<std::size_t>(source_token) * H, H,
                                    merged.data() + destination);
                    }
                }
            }
        }
        auto hidden = ops::host_linear(merged, proj_in);
        ops::host_gelu(hidden, true);
        auto output = ops::host_linear(hidden, proj_out);
        InklingVisionResult result;
        result.patch_width = grid_w;
        result.patch_height = grid_h;
        result.merged_count = merged_count;
        result.embeddings.resize(output.size());
        for (std::size_t i = 0; i < output.size(); ++i) {
            result.embeddings[i] = ops::host_float_to_bf16(output[i]);
        }
        return result;
    }
};

InklingVision::InklingVision(artifact::MaterializedArtifact& artifact, const VisionBindingPlan& plan)
    : impl_(std::make_unique<Impl>(artifact, plan)) {}
InklingVision::~InklingVision() = default;
InklingVisionResult InklingVision::encode(const media::decode::Image& image) const {
    return impl_->encode(image);
}

} // namespace ninfer::targets::inkling2_mini::detail
