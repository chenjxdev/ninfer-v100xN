#include "targets/inkling2_mini/impl/draft.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "core/arena.h"
#include "core/layout.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/inkling_pointwise.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"
#include "targets/inkling2_mini/impl/draft_kernels.h"
#include "targets/inkling2_mini/impl/runtime_kernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {
namespace {

constexpr int H = 2304;
constexpr int R = 256;
constexpr int QH = 3072;
constexpr int KVH = 1024;
constexpr int MLP = 9216;
constexpr int V = 201024;
constexpr int VALID_V = 200058;
constexpr int BLOCK = 8;
constexpr std::size_t WORKSPACE_BYTES = 256ULL * 1024ULL * 1024ULL;

struct DraftLayer {
    Weight q, k, v, o, context_k, context_v, gate, up, down;
    Tensor input_norm, post_norm, q_norm, k_norm, context_k_norm, feature_norm;
    Tensor k_cache, v_cache;
    bool local = false;
};

std::array<std::int32_t, 4> shape_of(const Tensor& tensor) {
    return {tensor.ne[0], tensor.ne[1], tensor.ne[2], tensor.ne[3]};
}

float bf16_host(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float round_bf16_host(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return bf16_host(static_cast<std::uint16_t>(bits >> 16));
}

} // namespace

struct InklingDraft::Impl {
    artifact::Reader reader;
    artifact::MaterializedArtifact backing;
    Weight target_embedding, target_lm_head;
    Tensor target_embedding_norm;
    DeviceContext& device;
    cudaStream_t stream;
    int max_context;
    int context_length = 0;
    std::array<DraftLayer, 3> layers;
    Tensor final_norm, mask, markov_norm;
    Weight markov_embedding, rnn_projection, hidden_out;
    Tensor rnn_bias, hidden_bias;
    Tensor fuse_alpha, f_rnn, markov_state;
    Weight folded_weight;
    Tensor folded_bias;
    std::unique_ptr<DeviceArena> persistent, workspace;
    std::unique_ptr<DeviceBuffer> alpha_storage, folded_weight_storage, folded_bias_storage;

    Impl(const std::string& path, const Weight& embedding, const Tensor& embedding_norm,
         const Weight& lm_head, int context, DeviceContext& device_context)
        : reader(path), target_embedding(embedding), target_embedding_norm(embedding_norm),
          target_lm_head(lm_head), device(device_context), stream(device_context.stream),
          max_context(context) {
        if (max_context < BLOCK + 1) {
            throw std::invalid_argument("Inkling MTP requires max_context >= 9");
        }
        CUDA_CHECK(cudaSetDevice(device.device));
        load_weights();
        allocate_state();
        prepare_fold();
    }

    void load_weights() {
        artifact::Binder binder(reader, 1);
        std::unordered_map<std::string, artifact::ObjectHandle> handles;
        const auto bind = [&](const std::string& name, std::initializer_list<std::uint64_t> shape) {
            const artifact::ObjectHandle handle = binder.require_tensor(name,
                artifact::NumericFormat::BF16, artifact::StorageLayout::ContiguousLeV1,
                std::span<const std::uint64_t>(shape.begin(), shape.size()));
            binder.materialize_on_device(handle);
            handles.emplace(name, handle);
        };
        const auto matrix = [&](const std::string& name, int n, int k) {
            bind(name, {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)});
        };
        const auto vector = [&](const std::string& name, int n) {
            bind(name, {static_cast<std::uint64_t>(n)});
        };
        bind("fuse_weight", {4, 4});
        vector("mask_embed", H);
        vector("norm.weight", H);
        vector("markov_head.dflare_norm.weight", H);
        matrix("markov_head.markov_w1.weight", V, R);
        matrix("markov_head.rnn_proj.weight", 3 * R, 2 * R + 2 * H);
        vector("markov_head.rnn_proj.bias", 3 * R);
        matrix("markov_head.hidden_out.weight", H, R);
        vector("markov_head.hidden_out.bias", H);
        for (int index = 0; index < 3; ++index) {
            const std::string prefix = "layers." + std::to_string(index) + ".";
            vector(prefix + "dflare_norm.weight", H);
            vector(prefix + "input_layernorm.weight", H);
            vector(prefix + "post_attention_layernorm.weight", H);
            matrix(prefix + "mlp.gate_proj.weight", MLP, H);
            matrix(prefix + "mlp.up_proj.weight", MLP, H);
            matrix(prefix + "mlp.down_proj.weight", H, MLP);
            vector(prefix + "self_attn.q_norm.weight", 128);
            vector(prefix + "self_attn.k_norm.weight", 128);
            vector(prefix + "self_attn.k_norm_ctx.weight", 128);
            matrix(prefix + "self_attn.q_proj.weight", QH, H);
            matrix(prefix + "self_attn.k_proj.weight", KVH, H);
            matrix(prefix + "self_attn.v_proj.weight", KVH, H);
            matrix(prefix + "self_attn.o_proj.weight", H, QH);
            matrix(prefix + "self_attn.k_proj_ctx.weight", KVH, H);
            matrix(prefix + "self_attn.v_proj_ctx.weight", KVH, H);
        }
        binder.validate_unconsumed_matching();
        backing = artifact::materialize(reader, binder.finish(), device);

        const auto W = [&](const std::string& name, int n, int k) {
            return artifact::materialized_weight(backing, handles.at(name),
                artifact::NumericFormat::BF16, n, k, 0);
        };
        const auto T = [&](const std::string& name, std::initializer_list<std::int32_t> shape) {
            return artifact::materialized_tensor(backing, handles.at(name),
                artifact::NumericFormat::BF16, shape, 0);
        };
        final_norm = T("norm.weight", {H});
        mask = T("mask_embed", {H});
        markov_norm = T("markov_head.dflare_norm.weight", {H});
        markov_embedding = W("markov_head.markov_w1.weight", V, R);
        rnn_projection = W("markov_head.rnn_proj.weight", 3 * R, 2 * R + 2 * H);
        rnn_bias = T("markov_head.rnn_proj.bias", {3 * R});
        hidden_out = W("markov_head.hidden_out.weight", H, R);
        hidden_bias = T("markov_head.hidden_out.bias", {H});
        for (int index = 0; index < 3; ++index) {
            const std::string prefix = "layers." + std::to_string(index) + ".";
            DraftLayer& layer = layers[index];
            layer.local = index != 1;
            layer.input_norm = T(prefix + "input_layernorm.weight", {H});
            layer.post_norm = T(prefix + "post_attention_layernorm.weight", {H});
            layer.feature_norm = T(prefix + "dflare_norm.weight", {H});
            layer.gate = W(prefix + "mlp.gate_proj.weight", MLP, H);
            layer.up = W(prefix + "mlp.up_proj.weight", MLP, H);
            layer.down = W(prefix + "mlp.down_proj.weight", H, MLP);
            layer.q_norm = T(prefix + "self_attn.q_norm.weight", {128});
            layer.k_norm = T(prefix + "self_attn.k_norm.weight", {128});
            layer.context_k_norm = T(prefix + "self_attn.k_norm_ctx.weight", {128});
            layer.q = W(prefix + "self_attn.q_proj.weight", QH, H);
            layer.k = W(prefix + "self_attn.k_proj.weight", KVH, H);
            layer.v = W(prefix + "self_attn.v_proj.weight", KVH, H);
            layer.o = W(prefix + "self_attn.o_proj.weight", H, QH);
            layer.context_k = W(prefix + "self_attn.k_proj_ctx.weight", KVH, H);
            layer.context_v = W(prefix + "self_attn.v_proj_ctx.weight", KVH, H);
        }

        Tensor fuse = T("fuse_weight", {4, 4});
        std::array<std::uint16_t, 16> fuse_host{};
        CUDA_CHECK(cudaMemcpyAsync(fuse_host.data(), fuse.data, sizeof(fuse_host),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::array<float, 16> alpha{};
        for (int row = 0; row < 4; ++row) {
            float maximum = -INFINITY;
            for (int tap = 0; tap < 4; ++tap) {
                maximum = std::max(maximum, bf16_host(fuse_host[row * 4 + tap]));
            }
            float denominator = 0.0F;
            for (int tap = 0; tap < 4; ++tap) {
                const float value = std::exp(bf16_host(fuse_host[row * 4 + tap]) - maximum);
                alpha[row * 4 + tap] = value;
                denominator += value;
            }
            for (int tap = 0; tap < 4; ++tap) {
                alpha[row * 4 + tap] = round_bf16_host(alpha[row * 4 + tap] / denominator);
            }
        }
        alpha_storage = std::make_unique<DeviceBuffer>(sizeof(alpha));
        alpha_storage->copy_from_host(alpha.data(), sizeof(alpha));
        fuse_alpha = Tensor(alpha_storage->p, DType::FP32, {4, 4});
    }

    void allocate_state() {
        WorkspaceLayoutBuilder layout;
        for (int index = 0; index < 3; ++index) {
            const int cache_tokens = layers[index].local ? 1024 : max_context;
            (void)layout.alloc(DType::BF16, {KVH, cache_tokens});
            (void)layout.alloc(DType::BF16, {KVH, cache_tokens});
        }
        (void)layout.alloc(DType::BF16, {H});
        (void)layout.alloc(DType::BF16, {R});
        persistent = std::make_unique<DeviceArena>(layout.peak_bytes(1));
        workspace = std::make_unique<DeviceArena>(WORKSPACE_BYTES);
        for (DraftLayer& layer : layers) {
            const int cache_tokens = layer.local ? 1024 : max_context;
            layer.k_cache = persistent->alloc(DType::BF16, {KVH, cache_tokens});
            layer.v_cache = persistent->alloc(DType::BF16, {KVH, cache_tokens});
        }
        f_rnn = persistent->alloc(DType::BF16, {H});
        markov_state = persistent->alloc(DType::BF16, {R});
    }

    void prepare_fold() {
        folded_weight_storage = std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(V) * R * sizeof(std::uint16_t));
        folded_bias_storage = std::make_unique<DeviceBuffer>(V * sizeof(std::uint16_t));
        folded_weight = Weight{};
        folded_weight.payload = folded_weight_storage->p;
        folded_weight.qdata = folded_weight_storage->p;
        folded_weight.payload_bytes = folded_weight_storage->bytes;
        folded_weight.qtype = QType::BF16_CTRL;
        folded_weight.ndim = 2;
        folded_weight.n = folded_weight.shape[0] = folded_weight.padded_shape[0] = V;
        folded_weight.k = folded_weight.shape[1] = folded_weight.padded_shape[1] = R;
        folded_weight.layout = QuantLayout::Contiguous;
        folded_weight.weight_scale_divisor = 1.0F;
        folded_weight.input_scale_divisor = 1.0F;
        folded_bias = Tensor(folded_bias_storage->p, DType::BF16, {V});
        draft_fold_lm_head(target_lm_head, hidden_out, hidden_bias, folded_weight, folded_bias, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape) {
        return workspace->alloc(dtype, shape);
    }
    Tensor alloc_like(const Tensor& source) {
        const auto shape = shape_of(source);
        return alloc(source.dtype, {shape[0], shape[1], shape[2], shape[3]});
    }
    Tensor linear(const Tensor& x, const Weight& weight, int rows) {
        Tensor out = alloc(DType::BF16, {rows, x.ne[1]});
        if (x.ne[1] <= 8) {
            bf16_gemv(x, weight, out, stream);
        } else {
            ops::linear(x, weight, out, stream);
        }
        return out;
    }
    Tensor rms(const Tensor& x, const Tensor& weight) {
        Tensor out = alloc_like(x);
        ops::rmsnorm(x, weight, 1.0e-6F, false, out, stream);
        return out;
    }
    Tensor alpha_row(int row) const {
        auto* pointer = static_cast<std::byte*>(fuse_alpha.data) + row * 4 * sizeof(float);
        return Tensor(pointer, DType::FP32, {4});
    }

    void commit(const std::array<Tensor, 4>& features, int count) {
        CUDA_CHECK(cudaSetDevice(device.device));
        if (count <= 0 || context_length + count > max_context) {
            throw std::invalid_argument("Inkling MTP feature commit exceeds max_context");
        }
        for (const Tensor& feature : features) {
            if (feature.dtype != DType::BF16 || feature.ne[0] != H ||
                feature.ne[1] != count || !feature.is_contiguous()) {
                throw std::invalid_argument("Inkling MTP expects four contiguous [H,count] BF16 features");
            }
        }
        int offset = 0;
        while (offset < count) {
            workspace->reset();
            const int chunk = std::min(256, count - offset);
            Tensor positions = alloc(DType::I32, {chunk});
            fill_positions(positions, context_length, stream);
            for (int index = 0; index < 3; ++index) {
                DraftLayer& layer = layers[index];
                Tensor mixed = alloc(DType::BF16, {H, chunk});
                draft_mix_rmsnorm(features, offset, chunk, alpha_row(index),
                                  layer.feature_norm, mixed, stream);
                Tensor key = linear(mixed, layer.context_k, KVH);
                Tensor value = linear(mixed, layer.context_v, KVH);
                Tensor key_heads = key.view({128, 8, chunk});
                Tensor normed_key = rms(key_heads, layer.context_k_norm);
                draft_rope_context_k(normed_key, positions, stream);
                if (layer.local) {
                    append_kv_ring(normed_key.view({KVH, chunk}), value, layer.k_cache, layer.v_cache,
                                   context_length, layer.k_cache.ne[1], stream);
                } else {
                    append_kv(normed_key.view({KVH, chunk}), value, layer.k_cache, layer.v_cache,
                              context_length, stream);
                }
            }
            context_length += chunk;
            offset += chunk;
        }
        workspace->reset();
        draft_mix_rmsnorm(features, count - 1, 1, alpha_row(3), markov_norm, f_rnn, stream);
    }

    std::array<int, 7> propose(int anchor, int draft_count) {
        if (draft_count <= 0 || draft_count > 7 || context_length <= 0 ||
            context_length + BLOCK > max_context) {
            throw std::invalid_argument("Inkling MTP proposal state is inconsistent");
        }
        CUDA_CHECK(cudaSetDevice(device.device));
        workspace->reset();
        Tensor anchor_id = alloc(DType::I32, {1});
        CUDA_CHECK(cudaMemcpyAsync(anchor_id.data, &anchor, sizeof(anchor), cudaMemcpyHostToDevice, stream));
        Tensor anchor_embedding = alloc(DType::BF16, {H, 1});
        ops::embedding(anchor_id, target_embedding, anchor_embedding, stream);
        Tensor normed_anchor = rms(anchor_embedding, target_embedding_norm);
        Tensor hidden = alloc(DType::BF16, {H, BLOCK});
        draft_prepare_embedding(normed_anchor, mask, hidden, stream);
        Tensor positions = alloc(DType::I32, {BLOCK});
        fill_positions(positions, context_length, stream);
        Tensor scores = alloc(DType::FP32, {context_length + BLOCK, 24, BLOCK});

        for (DraftLayer& layer : layers) {
            Tensor normed = rms(hidden, layer.input_norm);
            Tensor query = linear(normed, layer.q, QH).view({128, 24, BLOCK});
            Tensor key = linear(normed, layer.k, KVH).view({128, 8, BLOCK});
            Tensor value = linear(normed, layer.v, KVH).view({128, 8, BLOCK});
            query = rms(query, layer.q_norm);
            key = rms(key, layer.k_norm);
            draft_rope(query, key, positions, stream);
            Tensor attention = alloc(DType::BF16, {128, 24, BLOCK});
            Tensor context_k = layer.k_cache;
            Tensor context_v = layer.v_cache;
            int visible_context = context_length;
            if (layer.local) {
                visible_context = std::min(context_length, 1024);
                context_k = alloc(DType::BF16, {KVH, visible_context});
                context_v = alloc(DType::BF16, {KVH, visible_context});
                gather_kv_ring(layer.k_cache, layer.v_cache, context_length - visible_context,
                               visible_context, layer.k_cache.ne[1], context_k, context_v, stream);
            }
            draft_attention(query, key, value, context_k, context_v,
                            visible_context, layer.local, scores, attention, stream);
            Tensor projected = linear(attention.view({QH, BLOCK}), layer.o, H);
            ops::residual_add(hidden, projected, stream);
            hidden = projected;

            normed = rms(hidden, layer.post_norm);
            Tensor gate = linear(normed, layer.gate, MLP);
            Tensor up = linear(normed, layer.up, MLP);
            Tensor activated = alloc_like(gate);
            ops::silu_mul(gate, up, activated, stream);
            projected = linear(activated, layer.down, H);
            ops::residual_add(hidden, projected, stream);
            hidden = projected;
        }
        hidden = rms(hidden, final_norm);
        Tensor base_logits = linear(hidden.slice(1, 1, 7), target_lm_head, V);
        Tensor draft_tokens = alloc(DType::I32, {7});
        CUDA_CHECK(cudaMemsetAsync(markov_state.data, 0, markov_state.bytes(), stream));
        for (int step = 0; step < draft_count; ++step) {
            Tensor previous = step == 0 ? anchor_id : draft_tokens.slice(0, step - 1, 1);
            Tensor inputs = alloc(DType::BF16, {2 * R + 2 * H, 1});
            draft_rnn_inputs(markov_state, previous, markov_embedding,
                             hidden.slice(1, step + 1, 1), f_rnn, inputs, stream);
            Tensor projection = linear(inputs, rnn_projection, 3 * R);
            ops::residual_add(rnn_bias, projection, stream);
            Tensor latent = alloc(DType::BF16, {R, 1});
            draft_rnn_gates(projection, markov_state, latent, stream);
            Tensor bias = linear(latent, folded_weight, V);
            ops::residual_add(folded_bias, bias, stream);
            ops::residual_add(base_logits.slice(1, step, 1), bias, stream);
            Tensor capped = alloc(DType::BF16, {V, 1});
            ops::inkling_softcap(bias, 30.0F, capped, stream);
            Tensor next = draft_tokens.slice(0, step, 1);
            ops::argmax(capped, next, VALID_V, stream);
        }
        std::array<int, 7> result{};
        CUDA_CHECK(cudaMemcpyAsync(result.data(), draft_tokens.data, sizeof(result),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }
};

InklingDraft::InklingDraft(const std::string& path, const Weight& target_embedding,
    const Tensor& target_embedding_norm, const Weight& target_lm_head,
    int max_context, DeviceContext& device)
    : impl_(std::make_unique<Impl>(path, target_embedding, target_embedding_norm,
                                   target_lm_head, max_context, device)) {}
InklingDraft::~InklingDraft() = default;
void InklingDraft::reset() { impl_->context_length = 0; }
void InklingDraft::commit(const std::array<Tensor, 4>& target_features, int count) {
    impl_->commit(target_features, count);
}
std::array<int, 7> InklingDraft::propose(int anchor, int draft_count) {
    return impl_->propose(anchor, draft_count);
}

} // namespace ninfer::targets::inkling2_mini::detail
