#include "targets/inkling2_mini/impl/runtime_tp2.h"

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/inkling_attention_dense.h"
#include "ninfer/ops/inkling_conv1d.h"
#include "ninfer/ops/inkling_depth_mix.h"
#include "ninfer/ops/inkling_loop_control.h"
#include "ninfer/ops/inkling_moe.h"
#include "ninfer/ops/inkling_pointwise.h"
#include "ninfer/ops/inkling_router.h"
#include "ninfer/ops/l2_clip.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ops/launcher/inkling_moe.h"
#include "targets/inkling2_mini/impl/load/bindings.h"
#include "targets/inkling2_mini/impl/draft.h"
#include "targets/inkling2_mini/impl/runtime_kernels.h"
#include "targets/inkling2_mini/impl/vision.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {
namespace {

constexpr std::int32_t H = 2304;
constexpr std::int32_t QH = 1536;
constexpr std::int32_t KVH = 512;
constexpr std::int32_t RH = 192;
constexpr std::int32_t I = 576;
constexpr std::int32_t E = 128;
constexpr std::int32_t K = 8;
constexpr std::int32_t V = 201024;
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kAttentionScale = 0.08838834764831845F;
constexpr std::uint32_t kLogScaleFloor = 8192;
constexpr float kLogScaleAlpha = 0.1F;
constexpr std::size_t kWorkspaceBytes = 1024ULL * 1024ULL * 1024ULL;

struct LoopWeights {
    Tensor down;
    Tensor gain;
    Tensor forget;
};

struct LayerWeights {
    Weight q, k, v, o, rel, gate, sink;
    Tensor q_clip, rel_clip, rel_proj, sconv, norm;
    Tensor q_input_norm, k_input_norm, v_input_norm;
    Tensor k_sconv, v_sconv, k_sconv_static, v_sconv_static;
    Weight k_sconv_proj, v_sconv_proj;
    Tensor attention_query, mlp_query, moe_norm, moe_sconv, router, router_bias;
    std::vector<Weight> experts13, experts2;
    std::vector<Tensor> experts13_mxscale, experts2_mxscale;
    std::unique_ptr<DeviceBuffer> expert_view_storage;
    DeviceSpan expert_views;
    std::array<LoopWeights, 4> loop;
    bool full = false;
    bool core = false;
};

struct LayerState {
    Tensor k_cache, v_cache, k_conv, v_conv, attention_conv, ffn_conv;
    Tensor k_cache_i8, v_cache_i8, k_scales_i8, v_scales_i8;
    Tensor k_history, v_history, attention_history, ffn_history;
};

using TensorPair = std::array<Tensor, 2>;
using BlockPair = std::array<std::vector<Tensor>, 2>;

std::array<std::int32_t, 4> shape_of(const Tensor& tensor) {
    return {tensor.ne[0], tensor.ne[1], tensor.ne[2], tensor.ne[3]};
}

void enable_peer(int from, int to) {
    CUDA_CHECK(cudaSetDevice(from));
    const cudaError_t status = cudaDeviceEnablePeerAccess(to, 0);
    if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) { CUDA_CHECK(status); }
    (void)cudaGetLastError();
}

} // namespace

struct RuntimeTP2::Impl {
    ExecutionContext execution;
    artifact::Reader reader;
    artifact::MaterializedArtifact backing;
    BindingPlan bindings;
    LayerWeights layers[2][24];
    LayerState states[2][48];
    Weight embedding[2], output_head;
    Tensor embedding_norm[2], final_norm, output_query;
    TensorPair cross_q, cross_kv;
    std::unique_ptr<DeviceArena> persistent[2], workspace[2];
    std::unique_ptr<DeviceBuffer> ones_storage[2];
    Tensor ones[2];
    cudaEvent_t allreduce_events[2]{};
    std::string tokenizer_json, tokenizer_config_json, generation_config_json;
    std::unique_ptr<qwen3_6::frontend_internal::Tokenizer> tokenizer;
    std::unique_ptr<InklingVision> vision;
    std::unique_ptr<InklingDraft> draft;
    std::unique_ptr<DeviceBuffer> feature_storage;
    Tensor target_features[4];
    bool record_conv_history = false;
    bool quantize_full_kv = false;
    bool profile_layers = false;
    int profile_layer_count = 0;
    std::vector<std::array<cudaEvent_t, 6>> profile_events;
    int verify_rounds = 0;
    int proposed_drafts = 0;
    int accepted_drafts = 0;
    int native_fallback_tokens = 0;
    std::vector<std::uint16_t> visual_embeddings;
    int visual_begin = 0;
    int visual_count = 0;
    int max_context = 0;
    int length = 0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    double draft_seconds = 0.0;
    double verify_seconds = 0.0;

    Impl(const std::string& path, int context, bool enable_vision)
        : execution(std::vector<int>{0, 1}), reader(path), max_context(context) {
        if (max_context <= 0 || max_context > 262144) {
            throw std::invalid_argument("Inkling TP2 max_context must be in [1,262144]");
        }
        quantize_full_kv = max_context > 131072 || std::getenv("NINFER_INKLING_I8_KV") != nullptr;
        enable_peer(0, 1);
        enable_peer(1, 0);
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaEventCreateWithFlags(&allreduce_events[rank], cudaEventDisableTiming));
        }
        artifact::Binder binder(reader, 2);
        auto plan = bind_artifact(binder, artifact::TensorPlacement::Device,
                                  InklingDeviceMode::TensorParallel, enable_vision);
        backing = artifact::materialize(reader, plan.materialization, execution);
        bindings = std::move(plan.bindings);
        const auto take_resource = [&](artifact::ObjectHandle handle) {
            const auto bytes = backing.take_host_bytes(handle);
            return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        };
        tokenizer_json = take_resource(bindings.frontend.tokenizer_json);
        tokenizer_config_json = take_resource(bindings.frontend.tokenizer_config_json);
        generation_config_json = take_resource(bindings.frontend.generation_config_json);
        load_views();
        if (enable_vision) { vision = std::make_unique<InklingVision>(backing, bindings.vision); }
        allocate_state();
        if (const char* mtp_path = std::getenv("NINFER_INKLING_MTP");
            mtp_path != nullptr && *mtp_path != '\0') {
            CUDA_CHECK(cudaSetDevice(0));
            const std::size_t feature_bytes = static_cast<std::size_t>(H) * 128 *
                                              sizeof(std::uint16_t);
            feature_storage = std::make_unique<DeviceBuffer>(4 * feature_bytes);
            for (int tap = 0; tap < 4; ++tap) {
                auto* pointer = static_cast<std::byte*>(feature_storage->p) + tap * feature_bytes;
                target_features[tap] = Tensor(pointer, DType::BF16, {H, 128});
            }
            draft = std::make_unique<InklingDraft>(mtp_path, embedding[0], embedding_norm[0],
                                                   output_head, max_context, *execution.dev[0]);
        }
        reset_state();
    }

    ~Impl() {
        for (cudaEvent_t event : allreduce_events) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
    }

    cudaStream_t stream(int rank) const { return execution.dev[rank]->stream; }

    Tensor tensor(artifact::ObjectHandle handle, artifact::NumericFormat format,
                  std::initializer_list<std::int32_t> shape, int rank) const {
        return artifact::materialized_tensor(backing, handle, format, shape, rank);
    }

    Weight weight(artifact::ObjectHandle handle, artifact::NumericFormat format, std::int32_t n,
                  std::int32_t k, int rank, float divisor = 1.0F) const {
        if (format != artifact::NumericFormat::NVFP4) {
            return artifact::materialized_weight(backing, handle, format, n, k, rank);
        }
        const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(n),
                                                static_cast<std::uint64_t>(k)};
        const auto geometry = artifact::block_scale_geometry(format, shape);
        artifact::require_placement_bytes(backing, handle, rank, geometry.encoded_bytes);
        const auto* bytes = static_cast<const std::byte*>(backing.device_data(handle, rank));
        Weight out{};
        out.payload = bytes;
        out.payload_bytes = geometry.encoded_bytes;
        out.qdata = bytes;
        out.scales = bytes + geometry.scale_plane_offset;
        out.qtype = QType::NVFP4;
        out.group = out.group_size = 16;
        out.ndim = 2;
        out.n = out.shape[0] = out.padded_shape[0] = n;
        out.k = out.shape[1] = out.padded_shape[1] = k;
        out.layout = QuantLayout::BlockScaleK16M128x4;
        out.scale_dtype = DType::FP8_E4M3FN;
        out.weight_scale_divisor = divisor;
        out.input_scale_divisor = 1.0F;
        return out;
    }

    void load_views() {
        for (int rank = 0; rank < 2; ++rank) {
            embedding[rank] = weight(bindings.token_embedding, artifact::NumericFormat::BF16, V, H, rank);
            embedding_norm[rank] = tensor(bindings.embedding_norm, artifact::NumericFormat::BF16, {H}, rank);
            cross_q[rank] = tensor(bindings.cross_pass_query, artifact::NumericFormat::FP32, {24, H}, rank);
            cross_kv[rank] = tensor(bindings.cross_pass_kv_query, artifact::NumericFormat::FP32, {24, H}, rank);
        }
        output_head = weight(bindings.output_head, artifact::NumericFormat::BF16, V, H, 0);
        final_norm = tensor(bindings.final_norm, artifact::NumericFormat::BF16, {H}, 0);
        output_query = tensor(bindings.output_query, artifact::NumericFormat::FP32, {H, 1}, 0);
        for (int rank = 0; rank < 2; ++rank) {
            for (int physical = 0; physical < 24; ++physical) {
                const TextLayerPlan& plan = bindings.text_layers[physical];
                LayerWeights out;
                out.full = plan.full_attention;
                out.core = plan.loop_core;
                const auto W = [&](artifact::ObjectHandle handle, std::int32_t n, std::int32_t k) {
                    return weight(handle, artifact::NumericFormat::BF16, n, k, rank);
                };
                const auto T = [&](artifact::ObjectHandle handle,
                                   std::initializer_list<std::int32_t> shape) {
                    return tensor(handle, artifact::NumericFormat::BF16, shape, rank);
                };
                out.q = W(plan.attention.q, QH, H);
                out.k = W(plan.attention.k, KVH, H);
                out.v = W(plan.attention.v, KVH, H);
                out.o = W(plan.attention.o, H, QH);
                out.rel = W(plan.attention.rel, RH, H);
                out.gate = W(plan.attention.gate, QH, H);
                out.sink = W(plan.attention.sink, 12, H);
                out.q_clip = T(plan.attention.q_clip, {128, 12});
                out.rel_clip = T(plan.attention.rel_clip, {16, 12});
                out.rel_proj = T(plan.attention.rel_proj, {16, 1024});
                out.sconv = T(plan.attention.sconv, {H, 4});
                if (out.full) {
                    out.q_input_norm = T(plan.attention.q_input_norm, {H});
                    out.k_input_norm = T(plan.attention.k_input_norm, {H});
                    out.v_input_norm = T(plan.attention.v_input_norm, {H});
                    out.k_sconv_static = T(plan.attention.k_sconv_static, {4, KVH});
                    out.v_sconv_static = T(plan.attention.v_sconv_static, {4, KVH});
                    out.k_sconv_proj = W(plan.attention.k_sconv_proj, 16, H);
                    out.v_sconv_proj = W(plan.attention.v_sconv_proj, 16, H);
                } else {
                    out.norm = T(plan.attention.norm, {H});
                    out.k_sconv = T(plan.attention.k_sconv, {KVH, 4});
                    out.v_sconv = T(plan.attention.v_sconv, {KVH, 4});
                }
                if (physical > 0) {
                    out.attention_query = tensor(bindings.attention_queries[physical - 1],
                        artifact::NumericFormat::FP32, {H, out.full ? 3 : 1}, rank);
                }
                out.mlp_query = tensor(bindings.mlp_queries[physical], artifact::NumericFormat::FP32,
                                       {H, 1}, rank);
                out.moe_norm = T(plan.moe.norm, {H});
                out.moe_sconv = T(plan.moe.sconv, {H, 4});
                out.router = T(plan.moe.gate, {E, H});
                out.router_bias = tensor(plan.moe.gate_bias, artifact::NumericFormat::FP32, {E}, rank);
                out.experts13.reserve(E);
                out.experts2.reserve(E);
                out.experts13_mxscale.reserve(E);
                out.experts2_mxscale.reserve(E);
                for (int expert = 0; expert < E; ++expert) {
                    const ExpertPlan& source = plan.moe.experts[expert];
                    const auto format = physical >= 20 ? artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S
                                                       : artifact::NumericFormat::NVFP4;
                    out.experts13.push_back(weight(source.w13, format, 2 * I, H, rank, source.w13_divisor));
                    out.experts2.push_back(weight(source.w2, format, H, I, rank, source.w2_divisor));
                    if (format == artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S) {
                        out.experts13_mxscale.push_back(T(source.w13_mxscale, {2 * I, I / 8}));
                        out.experts2_mxscale.push_back(T(source.w2_mxscale, {H, I / 32}));
                    }
                }
                std::vector<ops::detail::InklingMoeWeightView> host_views;
                const auto make_view = [](const Weight& source, const Tensor* group_scales) {
                    return ops::detail::InklingMoeWeightView{
                        .qdata = source.qdata, .scales = source.scales,
                        .group_scales = group_scales == nullptr ? nullptr : group_scales->data,
                        .n = source.n, .k = source.k,
                        .divisor = source.qtype == QType::NVFP4 ? source.weight_scale_divisor : 1.0F,
                        .nvfp4 = source.qtype == QType::NVFP4,
                        .mxfp8 = source.qtype == QType::FP8_E4M3FN_ROW_BF16S};
                };
                for (int expert = 0; expert < E; ++expert) {
                    const Tensor* scale = out.experts13_mxscale.empty() ? nullptr : &out.experts13_mxscale[expert];
                    host_views.push_back(make_view(out.experts13[expert], scale));
                }
                for (int expert = 0; expert < E; ++expert) {
                    const Tensor* scale = out.experts2_mxscale.empty() ? nullptr : &out.experts2_mxscale[expert];
                    host_views.push_back(make_view(out.experts2[expert], scale));
                }
                CUDA_CHECK(cudaSetDevice(rank));
                out.expert_view_storage = std::make_unique<DeviceBuffer>(
                    host_views.size() * sizeof(ops::detail::InklingMoeWeightView));
                out.expert_view_storage->copy_from_host(host_views.data(), out.expert_view_storage->bytes);
                out.expert_views = {out.expert_view_storage->p, out.expert_view_storage->bytes};
                if (out.core) {
                    for (int site = 0; site < 4; ++site) {
                        out.loop[site].down = T(plan.loop[site].down, {64, 2 * H});
                        out.loop[site].gain = T(plan.loop[site].gain, {H, 64});
                        out.loop[site].forget = T(plan.loop[site].forget, {H, 64});
                    }
                }
                layers[rank][physical] = std::move(out);
            }
        }
    }

    void allocate_state() {
        for (int rank = 0; rank < 2; ++rank) {
            WorkspaceLayoutBuilder layout;
            for (int slot = 0; slot < 48; ++slot) {
                const bool full = is_full_attention(physical_layer_for_slot(slot));
                const int cache_tokens = full ? max_context : static_cast<int>(kSlidingWindow) + 64;
                if (full && quantize_full_kv) {
                    (void)layout.alloc(DType::I32, {KVH / 4, cache_tokens});
                    (void)layout.alloc(DType::I32, {KVH / 4, cache_tokens});
                    (void)layout.alloc(DType::FP32, {KVH / 128, cache_tokens});
                    (void)layout.alloc(DType::FP32, {KVH / 128, cache_tokens});
                } else {
                    (void)layout.alloc(DType::BF16, {KVH, cache_tokens});
                    (void)layout.alloc(DType::BF16, {KVH, cache_tokens});
                }
                (void)layout.alloc(DType::BF16, {KVH, 3});
                (void)layout.alloc(DType::BF16, {KVH, 3});
                (void)layout.alloc(DType::BF16, {H, 3});
                (void)layout.alloc(DType::BF16, {H, 3});
                (void)layout.alloc(DType::BF16, {KVH, 3, 9});
                (void)layout.alloc(DType::BF16, {KVH, 3, 9});
                (void)layout.alloc(DType::BF16, {H, 3, 9});
                (void)layout.alloc(DType::BF16, {H, 3, 9});
            }
            CUDA_CHECK(cudaSetDevice(rank));
            persistent[rank] = std::make_unique<DeviceArena>(layout.peak_bytes(1));
            workspace[rank] = std::make_unique<DeviceArena>(kWorkspaceBytes);
            ones_storage[rank] = std::make_unique<DeviceBuffer>(256);
            std::array<std::uint16_t, 128> host_ones{};
            host_ones.fill(0x3f80U);
            ones_storage[rank]->copy_from_host(host_ones.data(), sizeof(host_ones));
            ones[rank] = Tensor(ones_storage[rank]->p, DType::BF16, {128});
            for (int slot = 0; slot < 48; ++slot) {
                LayerState& state = states[rank][slot];
                const bool full = is_full_attention(physical_layer_for_slot(slot));
                const int cache_tokens = full ? max_context : static_cast<int>(kSlidingWindow) + 64;
                if (full && quantize_full_kv) {
                    state.k_cache_i8 = persistent[rank]->alloc(DType::I32, {KVH / 4, cache_tokens});
                    state.v_cache_i8 = persistent[rank]->alloc(DType::I32, {KVH / 4, cache_tokens});
                    state.k_scales_i8 = persistent[rank]->alloc(DType::FP32, {KVH / 128, cache_tokens});
                    state.v_scales_i8 = persistent[rank]->alloc(DType::FP32, {KVH / 128, cache_tokens});
                } else {
                    state.k_cache = persistent[rank]->alloc(DType::BF16, {KVH, cache_tokens});
                    state.v_cache = persistent[rank]->alloc(DType::BF16, {KVH, cache_tokens});
                }
                state.k_conv = persistent[rank]->alloc(DType::BF16, {KVH, 3});
                state.v_conv = persistent[rank]->alloc(DType::BF16, {KVH, 3});
                state.attention_conv = persistent[rank]->alloc(DType::BF16, {H, 3});
                state.ffn_conv = persistent[rank]->alloc(DType::BF16, {H, 3});
                state.k_history = persistent[rank]->alloc(DType::BF16, {KVH, 3, 9});
                state.v_history = persistent[rank]->alloc(DType::BF16, {KVH, 3, 9});
                state.attention_history = persistent[rank]->alloc(DType::BF16, {H, 3, 9});
                state.ffn_history = persistent[rank]->alloc(DType::BF16, {H, 3, 9});
            }
        }
    }

    Tensor alloc(int rank, DType dtype, std::initializer_list<std::int32_t> shape) {
        return workspace[rank]->alloc(dtype, shape);
    }
    Tensor alloc_like(int rank, const Tensor& source) {
        const auto s = shape_of(source);
        return alloc(rank, source.dtype, {s[0], s[1], s[2], s[3]});
    }
    Tensor linear(int rank, const Tensor& x, const Weight& w, std::int32_t rows) {
        Tensor out = alloc(rank, DType::BF16, {rows, x.ne[1]});
        if (x.ne[1] <= 8 && w.qtype == QType::BF16_CTRL) {
            bf16_gemv(x, w, out, stream(rank));
        } else {
            ops::linear(x, w, out, stream(rank));
        }
        return out;
    }
    Tensor rms(int rank, const Tensor& x, const Tensor& weight_tensor) {
        const auto s = shape_of(x);
        Tensor out = alloc(rank, DType::BF16, {s[0], s[1], s[2], s[3]});
        ops::rmsnorm(x, weight_tensor, kRmsEpsilon, false, out, stream(rank));
        return out;
    }
    Tensor mix(int rank, const std::vector<Tensor>& blocks, const Tensor& cur, const Tensor& query) {
        const int depth = static_cast<int>(blocks.size() + 1);
        const int tokens = cur.ne[1];
        std::array<const void*, 5> sources{};
        for (int index = 0; index < depth - 1; ++index) {
            sources[index] = blocks[blocks.size() - 1 - index].data;
        }
        sources[depth - 1] = cur.data;
        Tensor mixed = alloc(rank, DType::BF16, {H, tokens, query.ne[1]});
        mix_depth_states(sources, depth, query, mixed, stream(rank));
        return mixed;
    }
    Tensor extract(int rank, const Tensor& mixed, int query) {
        (void)rank;
        return mixed.slice(2, query, 1).view({H, mixed.ne[1]});
    }
    Tensor control(int rank, const Tensor& cur, const Tensor& message, const LoopWeights& weights) {
        Tensor out = alloc(rank, DType::BF16, {H, cur.ne[1]});
        ops::inkling_loop_control(cur, message, weights.down, weights.gain, weights.forget,
                                  *workspace[rank], out, stream(rank));
        return out;
    }

    void allreduce(TensorPair& values) {
        CUDA_CHECK(cudaSetDevice(0));
        Tensor remote0 = alloc_like(0, values[1]);
        CUDA_CHECK(cudaEventRecord(allreduce_events[0], stream(0)));
        CUDA_CHECK(cudaSetDevice(1));
        Tensor remote1 = alloc_like(1, values[0]);
        CUDA_CHECK(cudaEventRecord(allreduce_events[1], stream(1)));
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamWaitEvent(stream(0), allreduce_events[1], 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(remote0.data, 0, values[1].data, 1, values[1].bytes(), stream(0)));
        ops::residual_add(remote0, values[0], stream(0));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaStreamWaitEvent(stream(1), allreduce_events[0], 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(remote1.data, 1, values[0].data, 0, values[0].bytes(), stream(1)));
        ops::residual_add(remote1, values[1], stream(1));
    }

    Tensor attention_local(int rank, int physical, int slot, const Tensor& cur,
                           std::vector<Tensor>& blocks, const std::vector<Tensor>* bank,
                           const Tensor* bank_out, Tensor& ffn_message) {
        LayerWeights& layer = layers[rank][physical];
        LayerState& state = states[rank][slot];
        CUDA_CHECK(cudaSetDevice(rank));
        const int tokens = cur.ne[1];
        std::array<Tensor, 4> messages;
        if (bank != nullptr) {
            Tensor query = alloc(rank, DType::FP32, {H, 4});
            pack_loop_query(cross_q[rank], cross_kv[rank], physical - 6, query, stream(rank));
            Tensor mixed = mix(rank, *bank, *bank_out, query);
            for (int site = 0; site < 4; ++site) { messages[site] = extract(rank, mixed, site); }
            ffn_message = messages[3];
        }
        std::array<Tensor, 3> inputs{cur, cur, cur};
        if (physical > 0) {
            Tensor mixed = mix(rank, blocks, cur, layer.attention_query);
            if (layer.full) {
                for (int part = 0; part < 3; ++part) { inputs[part] = extract(rank, mixed, part); }
            } else {
                inputs[0] = inputs[1] = inputs[2] = extract(rank, mixed, 0);
            }
        }
        if (bank != nullptr) {
            for (int part = 0; part < 3; ++part) {
                inputs[part] = control(rank, inputs[part], messages[part], layer.loop[part]);
            }
        }
        if (!layer.full && bank == nullptr) {
            inputs[0] = rms(rank, inputs[0], layer.norm);
            inputs[1] = inputs[2] = inputs[0];
        } else if (!layer.full) {
            for (Tensor& input : inputs) { input = rms(rank, input, layer.norm); }
        } else {
            inputs[0] = rms(rank, inputs[0], layer.q_input_norm);
            inputs[1] = rms(rank, inputs[1], layer.k_input_norm);
            inputs[2] = rms(rank, inputs[2], layer.v_input_norm);
        }
        Tensor q = linear(rank, inputs[0], layer.q, QH).view({128, 12, tokens});
        Tensor k_flat = linear(rank, inputs[1], layer.k, KVH);
        Tensor v_flat = linear(rank, inputs[2], layer.v, KVH);
        Tensor r = linear(rank, inputs[0], layer.rel, RH).view({16, 12, tokens});
        Tensor attention_gate = linear(rank, inputs[0], layer.gate, QH);
        Tensor sinks = linear(rank, inputs[0], layer.sink, 12);
        Tensor k_dynamic, v_dynamic;
        if (layer.full) {
            k_dynamic = linear(rank, inputs[1], layer.k_sconv_proj, 16);
            v_dynamic = linear(rank, inputs[2], layer.v_sconv_proj, 16);
        }
        Tensor k_conv = alloc(rank, DType::BF16, {KVH, tokens});
        Tensor v_conv = alloc(rank, DType::BF16, {KVH, tokens});
        ops::inkling_conv1d(k_flat, layer.full ? layer.k_sconv_static : layer.k_sconv,
            layer.full ? &k_dynamic : nullptr, state.k_conv, state.k_conv, k_conv, stream(rank),
            record_conv_history ? &state.k_history : nullptr);
        ops::inkling_conv1d(v_flat, layer.full ? layer.v_sconv_static : layer.v_sconv,
            layer.full ? &v_dynamic : nullptr, state.v_conv, state.v_conv, v_conv, stream(rank),
            record_conv_history ? &state.v_history : nullptr);
        Tensor k = k_conv.view({128, 4, tokens});
        Tensor v = v_conv.view({128, 4, tokens});
        Tensor clipped_q = alloc(rank, DType::BF16, {128, 12, tokens});
        Tensor clipped_k = alloc(rank, DType::BF16, {128, 4, tokens});
        Tensor clipped_r = alloc(rank, DType::BF16, {16, 12, tokens});
        Tensor normed_v = alloc(rank, DType::BF16, {128, 4, tokens});
        ops::l2_clip(q, &layer.q_clip, kQkL2Clip, clipped_q, stream(rank));
        ops::l2_clip(k, nullptr, kQkL2Clip, clipped_k, stream(rank));
        ops::l2_clip(r, &layer.rel_clip, kRelativeL2Clip, clipped_r, stream(rank));
        ops::rmsnorm(v, ones[rank], kRmsEpsilon, false, normed_v, stream(rank));
        if (layer.full) {
            Tensor positions = alloc(rank, DType::I32, {tokens});
            fill_positions(positions, length, stream(rank));
            Tensor scaled_q = alloc(rank, DType::BF16, {128, 12, tokens});
            Tensor scaled_r = alloc(rank, DType::BF16, {16, 12, tokens});
            ops::inkling_log_position_scale(clipped_q, positions, kLogScaleFloor, kLogScaleAlpha,
                                            scaled_q, stream(rank));
            ops::inkling_log_position_scale(clipped_r, positions, kLogScaleFloor, kLogScaleAlpha,
                                            scaled_r, stream(rank));
            clipped_q = scaled_q;
            clipped_r = scaled_r;
        }
        Tensor k_history, v_history;
        int total_keys = 0;
        if (layer.full && quantize_full_kv) {
            total_keys = length + tokens;
            append_kv_int8(clipped_k.view({KVH, tokens}), normed_v.view({KVH, tokens}),
                state.k_cache_i8, state.k_scales_i8, state.v_cache_i8, state.v_scales_i8,
                length, stream(rank));
        } else if (layer.full) {
            append_kv(clipped_k.view({KVH, tokens}), normed_v.view({KVH, tokens}), state.k_cache,
                      state.v_cache, length, stream(rank));
            total_keys = length + tokens;
            k_history = Tensor(state.k_cache.data, DType::BF16, {KVH, total_keys});
            v_history = Tensor(state.v_cache.data, DType::BF16, {KVH, total_keys});
        } else {
            const int history = std::min(length, static_cast<int>(kSlidingWindow));
            total_keys = history + tokens;
            k_history = alloc(rank, DType::BF16, {KVH, total_keys});
            v_history = alloc(rank, DType::BF16, {KVH, total_keys});
            if (history > 0) {
                Tensor k_prefix(k_history.data, DType::BF16, {KVH, history});
                Tensor v_prefix(v_history.data, DType::BF16, {KVH, history});
                gather_kv_ring(state.k_cache, state.v_cache, length - history, history,
                               state.k_cache.ne[1], k_prefix, v_prefix, stream(rank));
            }
            auto* k_destination = static_cast<std::byte*>(k_history.data) +
                                  static_cast<std::size_t>(history) * KVH * sizeof(std::uint16_t);
            auto* v_destination = static_cast<std::byte*>(v_history.data) +
                                  static_cast<std::size_t>(history) * KVH * sizeof(std::uint16_t);
            const auto new_bytes = static_cast<std::size_t>(KVH) * tokens * sizeof(std::uint16_t);
            CUDA_CHECK(cudaMemcpyAsync(k_destination, clipped_k.data, new_bytes,
                                       cudaMemcpyDeviceToDevice, stream(rank)));
            CUDA_CHECK(cudaMemcpyAsync(v_destination, normed_v.data, new_bytes,
                                       cudaMemcpyDeviceToDevice, stream(rank)));
            append_kv_ring(clipped_k.view({KVH, tokens}), normed_v.view({KVH, tokens}),
                           state.k_cache, state.v_cache, length, state.k_cache.ne[1], stream(rank));
        }
        Tensor k_attention, v_attention;
        if (!(layer.full && quantize_full_kv)) {
            k_attention = k_history.view({128, 4, total_keys});
            v_attention = v_history.view({128, 4, total_keys});
        }
        Tensor attention = alloc(rank, DType::BF16, {128, 12, tokens});
        if (profile_layers && rank == 0) {
            CUDA_CHECK(cudaEventRecord(profile_events[profile_layer_count][4], stream(0)));
        }
        if (layer.full && quantize_full_kv) {
            Tensor k_codes(state.k_cache_i8.data, DType::I32, {KVH / 4, total_keys});
            Tensor v_codes(state.v_cache_i8.data, DType::I32, {KVH / 4, total_keys});
            Tensor k_scales(state.k_scales_i8.data, DType::FP32, {KVH / 128, total_keys});
            Tensor v_scales(state.v_scales_i8.data, DType::FP32, {KVH / 128, total_keys});
            inkling_attention_int8(clipped_q, k_codes, k_scales, v_codes, v_scales, clipped_r,
                layer.rel_proj, sinks, kAttentionScale, attention, stream(rank));
        } else {
            ops::inkling_attention_dense(clipped_q, k_attention, v_attention, clipped_r, layer.rel_proj,
                sinks, kAttentionScale, layer.full ? 0 : kSlidingWindow, attention, stream(rank));
        }
        if (profile_layers && rank == 0) {
            CUDA_CHECK(cudaEventRecord(profile_events[profile_layer_count][5], stream(0)));
        }
        Tensor attention_flat = attention.view({QH, tokens});
        ops::sigmoid_mul(attention_gate, attention_flat, stream(rank));
        return linear(rank, attention_flat, layer.o, H);
    }

    Tensor ffn_local(int rank, int physical, const Tensor& cur, std::vector<Tensor>& blocks,
                     bool has_bank, const Tensor& message) {
        LayerWeights& layer = layers[rank][physical];
        const int tokens = cur.ne[1];
        Tensor ffn = extract(rank, mix(rank, blocks, cur, layer.mlp_query), 0);
        if (has_bank) { ffn = control(rank, ffn, message, layer.loop[3]); }
        ffn = rms(rank, ffn, layer.moe_norm);
        Tensor selected = alloc(rank, DType::I32, {K, tokens});
        Tensor route_weights = alloc(rank, DType::FP32, {K, tokens});
        ops::inkling_router(ffn, layer.router, layer.router_bias, kRouteScale, *workspace[rank],
                            selected, route_weights, stream(rank));
        Tensor moe = alloc(rank, DType::BF16, {H, tokens});
        ops::inkling_moe_prepared(ffn, layer.experts13, layer.experts2, selected, route_weights,
            kGateSoftcap, kUpSoftcap, layer.expert_views, *workspace[rank], moe, stream(rank));
        return moe;
    }

    void build_layer_pair(int physical, std::uint32_t slot, TensorPair& cur, BlockPair& blocks,
                          bool has_bank, const BlockPair& bank, const TensorPair& bank_out) {
        TensorPair partial, messages;
        if (profile_layers) {
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaEventRecord(profile_events[profile_layer_count][0], stream(0)));
        }
        for (int rank = 0; rank < 2; ++rank) {
            partial[rank] = attention_local(rank, physical, slot, cur[rank], blocks[rank],
                has_bank ? &bank[rank] : nullptr, has_bank ? &bank_out[rank] : nullptr, messages[rank]);
        }
        allreduce(partial);
        if (profile_layers) {
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaEventRecord(profile_events[profile_layer_count][1], stream(0)));
        }
        TensorPair input_ffn;
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            LayerWeights& layer = layers[rank][physical];
            LayerState& state = states[rank][slot];
            const Tensor input_attention = cur[rank];
            Tensor attention_conv = alloc(rank, DType::BF16, {H, cur[rank].ne[1]});
            ops::inkling_conv1d(partial[rank], layer.sconv, nullptr, state.attention_conv,
                                state.attention_conv, attention_conv, stream(rank),
                                record_conv_history ? &state.attention_history : nullptr);
            cur[rank] = attention_conv;
            if (physical % kAttentionResidualBlockSize == 0) {
                blocks[rank].push_back(input_attention);
            } else {
                ops::residual_add(input_attention, cur[rank], stream(rank));
            }
            input_ffn[rank] = cur[rank];
        }
        if (profile_layers) {
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaEventRecord(profile_events[profile_layer_count][2], stream(0)));
        }
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            partial[rank] = ffn_local(rank, physical, cur[rank], blocks[rank], has_bank, messages[rank]);
        }
        allreduce(partial);
        if (profile_layers) {
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaEventRecord(profile_events[profile_layer_count][3], stream(0)));
            ++profile_layer_count;
        }
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            LayerWeights& layer = layers[rank][physical];
            LayerState& state = states[rank][slot];
            Tensor ffn_conv = alloc(rank, DType::BF16, {H, cur[rank].ne[1]});
            ops::inkling_conv1d(partial[rank], layer.moe_sconv, nullptr, state.ffn_conv,
                                state.ffn_conv, ffn_conv, stream(rank),
                                record_conv_history ? &state.ffn_history : nullptr);
            cur[rank] = ffn_conv;
            ops::residual_add(input_ffn[rank], cur[rank], stream(rank));
        }
    }

    void capture_feature(int tap, const Tensor& value) {
        if (draft == nullptr) { return; }
        CUDA_CHECK(cudaSetDevice(0));
        auto* destination = target_features[tap].data;
        CUDA_CHECK(cudaMemcpyAsync(destination, value.data, value.bytes(),
                                   cudaMemcpyDeviceToDevice, stream(0)));
    }

    std::array<Tensor, 4> feature_views(int committed_length) const {
        if (committed_length <= 0 || committed_length > 128) {
            throw std::invalid_argument("Inkling feature views are limited to the current 128-token block");
        }
        std::array<Tensor, 4> views;
        for (int tap = 0; tap < 4; ++tap) {
            views[tap] = Tensor(target_features[tap].data, DType::BF16, {H, committed_length});
        }
        return views;
    }

    void restore_convolution_state(int committed_tokens) {
        if (committed_tokens < 0 || committed_tokens > 8) {
            throw std::invalid_argument("Inkling speculative rollback expects 0..8 tokens");
        }
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            for (LayerState& state : states[rank]) {
                Tensor* current[] = {&state.k_conv, &state.v_conv,
                                     &state.attention_conv, &state.ffn_conv};
                Tensor* history[] = {&state.k_history, &state.v_history,
                                     &state.attention_history, &state.ffn_history};
                for (int site = 0; site < 4; ++site) {
                    const auto* source = static_cast<const std::byte*>(history[site]->data) +
                                         static_cast<std::size_t>(committed_tokens) * current[site]->bytes();
                    CUDA_CHECK(cudaMemcpyAsync(current[site]->data, source, current[site]->bytes(),
                                               cudaMemcpyDeviceToDevice, stream(rank)));
                }
            }
        }
    }

    std::vector<int> forward(const std::vector<int>& tokens, bool all_logits = false,
                             bool record_history = false, bool emit_logits = true) {
        if (tokens.empty() || length + static_cast<int>(tokens.size()) > max_context) {
            throw std::invalid_argument("Inkling TP2 forward chunk is empty or exceeds max_context");
        }
        const int count = static_cast<int>(tokens.size());
        if (!emit_logits && (all_logits || record_history)) {
            throw std::invalid_argument("Inkling verify forward must emit logits");
        }
        if (record_history && (draft == nullptr || count > 8)) {
            throw std::invalid_argument("Inkling convolution history requires an enabled MTP and at most 8 tokens");
        }
        record_conv_history = record_history;
        profile_layers = (record_history || count == 1) &&
                         std::getenv("NINFER_INKLING_PROFILE") != nullptr;
        if (profile_layers) {
            profile_layer_count = 0;
            profile_events.resize(48);
            CUDA_CHECK(cudaSetDevice(0));
            for (auto& events : profile_events) {
                for (cudaEvent_t& event : events) { CUDA_CHECK(cudaEventCreate(&event)); }
            }
        }
        TensorPair cur;
        BlockPair blocks;
        for (int rank = 0; rank < 2; ++rank) {
            workspace[rank]->reset();
            CUDA_CHECK(cudaSetDevice(rank));
            Tensor ids = alloc(rank, DType::I32, {count});
            CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data(), tokens.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, stream(rank)));
            cur[rank] = alloc(rank, DType::BF16, {H, count});
            ops::embedding(ids, embedding[rank], cur[rank], stream(rank));
            cur[rank] = rms(rank, cur[rank], embedding_norm[rank]);
            if (visual_count > 0 && length < visual_begin + visual_count &&
                length + count > visual_begin) {
                const int copy_begin = std::max(length, visual_begin);
                const int copy_end = std::min(length + count, visual_begin + visual_count);
                if (visual_begin < 0 ||
                    visual_embeddings.size() != static_cast<std::size_t>(visual_count) * H) {
                    throw std::invalid_argument("Inkling visual embedding range is inconsistent with the prompt");
                }
                const auto source_offset = static_cast<std::size_t>(copy_begin - visual_begin) * H;
                auto* destination = static_cast<std::byte*>(cur[rank].data) +
                                    static_cast<std::size_t>(copy_begin - length) * H * sizeof(std::uint16_t);
                const auto copy_values = static_cast<std::size_t>(copy_end - copy_begin) * H;
                CUDA_CHECK(cudaMemcpyAsync(destination, visual_embeddings.data() + source_offset,
                                           copy_values * sizeof(std::uint16_t),
                                           cudaMemcpyHostToDevice, stream(rank)));
            }
        }
        for (int physical = 0; physical < 6; ++physical) {
            if (physical == 5) { capture_feature(0, cur[0]); }
            TensorPair unused_bank;
            build_layer_pair(physical, state_slot(physical, 0), cur, blocks, false, {}, unused_bank);
        }
        const TensorPair prelude = cur;
        const BlockPair prelude_blocks = blocks;
        BlockPair bank;
        TensorPair bank_out;
        for (std::uint32_t pass = 0; pass < kLoopPasses; ++pass) {
            if (pass > 0) {
                for (int rank = 0; rank < 2; ++rank) {
                    bank[rank].assign(blocks[rank].begin() + prelude_blocks[rank].size() + 1,
                                      blocks[rank].end());
                    bank_out[rank] = cur[rank];
                    cur[rank] = prelude[rank];
                    blocks[rank] = prelude_blocks[rank];
                }
            }
            for (int physical = 6; physical < 18; ++physical) {
                if (pass == 1 && physical == 11) { capture_feature(1, cur[0]); }
                if (pass == 2 && physical == 17) { capture_feature(2, cur[0]); }
                build_layer_pair(physical, state_slot(physical, pass), cur, blocks, pass > 0, bank, bank_out);
            }
        }
        for (int physical = 18; physical < 24; ++physical) {
            build_layer_pair(physical, state_slot(physical, 0), cur, blocks, false, bank, bank_out);
        }
        if (!emit_logits && draft == nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(stream(0)));
            CUDA_CHECK(cudaStreamSynchronize(stream(1)));
            length += count;
            record_conv_history = false;
            return {};
        }
        CUDA_CHECK(cudaSetDevice(0));
        Tensor hidden;
        if (draft != nullptr || all_logits) {
            hidden = extract(0, mix(0, blocks[0], cur[0], output_query), 0);
            hidden = rms(0, hidden, final_norm);
            if (draft != nullptr) { capture_feature(3, hidden); }
            if (!all_logits) { hidden = hidden.slice(1, count - 1, 1); }
        } else {
            std::vector<Tensor> last_blocks;
            for (const Tensor& block : blocks[0]) { last_blocks.push_back(block.slice(1, count - 1, 1)); }
            hidden = extract(0, mix(0, last_blocks, cur[0].slice(1, count - 1, 1), output_query), 0);
            hidden = rms(0, hidden, final_norm);
        }
        if (!emit_logits) {
            CUDA_CHECK(cudaStreamSynchronize(stream(0)));
            CUDA_CHECK(cudaStreamSynchronize(stream(1)));
            length += count;
            record_conv_history = false;
            return {};
        }
        const int output_count = hidden.ne[1];
        Tensor logits = linear(0, hidden, output_head, V);
        Tensor capped = alloc(0, DType::BF16, {V, output_count});
        ops::inkling_softcap(logits, kLogitSoftcap, capped, stream(0));
        Tensor next = alloc(0, DType::I32, {output_count});
        ops::argmax(capped, next, static_cast<std::int32_t>(kUnpaddedVocab), stream(0));
        std::vector<int> result(output_count);
        CUDA_CHECK(cudaMemcpyAsync(result.data(), next.data, result.size() * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream(0)));
        CUDA_CHECK(cudaStreamSynchronize(stream(0)));
        CUDA_CHECK(cudaStreamSynchronize(stream(1)));
        if (profile_layers) {
            float attention_ms = 0.0F;
            float dense_ms = 0.0F;
            float ffn_ms = 0.0F;
            for (int index = 0; index < profile_layer_count; ++index) {
                auto& events = profile_events[index];
                float elapsed = 0.0F;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, events[0], events[1]));
                attention_ms += elapsed;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, events[4], events[5]));
                dense_ms += elapsed;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, events[2], events[3]));
                ffn_ms += elapsed;
            }
            float total_ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, profile_events[0][0],
                                             profile_events[profile_layer_count - 1][3]));
            std::fprintf(stderr, "layer_profile_tokens=%d attention_ms=%.3f dense_ms=%.3f ffn_ms=%.3f other_ms=%.3f total_ms=%.3f\n",
                         count, attention_ms, dense_ms, ffn_ms, total_ms - attention_ms - ffn_ms, total_ms);
            for (auto& events : profile_events) {
                for (cudaEvent_t event : events) { CUDA_CHECK(cudaEventDestroy(event)); }
            }
            profile_events.clear();
            profile_layers = false;
        }
        length += count;
        record_conv_history = false;
        return result;
    }

    void reset_state() {
        length = 0;
        verify_rounds = 0;
        proposed_drafts = 0;
        accepted_drafts = 0;
        native_fallback_tokens = 0;
        draft_seconds = 0.0;
        verify_seconds = 0.0;
        if (draft != nullptr) { draft->reset(); }
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            for (int slot = 0; slot < 48; ++slot) {
                LayerState& state = states[rank][slot];
                Tensor* tensors[6] = {nullptr};
                int count = 0;
                if (quantize_full_kv && is_full_attention(physical_layer_for_slot(slot))) {
                    tensors[count++] = &state.k_cache_i8;
                    tensors[count++] = &state.v_cache_i8;
                    tensors[count++] = &state.k_scales_i8;
                    tensors[count++] = &state.v_scales_i8;
                } else {
                    tensors[count++] = &state.k_cache;
                    tensors[count++] = &state.v_cache;
                }
                tensors[count++] = &state.k_conv;
                tensors[count++] = &state.v_conv;
                tensors[count++] = &state.attention_conv;
                tensors[count++] = &state.ffn_conv;
                for (int index = 0; index < count; ++index) {
                    CUDA_CHECK(cudaMemsetAsync(tensors[index]->data, 0, tensors[index]->bytes(), stream(rank)));
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream(rank)));
        }
    }
    void ensure_tokenizer() {
        if (tokenizer == nullptr) {
            tokenizer = std::make_unique<qwen3_6::frontend_internal::Tokenizer>(
                qwen3_6::frontend_internal::TokenizerResources{tokenizer_json, tokenizer_config_json,
                                                               generation_config_json});
        }
    }
    std::vector<int> encode_text(const std::string& text) { ensure_tokenizer(); return tokenizer->encode(text); }
    std::string decode_text(const std::vector<int>& tokens) {
        ensure_tokenizer();
        return tokenizer->decode(tokens, qwen3_6::frontend_internal::DecodeOptions{.skip_special_tokens = false});
    }
    bool is_stop(int token) const {
        if (tokenizer == nullptr) { return false; }
        const auto& stops = tokenizer->default_stop_token_ids();
        return std::find(stops.begin(), stops.end(), token) != stops.end();
    }
};

RuntimeTP2::RuntimeTP2(const std::string& artifact_path, int max_context, bool enable_vision)
    : impl_(std::make_unique<Impl>(artifact_path, max_context, enable_vision)) {}
RuntimeTP2::~RuntimeTP2() = default;
void RuntimeTP2::reset() { impl_->reset_state(); }
std::vector<int> RuntimeTP2::encode_text(const std::string& text) { return impl_->encode_text(text); }
std::string RuntimeTP2::decode_text(const std::vector<int>& tokens) { return impl_->decode_text(tokens); }
double RuntimeTP2::last_prefill_seconds() const { return impl_->prefill_seconds; }
double RuntimeTP2::last_decode_seconds() const { return impl_->decode_seconds; }
double RuntimeTP2::last_draft_seconds() const { return impl_->draft_seconds; }
double RuntimeTP2::last_verify_seconds() const { return impl_->verify_seconds; }
int RuntimeTP2::last_verify_rounds() const { return impl_->verify_rounds; }
int RuntimeTP2::last_proposed_drafts() const { return impl_->proposed_drafts; }
int RuntimeTP2::last_accepted_drafts() const { return impl_->accepted_drafts; }
int RuntimeTP2::last_native_fallback_tokens() const { return impl_->native_fallback_tokens; }

std::vector<int> RuntimeTP2::generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                                      OutputCallback callback) {
    impl_->visual_count = 0;
    impl_->visual_embeddings.clear();
    return generate_current(prompt_tokens, max_new_tokens, callback);
}

std::vector<int> RuntimeTP2::generate_with_image(const std::vector<int>& prompt_tokens,
    int image_token_id, const media::decode::Image& image, int max_new_tokens,
    OutputCallback callback) {
    if (impl_->vision == nullptr) { throw std::invalid_argument("Inkling vision is not enabled"); }
    const auto found = std::find(prompt_tokens.begin(), prompt_tokens.end(), image_token_id);
    if (found == prompt_tokens.end() || std::find(found + 1, prompt_tokens.end(), image_token_id) != prompt_tokens.end()) {
        throw std::invalid_argument("Inkling vision expects exactly one image placeholder");
    }
    const auto encoded = impl_->vision->encode(image);
    const std::size_t position = static_cast<std::size_t>(found - prompt_tokens.begin());
    std::vector<int> expanded;
    expanded.reserve(prompt_tokens.size() - 1 + encoded.merged_count);
    expanded.insert(expanded.end(), prompt_tokens.begin(), found);
    expanded.insert(expanded.end(), encoded.merged_count, image_token_id);
    expanded.insert(expanded.end(), found + 1, prompt_tokens.end());
    impl_->visual_begin = static_cast<int>(position);
    impl_->visual_count = encoded.merged_count;
    impl_->visual_embeddings = encoded.embeddings;
    return generate_current(expanded, max_new_tokens, callback);
}

std::vector<int> RuntimeTP2::generate_current(const std::vector<int>& prompt_tokens, int max_new_tokens,
                                               const OutputCallback& callback) {
    if (prompt_tokens.empty() || max_new_tokens <= 0 ||
        prompt_tokens.size() + static_cast<std::size_t>(max_new_tokens) > static_cast<std::size_t>(impl_->max_context)) {
        throw std::invalid_argument("Inkling TP2 generate requires a nonempty prompt and fitting output budget");
    }
    reset();
    std::vector<int> output;
    output.reserve(max_new_tokens);
    const auto prefill_start = std::chrono::steady_clock::now();
    int next = 0;
    constexpr std::size_t kPrefillChunk = 128;
    for (std::size_t begin = 0; begin < prompt_tokens.size(); begin += kPrefillChunk) {
        const std::size_t end = std::min(prompt_tokens.size(), begin + kPrefillChunk);
        const std::vector<int> chunk(prompt_tokens.begin() + static_cast<std::ptrdiff_t>(begin),
                                     prompt_tokens.begin() + static_cast<std::ptrdiff_t>(end));
        if (end == prompt_tokens.size()) {
            next = impl_->forward(chunk).back();
        } else {
            (void)impl_->forward(chunk, false, false, false);
        }
        if (impl_->draft != nullptr) {
            impl_->draft->commit(impl_->feature_views(static_cast<int>(chunk.size())),
                                 static_cast<int>(chunk.size()));
        }
    }
    impl_->prefill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_start).count();
    output.push_back(next); if (callback) { callback(next); }
    const auto decode_start = std::chrono::steady_clock::now();
    impl_->decode_seconds = 0.0;
    if (impl_->is_stop(next)) { return output; }
    if (impl_->draft == nullptr) {
        for (int index = 1; index < max_new_tokens; ++index) {
            next = impl_->forward(std::vector<int>{next}).front(); if (impl_->draft != nullptr) { impl_->draft->commit(impl_->feature_views(1), 1); }
            output.push_back(next); if (callback) { callback(next); }
            if (impl_->is_stop(next)) { break; }
        }
    } else {
        bool stopped = false;
        int native_remaining = 0;
        int window_rounds = 0;
        int window_tokens = 0;
        int gamma = 7;
        if (const char* configured = std::getenv("NINFER_INKLING_GAMMA")) {
            gamma = std::stoi(configured);
            if (gamma <= 0 || gamma > 7) {
                throw std::invalid_argument("NINFER_INKLING_GAMMA must be in [1,7]");
            }
        }
        while (static_cast<int>(output.size()) < max_new_tokens && !stopped) {
            const int remaining = max_new_tokens - static_cast<int>(output.size());
            if (native_remaining > 0 && remaining > 1) {
                next = impl_->forward(std::vector<int>{next}).front(); if (impl_->draft != nullptr) { impl_->draft->commit(impl_->feature_views(1), 1); }
                output.push_back(next); if (callback) { callback(next); }
                stopped = impl_->is_stop(next);
                --native_remaining;
                ++impl_->native_fallback_tokens;
                if (native_remaining == 0) {
                    window_rounds = 0;
                    window_tokens = 0;
                }
                continue;
            }
            if (remaining == 1 || impl_->length + 8 > impl_->max_context) {
                next = impl_->forward(std::vector<int>{next}).front(); if (impl_->draft != nullptr) { impl_->draft->commit(impl_->feature_views(1), 1); }
                output.push_back(next); if (callback) { callback(next); }
                stopped = impl_->is_stop(next);
                continue;
            }
            const int verify_begin = impl_->length;
            const auto draft_start = std::chrono::steady_clock::now();
            const auto drafts = impl_->draft->propose(next, gamma);
            impl_->draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - draft_start).count();
            std::vector<int> window{next};
            window.insert(window.end(), drafts.begin(), drafts.begin() + gamma);
            const auto verify_start = std::chrono::steady_clock::now();
            const std::vector<int> predictions = impl_->forward(window, true, true);
            impl_->verify_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - verify_start).count();
            int accepted = 0;
            while (accepted < gamma && predictions[accepted] == drafts[accepted]) { ++accepted; }
            impl_->restore_convolution_state(1 + accepted);
            impl_->length = verify_begin + 1 + accepted;
            impl_->draft->commit(impl_->feature_views(1 + accepted), 1 + accepted);
            ++impl_->verify_rounds;
            impl_->proposed_drafts += gamma;
            impl_->accepted_drafts += accepted;
            for (int step = 0; step <= accepted; ++step) {
                if (static_cast<int>(output.size()) >= max_new_tokens) { break; }
                next = predictions[step];
                output.push_back(next); if (callback) { callback(next); }
                if (impl_->is_stop(next)) {
                    stopped = true;
                    break;
                }
            }
            ++window_rounds;
            window_tokens += accepted + 1;
            if ((window_rounds == 1 && window_tokens <= 3) ||
                (window_rounds >= 2 && window_tokens <= 3 * window_rounds)) {
                if (std::getenv("NINFER_INKLING_SPEC_ALWAYS") == nullptr) {
                    native_remaining = 64;
                }
            }
        }
    }
    impl_->decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
    return output;
}

} // namespace ninfer::targets::inkling2_mini::detail
