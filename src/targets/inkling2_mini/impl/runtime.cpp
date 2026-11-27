#include "targets/inkling2_mini/impl/runtime.h"

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
#include "targets/inkling2_mini/impl/runtime_kernels.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {
namespace {

constexpr std::int32_t H = static_cast<std::int32_t>(kHidden);
constexpr std::int32_t Q_SIZE = static_cast<std::int32_t>(kQueryHeads * kHeadDim);
constexpr std::int32_t KV_SIZE = static_cast<std::int32_t>(kKvHeads * kHeadDim);
constexpr std::int32_t R_SIZE = static_cast<std::int32_t>(kRelativePositions * kQueryHeads);
constexpr std::int32_t I = static_cast<std::int32_t>(kExpertHidden);
constexpr std::int32_t E = static_cast<std::int32_t>(kExperts);
constexpr std::int32_t K = static_cast<std::int32_t>(kActiveExperts);
constexpr std::int32_t V = static_cast<std::int32_t>(kVocab);
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
    int device = 0;
    bool full = false;
    bool core = false;
    Weight q;
    Weight k;
    Weight v;
    Weight o;
    Weight rel;
    Weight gate;
    Weight sink;
    Tensor q_clip;
    Tensor rel_clip;
    Tensor rel_proj;
    Tensor sconv;
    Tensor norm;
    Tensor q_input_norm;
    Tensor k_input_norm;
    Tensor v_input_norm;
    Tensor k_sconv;
    Tensor v_sconv;
    Tensor k_sconv_static;
    Tensor v_sconv_static;
    Weight k_sconv_proj;
    Weight v_sconv_proj;
    Tensor attention_query;
    Tensor mlp_query;
    Tensor moe_norm;
    Tensor moe_sconv;
    Tensor router;
    Tensor router_bias;
    std::vector<Weight> experts13;
    std::vector<Weight> experts2;
    std::vector<Tensor> experts13_mxscale;
    std::vector<Tensor> experts2_mxscale;
    std::unique_ptr<DeviceBuffer> expert_view_storage;
    DeviceSpan expert_views;
    std::array<LoopWeights, 4> loop;
};

struct LayerState {
    int device = 0;
    Tensor k_cache;
    Tensor v_cache;
    Tensor k_conv;
    Tensor v_conv;
    Tensor attention_conv;
    Tensor ffn_conv;
};

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

struct Runtime::Impl {
    ExecutionContext execution;
    artifact::Reader reader;
    artifact::MaterializedArtifact backing;
    BindingPlan bindings;
    std::vector<LayerWeights> layers;
    std::vector<LayerState> states;
    Weight embedding;
    Weight output_head;
    Tensor embedding_norm;
    Tensor final_norm;
    Tensor output_query;
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string generation_config_json;
    std::unique_ptr<qwen3_6::frontend_internal::Tokenizer> tokenizer;
    std::array<Tensor, 2> cross_q;
    std::array<Tensor, 2> cross_kv;
    std::unique_ptr<DeviceArena> persistent[2];
    std::unique_ptr<DeviceArena> workspace[2];
    std::unique_ptr<DeviceBuffer> ones_storage[2];
    Tensor ones[2];
    int max_context = 0;
    int length = 0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;

    Impl(const std::string& path, int context)
        : execution(std::vector<int>{0, 1}), reader(path), max_context(context) {
        if (max_context <= 0 || max_context > 65536) {
            throw std::invalid_argument("Inkling eager runtime max_context must be in [1,65536]");
        }
        enable_peer(0, 1);
        enable_peer(1, 0);
        artifact::Binder binder(reader, 2);
        auto plan = bind_artifact(binder, artifact::TensorPlacement::Device,
                                  InklingDeviceMode::Pipeline16);
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
        allocate_state();
        reset_state();
    }

    cudaStream_t stream(int device) const { return execution.dev[device]->stream; }

    Tensor tensor(artifact::ObjectHandle handle, artifact::NumericFormat format,
                  std::initializer_list<std::int32_t> shape, int device) const {
        return artifact::materialized_tensor(backing, handle, format, shape, device);
    }

    Weight weight(artifact::ObjectHandle handle, artifact::NumericFormat format, std::int32_t n,
                  std::int32_t k, int device, float divisor = 1.0F) const {
        if (format != artifact::NumericFormat::NVFP4) {
            return artifact::materialized_weight(backing, handle, format, n, k, device);
        }
        const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(n),
                                                static_cast<std::uint64_t>(k)};
        const auto geometry = artifact::block_scale_geometry(format, shape);
        artifact::require_placement_bytes(backing, handle, device, geometry.encoded_bytes);
        const auto* bytes = static_cast<const std::byte*>(backing.device_data(handle, device));
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
        embedding = weight(bindings.token_embedding, artifact::NumericFormat::BF16, V, H, 0);
        output_head = weight(bindings.output_head, artifact::NumericFormat::BF16, V, H, 1);
        embedding_norm = tensor(bindings.embedding_norm, artifact::NumericFormat::BF16, {H}, 0);
        final_norm = tensor(bindings.final_norm, artifact::NumericFormat::BF16, {H}, 1);
        output_query = tensor(bindings.output_query, artifact::NumericFormat::FP32, {H, 1}, 1);
        for (int device = 0; device < 2; ++device) {
            cross_q[device] = tensor(bindings.cross_pass_query, artifact::NumericFormat::FP32,
                                     {24, H}, device);
            cross_kv[device] = tensor(bindings.cross_pass_kv_query, artifact::NumericFormat::FP32,
                                      {24, H}, device);
        }
        layers.reserve(kPhysicalLayers);
        for (std::size_t index = 0; index < kPhysicalLayers; ++index) {
            const TextLayerPlan& plan = bindings.text_layers[index];
            const int physical = static_cast<int>(index);
            LayerWeights out;
            out.device = physical < 16 ? 0 : 1;
            out.full = plan.full_attention;
            out.core = plan.loop_core;
            const int d = out.device;
            const auto W = [&](artifact::ObjectHandle handle, std::int32_t n, std::int32_t k) {
                return weight(handle, artifact::NumericFormat::BF16, n, k, d);
            };
            const auto T = [&](artifact::ObjectHandle handle,
                               std::initializer_list<std::int32_t> shape) {
                return tensor(handle, artifact::NumericFormat::BF16, shape, d);
            };
            out.q = W(plan.attention.q, Q_SIZE, H);
            out.k = W(plan.attention.k, KV_SIZE, H);
            out.v = W(plan.attention.v, KV_SIZE, H);
            out.o = W(plan.attention.o, H, Q_SIZE);
            out.rel = W(plan.attention.rel, R_SIZE, H);
            out.gate = W(plan.attention.gate, Q_SIZE, H);
            out.sink = W(plan.attention.sink, static_cast<std::int32_t>(kQueryHeads), H);
            out.q_clip = T(plan.attention.q_clip, {static_cast<std::int32_t>(kHeadDim),
                                                   static_cast<std::int32_t>(kQueryHeads)});
            out.rel_clip = T(plan.attention.rel_clip, {static_cast<std::int32_t>(kRelativePositions),
                                                       static_cast<std::int32_t>(kQueryHeads)});
            out.rel_proj = T(plan.attention.rel_proj, {static_cast<std::int32_t>(kRelativePositions),
                                                       static_cast<std::int32_t>(kRelativeExtent)});
            out.sconv = T(plan.attention.sconv, {H, 4});
            if (out.full) {
                out.q_input_norm = T(plan.attention.q_input_norm, {H});
                out.k_input_norm = T(plan.attention.k_input_norm, {H});
                out.v_input_norm = T(plan.attention.v_input_norm, {H});
                out.k_sconv_static = T(plan.attention.k_sconv_static, {4, KV_SIZE});
                out.v_sconv_static = T(plan.attention.v_sconv_static, {4, KV_SIZE});
                out.k_sconv_proj = W(plan.attention.k_sconv_proj, 4 * static_cast<std::int32_t>(kKvHeads), H);
                out.v_sconv_proj = W(plan.attention.v_sconv_proj, 4 * static_cast<std::int32_t>(kKvHeads), H);
            } else {
                out.norm = T(plan.attention.norm, {H});
                out.k_sconv = T(plan.attention.k_sconv, {KV_SIZE, 4});
                out.v_sconv = T(plan.attention.v_sconv, {KV_SIZE, 4});
            }
            if (physical > 0) {
                out.attention_query = tensor(bindings.attention_queries[physical - 1],
                    artifact::NumericFormat::FP32, {H, out.full ? 3 : 1}, d);
            }
            out.mlp_query = tensor(bindings.mlp_queries[physical], artifact::NumericFormat::FP32,
                                   {H, 1}, d);
            out.moe_norm = T(plan.moe.norm, {H});
            out.moe_sconv = T(plan.moe.sconv, {H, 4});
            out.router = T(plan.moe.gate, {E, H});
            out.router_bias = tensor(plan.moe.gate_bias, artifact::NumericFormat::FP32, {E}, d);
            out.experts13.reserve(E);
            out.experts2.reserve(E);
            out.experts13_mxscale.reserve(E);
            out.experts2_mxscale.reserve(E);
            for (std::size_t expert = 0; expert < E; ++expert) {
                const ExpertPlan& source = plan.moe.experts[expert];
                const auto format = has_bf16_experts(index)
                    ? artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S : artifact::NumericFormat::NVFP4;
                out.experts13.push_back(weight(source.w13, format, H, 2 * I, d, source.w13_divisor));
                out.experts2.push_back(weight(source.w2, format, H, I, d, source.w2_divisor));
                if (format == artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S) {
                    out.experts13_mxscale.push_back(T(source.w13_mxscale, {H, I / 16}));
                    out.experts2_mxscale.push_back(T(source.w2_mxscale, {H, I / 32}));
                }
            }
            std::vector<ops::detail::InklingMoeWeightView> host_views;
            host_views.reserve(2 * E);
            const auto make_view = [](const Weight& source, const Tensor* group_scales) {
                return ops::detail::InklingMoeWeightView{
                    .qdata = source.qdata,
                    .scales = source.scales,
                    .group_scales = group_scales == nullptr ? nullptr : group_scales->data,
                    .n = source.n,
                    .k = source.k,
                    .divisor = source.qtype == QType::NVFP4 ? source.weight_scale_divisor : 1.0F,
                    .nvfp4 = source.qtype == QType::NVFP4,
                    .mxfp8 = source.qtype == QType::FP8_E4M3FN_ROW_BF16S,
                };
            };
            for (std::size_t expert = 0; expert < E; ++expert) {
                const Tensor* scale = out.experts13_mxscale.empty() ? nullptr : &out.experts13_mxscale[expert];
                host_views.push_back(make_view(out.experts13[expert], scale));
            }
            for (std::size_t expert = 0; expert < E; ++expert) {
                const Tensor* scale = out.experts2_mxscale.empty() ? nullptr : &out.experts2_mxscale[expert];
                host_views.push_back(make_view(out.experts2[expert], scale));
            }
            CUDA_CHECK(cudaSetDevice(d));
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
            layers.push_back(std::move(out));
        }
    }

    void allocate_state() {
        WorkspaceLayoutBuilder layouts[2];
        for (std::uint32_t slot = 0; slot < kStateSlots; ++slot) {
            const int device = physical_layer_for_slot(slot) < 16 ? 0 : 1;
            (void)layouts[device].alloc(DType::BF16, {KV_SIZE, max_context});
            (void)layouts[device].alloc(DType::BF16, {KV_SIZE, max_context});
            (void)layouts[device].alloc(DType::BF16, {KV_SIZE, 3});
            (void)layouts[device].alloc(DType::BF16, {KV_SIZE, 3});
            (void)layouts[device].alloc(DType::BF16, {H, 3});
            (void)layouts[device].alloc(DType::BF16, {H, 3});
        }
        states.resize(kStateSlots);
        for (int device = 0; device < 2; ++device) {
            CUDA_CHECK(cudaSetDevice(device));
            persistent[device] = std::make_unique<DeviceArena>(layouts[device].peak_bytes(1));
            workspace[device] = std::make_unique<DeviceArena>(kWorkspaceBytes);
            ones_storage[device] = std::make_unique<DeviceBuffer>(128 * sizeof(std::uint16_t));
            std::array<std::uint16_t, 128> host_ones{};
            host_ones.fill(0x3f80U);
            ones_storage[device]->copy_from_host(host_ones.data(), sizeof(host_ones));
            ones[device] = Tensor(ones_storage[device]->p, DType::BF16, {128});
        }
        for (std::uint32_t slot = 0; slot < kStateSlots; ++slot) {
            LayerState& state = states[slot];
            state.device = physical_layer_for_slot(slot) < 16 ? 0 : 1;
            DeviceArena& arena = *persistent[state.device];
            state.k_cache = arena.alloc(DType::BF16, {KV_SIZE, max_context});
            state.v_cache = arena.alloc(DType::BF16, {KV_SIZE, max_context});
            state.k_conv = arena.alloc(DType::BF16, {KV_SIZE, 3});
            state.v_conv = arena.alloc(DType::BF16, {KV_SIZE, 3});
            state.attention_conv = arena.alloc(DType::BF16, {H, 3});
            state.ffn_conv = arena.alloc(DType::BF16, {H, 3});
        }
    }

    Tensor alloc(int device, DType dtype, std::initializer_list<std::int32_t> shape) {
        return workspace[device]->alloc(dtype, shape);
    }

    Tensor alloc_like(int device, const Tensor& source) {
        const auto shape = shape_of(source);
        return alloc(device, source.dtype, {shape[0], shape[1], shape[2], shape[3]});
    }

    Tensor linear(int device, const Tensor& x, const Weight& w, std::int32_t rows) {
        Tensor out = alloc(device, DType::BF16, {rows, x.ne[1]});
        if (x.ne[1] == 1 && w.qtype == QType::BF16_CTRL) {
            bf16_gemv(x, w, out, stream(device));
        } else {
            ops::linear(x, w, out, stream(device));
        }
        return out;
    }

    Tensor rms(int device, const Tensor& x, const Tensor& weight_tensor) {
        const auto shape = shape_of(x);
        Tensor out = alloc(device, DType::BF16, {shape[0], shape[1], shape[2], shape[3]});
        ops::rmsnorm(x, weight_tensor, kRmsEpsilon, false, out, stream(device));
        return out;
    }

    Tensor mix(int device, const std::vector<Tensor>& blocks, const Tensor& cur,
               const Tensor& query) {
        const std::int32_t depth = static_cast<std::int32_t>(blocks.size() + 1);
        const std::int32_t tokens = cur.ne[1];
        Tensor packed = alloc(device, DType::BF16, {H, depth, tokens});
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            const Tensor& source = blocks[blocks.size() - 1 - index];
            pack_depth_state(source, packed, static_cast<std::int32_t>(index), stream(device));
        }
        pack_depth_state(cur, packed, depth - 1, stream(device));
        Tensor mixed = alloc(device, DType::BF16, {H, query.ne[1], tokens});
        ops::inkling_depth_mix(packed, query, mixed, stream(device));
        return mixed;
    }

    Tensor extract(int device, const Tensor& mixed, std::int32_t query) {
        Tensor out = alloc(device, DType::BF16, {H, mixed.ne[2]});
        extract_depth_output(mixed, query, out, stream(device));
        return out;
    }

    Tensor control(int device, const Tensor& cur, const Tensor& message, const LoopWeights& weights) {
        Tensor out = alloc(device, DType::BF16, {H, cur.ne[1]});
        ops::inkling_loop_control(cur, message, weights.down, weights.gain, weights.forget,
                                  *workspace[device], out, stream(device));
        return out;
    }

    Tensor peer_copy_after_sync(const Tensor& source, int source_device, int destination_device) {
        Tensor destination = alloc_like(destination_device, source);
        CUDA_CHECK(cudaMemcpyPeerAsync(destination.data, destination_device, source.data,
                                       source_device, source.bytes(), stream(destination_device)));
        return destination;
    }

    void transfer(Tensor& cur, std::vector<Tensor>& blocks, int source, int destination) {
        if (source == destination) { return; }
        CUDA_CHECK(cudaStreamSynchronize(stream(source)));
        CUDA_CHECK(cudaSetDevice(destination));
        cur = peer_copy_after_sync(cur, source, destination);
        for (Tensor& block : blocks) { block = peer_copy_after_sync(block, source, destination); }
    }

    std::vector<Tensor> copy_bank_to_zero(const std::vector<Tensor>& source) {
        CUDA_CHECK(cudaStreamSynchronize(stream(1)));
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<Tensor> result;
        result.reserve(source.size());
        for (const Tensor& tensor : source) { result.push_back(peer_copy_after_sync(tensor, 1, 0)); }
        return result;
    }

    Tensor build_layer(int physical, std::uint32_t slot, Tensor cur, std::vector<Tensor>& blocks,
                       const std::vector<Tensor>* bank, const Tensor* bank_out) {
        LayerWeights& layer = layers[physical];
        LayerState& state = states[slot];
        const int device = layer.device;
        CUDA_CHECK(cudaSetDevice(device));
        const std::int32_t tokens = cur.ne[1];
        const Tensor input_attention = cur;
        std::array<Tensor, 4> messages;
        if (bank != nullptr) {
            Tensor query = alloc(device, DType::FP32, {H, 4});
            pack_loop_query(cross_q[device], cross_kv[device], physical - 6, query, stream(device));
            Tensor mixed = mix(device, *bank, *bank_out, query);
            for (int site = 0; site < 4; ++site) { messages[site] = extract(device, mixed, site); }
        }

        std::array<Tensor, 3> inputs{cur, cur, cur};
        if (physical > 0) {
            Tensor mixed = mix(device, blocks, cur, layer.attention_query);
            if (layer.full) {
                for (int part = 0; part < 3; ++part) { inputs[part] = extract(device, mixed, part); }
            } else {
                inputs[0] = inputs[1] = inputs[2] = extract(device, mixed, 0);
            }
        }
        if (bank != nullptr) {
            for (int part = 0; part < 3; ++part) {
                inputs[part] = control(device, inputs[part], messages[part], layer.loop[part]);
            }
        }
        if (!layer.full && bank == nullptr) {
            inputs[0] = rms(device, inputs[0], layer.norm);
            inputs[1] = inputs[2] = inputs[0];
        } else if (!layer.full) {
            for (Tensor& input : inputs) { input = rms(device, input, layer.norm); }
        } else {
            inputs[0] = rms(device, inputs[0], layer.q_input_norm);
            inputs[1] = rms(device, inputs[1], layer.k_input_norm);
            inputs[2] = rms(device, inputs[2], layer.v_input_norm);
        }

        Tensor q = linear(device, inputs[0], layer.q, Q_SIZE).view({128, 24, tokens});
        Tensor k_flat = linear(device, inputs[1], layer.k, KV_SIZE);
        Tensor v_flat = linear(device, inputs[2], layer.v, KV_SIZE);
        Tensor r = linear(device, inputs[0], layer.rel, R_SIZE).view({16, 24, tokens});
        Tensor attention_gate = linear(device, inputs[0], layer.gate, Q_SIZE);
        Tensor sinks = linear(device, inputs[0], layer.sink, 24);
        Tensor k_dynamic;
        Tensor v_dynamic;
        if (layer.full) {
            k_dynamic = linear(device, inputs[1], layer.k_sconv_proj, 32);
            v_dynamic = linear(device, inputs[2], layer.v_sconv_proj, 32);
        }
        Tensor k_conv = alloc(device, DType::BF16, {KV_SIZE, tokens});
        Tensor v_conv = alloc(device, DType::BF16, {KV_SIZE, tokens});
        ops::inkling_conv1d(k_flat, layer.full ? layer.k_sconv_static : layer.k_sconv,
                            layer.full ? &k_dynamic : nullptr, state.k_conv, state.k_conv, k_conv,
                            stream(device));
        ops::inkling_conv1d(v_flat, layer.full ? layer.v_sconv_static : layer.v_sconv,
                            layer.full ? &v_dynamic : nullptr, state.v_conv, state.v_conv, v_conv,
                            stream(device));
        Tensor k = k_conv.view({128, 8, tokens});
        Tensor v = v_conv.view({128, 8, tokens});
        Tensor clipped_q = alloc(device, DType::BF16, {128, 24, tokens});
        Tensor clipped_k = alloc(device, DType::BF16, {128, 8, tokens});
        Tensor clipped_r = alloc(device, DType::BF16, {16, 24, tokens});
        Tensor normed_v = alloc(device, DType::BF16, {128, 8, tokens});
        ops::l2_clip(q, &layer.q_clip, kQkL2Clip, clipped_q, stream(device));
        ops::l2_clip(k, nullptr, kQkL2Clip, clipped_k, stream(device));
        ops::l2_clip(r, &layer.rel_clip, kRelativeL2Clip, clipped_r, stream(device));
        ops::rmsnorm(v, ones[device], kRmsEpsilon, false, normed_v, stream(device));
        if (layer.full) {
            Tensor positions = alloc(device, DType::I32, {tokens});
            fill_positions(positions, length, stream(device));
            Tensor scaled_q = alloc(device, DType::BF16, {128, 24, tokens});
            Tensor scaled_r = alloc(device, DType::BF16, {16, 24, tokens});
            ops::inkling_log_position_scale(clipped_q, positions, kLogScaleFloor, kLogScaleAlpha,
                                            scaled_q, stream(device));
            ops::inkling_log_position_scale(clipped_r, positions, kLogScaleFloor, kLogScaleAlpha,
                                            scaled_r, stream(device));
            clipped_q = scaled_q;
            clipped_r = scaled_r;
        }
        append_kv(clipped_k.view({KV_SIZE, tokens}), normed_v.view({KV_SIZE, tokens}),
                  state.k_cache, state.v_cache, length, stream(device));
        const std::int32_t total_keys = length + tokens;
        Tensor k_history(state.k_cache.data, DType::BF16, {128, 8, total_keys});
        Tensor v_history(state.v_cache.data, DType::BF16, {128, 8, total_keys});
        Tensor attention = alloc(device, DType::BF16, {128, 24, tokens});
        ops::inkling_attention_dense(clipped_q, k_history, v_history, clipped_r, layer.rel_proj,
                                     sinks, kAttentionScale, layer.full ? 0 : kSlidingWindow,
                                     attention, stream(device));
        Tensor attention_flat = attention.view({Q_SIZE, tokens});
        ops::sigmoid_mul(attention_gate, attention_flat, stream(device));
        Tensor attention_out = linear(device, attention_flat, layer.o, H);
        Tensor attention_conv = alloc(device, DType::BF16, {H, tokens});
        ops::inkling_conv1d(attention_out, layer.sconv, nullptr, state.attention_conv,
                            state.attention_conv, attention_conv, stream(device));
        cur = attention_conv;
        if (physical % kAttentionResidualBlockSize == 0) {
            blocks.push_back(input_attention);
        } else {
            ops::residual_add(input_attention, cur, stream(device));
        }
        const Tensor input_ffn = cur;
        Tensor ffn = extract(device, mix(device, blocks, cur, layer.mlp_query), 0);
        if (bank != nullptr) { ffn = control(device, ffn, messages[3], layer.loop[3]); }
        ffn = rms(device, ffn, layer.moe_norm);
        Tensor selected = alloc(device, DType::I32, {K, tokens});
        Tensor route_weights = alloc(device, DType::FP32, {K, tokens});
        ops::inkling_router(ffn, layer.router, layer.router_bias, kRouteScale, *workspace[device],
                            selected, route_weights, stream(device));
        Tensor moe = alloc(device, DType::BF16, {H, tokens});
        ops::inkling_moe_prepared(ffn, layer.experts13, layer.experts2, selected, route_weights,
                                  kGateSoftcap, kUpSoftcap, layer.expert_views, *workspace[device],
                                  moe, stream(device));
        Tensor ffn_conv = alloc(device, DType::BF16, {H, tokens});
        ops::inkling_conv1d(moe, layer.moe_sconv, nullptr, state.ffn_conv, state.ffn_conv, ffn_conv,
                            stream(device));
        cur = ffn_conv;
        ops::residual_add(input_ffn, cur, stream(device));
        return cur;
    }

    int forward(const std::vector<int>& tokens) {
        if (tokens.empty() || length + static_cast<int>(tokens.size()) > max_context) {
            throw std::invalid_argument("Inkling forward chunk is empty or exceeds max_context");
        }
        const std::int32_t count = static_cast<std::int32_t>(tokens.size());
        for (int device = 0; device < 2; ++device) { workspace[device]->reset(); }
        CUDA_CHECK(cudaSetDevice(0));
        Tensor ids = alloc(0, DType::I32, {count});
        CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data(), tokens.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream(0)));
        Tensor cur = alloc(0, DType::BF16, {H, count});
        ops::embedding(ids, embedding, cur, stream(0));
        cur = rms(0, cur, embedding_norm);
        std::vector<Tensor> blocks;
        for (int physical = 0; physical < 6; ++physical) {
            cur = build_layer(physical, state_slot(physical, 0), cur, blocks, nullptr, nullptr);
        }
        const Tensor prelude = cur;
        const std::vector<Tensor> prelude_blocks = blocks;
        std::vector<Tensor> bank0;
        std::vector<Tensor> bank1;
        Tensor bank_out0;
        Tensor bank_out1;
        int current_device = 0;
        for (std::uint32_t pass = 0; pass < kLoopPasses; ++pass) {
            if (pass > 0) {
                bank1.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out1 = cur;
                bank0 = copy_bank_to_zero(bank1);
                bank_out0 = peer_copy_after_sync(bank_out1, 1, 0);
                cur = prelude;
                blocks = prelude_blocks;
                current_device = 0;
            }
            for (int physical = 6; physical < 18; ++physical) {
                const int destination = physical < 16 ? 0 : 1;
                transfer(cur, blocks, current_device, destination);
                current_device = destination;
                cur = build_layer(physical, state_slot(physical, pass), cur, blocks,
                    pass == 0 ? nullptr : (destination == 0 ? &bank0 : &bank1),
                    pass == 0 ? nullptr : (destination == 0 ? &bank_out0 : &bank_out1));
            }
        }
        for (int physical = 18; physical < 24; ++physical) {
            transfer(cur, blocks, current_device, 1);
            current_device = 1;
            cur = build_layer(physical, state_slot(physical, 0), cur, blocks, nullptr, nullptr);
        }
        CUDA_CHECK(cudaSetDevice(1));
        std::vector<Tensor> last_blocks;
        last_blocks.reserve(blocks.size());
        for (const Tensor& block : blocks) { last_blocks.push_back(block.slice(1, count - 1, 1)); }
        const Tensor last = cur.slice(1, count - 1, 1);
        Tensor hidden = extract(1, mix(1, last_blocks, last, output_query), 0);
        hidden = rms(1, hidden, final_norm);
        Tensor logits = linear(1, hidden, output_head, V);
        Tensor capped = alloc(1, DType::BF16, {V, 1});
        ops::inkling_softcap(logits, kLogitSoftcap, capped, stream(1));
        Tensor next = alloc(1, DType::I32, {1});
        ops::argmax(capped, next, static_cast<std::int32_t>(kUnpaddedVocab), stream(1));
        int result = 0;
        CUDA_CHECK(cudaMemcpyAsync(&result, next.data, sizeof(result), cudaMemcpyDeviceToHost,
                                   stream(1)));
        CUDA_CHECK(cudaStreamSynchronize(stream(1)));
        length += count;
        return result;
    }

    void ensure_tokenizer() {
        if (tokenizer == nullptr) {
            tokenizer = std::make_unique<qwen3_6::frontend_internal::Tokenizer>(
                qwen3_6::frontend_internal::TokenizerResources{
                    .tokenizer_json = tokenizer_json,
                    .tokenizer_config_json = tokenizer_config_json,
                    .generation_config_json = generation_config_json,
                });
        }
    }

    std::vector<int> encode_text(const std::string& text) {
        ensure_tokenizer();
        return tokenizer->encode(text);
    }

    std::string decode_text(const std::vector<int>& tokens) {
        ensure_tokenizer();
        return tokenizer->decode(tokens, qwen3_6::frontend_internal::DecodeOptions{
                                             .skip_special_tokens = false});
    }

    bool is_stop(int token) const {
        if (tokenizer == nullptr) { return false; }
        const auto& stops = tokenizer->default_stop_token_ids();
        return std::find(stops.begin(), stops.end(), token) != stops.end();
    }

    void reset_state() {
        length = 0;
        for (int device = 0; device < 2; ++device) {
            CUDA_CHECK(cudaSetDevice(device));
            for (LayerState& state : states) {
                if (state.device != device) { continue; }
                for (Tensor* tensor : {&state.k_cache, &state.v_cache, &state.k_conv, &state.v_conv,
                                       &state.attention_conv, &state.ffn_conv}) {
                    CUDA_CHECK(cudaMemsetAsync(tensor->data, 0, tensor->bytes(), stream(device)));
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream(device)));
        }
    }
};

Runtime::Runtime(const std::string& artifact_path, int max_context, bool enable_vision)
    : impl_(std::make_unique<Impl>(artifact_path, max_context)) {
    if (enable_vision) { throw std::invalid_argument("Pipeline vision is unavailable; use TP2 vision mode"); }
}

Runtime::~Runtime() = default;

void Runtime::reset() { impl_->reset_state(); }

std::vector<int> Runtime::encode_text(const std::string& text) { return impl_->encode_text(text); }

std::string Runtime::decode_text(const std::vector<int>& tokens) { return impl_->decode_text(tokens); }

double Runtime::last_prefill_seconds() const { return impl_->prefill_seconds; }

double Runtime::last_decode_seconds() const { return impl_->decode_seconds; }

std::vector<int> Runtime::generate(const std::vector<int>& prompt_tokens, int max_new_tokens) {
    if (prompt_tokens.empty() || max_new_tokens <= 0 ||
        prompt_tokens.size() + static_cast<std::size_t>(max_new_tokens) >
            static_cast<std::size_t>(impl_->max_context)) {
        throw std::invalid_argument("Inkling generate requires a nonempty prompt and fitting output budget");
    }
    reset();
    std::vector<int> output;
    output.reserve(max_new_tokens);
    const auto prefill_start = std::chrono::steady_clock::now();
    int next = impl_->forward(prompt_tokens);
    impl_->prefill_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_start).count();
    output.push_back(next);
    const auto decode_start = std::chrono::steady_clock::now();
    impl_->decode_seconds = 0.0;
    if (impl_->is_stop(next)) { return output; }
    for (int index = 1; index < max_new_tokens; ++index) {
        next = impl_->forward(std::vector<int>{next});
        output.push_back(next);
        if (impl_->is_stop(next)) { break; }
    }
    impl_->decode_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
    return output;
}

std::vector<int> Runtime::generate_with_image(const std::vector<int>&, int,
    const media::decode::Image&, int) {
    throw std::invalid_argument("Pipeline vision is unavailable; use TP2 vision mode");
}

} // namespace ninfer::targets::inkling2_mini::detail
