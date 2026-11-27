#include "targets/inkling2_mini/impl/load/bindings.h"

#include "artifact/typed_binding.h"
#include "targets/inkling2_mini/impl/load/shard.h"

#include <bit>
#include <cstdint>
#include <cmath>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets::inkling2_mini::detail {
namespace {

float read_nvfp4_divisor(artifact::Binder& binder, artifact::ObjectHandle handle) {
    const auto* tensor = std::get_if<artifact::TensorDescriptor>(&binder.descriptor(handle));
    if (tensor == nullptr || tensor->format != artifact::NumericFormat::NVFP4) {
        throw artifact::ArtifactError("Inkling divisor requested for a non-NVFP4 object");
    }
    const auto geometry = artifact::block_scale_geometry(tensor->format, tensor->shape);
    const auto payload = binder.payload(handle).data;
    if (geometry.weight_divisor_offset + 4 > payload.size()) {
        throw artifact::ArtifactError("Inkling NVFP4 divisor is outside the payload");
    }
    const std::byte* word = payload.data() + geometry.weight_divisor_offset;
    const std::uint32_t bits = std::to_integer<std::uint32_t>(word[0]) |
                               (std::to_integer<std::uint32_t>(word[1]) << 8U) |
                               (std::to_integer<std::uint32_t>(word[2]) << 16U) |
                               (std::to_integer<std::uint32_t>(word[3]) << 24U);
    const float divisor = std::bit_cast<float>(bits);
    if (!std::isfinite(divisor) || divisor <= 0.0F) {
        throw artifact::ArtifactError("Inkling NVFP4 divisor must be finite and positive");
    }
    return divisor;
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder,
                               artifact::TensorPlacement text_placement,
                               InklingDeviceMode device_mode,
                               bool vision_on_host) {
    using artifact::NumericFormat;
    using artifact::TensorPlacement;

    if (text_placement != TensorPlacement::Device &&
        text_placement != TensorPlacement::ValidateOnly) {
        throw std::invalid_argument("Inkling text tensors must be device-resident or validate-only");
    }
    if (text_placement == TensorPlacement::Device && binder.device_count() != 1 &&
        binder.device_count() != 2) {
        throw std::invalid_argument("Inkling materialization supports one device or execution TP2");
    }
    if (text_placement == TensorPlacement::Device && device_mode == InklingDeviceMode::Pipeline16 &&
        binder.device_count() != 2) {
        throw std::invalid_argument("Inkling Pipeline16 materialization requires two devices");
    }

    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    auto shard_plans = std::make_shared<std::map<std::string, artifact::ShardPlacement, std::less<>>>();
    if (text_placement == TensorPlacement::Device && binder.device_count() == 2 &&
        device_mode == InklingDeviceMode::TensorParallel) {
        binder.set_shard_resolver([shard_plans](std::string_view name) {
            const auto found = shard_plans->find(name);
            if (found == shard_plans->end()) {
                throw artifact::ArtifactError("missing Inkling shard plan for " + std::string(name));
            }
            return found->second;
        });
    }
    if (text_placement == TensorPlacement::Device && binder.device_count() == 2 &&
        device_mode == InklingDeviceMode::Pipeline16) {
        binder.set_device_mask_resolver(pipeline16_device_mask);
    }
    out.frontend = {
        .tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json"),
        .tokenizer_config_json =
            artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json"),
        .chat_template_jinja =
            artifact::bind_raw_resource(binder, "frontend/chat_template.jinja"),
        .generation_config_json =
            artifact::bind_raw_resource(binder, "frontend/generation_config.json"),
        .preprocessor_config_json =
            artifact::bind_raw_resource(binder, "frontend/preprocessor_config.json"),
    };

    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        if (text_placement == TensorPlacement::ValidateOnly || binder.device_count() == 1 ||
            device_mode == InklingDeviceMode::Pipeline16) {
            return artifact::bind_tensor(binder, name, format, shape, text_placement);
        }
        const auto layout = format == NumericFormat::NVFP4
                                ? artifact::StorageLayout::BlockScaleK16M128x4V1
                                : format == NumericFormat::FP8_E4M3FN_ROW_BF16S
                                      ? artifact::StorageLayout::RowScaleV1
                                      : artifact::StorageLayout::ContiguousLeV1;
        const std::span<const std::uint64_t> dimensions(shape.begin(), shape.size());
        const auto handle = binder.require_tensor(name, format, layout, dimensions);
        auto placement = shard_placement_for(shard_kind_for(name), dimensions, binder.device_count());
        shard_plans->insert_or_assign(std::string(name), std::move(placement));
        binder.materialize_on_device(handle);
        return handle;
    };
    const auto bind_vision = [&](std::string_view name, NumericFormat format,
                                 std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape,
                                     vision_on_host ? TensorPlacement::Host : TensorPlacement::ValidateOnly);
    };

    out.token_embedding = bind("text/embed", NumericFormat::BF16, {kVocab, kHidden});
    out.embedding_norm  = bind("text/embed_norm", NumericFormat::BF16, {kHidden});
    out.final_norm      = bind("text/final_norm", NumericFormat::BF16, {kHidden});
    out.output_head     = bind("text/output_head", NumericFormat::BF16, {kVocab, kHidden});
    for (std::size_t index = 0; index < out.attention_queries.size(); ++index) {
        const std::uint64_t width =
            index == 4 || index == 10 || index == 16 || index == 22 ? 3 : 1;
        out.attention_queries[index] =
            bind("text/attn_res/attn_query/" + std::to_string(index), NumericFormat::FP32,
                 {width * kHidden});
    }
    for (std::size_t index = 0; index < out.mlp_queries.size(); ++index) {
        out.mlp_queries[index] = bind("text/attn_res/mlp_query/" + std::to_string(index),
                                      NumericFormat::FP32, {kHidden});
    }
    out.output_query = bind("text/attn_res/output_query", NumericFormat::FP32, {kHidden});
    out.cross_pass_query =
        bind("text/cross_pass/query", NumericFormat::FP32, {kPhysicalLayers, kHidden});
    out.cross_pass_kv_query =
        bind("text/cross_pass/kv_query", NumericFormat::FP32, {kPhysicalLayers, kHidden});

    for (std::size_t layer = 0; layer < kPhysicalLayers; ++layer) {
        TextLayerPlan& target = out.text_layers[layer];
        target.full_attention   = is_full_attention(static_cast<std::uint32_t>(layer));
        target.loop_core        = is_core_layer(static_cast<std::uint32_t>(layer));
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        AttentionPlan& attention = target.attention;
        attention.q       = bind(prefix + "attn/q", NumericFormat::BF16,
                                 {kQueryHeads * kHeadDim, kHidden});
        attention.k       = bind(prefix + "attn/k", NumericFormat::BF16,
                                 {kKvHeads * kHeadDim, kHidden});
        attention.v       = bind(prefix + "attn/v", NumericFormat::BF16,
                                 {kKvHeads * kHeadDim, kHidden});
        attention.o       = bind(prefix + "attn/o", NumericFormat::BF16,
                                 {kHidden, kQueryHeads * kHeadDim});
        attention.rel     = bind(prefix + "attn/rel", NumericFormat::BF16,
                                 {kRelativePositions * kQueryHeads, kHidden});
        attention.gate    = bind(prefix + "attn/gate", NumericFormat::BF16,
                                 {kQueryHeads * kHeadDim, kHidden});
        attention.sink    = bind(prefix + "attn/sink", NumericFormat::BF16, {kQueryHeads, kHidden});
        attention.q_clip  = bind(prefix + "attn/q_clip", NumericFormat::BF16,
                                 {kQueryHeads, kHeadDim});
        attention.rel_clip = bind(prefix + "attn/rel_clip", NumericFormat::BF16,
                                  {kQueryHeads, kRelativePositions});
        attention.rel_proj = bind(prefix + "attn/rel_proj", NumericFormat::BF16,
                                  {kRelativePositions, kKvHeads * kHeadDim});
        attention.sconv    = bind(prefix + "attn/sconv", NumericFormat::BF16,
                                  {kHidden, kConvolutionTaps});
        if (target.full_attention) {
            attention.q_input_norm =
                bind(prefix + "attn/q_input_norm", NumericFormat::BF16, {kHidden});
            attention.k_input_norm =
                bind(prefix + "attn/k_input_norm", NumericFormat::BF16, {kHidden});
            attention.v_input_norm =
                bind(prefix + "attn/v_input_norm", NumericFormat::BF16, {kHidden});
            attention.k_sconv_static =
                bind(prefix + "attn/k_sconv_static", NumericFormat::BF16,
                     {kConvolutionTaps, kKvHeads * kHeadDim});
            attention.k_sconv_proj =
                bind(prefix + "attn/k_sconv_proj", NumericFormat::BF16,
                     {kConvolutionTaps * kKvHeads, kHidden});
            attention.v_sconv_static =
                bind(prefix + "attn/v_sconv_static", NumericFormat::BF16,
                     {kConvolutionTaps, kKvHeads * kHeadDim});
            attention.v_sconv_proj =
                bind(prefix + "attn/v_sconv_proj", NumericFormat::BF16,
                     {kConvolutionTaps * kKvHeads, kHidden});
        } else {
            attention.norm = bind(prefix + "attn/norm", NumericFormat::BF16, {kHidden});
            attention.k_sconv = bind(prefix + "attn/k_sconv", NumericFormat::BF16,
                                     {kKvHeads * kHeadDim, kConvolutionTaps});
            attention.v_sconv = bind(prefix + "attn/v_sconv", NumericFormat::BF16,
                                     {kKvHeads * kHeadDim, kConvolutionTaps});
        }

        MoePlan& moe = target.moe;
        moe.norm  = bind(prefix + "mlp/norm", NumericFormat::BF16, {kHidden});
        moe.sconv = bind(prefix + "mlp/sconv", NumericFormat::BF16,
                         {kHidden, kConvolutionTaps});
        moe.gate  = bind(prefix + "mlp/gate", NumericFormat::BF16, {kExperts, kHidden});
        moe.gate_bias = bind(prefix + "mlp/gate_bias", NumericFormat::FP32, {kExperts});
        const NumericFormat expert_format = has_bf16_experts(static_cast<std::uint32_t>(layer))
            ? NumericFormat::FP8_E4M3FN_ROW_BF16S : NumericFormat::NVFP4;
        for (std::size_t expert = 0; expert < kExperts; ++expert) {
            const std::string expert_prefix =
                prefix + "mlp/experts/" + std::to_string(expert) + "/";
            moe.experts[expert].w13 =
                bind(expert_prefix + "w13", expert_format, {kHidden, 2 * kExpertHidden});
            moe.experts[expert].w2 =
                bind(expert_prefix + "w2", expert_format, {kHidden, kExpertHidden});
            if (expert_format == NumericFormat::FP8_E4M3FN_ROW_BF16S) {
                moe.experts[expert].w13_mxscale =
                    bind(expert_prefix + "w13_mxscale", NumericFormat::BF16,
                         {kHidden, kExpertHidden / 16});
                moe.experts[expert].w2_mxscale =
                    bind(expert_prefix + "w2_mxscale", NumericFormat::BF16,
                         {kHidden, kExpertHidden / 32});
            }
            if (expert_format == NumericFormat::NVFP4) {
                moe.experts[expert].w13_divisor =
                    read_nvfp4_divisor(binder, moe.experts[expert].w13);
                moe.experts[expert].w2_divisor =
                    read_nvfp4_divisor(binder, moe.experts[expert].w2);
            }
        }

        if (target.loop_core) {
            for (std::size_t site = 0; site < target.loop.size(); ++site) {
                static constexpr std::array<std::string_view, 4> sites = {"attn", "k", "v", "mlp"};
                const std::string loop_prefix = prefix + "loop/" + std::string(sites[site]) + "/";
                target.loop[site].down =
                    bind(loop_prefix + "down", NumericFormat::BF16, {64, 2 * kHidden});
                target.loop[site].gain =
                    bind(loop_prefix + "gain", NumericFormat::BF16, {kHidden, 64});
                target.loop[site].forget =
                    bind(loop_prefix + "forget", NumericFormat::BF16, {kHidden, 64});
            }
        }
    }

    // The first milestone is text-only, but consume and validate the complete vision inventory so
    // Binder::finish still proves that the artifact has no unknown or omitted objects.
    out.vision.class_embedding = bind_vision("vision/class_embedding", NumericFormat::BF16, {1024});
    out.vision.cls_position_embedding = bind_vision("vision/cls_position_embedding", NumericFormat::BF16, {1024});
    out.vision.patch_embedding = bind_vision("vision/patch_embedding", NumericFormat::BF16, {1024, 1200});
    out.vision.position_embedding = bind_vision("vision/position_embedding", NumericFormat::BF16, {1024, 1024});
    out.vision.ln_pre_weight = bind_vision("vision/ln_pre/weight", NumericFormat::BF16, {1024});
    out.vision.ln_pre_bias = bind_vision("vision/ln_pre/bias", NumericFormat::BF16, {1024});
    out.vision.proj_in = bind_vision("vision/proj_in", NumericFormat::BF16, {4096, 4096});
    out.vision.proj_out = bind_vision("vision/proj_out", NumericFormat::BF16, {kHidden, 4096});
    for (std::size_t layer = 0; layer < 23; ++layer) {
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        VisionLayerPlan& target = out.vision.layers[layer];
        target.ln1_weight = bind_vision(prefix + "ln_1/weight", NumericFormat::BF16, {1024});
        target.ln1_bias = bind_vision(prefix + "ln_1/bias", NumericFormat::BF16, {1024});
        target.ln2_weight = bind_vision(prefix + "ln_2/weight", NumericFormat::BF16, {1024});
        target.ln2_bias = bind_vision(prefix + "ln_2/bias", NumericFormat::BF16, {1024});
        target.ls1 = bind_vision(prefix + "ls_1/gamma", NumericFormat::BF16, {1024});
        target.ls2 = bind_vision(prefix + "ls_2/gamma", NumericFormat::BF16, {1024});
        target.q_weight = bind_vision(prefix + "q_proj/weight", NumericFormat::BF16, {1024, 1024});
        target.q_bias = bind_vision(prefix + "q_proj/bias", NumericFormat::BF16, {1024});
        target.k_weight = bind_vision(prefix + "k_proj/weight", NumericFormat::BF16, {1024, 1024});
        target.k_bias = bind_vision(prefix + "k_proj/bias", NumericFormat::BF16, {1024});
        target.v_weight = bind_vision(prefix + "v_proj/weight", NumericFormat::BF16, {1024, 1024});
        target.v_bias = bind_vision(prefix + "v_proj/bias", NumericFormat::BF16, {1024});
        target.out_weight = bind_vision(prefix + "out_proj/weight", NumericFormat::BF16, {1024, 1024});
        target.out_bias = bind_vision(prefix + "out_proj/bias", NumericFormat::BF16, {1024});
        target.c_proj_weight = bind_vision(prefix + "mlp/c_proj/weight", NumericFormat::BF16, {1024, 4096});
        target.c_proj_bias = bind_vision(prefix + "mlp/c_proj/bias", NumericFormat::BF16, {1024});
        target.c_fc_weight = bind_vision(prefix + "mlp/c_fc/weight", NumericFormat::BF16, {4096, 1024});
        target.c_fc_bias = bind_vision(prefix + "mlp/c_fc/bias", NumericFormat::BF16, {4096});
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

} // namespace ninfer::targets::inkling2_mini::detail
