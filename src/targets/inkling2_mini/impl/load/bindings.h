#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "targets/inkling2_mini/impl/topology.h"

#include <array>
#include <cstddef>

namespace ninfer::targets::inkling2_mini::detail {

inline constexpr std::size_t kArtifactObjectCount = 8241;

enum class InklingDeviceMode : std::uint8_t { TensorParallel, Pipeline16 };

struct FrontendResourcePlan {
    artifact::ObjectHandle tokenizer_json;
    artifact::ObjectHandle tokenizer_config_json;
    artifact::ObjectHandle chat_template_jinja;
    artifact::ObjectHandle generation_config_json;
    artifact::ObjectHandle preprocessor_config_json;
};

struct ExpertPlan {
    artifact::ObjectHandle w13;
    artifact::ObjectHandle w2;
    artifact::ObjectHandle w13_mxscale;
    artifact::ObjectHandle w2_mxscale;
    float w13_divisor = 1.0F;
    float w2_divisor  = 1.0F;
};

struct LoopControllerPlan {
    artifact::ObjectHandle down;
    artifact::ObjectHandle gain;
    artifact::ObjectHandle forget;
};

struct AttentionPlan {
    artifact::ObjectHandle q;
    artifact::ObjectHandle k;
    artifact::ObjectHandle v;
    artifact::ObjectHandle o;
    artifact::ObjectHandle rel;
    artifact::ObjectHandle gate;
    artifact::ObjectHandle sink;
    artifact::ObjectHandle q_clip;
    artifact::ObjectHandle rel_clip;
    artifact::ObjectHandle rel_proj;
    artifact::ObjectHandle sconv;

    // Full-attention layers normalize Q/K/V independently and derive K/V convolution taps.
    artifact::ObjectHandle q_input_norm;
    artifact::ObjectHandle k_input_norm;
    artifact::ObjectHandle v_input_norm;
    artifact::ObjectHandle k_sconv_static;
    artifact::ObjectHandle k_sconv_proj;
    artifact::ObjectHandle v_sconv_static;
    artifact::ObjectHandle v_sconv_proj;

    // Sliding-window layers use one input norm and static K/V convolutions instead.
    artifact::ObjectHandle norm;
    artifact::ObjectHandle k_sconv;
    artifact::ObjectHandle v_sconv;
};

struct MoePlan {
    artifact::ObjectHandle norm;
    artifact::ObjectHandle sconv;
    artifact::ObjectHandle gate;
    artifact::ObjectHandle gate_bias;
    std::array<ExpertPlan, kExperts> experts;
};

struct TextLayerPlan {
    AttentionPlan attention;
    MoePlan moe;
    // Sites are attention, K, V and MLP, in that order. Only physical core layers own these.
    std::array<LoopControllerPlan, 4> loop;
    bool full_attention = false;
    bool loop_core      = false;
};

struct VisionLayerPlan {
    artifact::ObjectHandle ln1_weight, ln1_bias, ln2_weight, ln2_bias;
    artifact::ObjectHandle ls1, ls2;
    artifact::ObjectHandle q_weight, q_bias, k_weight, k_bias, v_weight, v_bias;
    artifact::ObjectHandle out_weight, out_bias;
    artifact::ObjectHandle c_proj_weight, c_proj_bias, c_fc_weight, c_fc_bias;
};

struct VisionBindingPlan {
    artifact::ObjectHandle class_embedding, cls_position_embedding, patch_embedding, position_embedding;
    artifact::ObjectHandle ln_pre_weight, ln_pre_bias, proj_in, proj_out;
    std::array<VisionLayerPlan, 23> layers;
};

struct BindingPlan {
    FrontendResourcePlan frontend;
    artifact::ObjectHandle token_embedding;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle final_norm;
    artifact::ObjectHandle output_head;
    std::array<artifact::ObjectHandle, 23> attention_queries;
    std::array<artifact::ObjectHandle, kPhysicalLayers> mlp_queries;
    artifact::ObjectHandle output_query;
    artifact::ObjectHandle cross_pass_query;
    artifact::ObjectHandle cross_pass_kv_query;
    std::array<TextLayerPlan, kPhysicalLayers> text_layers;
    VisionBindingPlan vision;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

// Binds the native v2 inventory without coalescing per-expert objects. ValidateOnly verifies every
// name, format, shape and placement without requiring 23.5 GB of device memory. Device supports a
// full TP1 copy or the execution-aware TP2 row/column shard map.
[[nodiscard]] ArtifactLoadPlan bind_artifact(
    artifact::Binder& binder,
    artifact::TensorPlacement text_placement = artifact::TensorPlacement::Device,
    InklingDeviceMode device_mode = InklingDeviceMode::TensorParallel,
    bool vision_on_host = false);

} // namespace ninfer::targets::inkling2_mini::detail
