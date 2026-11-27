#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::targets::inkling2_mini::detail {

// Physical weights are shared by the repeated middle-cycle passes. KV and short-convolution state
// are not: every pass owns a separate logical slot, exactly as llama.cpp's inkling2-loop does.
inline constexpr std::uint32_t kPhysicalLayers = 24;
inline constexpr std::uint32_t kLoopCoreBegin  = 6;
inline constexpr std::uint32_t kLoopCoreEnd    = 18;
inline constexpr std::uint32_t kLoopPasses     = 3;
inline constexpr std::uint32_t kLoopCoreLayers = kLoopCoreEnd - kLoopCoreBegin;
inline constexpr std::uint32_t kStateSlots =
    kPhysicalLayers + (kLoopPasses - 1) * kLoopCoreLayers;
inline constexpr std::uint32_t kInvalidStateSlot =
    std::numeric_limits<std::uint32_t>::max();

inline constexpr std::uint32_t kHidden             = 2304;
inline constexpr std::uint32_t kQueryHeads         = 24;
inline constexpr std::uint32_t kKvHeads            = 8;
inline constexpr std::uint32_t kHeadDim            = 128;
inline constexpr std::uint32_t kRelativePositions  = 16;
inline constexpr std::uint32_t kRelativeExtent     = 1024;
inline constexpr std::uint32_t kSlidingWindow      = 1024;
inline constexpr std::uint32_t kConvolutionTaps    = 4;
inline constexpr std::uint32_t kExperts            = 128;
inline constexpr std::uint32_t kActiveExperts      = 8;
inline constexpr std::uint32_t kExpertHidden       = 1152;
inline constexpr std::uint32_t kVocab              = 201024;
inline constexpr std::uint32_t kUnpaddedVocab      = 200058;
inline constexpr float         kGateSoftcap        = 4.0F;
inline constexpr float         kUpSoftcap          = 25.0F;
inline constexpr float         kLogitSoftcap       = 30.0F;
inline constexpr float         kQkL2Clip           = 22.627416997969522F;
inline constexpr float         kRelativeL2Clip     = 8.0F;
inline constexpr float         kRouteScale         = 8.0F;
inline constexpr std::uint32_t kAttentionResidualBlockSize = 6;
inline constexpr std::uint32_t kMaximumDepthStates =
    kPhysicalLayers / kAttentionResidualBlockSize + 1;

[[nodiscard]] constexpr bool is_core_layer(std::uint32_t physical_layer) {
    return physical_layer >= kLoopCoreBegin && physical_layer < kLoopCoreEnd;
}

[[nodiscard]] constexpr bool is_full_attention(std::uint32_t physical_layer) {
    return physical_layer == 5 || physical_layer == 11 || physical_layer == 17 ||
           physical_layer == 23;
}

[[nodiscard]] constexpr bool has_dynamic_kv_convolution(std::uint32_t physical_layer) {
    return is_full_attention(physical_layer);
}

[[nodiscard]] constexpr bool has_bf16_experts(std::uint32_t physical_layer) {
    return physical_layer >= 20 && physical_layer < kPhysicalLayers;
}

[[nodiscard]] constexpr std::uint32_t state_slot(std::uint32_t physical_layer,
                                                 std::uint32_t pass) {
    if (physical_layer >= kPhysicalLayers || pass >= kLoopPasses ||
        (pass != 0 && !is_core_layer(physical_layer))) {
        return kInvalidStateSlot;
    }
    return pass == 0
               ? physical_layer
               : kPhysicalLayers + (pass - 1) * kLoopCoreLayers +
                     (physical_layer - kLoopCoreBegin);
}

[[nodiscard]] constexpr std::uint32_t physical_layer_for_slot(std::uint32_t slot) {
    if (slot >= kStateSlots) { return kPhysicalLayers; }
    return slot < kPhysicalLayers
               ? slot
               : kLoopCoreBegin + (slot - kPhysicalLayers) % kLoopCoreLayers;
}

[[nodiscard]] constexpr std::uint32_t pass_for_slot(std::uint32_t slot) {
    if (slot >= kStateSlots) { return kLoopPasses; }
    return slot < kPhysicalLayers ? 0 : 1 + (slot - kPhysicalLayers) / kLoopCoreLayers;
}

enum class LoopMessage : std::uint32_t { Attention = 0, K = 1, V = 2, Mlp = 3 };

struct LoopQuerySource {
    bool kv;
    std::uint32_t row;
};

inline constexpr std::uint32_t kCrossPassQueryRows = 24;

// Native checkpoint order is q[2*i], kv[2*i], kv[2*i+1], q[2*i+1] for attention, K, V and MLP.
[[nodiscard]] constexpr LoopQuerySource loop_query_source(std::uint32_t physical_layer,
                                                          LoopMessage message) {
    if (!is_core_layer(physical_layer)) { return {false, kCrossPassQueryRows}; }
    const std::uint32_t pair = 2 * (physical_layer - kLoopCoreBegin);
    switch (message) {
    case LoopMessage::Attention: return {false, pair};
    case LoopMessage::K:         return {true, pair};
    case LoopMessage::V:         return {true, pair + 1};
    case LoopMessage::Mlp:       return {false, pair + 1};
    }
    return {false, kCrossPassQueryRows};
}

struct LayerExecution {
    std::uint32_t physical_layer;
    std::uint32_t pass;
    std::uint32_t state_slot;
};

using ExecutionSchedule = std::array<LayerExecution, kStateSlots>;

[[nodiscard]] constexpr ExecutionSchedule make_execution_schedule() {
    ExecutionSchedule schedule{};
    std::size_t cursor = 0;
    for (std::uint32_t layer = 0; layer < kLoopCoreBegin; ++layer) {
        schedule[cursor++] = {layer, 0, state_slot(layer, 0)};
    }
    for (std::uint32_t pass = 0; pass < kLoopPasses; ++pass) {
        for (std::uint32_t layer = kLoopCoreBegin; layer < kLoopCoreEnd; ++layer) {
            schedule[cursor++] = {layer, pass, state_slot(layer, pass)};
        }
    }
    for (std::uint32_t layer = kLoopCoreEnd; layer < kPhysicalLayers; ++layer) {
        schedule[cursor++] = {layer, 0, state_slot(layer, 0)};
    }
    return schedule;
}

[[nodiscard]] constexpr bool execution_schedule_is_valid() {
    constexpr auto schedule = make_execution_schedule();
    std::array<bool, kStateSlots> seen_slots{};
    std::array<bool, kPhysicalLayers> seen_weights{};
    for (const LayerExecution& step : schedule) {
        if (step.physical_layer >= kPhysicalLayers || step.pass >= kLoopPasses ||
            step.state_slot >= kStateSlots || seen_slots[step.state_slot] ||
            state_slot(step.physical_layer, step.pass) != step.state_slot ||
            physical_layer_for_slot(step.state_slot) != step.physical_layer ||
            pass_for_slot(step.state_slot) != step.pass) {
            return false;
        }
        seen_slots[step.state_slot] = true;
        seen_weights[step.physical_layer] = true;
    }
    for (const bool seen : seen_slots) {
        if (!seen) { return false; }
    }
    for (const bool seen : seen_weights) {
        if (!seen) { return false; }
    }
    return true;
}

static_assert(kStateSlots == 48);
static_assert(kMaximumDepthStates == 5);
static_assert(execution_schedule_is_valid());

} // namespace ninfer::targets::inkling2_mini::detail
