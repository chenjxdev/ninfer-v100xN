#include "targets/inkling2_mini/impl/topology.h"

#include <array>
#include <cstdint>
#include <iostream>

namespace {

using namespace ninfer::targets::inkling2_mini::detail;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    constexpr auto schedule = make_execution_schedule();

    std::array<bool, kStateSlots> slots{};
    std::array<std::uint32_t, kPhysicalLayers> executions{};
    std::size_t cursor = 0;
    for (std::uint32_t layer = 0; layer < kLoopCoreBegin; ++layer) {
        const auto& step = schedule[cursor++];
        failures += check(step.physical_layer == layer && step.pass == 0 &&
                              step.state_slot == layer,
                          "prelude must execute once before the middle cycle");
    }
    for (std::uint32_t pass = 0; pass < kLoopPasses; ++pass) {
        for (std::uint32_t layer = kLoopCoreBegin; layer < kLoopCoreEnd; ++layer) {
            const auto& step = schedule[cursor++];
            failures += check(step.physical_layer == layer && step.pass == pass &&
                                  step.state_slot == state_slot(layer, pass),
                              "middle-cycle order or state mapping is wrong");
        }
    }
    for (std::uint32_t layer = kLoopCoreEnd; layer < kPhysicalLayers; ++layer) {
        const auto& step = schedule[cursor++];
        failures += check(step.physical_layer == layer && step.pass == 0 &&
                              step.state_slot == layer,
                          "coda must execute once after all middle-cycle passes");
    }

    for (const auto& step : schedule) {
        failures += check(!slots[step.state_slot], "two executions share mutable state");
        slots[step.state_slot] = true;
        ++executions[step.physical_layer];
    }
    for (const bool seen : slots) {
        failures += check(seen, "a logical state slot was not scheduled");
    }
    for (std::uint32_t layer = 0; layer < kPhysicalLayers; ++layer) {
        const std::uint32_t expected = is_core_layer(layer) ? kLoopPasses : 1;
        failures += check(executions[layer] == expected,
                          "physical weight execution count does not match the loop topology");
    }

    failures += check(state_slot(6, 1) == 24 && state_slot(17, 1) == 35 &&
                          state_slot(6, 2) == 36 && state_slot(17, 2) == 47,
                      "repeated core slots are not contiguous per pass");
    failures += check(state_slot(5, 1) == kInvalidStateSlot &&
                          state_slot(18, 1) == kInvalidStateSlot &&
                          state_slot(6, 3) == kInvalidStateSlot &&
                          state_slot(24, 0) == kInvalidStateSlot,
                      "invalid physical-layer/pass pairs were accepted");
    failures += check(physical_layer_for_slot(24) == 6 && pass_for_slot(24) == 1 &&
                          physical_layer_for_slot(47) == 17 && pass_for_slot(47) == 2 &&
                          physical_layer_for_slot(48) == kPhysicalLayers &&
                          pass_for_slot(48) == kLoopPasses,
                      "logical-to-physical slot lookup is wrong");

    const auto q0 = loop_query_source(6, LoopMessage::Attention);
    const auto k0 = loop_query_source(6, LoopMessage::K);
    const auto v0 = loop_query_source(6, LoopMessage::V);
    const auto m0 = loop_query_source(6, LoopMessage::Mlp);
    failures += check(!q0.kv && q0.row == 0 && k0.kv && k0.row == 0 && v0.kv && v0.row == 1 &&
                          !m0.kv && m0.row == 1,
                      "first core-layer cross-pass query order is wrong");
    const auto q11 = loop_query_source(17, LoopMessage::Attention);
    const auto k11 = loop_query_source(17, LoopMessage::K);
    const auto v11 = loop_query_source(17, LoopMessage::V);
    const auto m11 = loop_query_source(17, LoopMessage::Mlp);
    failures += check(!q11.kv && q11.row == 22 && k11.kv && k11.row == 22 && v11.kv &&
                          v11.row == 23 && !m11.kv && m11.row == 23,
                      "last core-layer cross-pass query order is wrong");
    failures += check(loop_query_source(5, LoopMessage::Attention).row == kCrossPassQueryRows &&
                          loop_query_source(18, LoopMessage::Mlp).row == kCrossPassQueryRows,
                      "non-core layers unexpectedly own loop-message queries");

    for (std::uint32_t layer = 0; layer < kPhysicalLayers; ++layer) {
        const bool full = layer == 5 || layer == 11 || layer == 17 || layer == 23;
        failures += check(is_full_attention(layer) == full &&
                              has_dynamic_kv_convolution(layer) == full,
                          "full-attention or dynamic-convolution pattern is wrong");
        failures += check(has_bf16_experts(layer) == (layer >= 20),
                          "NVFP4/BF16 expert boundary is wrong");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
