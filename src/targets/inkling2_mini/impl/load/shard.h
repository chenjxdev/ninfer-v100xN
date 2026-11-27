#pragma once

#include "artifact/binder.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::targets::inkling2_mini::detail {

enum class InklingShardKind : std::uint8_t { Replicated, Rows, Columns };

[[nodiscard]] InklingShardKind shard_kind_for(std::string_view name);

[[nodiscard]] artifact::ShardPlacement shard_placement_for(
    InklingShardKind kind, std::span<const std::uint64_t> shape, int device_count);

// Layers 0..15 live on device 0; layers 16..23 and the LM head live on device 1.
[[nodiscard]] std::uint32_t pipeline16_device_mask(std::string_view name);

} // namespace ninfer::targets::inkling2_mini::detail
