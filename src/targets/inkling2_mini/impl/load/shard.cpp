#include "targets/inkling2_mini/impl/load/shard.h"

#include <stdexcept>
#include <string>

namespace ninfer::targets::inkling2_mini::detail {

InklingShardKind shard_kind_for(std::string_view name) {
    if (name == "text/embed" || name == "text/output_head") {
        return InklingShardKind::Replicated;
    }
    if (!name.starts_with("text/layers/")) {
        if (name == "text/embed_norm" || name == "text/final_norm" ||
            name.starts_with("text/attn_res/") || name.starts_with("text/cross_pass/")) {
            return InklingShardKind::Replicated;
        }
        throw std::invalid_argument("Inkling TP2 shard map: unknown global object " + std::string(name));
    }
    const std::size_t layer_end = name.find('/', 12);
    if (layer_end == std::string_view::npos || layer_end + 1 >= name.size()) {
        throw std::invalid_argument("Inkling TP2 shard map: invalid layer object " + std::string(name));
    }
    const std::string_view role = name.substr(layer_end + 1);
    if (role.starts_with("attn/")) {
        const std::string_view leaf = role.substr(5);
        if (leaf == "q" || leaf == "k" || leaf == "v" || leaf == "rel" || leaf == "gate" ||
            leaf == "sink" || leaf == "q_clip" || leaf == "rel_clip" ||
            leaf == "k_sconv" || leaf == "v_sconv" || leaf == "k_sconv_proj" ||
            leaf == "v_sconv_proj") {
            return InklingShardKind::Rows;
        }
        if (leaf == "o" || leaf == "k_sconv_static" || leaf == "v_sconv_static") {
            return InklingShardKind::Columns;
        }
        if (leaf == "rel_proj" || leaf == "sconv" || leaf == "norm" || leaf == "q_input_norm" ||
            leaf == "k_input_norm" || leaf == "v_input_norm") {
            return InklingShardKind::Replicated;
        }
    } else if (role.starts_with("mlp/")) {
        if (role == "mlp/sconv") { return InklingShardKind::Replicated; }
        if (role == "mlp/norm" || role == "mlp/gate" || role == "mlp/gate_bias") {
            return InklingShardKind::Replicated;
        }
        if (role.ends_with("/w13") || role.ends_with("/w13_mxscale")) {
            return InklingShardKind::Rows;
        }
        if (role.ends_with("/w2") || role.ends_with("/w2_mxscale")) {
            return InklingShardKind::Columns;
        }
    } else if (role.starts_with("loop/")) {
        if (role.ends_with("/down") || role.ends_with("/gain") || role.ends_with("/forget")) {
            return InklingShardKind::Replicated;
        }
    }
    throw std::invalid_argument("Inkling TP2 shard map: unknown layer role " + std::string(role));
}

artifact::ShardPlacement shard_placement_for(InklingShardKind kind,
                                             std::span<const std::uint64_t> shape,
                                             int device_count) {
    if (device_count != 2) {
        throw std::invalid_argument("Inkling shard placement requires exactly two devices");
    }
    artifact::ShardPlacement placement;
    placement.axis = kind == InklingShardKind::Rows
                         ? artifact::ShardAxis::Rows
                         : kind == InklingShardKind::Columns ? artifact::ShardAxis::Columns
                                                            : artifact::ShardAxis::Replicated;
    if (kind == InklingShardKind::Replicated) { return placement; }
    if (shape.size() != 2) {
        throw std::invalid_argument("Inkling row/column sharding requires a rank-two tensor");
    }
    const std::size_t axis = kind == InklingShardKind::Rows ? 0 : 1;
    const std::uint64_t extent = shape[axis];
    if (extent == 0 || extent % 2 != 0) {
        throw std::invalid_argument("Inkling shard extent must be positive and even");
    }
    const std::uint64_t half = extent / 2;
    placement.device_ranges[0].push_back({0, half});
    placement.device_ranges[1].push_back({half, half});
    return placement;
}

std::uint32_t pipeline16_device_mask(std::string_view name) {
    const auto layer_mask = [](std::uint32_t layer) {
        if (layer >= 24) { throw std::invalid_argument("Inkling pipeline layer is out of range"); }
        return layer < 16 ? 1U : 2U;
    };
    const auto parse_index = [](std::string_view value, std::string_view prefix) {
        if (!value.starts_with(prefix)) { throw std::invalid_argument("Inkling pipeline index prefix mismatch"); }
        const std::string tail(value.substr(prefix.size()));
        const std::size_t end = tail.find('/');
        return static_cast<std::uint32_t>(std::stoul(tail.substr(0, end)));
    };
    if (name.starts_with("text/layers/")) {
        return layer_mask(parse_index(name, "text/layers/"));
    }
    if (name == "text/embed" || name == "text/embed_norm") { return 1; }
    if (name == "text/output_head" || name == "text/final_norm" ||
        name == "text/attn_res/output_query") {
        return 2;
    }
    if (name.starts_with("text/attn_res/attn_query/")) {
        return layer_mask(parse_index(name, "text/attn_res/attn_query/") + 1);
    }
    if (name.starts_with("text/attn_res/mlp_query/")) {
        return layer_mask(parse_index(name, "text/attn_res/mlp_query/"));
    }
    if (name.starts_with("text/cross_pass/")) { return 3; }
    throw std::invalid_argument("Inkling pipeline placement: unknown object " + std::string(name));
}

} // namespace ninfer::targets::inkling2_mini::detail
