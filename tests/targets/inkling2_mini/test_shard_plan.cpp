#include "targets/inkling2_mini/impl/load/shard.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::targets::inkling2_mini::detail;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int expect_kind(std::string_view name, InklingShardKind expected) {
    return check(shard_kind_for(name) == expected, "unexpected shard kind");
}

int expect_ranges(std::string_view name, std::vector<std::uint64_t> shape,
                  artifact::ShardAxis expected_axis, std::uint64_t expected_half) {
    const auto placement = shard_placement_for(shard_kind_for(name), shape, 2);
    int failures = check(placement.axis == expected_axis, "unexpected shard axis");
    if (expected_axis == artifact::ShardAxis::Replicated) {
        return failures + check(placement.device_ranges[0].empty() &&
                                    placement.device_ranges[1].empty(),
                                "replicated object has shard ranges");
    }
    const auto& first = placement.device_ranges[0];
    const auto& second = placement.device_ranges[1];
    return failures + check(first.size() == 1 && second.size() == 1 && first[0].begin == 0 &&
                                first[0].count == expected_half && second[0].begin == expected_half &&
                                second[0].count == expected_half,
                            "incorrect TP2 shard ranges");
}

} // namespace

int main() {
    int failures = 0;
    failures += expect_ranges("text/embed", {201024, 2304}, artifact::ShardAxis::Replicated, 0);
    failures += expect_ranges("text/layers/0/attn/q", {3072, 2304}, artifact::ShardAxis::Rows, 1536);
    failures += expect_ranges("text/layers/0/attn/o", {2304, 3072}, artifact::ShardAxis::Columns, 1536);
    failures += expect_ranges("text/layers/5/attn/k_sconv_static", {4, 1024},
                              artifact::ShardAxis::Columns, 512);
    failures += expect_ranges("text/layers/0/mlp/experts/0/w13", {2304, 2304},
                              artifact::ShardAxis::Rows, 1152);
    failures += expect_ranges("text/layers/0/mlp/experts/0/w2", {2304, 1152},
                              artifact::ShardAxis::Columns, 576);
    failures += expect_ranges("text/layers/6/loop/attn/gain", {2304, 64},
                              artifact::ShardAxis::Replicated, 0);
    failures += expect_ranges("text/layers/6/loop/attn/down", {64, 4608},
                              artifact::ShardAxis::Replicated, 0);
    failures += expect_ranges("text/layers/0/attn/rel_proj", {16, 1024},
                              artifact::ShardAxis::Replicated, 0);
    failures += expect_ranges("text/layers/0/mlp/gate", {128, 2304},
                              artifact::ShardAxis::Replicated, 0);
    failures += expect_kind("text/cross_pass/query", InklingShardKind::Replicated);
    bool rejected = false;
    try { (void)shard_kind_for("text/layers/0/unknown"); } catch (const std::invalid_argument&) {
        rejected = true;
    }
    failures += check(rejected, "unknown object was silently replicated");
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
