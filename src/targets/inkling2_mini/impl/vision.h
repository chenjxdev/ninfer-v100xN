#pragma once

#include "media/decode/decode.h"
#include "targets/inkling2_mini/impl/load/bindings.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {

struct InklingVisionResult {
    int patch_width = 0;
    int patch_height = 0;
    int merged_count = 0;
    std::vector<std::uint16_t> embeddings;
};

class InklingVision {
public:
    InklingVision(artifact::MaterializedArtifact& artifact, const VisionBindingPlan& plan);
    ~InklingVision();

    InklingVision(const InklingVision&) = delete;
    InklingVision& operator=(const InklingVision&) = delete;

    [[nodiscard]] InklingVisionResult encode(const media::decode::Image& image) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::inkling2_mini::detail
