#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <array>
#include <memory>
#include <string>

namespace ninfer::targets::inkling2_mini::detail {

// The official three-layer, eight-position Inkling MTP block. The draft owns its context KV;
// target hidden features remain owned by RuntimeTP2 and are committed only after verification.
class InklingDraft {
public:
    InklingDraft(const std::string& artifact_path, const Weight& target_embedding,
                 const Tensor& target_embedding_norm, const Weight& target_lm_head,
                 int max_context, DeviceContext& device);
    ~InklingDraft();

    InklingDraft(const InklingDraft&) = delete;
    InklingDraft& operator=(const InklingDraft&) = delete;

    void reset();
    void commit(const std::array<Tensor, 4>& target_features, int count);
    [[nodiscard]] std::array<int, 7> propose(int anchor_token, int draft_count = 7);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::inkling2_mini::detail
