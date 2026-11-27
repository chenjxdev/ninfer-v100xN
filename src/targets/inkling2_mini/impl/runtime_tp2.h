#pragma once

#include "media/decode/decode.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {

class RuntimeTP2 {
public:
    RuntimeTP2(const std::string& artifact_path, int max_context = 1024, bool enable_vision = false);
    ~RuntimeTP2();

    RuntimeTP2(const RuntimeTP2&) = delete;
    RuntimeTP2& operator=(const RuntimeTP2&) = delete;

    void reset();
    [[nodiscard]] std::vector<int> encode_text(const std::string& text);
    [[nodiscard]] std::string decode_text(const std::vector<int>& tokens);
    using OutputCallback = std::function<void(int token)>;
    [[nodiscard]] std::vector<int> generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                                            OutputCallback callback = {});
    [[nodiscard]] std::vector<int> generate_with_image(const std::vector<int>& prompt_tokens,
        int image_token_id, const media::decode::Image& image, int max_new_tokens,
        OutputCallback callback = {});
    [[nodiscard]] double last_prefill_seconds() const;
    [[nodiscard]] double last_decode_seconds() const;
    [[nodiscard]] double last_draft_seconds() const;
    [[nodiscard]] double last_verify_seconds() const;
    [[nodiscard]] int last_verify_rounds() const;
    [[nodiscard]] int last_proposed_drafts() const;
    [[nodiscard]] int last_accepted_drafts() const;
    [[nodiscard]] int last_native_fallback_tokens() const;

private:
    [[nodiscard]] std::vector<int> generate_current(const std::vector<int>& prompt_tokens,
                                                     int max_new_tokens,
                                                     const OutputCallback& callback);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::inkling2_mini::detail
