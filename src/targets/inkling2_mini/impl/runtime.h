#pragma once

#include "media/decode/decode.h"

#include <memory>
#include <string>
#include <vector>

namespace ninfer::targets::inkling2_mini::detail {

// Minimal eager text runtime used to bring up the architecture before integrating the product
// request scheduler. It supports greedy prefill/decode on two devices with Pipeline16 placement.
class Runtime {
public:
    Runtime(const std::string& artifact_path, int max_context = 1024, bool enable_vision = false);
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void reset();
    [[nodiscard]] std::vector<int> encode_text(const std::string& text);
    [[nodiscard]] std::string decode_text(const std::vector<int>& tokens);
    [[nodiscard]] std::vector<int> generate(const std::vector<int>& prompt_tokens, int max_new_tokens);
    [[nodiscard]] std::vector<int> generate_with_image(const std::vector<int>& prompt_tokens,
        int image_token_id, const media::decode::Image& image, int max_new_tokens);
    [[nodiscard]] double last_prefill_seconds() const;
    [[nodiscard]] double last_decode_seconds() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::inkling2_mini::detail
