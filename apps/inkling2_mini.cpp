#include "targets/inkling2_mini/impl/runtime.h"
#include "targets/inkling2_mini/impl/runtime_tp2.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

std::vector<int> parse_ids(const std::string& text) {
    std::vector<int> ids;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (!item.empty()) { ids.push_back(std::stoi(item)); }
    }
    return ids;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::invalid_argument("cannot open image: " + path); }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), {});
}

template <class Runtime>
int run(int argc, char** argv) {
    const bool text = std::string(argv[2]) == "--text";
    const bool chat = std::string(argv[2]) == "--chat";
    const bool image_mode = std::string(argv[2]) == "--chat-image";
    if ((text || chat) && argc < 4) { throw std::invalid_argument("missing prompt text"); }
    if (image_mode && argc < 5) { throw std::invalid_argument("missing image path or prompt text"); }
    const int value_index = image_mode ? 4 : (text || chat ? 3 : 2);
    const int max_new = argc > value_index + 1 ? std::stoi(argv[value_index + 1])
                                               : (chat || image_mode ? 16 : 8);
    const int max_context = argc > value_index + 2 ? std::stoi(argv[value_index + 2]) : 1024;
    Runtime runtime(argv[1], max_context, image_mode);
    std::vector<int> prompt;
    ninfer::media::decode::Image image;
    if (image_mode) {
        const auto bytes = read_file(argv[3]);
        image = ninfer::media::decode::decode_image(bytes, ninfer::media::decode::Policy{});
        const std::string rendered =
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            "<|message_user|><|content_image|><|unused_200054|><|end_message|>"
            "<|message_user|><|content_text|>" + std::string(argv[value_index]) + "<|end_message|>";
        prompt = runtime.encode_text(rendered);
    } else if (text) {
        prompt = runtime.encode_text(argv[value_index]);
    } else if (chat) {
        const std::string rendered =
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            "<|message_user|><|content_text|>" + std::string(argv[value_index]) + "<|end_message|>";
        prompt = runtime.encode_text(rendered);
    } else {
        prompt = parse_ids(argv[2]);
    }
    const auto started = std::chrono::steady_clock::now();
    std::cout << "prompt_ids=";
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        if (i != 0) { std::cout << ','; }
        std::cout << prompt[i];
    }
    std::cout << '\n';
    const auto output = image_mode ? runtime.generate_with_image(prompt, 200054, image, max_new)
                                   : runtime.generate(prompt, max_new);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "output_ids=";
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (i != 0) { std::cout << ','; }
        std::cout << output[i];
    }
    if (text || chat || image_mode) { std::cout << "\noutput_text=" << runtime.decode_text(output); }
    std::cout << "\nelapsed_seconds=" << seconds
              << "\nprefill_seconds=" << runtime.last_prefill_seconds()
              << "\ndecode_seconds=" << runtime.last_decode_seconds()
              << "\ntokens_per_second=" << output.size() / seconds;
    if (output.size() > 1 && runtime.last_decode_seconds() > 0.0) {
        std::cout << "\ndecode_tokens_per_second=" << (output.size() - 1) / runtime.last_decode_seconds();
    }
    if constexpr (std::is_same_v<Runtime, ninfer::targets::inkling2_mini::detail::RuntimeTP2>) {
        if (runtime.last_verify_rounds() > 0) {
            std::cout << "\nverify_rounds=" << runtime.last_verify_rounds()
                      << "\ndraft_seconds=" << runtime.last_draft_seconds()
                      << "\nverify_seconds=" << runtime.last_verify_seconds()
                      << "\nproposed_drafts=" << runtime.last_proposed_drafts()
                      << "\naccepted_drafts=" << runtime.last_accepted_drafts()
                      << "\nnative_fallback_tokens=" << runtime.last_native_fallback_tokens()
                      << "\ndraft_acceptance_rate="
                      << static_cast<double>(runtime.last_accepted_drafts()) /
                         runtime.last_proposed_drafts()
                      << "\naccepted_tokens_per_verify="
                      << static_cast<double>(runtime.last_accepted_drafts() +
                                             runtime.last_verify_rounds()) /
                         runtime.last_verify_rounds();
        }
    }
    std::cout << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage:\n  " << argv[0]
                  << " MODEL.ninfer TOKEN_ID[,TOKEN_ID...] [MAX_NEW=8] [MAX_CONTEXT=1024]\n  "
                  << argv[0] << " MODEL.ninfer --text PROMPT [MAX_NEW=8] [MAX_CONTEXT=1024]\n  "
                  << argv[0] << " MODEL.ninfer --chat PROMPT [MAX_NEW=16] [MAX_CONTEXT=1024]\n  "
                  << argv[0] << " MODEL.ninfer --chat-image IMAGE PROMPT [MAX_NEW=16] [MAX_CONTEXT=1024]\n"
                  << "set NINFER_INKLING_TP2=1 for layer-parallel dual-GPU execution\n";
        return 2;
    }
    try {
        const char* tp2 = std::getenv("NINFER_INKLING_TP2");
        if (tp2 != nullptr && std::string(tp2) == "1") {
            return run<ninfer::targets::inkling2_mini::detail::RuntimeTP2>(argc, argv);
        }
        return run<ninfer::targets::inkling2_mini::detail::Runtime>(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
