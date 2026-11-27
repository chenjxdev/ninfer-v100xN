#include "targets/inkling2_mini/impl/runtime_tp2.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using json = nlohmann::json;
using Runtime = ninfer::targets::inkling2_mini::detail::RuntimeTP2;

std::vector<std::uint8_t> decode_base64(std::string_view text) {
    static constexpr std::array<std::int8_t, 256> table = [] {
        std::array<std::int8_t, 256> out{};
        out.fill(-1);
        for (int i = 0; i < 26; ++i) {
            out[static_cast<std::size_t>('A' + i)] = static_cast<std::int8_t>(i);
            out[static_cast<std::size_t>('a' + i)] = static_cast<std::int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i) {
            out[static_cast<std::size_t>('0' + i)] = static_cast<std::int8_t>(52 + i);
        }
        out[static_cast<std::size_t>('+')] = 62;
        out[static_cast<std::size_t>('/')] = 63;
        return out;
    }();
    std::vector<std::uint8_t> out;
    out.reserve(text.size() * 3 / 4);
    std::uint32_t bits = 0;
    int count = 0;
    bool padded = false;
    for (const unsigned char c : text) {
        if (c == '=') { padded = true; continue; }
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { continue; }
        if (padded || table[c] < 0) { throw std::invalid_argument("malformed base64 image"); }
        bits = (bits << 6U) | static_cast<std::uint32_t>(table[c]);
        count += 6;
        if (count >= 8) {
            count -= 8;
            out.push_back(static_cast<std::uint8_t>((bits >> count) & 0xffU));
        }
    }
    if (count >= 6) { throw std::invalid_argument("malformed base64 image padding"); }
    return out;
}

std::vector<std::uint8_t> image_bytes(std::string_view source) {
    const std::size_t comma = source.find(',');
    if (source.starts_with("data:")) {
        if (comma == std::string_view::npos || source.substr(0, comma).find(";base64") == std::string_view::npos) {
            throw std::invalid_argument("image data URL must use base64 encoding");
        }
        return decode_base64(source.substr(comma + 1));
    }
    return decode_base64(source);
}

struct TextAndImage {
    std::string text;
    std::vector<std::uint8_t> image;
};

TextAndImage message_content(const json& content) {
    TextAndImage out;
    if (content.is_string()) {
        out.text = content.get<std::string>();
        return out;
    }
    if (!content.is_array()) { throw std::invalid_argument("message content must be a string or parts array"); }
    for (const json& part : content) {
        const std::string type = part.value("type", "");
        if (type == "text") {
            out.text += part.at("text").get<std::string>();
        } else if (type == "image_url") {
            if (!out.image.empty()) { throw std::invalid_argument("exactly one image per request is supported"); }
            const auto& url = part.at("image_url");
            const std::string source = url.is_string() ? url.get<std::string>() : url.at("url").get<std::string>();
            out.image = image_bytes(source);
        } else {
            throw std::invalid_argument("unsupported message content part: " + type);
        }
    }
    return out;
}

std::string render_messages(const json& messages, std::vector<std::uint8_t>& image) {
    if (!messages.is_array() || messages.empty()) { throw std::invalid_argument("messages must be nonempty"); }
    std::string rendered =
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>";
    for (const json& message : messages) {
        const std::string role = message.at("role").get<std::string>();
        TextAndImage content = message_content(message.at("content"));
        if (!content.image.empty()) {
            if (!image.empty()) { throw std::invalid_argument("exactly one image per request is supported"); }
            image = std::move(content.image);
            rendered += "<|message_user|><|content_image|><|unused_200054|><|end_message|>";
        }
        if (!content.text.empty()) {
            if (role == "system") {
                rendered += "<|message_system|><|content_text|>";
            } else if (role == "user") {
                rendered += "<|message_user|><|content_text|>";
            } else if (role == "assistant") {
                rendered += "<|message_model|><|content_text|>";
            } else {
                throw std::invalid_argument("unsupported message role: " + role);
            }
            rendered += content.text;
            rendered += "<|end_message|>";
        }
    }
    return rendered;
}

json execute(Runtime& runtime, const std::string& rendered, std::vector<std::uint8_t> image,
             int max_new_tokens, Runtime::OutputCallback callback = {}) {
    if (max_new_tokens <= 0 || max_new_tokens > 4096) {
        throw std::invalid_argument("max_new_tokens must be in [1,4096]");
    }
    const std::vector<int> prompt = runtime.encode_text(rendered);
    std::cerr << "request prompt_tokens=" << prompt.size() << " max_new_tokens=" << max_new_tokens
              << " image=" << (image.empty() ? 0 : 1) << std::endl;
    std::vector<int> output;
    if (image.empty()) {
        output = runtime.generate(prompt, max_new_tokens, std::move(callback));
    } else {
        const auto decoded = ninfer::media::decode::decode_image(image, ninfer::media::decode::Policy{});
        output = runtime.generate_with_image(prompt, 200054, decoded, max_new_tokens, std::move(callback));
    }
    json result;
    result["text"] = runtime.decode_text(output);
    result["token_ids"] = output;
    result["prompt_tokens"] = prompt.size();
    result["completion_tokens"] = output.size();
    result["prefill_seconds"] = runtime.last_prefill_seconds();
    result["decode_seconds"] = runtime.last_decode_seconds();
    if (output.size() > 1 && runtime.last_decode_seconds() > 0.0) {
        result["decode_tokens_per_second"] = (output.size() - 1) / runtime.last_decode_seconds();
    }
    result["speculative"] = {
        {"verify_rounds", runtime.last_verify_rounds()},
        {"draft_seconds", runtime.last_draft_seconds()},
        {"verify_seconds", runtime.last_verify_seconds()},
        {"proposed_drafts", runtime.last_proposed_drafts()},
        {"accepted_drafts", runtime.last_accepted_drafts()},
        {"native_fallback_tokens", runtime.last_native_fallback_tokens()},
    };
    return result;
}

void send_error(httplib::Response& response, int status, const std::string& message) {
    response.status = status;
    response.set_content(json{{"error", {{"message", message}, {"type", "invalid_request_error"}}}}.dump(),
                         "application/json");
}

struct StreamState {
    std::string rendered, model, id;
    std::vector<std::uint8_t> image;
    std::vector<int> tokens;
    std::string streamed_text;
    int max_new = 0;
    std::int64_t created = 0;
    bool include_usage = false;
    bool started = false;
    std::atomic<bool> cancelled{false};
};

void write_sse(httplib::DataSink& sink, StreamState& state, const std::string& data) {
    if (state.cancelled.load() || (sink.is_writable && !sink.is_writable()) ||
        !sink.write(data.data(), data.size())) {
        state.cancelled.store(true);
        throw std::runtime_error("streaming client disconnected");
    }
}

void write_chunk(httplib::DataSink& sink, StreamState& state, const json& delta,
                 const char* finish_reason = nullptr) {
    json choice = {{"index", 0}, {"delta", delta}};
    if (finish_reason == nullptr) { choice["finish_reason"] = nullptr; }
    else { choice["finish_reason"] = finish_reason; }
    const json chunk = {{"id", state.id}, {"object", "chat.completion.chunk"},
                        {"created", state.created}, {"model", state.model},
                        {"choices", json::array({choice})}};
    write_sse(sink, state, "data: " + chunk.dump() + "\n\n");
}

int run_server(int argc, char** argv) {
    const int port = argc > 3 ? std::stoi(argv[3]) : 8080;
    const int max_context = argc > 4 ? std::stoi(argv[4]) : 262144;
    Runtime runtime(argv[1], max_context, true);
    std::mutex runtime_mutex;
    const char* api_key = std::getenv("NINFER_INKLING_API_KEY");
    httplib::Server server;
    server.set_payload_max_length(32 * 1024 * 1024);

    server.Get("/health", [max_context](const httplib::Request&, httplib::Response& response) {
        response.set_content(json{{"status", "ok"}, {"model", "inkling2-mini"},
                                  {"vision", true}, {"max_context", max_context},
                                  {"speculative", std::getenv("NINFER_INKLING_MTP") != nullptr}}.dump(),
                             "application/json");
    });
    const auto authorize = [api_key](const httplib::Request& request) {
        if (api_key == nullptr || *api_key == '\0') { return true; }
        return request.get_header_value("Authorization") == std::string("Bearer ") + api_key;
    };
    server.Get("/v1/models", [&](const httplib::Request& request, httplib::Response& response) {
        if (!authorize(request)) { send_error(response, 401, "invalid API key"); return; }
        const auto created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const json model = {{"id", "inkling2-mini"}, {"object", "model"},
                            {"created", created}, {"owned_by", "ninfer"},
                            {"permission", json::array()}};
        response.set_content(json{{"object", "list"}, {"data", json::array({model})}}.dump(),
                             "application/json");
    });
    server.Post("/generate", [&](const httplib::Request& request, httplib::Response& response) {
        try {
            if (!authorize(request)) { send_error(response, 401, "invalid API key"); return; }
            const json body = json::parse(request.body);
            const std::string prompt = body.at("prompt").get<std::string>();
            const bool chat = body.value("chat", true);
            std::vector<std::uint8_t> image;
            if (body.contains("image_base64")) { image = image_bytes(body.at("image_base64").get<std::string>()); }
            else if (body.contains("image_url")) { image = image_bytes(body.at("image_url").get<std::string>()); }
            std::string rendered = prompt;
            if (chat || !image.empty()) {
                rendered = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>";
                if (!image.empty()) {
                    rendered += "<|message_user|><|content_image|><|unused_200054|><|end_message|>";
                }
                rendered += "<|message_user|><|content_text|>" + prompt + "<|end_message|>";
            }
            std::lock_guard lock(runtime_mutex);
            const json result = execute(runtime, rendered, std::move(image),
                                        body.value("max_new_tokens", 64));
            response.set_content(result.dump(), "application/json");
        } catch (const std::exception& error) {
            send_error(response, 400, error.what());
        }
    });
    server.Post("/v1/chat/completions", [&](const httplib::Request& request, httplib::Response& response) {
        try {
            if (!authorize(request)) { send_error(response, 401, "invalid API key"); return; }
            const json body = json::parse(request.body);
            const bool stream = body.value("stream", false);
            std::vector<std::uint8_t> image;
            const std::string rendered = render_messages(body.at("messages"), image);
            const int max_new = body.contains("max_tokens") ? body.at("max_tokens").get<int>()
                                                             : body.value("max_completion_tokens", 64);
            const auto created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            const std::string model = body.value("model", "inkling2-mini");
            if (stream) {
                auto state = std::make_shared<StreamState>();
                state->rendered = rendered;
                state->model = model;
                state->id = "chatcmpl-inkling2-mini";
                state->image = std::move(image);
                state->max_new = max_new;
                state->created = created;
                state->include_usage = body.contains("stream_options") &&
                    body.at("stream_options").value("include_usage", false);
                response.set_header("Cache-Control", "no-cache");
                response.set_header("X-Accel-Buffering", "no");
                response.set_chunked_content_provider(
                    "text/event-stream",
                    [&, state](std::size_t, httplib::DataSink& sink) -> bool {
                        if (state->started) { sink.done(); return true; }
                        state->started = true;
                        try {
                            write_chunk(sink, *state, json{{"role", "assistant"}, {"content", ""}});
                            json generated;
                            {
                                std::lock_guard lock(runtime_mutex);
                                generated = execute(runtime, state->rendered, std::move(state->image),
                                    state->max_new, [&, state](int token) {
                                        if (state->cancelled.load()) {
                                            throw std::runtime_error("streaming client disconnected");
                                        }
                                        state->tokens.push_back(token);
                                        const std::string text = runtime.decode_text(state->tokens);
                                        if (text.size() < state->streamed_text.size()) {
                                            throw std::runtime_error("tokenizer stream moved backwards");
                                        }
                                        const std::string delta = text.substr(state->streamed_text.size());
                                        state->streamed_text = text;
                                        if (!delta.empty()) {
                                            write_chunk(sink, *state, json{{"content", delta}});
                                        }
                                    });
                            }
                            const std::string full_text = generated.at("text").get<std::string>();
                            if (full_text.size() < state->streamed_text.size()) {
                                throw std::runtime_error("streamed text exceeds final output");
                            }
                            const std::string remaining = full_text.substr(state->streamed_text.size());
                            if (!remaining.empty()) {
                                write_chunk(sink, *state, json{{"content", remaining}});
                            }
                            const char* finish = generated.at("completion_tokens").get<int>() < state->max_new
                                ? "stop" : "length";
                            write_chunk(sink, *state, json::object(), finish);
                            if (state->include_usage) {
                                const json usage = {
                                    {"prompt_tokens", generated.at("prompt_tokens")},
                                    {"completion_tokens", generated.at("completion_tokens")},
                                    {"total_tokens", generated.at("prompt_tokens").get<int>() +
                                                     generated.at("completion_tokens").get<int>()}};
                                const json usage_chunk = {
                                    {"id", state->id}, {"object", "chat.completion.chunk"},
                                    {"created", state->created}, {"model", state->model},
                                    {"choices", json::array()}, {"usage", usage}};
                                write_sse(sink, *state, "data: " + usage_chunk.dump() + "\n\n");
                            }
                            write_sse(sink, *state, "data: [DONE]\n\n");
                            sink.done();
                            return true;
                        } catch (const std::exception& error) {
                            std::cerr << "stream request error: " << error.what() << std::endl;
                            if (state->cancelled.load()) { return false; }
                            try {
                                const json event = {{"error", {{"message", error.what()},
                                                               {"type", "internal_error"}}}};
                                write_sse(sink, *state, "data: " + event.dump() + "\n\n");
                                sink.done();
                                return true;
                            } catch (const std::exception&) { return false; }
                        }
                    },
                    [state](bool) { state->cancelled.store(true); });
                return;
            }
            json generated;
            {
                std::lock_guard lock(runtime_mutex);
                generated = execute(runtime, rendered, std::move(image), max_new);
            }
            json result = {
                {"id", "chatcmpl-inkling2-mini"}, {"object", "chat.completion"}, {"created", created},
                {"model", model},
                {"choices", json::array({{{"index", 0},
                    {"message", {{"role", "assistant"}, {"content", generated.at("text")}}},
                    {"finish_reason", generated.at("completion_tokens").get<int>() < max_new ? "stop" : "length"}}})},
                {"usage", {{"prompt_tokens", generated.at("prompt_tokens")},
                           {"completion_tokens", generated.at("completion_tokens")},
                           {"total_tokens", generated.at("prompt_tokens").get<int>() +
                                            generated.at("completion_tokens").get<int>()}}},
                {"ninfer", generated},
            };
            response.set_content(result.dump(), "application/json");
        } catch (const std::exception& error) {
            send_error(response, 400, error.what());
        }
    });
    std::cout << "Inkling-2-Mini server listening on 0.0.0.0:" << port
              << " context=" << max_context << " vision=1 speculative="
              << (std::getenv("NINFER_INKLING_MTP") != nullptr ? 1 : 0) << std::endl;
    return server.listen("0.0.0.0", port) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[2]) != "--serve") {
        std::cerr << "usage: " << argv[0]
                  << " MODEL.ninfer --serve [PORT=8080] [MAX_CONTEXT=262144]\n";
        return 2;
    }
    try {
        return run_server(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
