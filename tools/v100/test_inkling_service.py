"""Black-box HTTP regression for the persistent Inkling-2-Mini service."""
from __future__ import annotations

import argparse
import base64
import json
import urllib.request


def post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def stream_chat(base: str, body: dict) -> dict:
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    text = ""
    finish_reason = None
    usage = None
    chunks = 0
    done = False
    with urllib.request.urlopen(request, timeout=300) as response:
        if not response.headers.get("Content-Type", "").startswith("text/event-stream"):
            raise ValueError("streaming response is not text/event-stream")
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                continue
            chunk = json.loads(payload)
            chunks += 1
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text += choice.get("delta", {}).get("content", "") or ""
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    if not done:
        raise ValueError("SSE stream did not terminate with [DONE]")
    return {"text": text, "finish_reason": finish_reason, "usage": usage, "chunks": chunks}


def summarize(label: str, result: dict) -> None:
    spec = result.get("speculative", {})
    print(json.dumps({"label": label, "text": result["text"],
                      "decode_tokens_per_second": result.get("decode_tokens_per_second"),
                      "completion_tokens": result["completion_tokens"],
                      "accepted_drafts": spec.get("accepted_drafts"),
                      "proposed_drafts": spec.get("proposed_drafts"),
                      "native_fallback_tokens": spec.get("native_fallback_tokens")},
                     ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--image")
    args = parser.parse_args()
    with urllib.request.urlopen(args.base + "/health", timeout=30) as response:
        print("health", json.load(response))
    summarize("adaptive_chat", post(args.base, "/generate", {
        "prompt": "Hello, who are you?", "max_new_tokens": 32}))
    summarize("repeated_text", post(args.base, "/generate", {
        "prompt": "a a a a a a a a a a a a a a a a", "chat": False,
        "max_new_tokens": 32}))
    if args.image:
        with open(args.image, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        summarize("vision", post(args.base, "/generate", {
            "prompt": "What shapes and colors are in the image?", "image_base64": encoded,
            "max_new_tokens": 64}))
    result = post(args.base, "/v1/chat/completions", {
        "model": "inkling2-mini", "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8})
    print(json.dumps({"label": "openai", "choices": result["choices"], "usage": result["usage"]},
                     ensure_ascii=True))
    streamed = stream_chat(args.base, {
        "model": "inkling2-mini", "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True}})
    if streamed["text"] != result["choices"][0]["message"]["content"]:
        raise ValueError("streamed text differs from the non-streaming response")
    print(json.dumps({"label": "openai_stream", **streamed}, ensure_ascii=True))
    eos = post(args.base, "/v1/chat/completions", {
        "model": "inkling2-mini", "messages": [{"role": "user", "content": "Reply with only hello."}],
        "max_tokens": 256})
    print(json.dumps({"label": "eos", "finish_reason": eos["choices"][0]["finish_reason"],
                      "completion_tokens": eos["usage"]["completion_tokens"],
                      "text": eos["choices"][0]["message"]["content"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
