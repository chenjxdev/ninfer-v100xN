"""Vision verification for the ninfer OpenAI-compatible endpoint.

Same test images as tools/v100/check_vision_http.py (a red circle and a blue
square on white), but without that script's start-up assertions: it expects a
2-way concurrent engine with a 1024-token prefill chunk, while the production
configuration is single-lane.

Checks: single image, a large image, two images in one prompt, and a text
request concurrent with an image request.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import threading
import time
import urllib.request


def image_uri(color: str, size: int = 384) -> str:
    foreground = (255, 0, 0) if color == "red" else (0, 0, 255)
    pixels = bytearray()
    for row in range(size):
        for column in range(size):
            if color == "red":
                inside = (row - size // 2) ** 2 + (column - size // 2) ** 2 < (size // 3) ** 2
            else:
                inside = size // 4 <= row < 3 * size // 4 and size // 4 <= column < 3 * size // 4
            pixels.extend(foreground if inside else (255, 255, 255))
    image = f"P6\n{size} {size}\n255\n".encode() + pixels
    return "data:image/x-portable-pixmap;base64," + base64.b64encode(image).decode()


def request(base_url, model, images, prompt, max_tokens, barrier=None, stream=False):
    content = [{"type": "image_url", "image_url": {"url": image}} for image in images]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": stream,
    }
    http_request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait(timeout=60)
    started = time.monotonic()
    first_content = None
    output = []
    with urllib.request.urlopen(http_request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise AssertionError(event["error"])
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or delta.get("reasoning_content") or ""
                if text:
                    first_content = first_content or time.monotonic()
                    output.append(text)
    finished = time.monotonic()
    assert first_content is not None
    return {
        "text": "".join(output), "ttft_seconds": first_content - started,
        "total_seconds": finished - started,
    }


def scene_ok(text: str, color: str) -> bool:
    low = text.lower()
    shape = "circle" if color == "red" else "square"
    shapes = ("circle", "round", "disk", "disc", "圆", "球") if color == "red" \
        else ("square", "rectangle", "box", "方", "正方")
    colors = ("red", "红") if color == "red" else ("blue", "蓝")
    backgrounds = ("white", "白")
    return (any(s in low for s in shapes)
            and any(c in low for c in colors)
            and any(b in low for b in backgrounds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://172.16.3.88:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    args = parser.parse_args()

    images = {color: image_uri(color) for color in ("red", "blue")}
    failures = []

    print("=== 1. 单图识别 ===")
    for color in ("red", "blue"):
        result = request(
            args.base_url, args.model, [images[color]],
            "Identify the shape, its color and the background color in this image.",
            400, stream=True)
        ok = scene_ok(result["text"], color)
        print("  %-5s %s  ttft %.2fs  total %.2fs" % (
            color, "PASS" if ok else "FAIL", result["ttft_seconds"], result["total_seconds"]))
        print("      %s" % result["text"].replace("\n", " ")[:160])
        if not ok:
            failures.append("single-" + color)

    print()
    print("=== 2. 大图 (1280) ===")
    large = request(
        args.base_url, args.model, [image_uri("blue", 1280)],
        "Identify the shape, its color and the background color in this image.",
        400, stream=True)
    ok = scene_ok(large["text"], "blue")
    print("  1280 %s  ttft %.2fs" % ("PASS" if ok else "FAIL", large["ttft_seconds"]))
    print("      %s" % large["text"].replace("\n", " ")[:160])
    if not ok:
        failures.append("large")

    print()
    print("=== 3. 一张图里两个形状 ===")
    both = request(
        args.base_url, args.model, list(images.values()),
        "Describe both images, identifying each shape, its color, and background.",
        400, stream=True)
    text = both["text"].lower()
    ok = ("red" in text or "红" in text) and ("blue" in text or "蓝" in text) \
        and ("circle" in text or "round" in text or "圆" in text) \
        and ("square" in text or "box" in text or "rectangle" in text or "方" in text)
    print("  两图 %s  ttft %.2fs" % ("PASS" if ok else "FAIL", both["ttft_seconds"]))
    print("      %s" % both["text"].replace("\n", " ")[:200])
    if not ok:
        failures.append("two-images")

    print()
    print("=== 4. 文本与图像并发 ===")
    barrier = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        text_future = pool.submit(
            request, args.base_url, args.model, [],
            "用一句话解释什么是张量并行。", 400, barrier, True)
        image_future = pool.submit(
            request, args.base_url, args.model, [images["red"]],
            "Identify the shape, its color and the background color in this image.",
            400, barrier, True)
        text_result = text_future.result()
        image_result = image_future.result()
    print("  文本: %.2fs  %s" % (text_result["total_seconds"],
                                 text_result["text"].replace("\n", " ")[:80]))
    ok = scene_ok(image_result["text"], "red")
    print("  图像: %.2fs  %s  %s" % (image_result["total_seconds"],
                                     "PASS" if ok else "FAIL",
                                     image_result["text"].replace("\n", " ")[:100]))
    if not ok:
        failures.append("concurrent-image")

    print()
    print("=== 结果 ===")
    print("  全部通过" if not failures else "  失败项: %s" % failures)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
