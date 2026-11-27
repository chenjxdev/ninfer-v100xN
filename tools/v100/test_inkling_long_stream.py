import json
import sys
import time
import urllib.request

repeat = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
body = {
    "model": "inkling2-mini",
    "messages": [{"role": "user", "content": "hello " * repeat}],
    "max_tokens": 4,
    "stream": True,
    "stream_options": {"include_usage": True},
}
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
started = time.monotonic()
chunks = 0
done = False
errors = []
with urllib.request.urlopen(request, timeout=600) as response:
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            done = True
            continue
        event = json.loads(payload)
        if "error" in event:
            errors.append(event["error"])
        chunks += 1
print(json.dumps({"chunks": chunks, "done": done, "errors": errors,
                  "seconds": time.monotonic() - started}))
if errors or not done:
    raise SystemExit(1)
