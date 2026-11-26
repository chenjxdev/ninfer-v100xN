#!/bin/bash
# 向指定端口发一个问题，打印回答与引擎日志里的那一行统计。用法: ask.sh <port> <max_tokens> <问题>
PORT=$1; MAXTOK=$2; Q=$3
python3 - "$PORT" "$MAXTOK" "$Q" <<'PY'
import json, sys, urllib.request
port, maxtok, q = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": q}],
        "max_tokens": int(maxtok), "temperature": 0}
req = urllib.request.Request(f"http://172.16.3.88:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=900))
except Exception as e:
    print("REQUEST FAILED:", type(e).__name__, e); sys.exit(1)
m = d["choices"][0]["message"]
text = (m.get("content") or "").strip()
reason = (m.get("reasoning_content") or "").strip()
print("usage:", d.get("usage"))
print("finish:", d["choices"][0].get("finish_reason"), "| content_len:", len(text), "| reasoning_len:", len(reason))
print("--- content ---")
print(text[:700] if text else "(empty; reasoning head follows) " + reason[:400])
PY
