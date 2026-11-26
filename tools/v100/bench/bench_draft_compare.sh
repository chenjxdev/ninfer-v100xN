#!/usr/bin/env bash
# MMFlash 版基准：与之前 MTP A/B 同一段 prompt，可直接对比。
# 只打 18099 端口，不碰 8080。
set -u
PORT=${PORT:-18099}
LOG=${LOG:-/data/models/logs/heretic-mmflash.log}
PROMPT='用中文写一段 600 字的说明，介绍 KV cache 在大模型推理中的作用与原理，不要分点，写成连贯的段落。'

ask() {
  curl -s -m 600 http://172.16.3.88:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":900,\"temperature\":0}" \
    -o /tmp/mm_reply.json
}
echo "=== 预热 2 次"; ask; sleep 1; ask; sleep 1
echo "=== 计时 3 次 (900 token, 与 MTP A/B 同 prompt)"; ask; sleep 1; ask; sleep 1; ask
grep "done finish" "$LOG" | tail -3 | sed 's/.*\[info\] ninfer-serve: //'

echo; echo "=== 第二组: 另一段内容 (代码说明, 看接受率是否内容相关)"
curl -s -m 600 http://172.16.3.88:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"用中文详细解释什么是张量并行（Tensor Parallelism），包括它与数据并行的区别、在显存带宽受限场景下的意义，以及在两卡推理时的切分方式。"}],"max_tokens":900,"temperature":0}' -o /tmp/mm_reply2.json
sleep 1
grep "done finish" "$LOG" | tail -1 | sed 's/.*\[info\] ninfer-serve: //'

echo; echo "=== 回复抽样 (确认输出正常):"
python3 - <<'EOF'
import json
for f in ("/tmp/mm_reply.json", "/tmp/mm_reply2.json"):
    try:
        d = json.load(open(f)); m = d["choices"][0]["message"]
        t = (m.get("content") or "").strip() or (m.get("reasoning_content") or "").strip()
        print(f"  {f}: len={len(t)}  head: {t[:110]!r}")
    except Exception as e:
        print(f"  {f}: ERR {e}")
EOF
echo; echo "=== 显存:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
