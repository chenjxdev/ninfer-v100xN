#!/usr/bin/env bash
# Qwen3.8-27B QAD on two V100s: Q6 input embedding, W8 output head and Q4 MTP matrices.
# GPU vision, a native 256K INT8 KV pool and one request lane. Prefix reuse is enabled by default.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)
if [[ ! -f "$repo_dir/tools/v100/prepare_qad_q6emb_w8head.py" ]]; then
  repo_dir=/home/chenjx/ninfer-jack
fi
executable=${NINFER_QAD_EXECUTABLE:-"$repo_dir/pkg/libexec/ninfer-serve"}
lib_dir=${NINFER_QAD_LIB_DIR:-"$repo_dir/pkg/lib"}
artifact=${NINFER_QAD_ARTIFACT:-}
if [[ -z "$artifact" ]]; then
  cache=$(PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" python3 "$repo_dir/tools/v100/prepare_qad_q6emb_w8head.py")
  # Keep the model available for future reloads if logind clears /dev/shm after logout.
  exec 9<"$cache"
  run_dir=$(mktemp -d "${TMPDIR:-/tmp}/ninfer-qad-XXXXXXXX")
  artifact="$run_dir/${cache##*/}"
  ln -s "/proc/$$/fd/9" "$artifact"
fi
export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$executable" "$artifact" \
  --host "${NINFER_QAD_HOST:-172.16.3.88}" --port "${NINFER_QAD_PORT:-8080}" \
  --tp 2 --devices "${NINFER_QAD_DEVICES:-1,0}" \
  --max-concurrency "${NINFER_QAD_MAX_CONCURRENCY:-1}" \
  --max-context "${NINFER_QAD_MAX_CONTEXT:-262144}" --kv-dtype "${NINFER_QAD_KV_DTYPE:-int8}" \
  --spec mtp --draft-tokens 4 --lm-head-draft --volta-long-attention \
  --vision --vision-device "${NINFER_QAD_VISION_DEVICE:-gpu}" --vision-max-tokens "${NINFER_QAD_VISION_MAX_TOKENS:-4096}" \
  --pending-timeout-ms 600000 "$@"
