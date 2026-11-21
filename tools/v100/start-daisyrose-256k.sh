#!/usr/bin/env bash
# DaisyRose-27B on two V100s: NVFP4 body, W8 input embedding, Q6 output head and Q4 MTP matrices.
# GPU vision, a native 256K INT8 KV pool and one request lane. Prefix reuse is enabled by default.
#
# The artifact is produced in one pass by the closed source profile:
#   NINFER_NVFP4_SOURCE=daisyrose python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
#       --model /data/models/daisyrose-27b-nvfp4 \
#       --out /data/models/daisyrose_27b_nvfp4_q6emb_q6head_q4mtp.ninfer --device cpu
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
if [[ ! -f "$repo_dir/tools/convert/qwen3_8_27b/source_profile.py" ]]; then
  repo_dir=/home/chenjx/ninfer-jack
fi
executable=${NINFER_DAISY_EXECUTABLE:-"$repo_dir/pkg/libexec/ninfer-serve"}
lib_dir=${NINFER_DAISY_LIB_DIR:-"$repo_dir/pkg/lib"}
artifact=${NINFER_DAISY_ARTIFACT:-/data/models/daisyrose_27b_nvfp4_q6emb_q6head_q4mtp.ninfer}
if [[ ! -f "$artifact" ]]; then
  echo "missing artifact: $artifact" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$executable" "$artifact" \
  --host "${NINFER_DAISY_HOST:-172.16.3.88}" --port "${NINFER_DAISY_PORT:-8080}" \
  --tp 2 --devices "${NINFER_DAISY_DEVICES:-1,0}" \
  --max-concurrency "${NINFER_DAISY_MAX_CONCURRENCY:-1}" \
  --max-context "${NINFER_DAISY_MAX_CONTEXT:-262144}" --kv-dtype "${NINFER_DAISY_KV_DTYPE:-int8}" \
  --spec mtp --draft-tokens 4 --lm-head-draft --volta-long-attention \
  --vision --vision-device "${NINFER_DAISY_VISION_DEVICE:-gpu}" --vision-max-tokens "${NINFER_DAISY_VISION_MAX_TOKENS:-4096}" \
  --pending-timeout-ms 600000 "$@"
