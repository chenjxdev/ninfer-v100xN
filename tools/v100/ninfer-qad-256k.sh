#!/usr/bin/env bash
# Launcher for the NVIDIA QAD (Qwen3.8-27B-NVFP4-QAD) artifact on 2x Volta.
#
# The artifact is /data/models/qwen3_8_27b_nvfp4.ninfer (17.5 GiB / 1216 objects):
#   - 204 NVFP4 text projections (byte-for-byte passthrough of the QAT weights)
#   -  52 FP8 row-scaled attention projections (byte-for-byte passthrough, SNR 52.8-141 dB)
#   -   2 W8 vocabulary endpoints (lm_head / embed_tokens)
#   - 333 Vision objects (Q4/Q5/Q6 groupwise) + BF16 controls
#
# Runtime notes (measured on this box, 2x V100-SXM2-16GB):
#
#   --max-context 262144   The full native context.  It only fits because of the two
#                           flags below; the default reservation is ~36 MiB short.
#   --no-cuda-graph         Frees the captured CUDA graphs (16/94 MiB).  That is exactly
#                           the room the Vision lane-local prefix checkpoints need:
#                           they are retained on-device and scale with
#                           --vision-max-tokens at ~10.5 MiB per 1024 tokens.
#                           Cost is ~0.4% decode (121.3 vs 121.8 tok/s): a 27B model
#                           is HBM-bandwidth bound, so replaying a captured graph buys
#                           almost nothing.  Token output is identical.
#   --vision-device cpu     Keeps the Vision tower in host memory.  Decode is unaffected
#                           (the tower is not on the text path) and it frees ~0.4 GiB of
#                           device memory for the KV budget.
#   --vision-max-tokens 4096  Full image fidelity.  Lowering this is the other way to
#                           buy memory, at the cost of image detail.
#
# Optional: --ram-kv-window moves long-history attention to host RAM (approximate),
# which buys several GiB if the context ever has to grow beyond 256K.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)

executable=${NINFER_QAD_EXECUTABLE:-"${repo_dir}/pkg/libexec/ninfer-serve"}
artifact=${NINFER_QAD_ARTIFACT:-/data/models/qwen3_8_27b_nvfp4.ninfer}
lib_dir=${NINFER_QAD_LIB_DIR:-"${repo_dir}/pkg/lib"}
host=${NINFER_QAD_HOST:-172.16.3.88}
port=${NINFER_QAD_PORT:-8080}
devices=${NINFER_QAD_DEVICES:-1,0}
max_context=${NINFER_QAD_MAX_CONTEXT:-262144}
vision_max_tokens=${NINFER_QAD_VISION_MAX_TOKENS:-4096}
media_cache_mib=${NINFER_QAD_MEDIA_CACHE_MIB:-512}
kv_dtype=${NINFER_QAD_KV_DTYPE:-int8}

export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${executable}" "${artifact}" \
  --host "${host}" --port "${port}" \
  --tp 2 --devices "${devices}" \
  --max-context "${max_context}" --kv-dtype "${kv_dtype}" \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --volta-long-attention \
  --vision --vision-device cpu --vision-max-tokens "${vision_max_tokens}" \
  --media-cache-mib "${media_cache_mib}" \
  --no-cuda-graph \
  --pending-timeout-ms 600000 "$@"
