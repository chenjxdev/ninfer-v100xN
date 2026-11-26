#!/usr/bin/env bash
# DaisyRose-27B Heretic v2 (kelsey1337/DaisyRose-27B-Heretic-v2) on two V100s.
#
# The artifact is converted from the plain BF16 checkpoint into the production
# format map: NVFP4 block-scaled MLP and GDN projections, row-scaled FP8 for
# every full-attention projection and ten GDN layers, Q6 vocabulary endpoints
# and Q4 MTP matrices -- the same object map as the production artifact.
#
# 256K INT8 KV pool, one request lane, GPU vision, MTP speculation with the
# shortlist proposal head. CUDA graphs stay on by default.
set -euo pipefail

repo_dir=${NINFER_REPO_DIR:-/home/chenjx/ninfer-jack}
executable=${NINFER_EXECUTABLE:-"$repo_dir/pkg/libexec/ninfer-serve"}
lib_dir=${NINFER_LIB_DIR:-"$repo_dir/pkg/lib"}
artifact=${NINFER_ARTIFACT:-/data/models/heretic_v2_27b_nvfp4_q6emb_q6head_q4mtp.ninfer}

export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$executable" "$artifact" \
  --host "${NINFER_HOST:-172.16.3.88}" --port "${NINFER_PORT:-8080}" \
  --tp 2 --devices "${NINFER_DEVICES:-1,0}" \
  --max-concurrency "${NINFER_MAX_CONCURRENCY:-1}" \
  --max-context "${NINFER_MAX_CONTEXT:-262144}" --kv-dtype "${NINFER_KV_DTYPE:-int8}" \
  --spec mtp --draft-tokens 4 --lm-head-draft --volta-long-attention \
  --vision --vision-device "${NINFER_VISION_DEVICE:-gpu}" \
  --vision-max-tokens "${NINFER_VISION_MAX_TOKENS:-4096}" \
  --pending-timeout-ms 600000 "$@"
