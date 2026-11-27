#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/home/chenjx/ninfer-jack/pkg/lib:${LD_LIBRARY_PATH:-}
binary=${NINFER_INKLING_BIN:-/home/chenjx/ninfer-jack/build-v100/src/targets/inkling2_mini/ninfer-inkling2-mini}
artifact=${NINFER_INKLING_ARTIFACT:-/data/models/inkling2_mini_nvfp4.ninfer}
export NINFER_INKLING_TP2=${NINFER_INKLING_TP2:-1}
if [[ ${1:-} == --image ]]; then
  image=${2:?image path is required}
  prompt=${3:-What is in the image?}
  max_new=${4:-32}
  max_context=${5:-1024}
  exec "$binary" "$artifact" --chat-image "$image" "$prompt" "$max_new" "$max_context"
fi
prompt=${1:-Hello, who are you?}
max_new=${2:-16}
max_context=${3:-1024}
exec "$binary" "$artifact" --chat "$prompt" "$max_new" "$max_context"
