#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${PARSER_IMAGE:-pdf-structure-parser:gpu}"
GPU_REQUEST="${PARSER_GPUS:-all}"
mkdir -p "$ROOT"/{cache,envs,models,vendor,data,runs,logs,manifests}
exec docker run --rm --network host --shm-size 16g --gpus "$GPU_REQUEST" --user "$(id -u):$(id -g)" \
  -v "$ROOT:/workspace" -w /workspace \
  -e HF_HOME=/workspace/cache/huggingface -e XDG_CACHE_HOME=/workspace/cache \
  -e MODELSCOPE_CACHE=/workspace/cache/modelscope \
  -e MODELSCOPE_CREDENTIALS_PATH=/workspace/cache/modelscope/credentials \
  -e PIP_CACHE_DIR=/workspace/cache/pip -e PYTHONPATH=/workspace \
  -e USER=parser -e LOGNAME=parser -e TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor \
  -e PARSER_DEVICE="${PARSER_DEVICE:-cuda:0}" -e PARSER_DTYPE="${PARSER_DTYPE:-auto}" \
  -e PARSER_MINERU_DTYPE="${PARSER_MINERU_DTYPE:-auto}" -e RAPIDOCR_DEVICE="${RAPIDOCR_DEVICE:-auto}" \
  -e PARSER_MIN_FREE_GIB="${PARSER_MIN_FREE_GIB:-6}" \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
  -e http_proxy -e https_proxy -e HTTP_PROXY -e HTTPS_PROXY -e ALL_PROXY -e no_proxy -e NO_PROXY \
  "$IMAGE" "$@"
