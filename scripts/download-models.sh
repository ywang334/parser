#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  bash scripts/container.sh envs/pipeline/bin/python scripts/download_modelscope.py; then
  source /data/wy/tools/vpn/use_proxy.sh
  bash scripts/container.sh envs/pipeline/bin/python scripts/download.py mineru
fi

if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  bash scripts/container.sh envs/pipeline/bin/python scripts/download.py tatr tatr-detection; then
  source /data/wy/tools/vpn/use_proxy.sh
  bash scripts/container.sh envs/pipeline/bin/python scripts/download.py tatr tatr-detection
fi

if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  bash scripts/container.sh envs/pipeline/bin/python scripts/prepare_rapidocr_models.py; then
  source /data/wy/tools/vpn/use_proxy.sh
  bash scripts/container.sh envs/pipeline/bin/python scripts/prepare_rapidocr_models.py
fi
