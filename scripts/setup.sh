#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! bash scripts/build-image.sh; then
  source /data/wy/tools/vpn/use_proxy.sh
  bash scripts/build-image.sh
fi

bash scripts/bootstrap.sh

if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  bash scripts/container.sh envs/pipeline/bin/python scripts/source_snapshots.py; then
  source /data/wy/tools/vpn/use_proxy.sh
  bash scripts/container.sh envs/pipeline/bin/python scripts/source_snapshots.py
fi

bash scripts/download-models.sh
bash scripts/container.sh envs/pipeline/bin/python scripts/record_manifests.py
