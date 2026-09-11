#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

bash scripts/container.sh python -m venv /workspace/envs/pipeline

install_direct() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    bash scripts/container.sh envs/pipeline/bin/pip install --index-url "$TORCH_INDEX_URL" \
      'torch==2.10.0' 'torchvision==0.25.0'
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    bash scripts/container.sh envs/pipeline/bin/pip install -i "$PYPI_INDEX_URL" -r requirements.txt
}

if ! install_direct; then
  source /data/wy/tools/vpn/use_proxy.sh
  bash scripts/container.sh envs/pipeline/bin/pip install --index-url "$TORCH_INDEX_URL" \
    'torch==2.10.0' 'torchvision==0.25.0'
  bash scripts/container.sh envs/pipeline/bin/pip install -i "$PYPI_INDEX_URL" -r requirements.txt
fi

# RapidOCR depends on the GUI OpenCV distribution by package name. Keep the
# headless build in this server/container runtime after resolving dependencies.
bash scripts/container.sh envs/pipeline/bin/pip uninstall -y opencv-python || true
bash scripts/container.sh envs/pipeline/bin/pip install -i "$PYPI_INDEX_URL" --force-reinstall --no-deps 'opencv-python-headless==4.12.0.88'
bash scripts/container.sh envs/pipeline/bin/pip freeze > manifests/pipeline.freeze.txt
docker image inspect "${PARSER_IMAGE:-pdf-structure-parser:gpu}" > manifests/image.json
