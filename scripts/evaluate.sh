#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 3 ]; then
  echo "usage: scripts/evaluate.sh GROUND_TRUTH.json PREDICTION.json METRICS.json" >&2
  exit 2
fi
exec bash scripts/container.sh envs/pipeline/bin/python -m pdfpipe.metrics "$1" "$2" --out "$3"
