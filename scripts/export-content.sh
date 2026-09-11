#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 1 ]; then
  echo "usage: scripts/export-content.sh DOCUMENT_JSON_OR_RESULT_DIRECTORY" >&2
  exit 2
fi

exec bash scripts/container.sh envs/pipeline/bin/python -m content_export.cli "$1"
