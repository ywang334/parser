#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 1 ] || [ ! -d "$1" ]; then
  echo "usage: scripts/run.sh PDF_DIRECTORY --run-id ID --parallel-gpus 0,1 [pdfpipe options]" >&2
  exit 2
fi

INPUT_DIR="$1"
shift
GPU_SPEC=""
RUN_ID=""
RESUME=0
FORWARD_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --parallel-gpus)
      [ "$#" -ge 2 ] || { echo "--parallel-gpus requires a comma-separated value" >&2; exit 2; }
      GPU_SPEC="$2"
      shift 2
      ;;
    --parallel-gpus=*)
      GPU_SPEC="${1#*=}"
      shift
      ;;
    --run-id)
      [ "$#" -ge 2 ] || { echo "--run-id requires a value" >&2; exit 2; }
      RUN_ID="$2"
      shift 2
      ;;
    --run-id=*)
      RUN_ID="${1#*=}"
      shift
      ;;
    --resume)
      RESUME=1
      FORWARD_ARGS+=("$1")
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

[ -n "$GPU_SPEC" ] || { echo "--parallel-gpus cannot be empty" >&2; exit 2; }
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "parallel --run-id may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_SPEC"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "no GPU ids supplied" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU id: $gpu" >&2
    exit 2
  fi
  if [ -n "${SEEN_GPUS[$gpu]:-}" ]; then
    echo "duplicate GPU id: $gpu" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done

PDFS=()
while IFS= read -r -d '' pdf; do
  PDFS+=("$pdf")
done < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.pdf' -print0 | sort -z)
if [ "${#PDFS[@]}" -eq 0 ]; then
  echo "no PDF files found in $INPUT_DIR" >&2
  exit 2
fi

WORKER_COUNT="${#GPU_IDS[@]}"
if [ "$WORKER_COUNT" -gt "${#PDFS[@]}" ]; then
  WORKER_COUNT="${#PDFS[@]}"
fi

RUN_ROOT="runs/$RUN_ID"
if [ -e "$RUN_ROOT" ] && [ "$RESUME" -eq 0 ]; then
  echo "Run already exists; choose another --run-id or use --resume" >&2
  exit 2
fi
if [ -e "$RUN_ROOT" ] && [ "$RESUME" -eq 1 ] && [ ! -f "$RUN_ROOT/.parallel-status/worker-count" ]; then
  echo "Existing run is not a resumable parallel batch: $RUN_ROOT" >&2
  exit 2
fi
STATUS_ROOT="$RUN_ROOT/.parallel-status"
mkdir -p "$STATUS_ROOT"
COUNT_FILE="$STATUS_ROOT/worker-count"
if [ -f "$COUNT_FILE" ] && [ "$(<"$COUNT_FILE")" != "$WORKER_COUNT" ]; then
  echo "Parallel resume requires the original worker count $(<"$COUNT_FILE"); got $WORKER_COUNT" >&2
  exit 2
fi
printf '%s\n' "$WORKER_COUNT" > "$COUNT_FILE"
INPUT_LIST_FILE="$STATUS_ROOT/input-files.nul"
if [ -f "$INPUT_LIST_FILE" ] && ! cmp -s "$INPUT_LIST_FILE" <(printf '%s\0' "${PDFS[@]}"); then
  echo "Parallel resume requires the original ordered PDF list; start a new run id" >&2
  exit 2
fi
printf '%s\0' "${PDFS[@]}" > "$INPUT_LIST_FILE"

echo "parallel batch: ${#PDFS[@]} PDFs, $WORKER_COUNT workers, GPUs $GPU_SPEC"
pids=()
for ((worker=0; worker<WORKER_COUNT; worker++)); do
  gpu="${GPU_IDS[$worker]}"
  worker_name="$(printf 'worker-%03d' "$worker")"
  (
    status=0
    PARSER_GPUS="device=$gpu" PARSER_DEVICE=cuda:0 \
      bash scripts/container.sh envs/pipeline/bin/python -m pdfpipe.cli "$INPUT_DIR" \
      --run-id "$RUN_ID/.parallel/$worker_name" \
      --shard-index "$worker" --shard-count "$WORKER_COUNT" \
      "${FORWARD_ARGS[@]}" || status=$?
    printf '%s\n' "$status" > "$STATUS_ROOT/$worker_name.exit"
    exit "$status"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

USED_GPU_SPEC="$(IFS=,; echo "${GPU_IDS[*]:0:$WORKER_COUNT}")"
PARSER_GPUS="device=${GPU_IDS[0]}" PARSER_DEVICE=cuda:0 \
  bash scripts/container.sh envs/pipeline/bin/python -m pdfpipe.aggregate_parallel \
  "/workspace/$RUN_ROOT" --gpus "$USED_GPU_SPEC" || status=1

echo "$RUN_ROOT"
exit "$status"
