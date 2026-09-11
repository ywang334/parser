#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
  echo "usage: scripts/run.sh FILE_OR_DIRECTORY [--format pdf|word] [parser options]" >&2
  exit 2
fi

INPUT="$1"
shift
FORMAT=""
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --format)
      [ "$#" -ge 2 ] || { echo "--format requires pdf or word" >&2; exit 2; }
      FORMAT="$2"
      shift 2
      ;;
    --format=*)
      FORMAT="${1#*=}"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$FORMAT" ]; then
  if [ -f "$INPUT" ]; then
    extension="${INPUT##*.}"
    case "${extension,,}" in
      pdf) FORMAT="pdf" ;;
      doc|docx) FORMAT="word" ;;
      *) echo "unsupported input file: $INPUT" >&2; exit 2 ;;
    esac
  elif [ -d "$INPUT" ]; then
    has_pdf=0
    has_word=0
    find "$INPUT" -maxdepth 1 -type f -iname '*.pdf' -print -quit | read -r _ && has_pdf=1 || true
    find "$INPUT" -maxdepth 1 -type f \( -iname '*.doc' -o -iname '*.docx' \) -print -quit | read -r _ && has_word=1 || true
    if [ "$has_pdf" -eq 1 ] && [ "$has_word" -eq 1 ]; then
      echo "directory contains both PDF and Word files; specify --format pdf or --format word" >&2
      exit 2
    elif [ "$has_pdf" -eq 1 ]; then
      FORMAT="pdf"
    elif [ "$has_word" -eq 1 ]; then
      FORMAT="word"
    else
      echo "no supported PDF/DOC/DOCX files found in $INPUT" >&2
      exit 2
    fi
  else
    echo "input does not exist: $INPUT" >&2
    exit 2
  fi
fi

if [ "$FORMAT" != "pdf" ] && [ "$FORMAT" != "word" ]; then
  echo "--format must be pdf or word" >&2
  exit 2
fi

if [ "$FORMAT" = "word" ]; then
  for argument in "${ARGS[@]}"; do
    case "$argument" in
      --parallel-gpus|--parallel-gpus=*)
        echo "--parallel-gpus is only available for PDF directory batches" >&2
        exit 2
        ;;
    esac
  done
  exec bash scripts/container.sh envs/pipeline/bin/python -m wordpipe.cli "$INPUT" "${ARGS[@]}"
fi

for argument in "${ARGS[@]}"; do
  case "$argument" in
    --parallel-gpus|--parallel-gpus=*)
      exec bash scripts/run_parallel.sh "$INPUT" "${ARGS[@]}"
      ;;
  esac
done
exec bash scripts/container.sh envs/pipeline/bin/python -m pdfpipe.cli "$INPUT" "${ARGS[@]}"
