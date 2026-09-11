"""Command-line entry point for the single integrated pipeline."""

from __future__ import annotations

import argparse
import datetime
import html
import json
import math
import os
import pathlib
import sys

from content_export import export_document

from .pipeline import PDFPipeline, atomic_json, document_output_name

ROOT = pathlib.Path(__file__).resolve().parents[1]


def probability(value):
    """Parse a finite probability value for CLI threshold overrides."""
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return result


def page_selection(value):
    """Parse a 1-based page selector such as 1,3-5."""
    pages = set()
    try:
        for raw_part in value.split(","):
            part = raw_part.strip()
            bounds = part.split("-")
            if not part or len(bounds) > 2 or any(not bound.strip() for bound in bounds):
                raise ValueError
            start = int(bounds[0])
            end = int(bounds[-1])
            if start < 1 or end < start:
                raise ValueError
            if end - start >= 100000:
                raise ValueError
            pages.update(range(start, end + 1))
            if len(pages) > 100000:
                raise ValueError
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "must contain positive page numbers/ranges such as 1,3-5"
        ) from exc
    return sorted(pages)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def cuda_preflight():
    import torch

    requested = os.environ.get("PARSER_DEVICE", "cuda:0")
    if not requested.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; stopped before parsing (TATR/MinerU CPU fallback is disabled)")
    device = torch.device("cuda:0" if requested == "cuda" else requested)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    required_gib = float(os.environ.get("PARSER_MIN_FREE_GIB", "6"))
    if free < required_gib * 1024**3:
        raise SystemExit(
            f"Only {free / 1024**3:.2f} GiB GPU memory is free; "
            f"need at least {required_gib:.2f} GiB. Stopped without CPU/model downgrade."
        )
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "free_gib": round(free / 1024**3, 3),
        "total_gib": round(total / 1024**3, 3),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "minimum_free_gib": required_gib,
    }


def input_files(value):
    path = pathlib.Path(value)
    if path.is_dir():
        files = sorted(path.glob("*.pdf"))
    else:
        files = [path]
    missing = [item for item in files if not item.is_file()]
    if missing or not files:
        raise SystemExit(f"No readable PDF input: {value}")
    return files


def shard_files(files, index, count):
    """Return one deterministic round-robin shard."""
    if count < 1 or index < 0 or index >= count:
        raise ValueError("shard index/count must satisfy 0 <= index < count")
    return files[index::count]


def manifests():
    result = {}
    for name in ("tatr", "tatr-detection", "mineru", "rapidocr"):
        path = ROOT / "manifests" / f"model-{name}.json"
        result[name] = json.loads(path.read_text()) if path.exists() else {"status": "manifest_missing"}
    return result


def main(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(ROOT / "config" / "default.json"))
    config_args, _ = config_parser.parse_known_args(argv)
    config_path = pathlib.Path(config_args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config does not exist: {config_path}")
    config = json.loads(config_path.read_text())
    parser = argparse.ArgumentParser(description="逐页 PDF 内容与复杂表格结构化解析")
    parser.add_argument("input", help="PDF 文件或包含 PDF 的目录")
    parser.add_argument("--config", default=str(config_path), help="流水线 JSON 配置")
    parser.add_argument("--run-id", help="输出到 runs/<run-id>；默认使用 UTC 时间")
    parser.add_argument("--resume", action="store_true", help="复用同一 run-id 中已完成的 page.json")
    parser.add_argument("--dpi", type=int, default=int(config["defaults"]["dpi"]), help="页面渲染 DPI")
    parser.add_argument(
        "--tatr-detection-threshold",
        type=probability,
        help="覆盖配置中的 TATR Detection 候选阈值（0～1）",
    )
    page_group = parser.add_mutually_exclusive_group()
    page_group.add_argument("--pages", type=page_selection, metavar="SPEC", help="指定页码，如 1,3-5")
    page_group.add_argument("--max-pages", type=int, help="仅调试使用；默认不截断")
    parser.add_argument("--disable-mineru", action="store_true", help="不运行最终 MinerU 表格区域补救")
    parser.add_argument("--parallel-gpus", metavar="IDS", help="目录批量多 GPU，如 0,1；仅由 scripts/run.sh 调度")
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.parallel_gpus is not None:
        parser.error("--parallel-gpus must be launched through scripts/run.sh")
    if args.dpi < 96:
        parser.error("--dpi must be at least 96")
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be positive")
    if args.tatr_detection_threshold is not None:
        config["defaults"]["tatr_detection_threshold"] = args.tatr_detection_threshold
    if (args.shard_index is None) != (args.shard_count is None):
        parser.error("--shard-index and --shard-count must be used together")
    files = input_files(args.input)
    if args.shard_count is not None:
        try:
            files = shard_files(files, args.shard_index, args.shard_count)
        except ValueError as exc:
            parser.error(str(exc))
        if not files:
            parser.error("the selected worker shard contains no PDF files")
    preflight = cuda_preflight()
    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit("Run already exists; choose another --run-id or use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("a", buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    print(f"GPU preflight: {preflight}", flush=True)
    summaries = []
    pipeline = PDFPipeline(
        dpi=args.dpi,
        max_pages=args.max_pages,
        pages=args.pages,
        use_mineru=not args.disable_mineru,
        resume=args.resume,
        config=config,
    )
    for input_path in files:
        name, digest = document_output_name(input_path)
        output = run_dir / name
        result, runtime = pipeline.run_document(input_path, output)
        export_document(result, output)
        previous_manifest_path = output / "manifest.json"
        previous_runtime = {}
        if args.resume and previous_manifest_path.exists():
            previous_runtime = json.loads(previous_manifest_path.read_text()).get("runtime", {})
        runtime = {
            key: value if value is not None else previous_runtime.get(key)
            for key, value in runtime.items()
        }
        manifest = {
            "input": str(input_path.resolve()),
            "sha256": digest,
            "parameters": vars(args),
            "config": config,
            "runtime": runtime,
            "gpu_preflight": preflight,
            "models": manifests(),
            "coordinate_system": "PDF points, top-left origin, display orientation",
            "page_processing": "streaming; no default page limit; atomic page.json checkpoints",
            "metrics": result["metrics"],
        }
        atomic_json(output / "manifest.json", manifest)
        summaries.append({"input": str(input_path), "output": name, "status": result["status"], "pages": result["processed_pages"], "tables": len(result["tables"])})
        atomic_json(run_dir / "summary.json", summaries)
    links = ["<meta charset='utf-8'><h1>PDF 解析批次</h1>"]
    for summary in summaries:
        links.append(f"<p><a href='{html.escape(summary['output'])}/index.html'>{html.escape(summary['input'])}</a> · {summary['status']} · {summary['pages']} 页 · {summary['tables']} 表</p>")
    (run_dir / "index.html").write_text("\n".join(links))
    print(run_dir, flush=True)
    if any(summary["status"] != "ok" for summary in summaries):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
