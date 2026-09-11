"""Command-line entry point for DOC/DOCX parsing."""

from __future__ import annotations

import argparse
import datetime
import html
import json
import pathlib
import platform
import sys

import lxml

from content_export import export_document

from .pipeline import WordPipeline, atomic_json, document_output_name

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUPPORTED = {".doc", ".docx"}


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


def input_files(value):
    path = pathlib.Path(value)
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED)
    else:
        files = [path]
    if not files or any(not item.is_file() or item.suffix.lower() not in SUPPORTED for item in files):
        raise SystemExit(f"No readable DOC/DOCX input: {value}")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description="DOC/DOCX 原生文字与表格结构化解析")
    parser.add_argument("input", help="DOC/DOCX 文件或包含 Word 文件的目录")
    parser.add_argument("--run-id", help="输出到 runs/<run-id>；默认使用 UTC 时间")
    parser.add_argument("--resume", action="store_true", help="复用已经完成的 document.json")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.json"), help="图像型回退使用的 PDF 配置")
    parser.add_argument("--dpi", type=int, default=144, help="仅图像型 PDF 回退使用")
    parser.add_argument("--disable-mineru", action="store_true", help="仅图像型 PDF 回退使用")
    args = parser.parse_args(argv)
    if args.dpi < 96:
        parser.error("--dpi must be at least 96")
    if not pathlib.Path(args.config).is_file():
        parser.error(f"config does not exist: {args.config}")
    files = input_files(args.input)
    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit("Run already exists; choose another --run-id or use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("a", buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    pipeline = WordPipeline(
        resume=args.resume,
        pdf_config=args.config,
        dpi=args.dpi,
        use_mineru=not args.disable_mineru,
    )
    summaries = []
    for input_path in files:
        name, digest = document_output_name(input_path)
        output = run_dir / name
        document_path = output / "document.json"
        if args.resume and document_path.exists():
            result = json.loads(document_path.read_text())
            runtime = {"resumed": True}
            print(f"{input_path}: resumed", flush=True)
        else:
            result, runtime = pipeline.run_document(input_path, output)
            print(
                f"{input_path}: {len(result.get('blocks', []))} blocks, "
                f"{len(result.get('tables', []))} tables, {result['parse_mode']}",
                flush=True,
            )
        export_document(result, output)
        manifest = {
            "input": str(input_path.resolve()),
            "sha256": digest,
            "parameters": vars(args),
            "runtime": runtime,
            "environment": {
                "python": platform.python_version(),
                "lxml": lxml.__version__,
                "libreoffice": pipeline.converter.version(),
            },
            "coordinate_system": "flow order and OpenXML path; no fabricated page/cell coordinates",
            "image_handling": "images skipped; image-only Word documents use the frozen PDF pipeline",
        }
        atomic_json(output / "manifest.json", manifest)
        summaries.append(
            {
                "input": str(input_path),
                "output": name,
                "status": result["status"],
                "blocks": len(result.get("blocks", [])),
                "tables": len(result.get("tables", [])),
                "mode": result["parse_mode"],
            }
        )
        atomic_json(run_dir / "summary.json", summaries)
    links = ["<meta charset='utf-8'><h1>Word 解析批次</h1>"]
    for summary in summaries:
        links.append(
            f"<p><a href='{html.escape(summary['output'])}/index.html'>"
            f"{html.escape(summary['input'])}</a> · {summary['status']} · "
            f"{summary['blocks']} 块 · {summary['tables']} 表 · {html.escape(summary['mode'])}</p>"
        )
    (run_dir / "index.html").write_text("\n".join(links))
    print(run_dir, flush=True)
    return 1 if any(summary["status"] != "ok" for summary in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
