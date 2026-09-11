"""Regenerate content-only exports from one result or a result tree."""

from __future__ import annotations

import argparse
import json
import pathlib

from .exporter import export_document


def _documents(path):
    path = pathlib.Path(path)
    if path.is_file():
        if path.name != "document.json":
            raise SystemExit("input file must be document.json")
        candidates = [path]
    elif path.is_dir():
        direct = path / "document.json"
        candidates = [direct] if direct.is_file() else sorted(path.rglob("document.json"))
    else:
        raise SystemExit(f"input does not exist: {path}")
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description="从解析结果生成知识库内容 JSON/HTML/Markdown")
    parser.add_argument("input", help="document.json、单文档结果目录或批次/结果根目录")
    args = parser.parse_args(argv)
    count = 0
    for document_path in _documents(args.input):
        document = json.loads(document_path.read_text(encoding="utf-8"))
        export_document(document, document_path.parent)
        count += 1
        print(document_path.parent)
    if not count:
        raise SystemExit(f"no document.json found under: {args.input}")
    print(f"exported {count} document(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
