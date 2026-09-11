"""Aggregate isolated multi-GPU worker runs into one batch index."""

from __future__ import annotations

import argparse
import html
import json
import os
import pathlib

from .pipeline import atomic_json


def _link_document(run_root, worker_dir, output):
    link = run_root / output
    target = pathlib.Path(".parallel") / worker_dir.name / output
    if link.is_symlink():
        if pathlib.Path(os.readlink(link)) != target:
            raise RuntimeError(f"conflicting result link: {link}")
        return
    if link.exists():
        raise RuntimeError(f"result already exists and is not a worker link: {link}")
    link.symlink_to(target, target_is_directory=True)


def aggregate(run_root, gpu_spec):
    run_root = pathlib.Path(run_root)
    gpu_ids = gpu_spec.split(",")
    worker_root = run_root / ".parallel"
    status_root = run_root / ".parallel-status"
    worker_names = {path.name for path in worker_root.glob("worker-*")}
    worker_names.update(path.name.removesuffix(".exit") for path in status_root.glob("worker-*.exit"))
    summaries = []
    worker_status = []
    for worker_number, worker_name in enumerate(sorted(worker_names)):
        worker = worker_root / worker_name
        status_path = status_root / f"{worker_name}.exit"
        exit_code = int(status_path.read_text().strip()) if status_path.exists() else None
        worker_status.append(
            {
                "worker": worker_name,
                "gpu_id": gpu_ids[worker_number] if worker_number < len(gpu_ids) else None,
                "exit_code": exit_code,
                "run_log": f".parallel/{worker_name}/run.log",
            }
        )
        summary_path = worker / "summary.json"
        if not summary_path.exists():
            continue
        for item in json.loads(summary_path.read_text()):
            _link_document(run_root, worker, item["output"])
            summaries.append({**item, "worker": worker.name})

    summaries.sort(key=lambda item: item["input"])
    atomic_json(run_root / "summary.json", summaries)
    atomic_json(
        run_root / "parallel.json",
        {
            "gpu_ids": gpu_ids,
            "worker_count": len(worker_names),
            "workers": worker_status,
            "documents": len(summaries),
        },
    )
    links = ["<meta charset='utf-8'><h1>PDF 多 GPU 解析批次</h1>"]
    for item in summaries:
        links.append(
            f"<p><a href='{html.escape(item['output'])}/index.html'>"
            f"{html.escape(item['input'])}</a> · {item['status']} · "
            f"{item['pages']} 页 · {item['tables']} 表 · {item['worker']}</p>"
        )
    (run_root / "index.html").write_text("\n".join(links))
    (run_root / "run.log").write_text(
        "Parallel worker logs:\n"
        + "\n".join(
            f"{item['worker']}: exit={item['exit_code']} {item['run_log']}"
            for item in worker_status
        )
        + "\n"
    )
    failed = any(item["exit_code"] not in (0,) for item in worker_status)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="汇总多 GPU PDF 批次")
    parser.add_argument("run_root")
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args(argv)
    return aggregate(args.run_root, args.gpus)


if __name__ == "__main__":
    raise SystemExit(main())
