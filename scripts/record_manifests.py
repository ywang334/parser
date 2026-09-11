"""Hash all model bytes used by the current pipeline."""

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOS = {
    "mineru": ("OpenDataLab/MinerU2.5-Pro-2605-1.2B", "ModelScope/Hugging Face"),
    "tatr": ("microsoft/table-transformer-structure-recognition-v1.1-all", "Hugging Face"),
    "tatr-detection": ("microsoft/table-transformer-detection", "Hugging Face"),
    "rapidocr": ("RapidAI/RapidOCR model registry", "RapidOCR official model registry"),
}

for name, (repository, source) in REPOS.items():
    directory = ROOT / "models" / name
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "._____temp" in path.parts or path.name in (".msc", ".mv"):
            continue
        files.append({
            "file": str(path.relative_to(directory)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        })
    record = {"repository": repository, "source": source, "files": files}
    (ROOT / "manifests" / f"model-{name}.json").write_text(json.dumps(record, indent=2))
    print(name, sum(item["bytes"] for item in files), flush=True)
