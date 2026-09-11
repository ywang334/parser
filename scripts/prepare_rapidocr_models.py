"""Materialize both RapidOCR inference formats in /workspace/models/rapidocr."""

from pathlib import Path

from rapidocr import RapidOCR
from rapidocr.utils.typings import EngineType

root = Path("/workspace/models/rapidocr")
root.mkdir(parents=True, exist_ok=True)

for engine_type in (EngineType.TORCH, EngineType.ONNXRUNTIME):
    params = {
        "Global.model_root_dir": root,
        "Global.log_level": "warning",
        "Det.engine_type": engine_type,
        "Cls.engine_type": engine_type,
        "Rec.engine_type": engine_type,
    }
    if engine_type == EngineType.TORCH:
        params["EngineConfig.torch.use_cuda"] = False
    else:
        params["EngineConfig.onnxruntime.use_cuda"] = False
    RapidOCR(params=params)
    print(f"prepared {engine_type.value}", flush=True)
