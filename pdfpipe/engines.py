"""GPU model adapters. Only RapidOCR is allowed to fall back to CPU."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import numpy as np
from PIL import Image

from .official import tatr_functions
from .tables import check, parse_html

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def require_cuda(torch):
    requested = os.environ.get("PARSER_DEVICE", "cuda:0")
    if not requested.startswith("cuda"):
        raise RuntimeError(f"PARSER_DEVICE={requested!r}: TATR and MinerU require CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; TATR/MinerU CPU fallback is disabled")
    device = torch.device("cuda:0" if requested == "cuda" else requested)
    torch.cuda.set_device(device)
    return device


class RapidOCREngine:
    """Unified RapidOCR with Torch CUDA first and ONNX Runtime CPU fallback."""

    def __init__(self):
        self.engine = None
        self.provider = None
        self.events = []
        self.model_root = ROOT / "models" / "rapidocr"
        self._start_gpu_or_cpu()

    @staticmethod
    def _params(provider, model_root):
        from rapidocr.utils.typings import EngineType

        engine_type = EngineType.TORCH if provider == "torch-cuda" else EngineType.ONNXRUNTIME
        params = {
            # RapidOCR 3.9.2's Torch loader applies pathlib's `/` operator.
            "Global.model_root_dir": model_root,
            "Global.log_level": "warning",
            "Global.return_word_box": False,
            "Det.engine_type": engine_type,
            "Cls.engine_type": engine_type,
            "Rec.engine_type": engine_type,
        }
        if provider == "torch-cuda":
            params["EngineConfig.torch.use_cuda"] = True
            params["EngineConfig.torch.cuda_ep_cfg.device_id"] = 0
        else:
            params["EngineConfig.onnxruntime.use_cuda"] = False
        return params

    def _create(self, provider):
        from rapidocr import RapidOCR

        return RapidOCR(params=self._params(provider, self.model_root))

    def _start_gpu_or_cpu(self):
        requested = os.environ.get("RAPIDOCR_DEVICE", "auto").lower()
        if requested not in {"auto", "cuda", "cpu"}:
            raise RuntimeError(f"Unknown RAPIDOCR_DEVICE={requested!r}")
        if requested != "cpu":
            try:
                self.engine = self._create("torch-cuda")
                self.provider = "torch-cuda"
                return
            except Exception as exc:  # explicitly authorized silent operational fallback
                self.events.append({"event": "gpu_init_failed_cpu_fallback", "error": repr(exc)})
        self.engine = self._create("onnxruntime-cpu")
        self.provider = "onnxruntime-cpu"

    def _fallback(self, exc):
        self.events.append({"event": "gpu_inference_failed_cpu_fallback", "error": repr(exc)})
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.engine = self._create("onnxruntime-cpu")
        self.provider = "onnxruntime-cpu"

    def words(self, image: Image.Image, page_width: float, page_height: float):
        try:
            output = self.engine(np.asarray(image))
        except Exception as exc:
            if self.provider != "torch-cuda":
                raise
            self._fallback(exc)
            output = self.engine(np.asarray(image))
        sx, sy = image.width / page_width, image.height / page_height
        boxes = getattr(output, "boxes", None)
        texts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        words = []
        if boxes is not None and texts is not None:
            scores = scores if scores is not None else [None] * len(texts)
            for polygon, text, score in zip(boxes, texts, scores):
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                words.append(
                    {
                        "text": str(text),
                        "bbox": [min(xs) / sx, min(ys) / sy, max(xs) / sx, max(ys) / sy],
                        "confidence": None if score is None else float(score),
                        "source": "rapidocr",
                    }
                )
        words.sort(key=lambda word: (word["bbox"][1], word["bbox"][0]))
        return words, {
            "provider": self.provider,
            "gpu_preferred": True,
            "cpu_fallback_allowed": True,
            "events": list(self.events),
            "raw": output.to_json() if hasattr(output, "to_json") else _plain(output),
        }


class TATREngine:
    def __init__(self, detection_threshold=0.6, structure_threshold=0.5):
        import torch
        from transformers import AutoImageProcessor, TableTransformerConfig, TableTransformerForObjectDetection

        self.torch = torch
        self.device = require_cuda(torch)
        self.detection_threshold = detection_threshold
        self.structure_threshold = structure_threshold
        self.processors, self.models = {}, {}
        compatibility = []
        for name in ("tatr", "tatr-detection"):
            path = ROOT / "models" / name
            self.processors[name] = AutoImageProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
            config_data = json.loads((path / "config.json").read_text())
            if config_data.get("dilation", "absent") is None:
                config_data.pop("dilation")
                compatibility.append(f"{name}_null_dilation_uses_default")
            config = TableTransformerConfig.from_dict(config_data)
            self.models[name] = TableTransformerForObjectDetection.from_pretrained(
                path, config=config, local_files_only=True
            ).to(self.device, dtype=torch.float32).eval()
        self.postprocess = tatr_functions()
        self.runtime = {
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device),
            "dtype": "float32",
            "cpu_fallback": False,
            "compatibility": compatibility,
        }

    def _objects(self, image, name, threshold):
        processor = self.processors[name]
        inputs = processor(
            images=image,
            return_tensors="pt",
            size={"shortest_edge": 800, "longest_edge": 800},
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.models[name](**inputs)
        target = self.torch.tensor([[image.height, image.width]], device=self.device)
        result = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target)[0]
        return [
            {
                "label": self.models[name].config.id2label[int(label)],
                "score": float(score),
                "bbox": [float(value) for value in box.tolist()],
            }
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"])
        ]

    def detect(self, image, page_width, page_height):
        sx, sy = image.width / page_width, image.height / page_height
        raw = self._objects(image, "tatr-detection", self.detection_threshold)
        detections = []
        for item in raw:
            if item["label"] not in {"table", "table rotated"}:
                continue
            box = item["bbox"]
            detections.append(
                {
                    "label": item["label"],
                    "score": item["score"],
                    "bbox": [box[0] / sx, box[1] / sy, box[2] / sx, box[3] / sy],
                    "model": "microsoft/table-transformer-detection",
                }
            )
        return detections, raw

    @staticmethod
    def _rotate_box_ccw(box, width):
        return [box[1], width - box[2], box[3], width - box[0]]

    @staticmethod
    def _unrotate_box_ccw(box, width):
        return [width - box[3], box[0], width - box[1], box[2]]

    def structure(self, image, page, candidate, words):
        detections = sorted(candidate["tatr_detections"], key=lambda item: item["score"], reverse=True)
        native = sorted(
            candidate["native_tables"],
            key=lambda item: (item.get("quality", {}).get("passed", False), len(item.get("cells", []))),
            reverse=True,
        )
        proposal = detections[0] if detections else native[0]
        region = proposal["bbox"]
        sx, sy = image.width / page["width"], image.height / page["height"]
        padding = 10
        pixel_box = [
            max(0, int(region[0] * sx) - padding),
            max(0, int(region[1] * sy) - padding),
            min(image.width, int(region[2] * sx) + padding),
            min(image.height, int(region[3] * sy) + padding),
        ]
        crop = image.crop(pixel_box)
        rotated = proposal.get("label") == "table rotated"
        inference_image = crop.rotate(90, expand=True) if rotated else crop
        local_tokens = []
        for index, word in enumerate(words):
            box = word["bbox"]
            local = [
                box[0] * sx - pixel_box[0], box[1] * sy - pixel_box[1],
                box[2] * sx - pixel_box[0], box[3] * sy - pixel_box[1],
            ]
            if not (0 <= (local[0] + local[2]) / 2 <= crop.width and 0 <= (local[1] + local[3]) / 2 <= crop.height):
                continue
            if rotated:
                local = self._rotate_box_ccw(local, crop.width)
            local_tokens.append(
                {"bbox": local, "text": word["text"], "span_num": index, "line_num": index, "block_num": 0}
            )
        objects = self._objects(inference_image, "tatr", 0.3)
        thresholds = {label: self.structure_threshold for label in self.models["tatr"].config.id2label.values()}
        structures = self.postprocess.objects_to_structures(objects, local_tokens, thresholds)
        outputs = []
        for structure in structures:
            cells, confidence = self.postprocess.structure_to_cells(structure, local_tokens)
            normalized = []
            for cell in cells:
                box = [float(value) for value in cell["bbox"]]
                if rotated:
                    box = self._unrotate_box_ccw(box, crop.width)
                box = [
                    (box[0] + pixel_box[0]) / sx, (box[1] + pixel_box[1]) / sy,
                    (box[2] + pixel_box[0]) / sx, (box[3] + pixel_box[1]) / sy,
                ]
                normalized.append(
                    {
                        "row": min(cell["row_nums"]),
                        "col": min(cell["column_nums"]),
                        "rowspan": len(cell["row_nums"]),
                        "colspan": len(cell["column_nums"]),
                        "text": cell.get("cell text", ""),
                        "bbox": box,
                        "header": bool(cell.get("column header", False)),
                        "source": {
                            "parser": "tatr-v1.1-all",
                            "page": page["page"],
                            "geometry": "model",
                            "text_source": page["text_source"],
                        },
                    }
                )
            table = {
                "id": candidate["id"],
                "page": page["page"],
                "bbox": [pixel_box[0] / sx, pixel_box[1] / sy, pixel_box[2] / sx, pixel_box[3] / sy],
                "parser": "tatr-v1.1-all",
                "nrows": max((cell["row"] + cell["rowspan"] for cell in normalized), default=0),
                "ncols": max((cell["col"] + cell["colspan"] for cell in normalized), default=0),
                "cells": normalized,
                "confidence": float(confidence),
                "rotated": rotated,
                "region_sources": {
                    "pdfplumber": [item["id"] for item in candidate["native_tables"]],
                    "tatr_detection": candidate["tatr_detections"],
                },
            }
            table["issues"] = check(table)
            outputs.append(table)
        outputs.sort(key=lambda table: (table["confidence"], len(table["cells"])), reverse=True)
        raw = {
            "candidate": candidate,
            "selected_bbox": region,
            "pixel_crop": pixel_box,
            "rotated": rotated,
            "objects": objects,
            "tokens": local_tokens,
            "structures_found": len(structures),
        }
        return (outputs[0] if outputs else None), raw, crop


class MinerUTableEngine:
    def __init__(self):
        import torch
        from mineru_vl_utils import MinerUClient
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = require_cuda(torch)
        requested = os.environ.get("PARSER_MINERU_DTYPE", os.environ.get("PARSER_DTYPE", "auto")).lower()
        choices = {
            "fp32": torch.float32, "float32": torch.float32,
            "fp16": torch.float16, "float16": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        }
        if requested == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        elif requested in choices:
            self.dtype = choices[requested]
        else:
            raise RuntimeError(f"Unknown PARSER_MINERU_DTYPE={requested!r}")
        if self.dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested for MinerU but torch reports it unsupported")
        path = ROOT / "models" / "mineru"
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device).eval()
        self.client = MinerUClient(backend="transformers", model=self.model, processor=self.processor, use_tqdm=False)
        self.runtime = {
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "cpu_fallback": False,
            "entry": "MinerUClient.content_extract(type='table')",
        }

    def recognize(self, crop, page, candidate):
        output = self.client.content_extract(crop, type="table")
        raw = "" if output is None else str(output)
        table = parse_html(raw, page["page"], candidate["bbox"], "mineru-table-fallback", candidate["id"])
        table["region_sources"] = {
            "pdfplumber": [item["id"] for item in candidate["native_tables"]],
            "tatr_detection": candidate["tatr_detections"],
        }
        return table, {"output": raw, "entry": "MinerUClient.content_extract", "type": "table"}
