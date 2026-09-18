"""Accuracy-first, page-streaming PDF parsing pipeline."""

from __future__ import annotations

import hashlib
import html
import json
import pathlib
import time
import traceback
from collections import defaultdict

import fitz
import numpy as np
import pdfplumber
from PIL import Image, ImageDraw

from .engines import MinerUTableEngine, RapidOCREngine, TATREngine
from .tables import center_in, continuation_candidates, iou, merge_candidates, quality_check, render, words_in

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))
    temporary.replace(path)


def native_words(page):
    result = []
    for item in page.get_text("words", sort=True):
        x0, y0, x1, y1, text, block, line, number = item[:8]
        if not str(text).strip():
            continue
        result.append(
            {
                "text": str(text),
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "block": int(block),
                "line": int(line),
                "word": int(number),
                "source": "pymupdf",
            }
        )
    return result


def text_layer_quality(words):
    text = "".join(word["text"] for word in words)
    visible = [char for char in text if not char.isspace()]
    printable_ratio = sum(char.isprintable() and char != "\ufffd" for char in visible) / max(len(visible), 1)
    semantic_ratio = sum(char.isalnum() for char in visible) / max(len(visible), 1)
    replacement_ratio = text.count("\ufffd") / max(len(visible), 1)
    reliable = len(visible) >= 20 and len(words) >= 3 and printable_ratio >= 0.95 and semantic_ratio >= 0.25 and replacement_ratio <= 0.01
    return {
        "reliable": reliable,
        "visible_characters": len(visible),
        "word_count": len(words),
        "printable_ratio": round(printable_ratio, 4),
        "semantic_ratio": round(semantic_ratio, 4),
        "replacement_ratio": round(replacement_ratio, 4),
        "classification": "reliable_native" if reliable else "scan_or_unreliable_native",
    }


def render_page(page, dpi):
    scale = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _table_from_pdfplumber(found, words, page_number, table_id, strategy):
    boxes = [list(cell) for cell in found.cells if cell and cell[2] > cell[0] and cell[3] > cell[1]]
    xs = sorted({round(value, 3) for box in boxes for value in (box[0], box[2])})
    ys = sorted({round(value, 3) for box in boxes for value in (box[1], box[3])})
    cells = []
    for box in boxes:
        rounded = [round(value, 3) for value in box]
        row, col = ys.index(rounded[1]), xs.index(rounded[0])
        cell_words = words_in(words, box)
        cells.append(
            {
                "row": row,
                "col": col,
                "rowspan": ys.index(rounded[3]) - row,
                "colspan": xs.index(rounded[2]) - col,
                "text": " ".join(word["text"] for word in cell_words),
                "bbox": [float(value) for value in box],
                "header": row == 0,
                "source": {
                    "parser": "pdfplumber",
                    "page": page_number,
                    "geometry": "native_pdf",
                    "text_source": "pymupdf",
                },
            }
        )
    table = {
        "id": table_id,
        "page": page_number,
        "bbox": [float(value) for value in found.bbox],
        "parser": "pdfplumber",
        "strategy": strategy,
        "nrows": max((cell["row"] + cell["rowspan"] for cell in cells), default=0),
        "ncols": max((cell["col"] + cell["colspan"] for cell in cells), default=0),
        "cells": cells,
    }
    table["quality"] = quality_check(table, words_in(words, table["bbox"]), "pdfplumber")
    table["issues"] = table["quality"]["issues"]
    return table


def pdfplumber_tables(page, words, page_number):
    settings = [
        ("lines", {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 3, "join_tolerance": 3}),
    ]
    tables = []
    for strategy, config in settings:
        for index, found in enumerate(page.find_tables(table_settings=config)):
            table = _table_from_pdfplumber(found, words, page_number, f"pdfplumber-p{page_number:04d}-{strategy}-{index + 1:03d}", strategy)
            duplicate = next(
                (other for other in tables if iou(other["bbox"], table["bbox"]) > 0.90 and other["nrows"] == table["nrows"] and other["ncols"] == table["ncols"]),
                None,
            )
            if duplicate is None:
                tables.append(table)
    return tables


def text_blocks(words, excluded_boxes):
    kept = [word for word in words if not any(center_in(word["bbox"], box) for box in excluded_boxes)]
    groups = defaultdict(list)
    if kept and "block" in kept[0]:
        for word in kept:
            groups[(word.get("block", 0), word.get("line", 0))].append(word)
    else:
        line_id, last_y, tolerance = 0, None, 4.0
        for word in sorted(kept, key=lambda item: (item["bbox"][1], item["bbox"][0])):
            middle = (word["bbox"][1] + word["bbox"][3]) / 2
            if last_y is not None and abs(middle - last_y) > tolerance:
                line_id += 1
            groups[(line_id, 0)].append(word)
            last_y = middle if last_y is None else (last_y + middle) / 2
    blocks = []
    for index, (_, line_words) in enumerate(sorted(groups.items(), key=lambda item: (min(w["bbox"][1] for w in item[1]), min(w["bbox"][0] for w in item[1])))):
        line_words.sort(key=lambda word: word["bbox"][0])
        blocks.append(
            {
                "id": f"text-{index + 1:04d}",
                "type": "text",
                "text": " ".join(word["text"] for word in line_words),
                "bbox": [
                    min(word["bbox"][0] for word in line_words), min(word["bbox"][1] for word in line_words),
                    max(word["bbox"][2] for word in line_words), max(word["bbox"][3] for word in line_words),
                ],
                "source": line_words[0]["source"],
            }
        )
    return blocks


def visual_grid_evidence(crop):
    """Measure long horizontal/vertical rules; this is evidence, not parsing."""
    try:
        import cv2

        gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, crop.width // 18), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, crop.height // 18)))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        h_contours = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        v_contours = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        h_lines = sum(cv2.boundingRect(contour)[2] >= 0.35 * crop.width for contour in h_contours)
        v_lines = sum(cv2.boundingRect(contour)[3] >= 0.35 * crop.height for contour in v_contours)
        intersections = int(np.count_nonzero(cv2.bitwise_and(horizontal, vertical)))
        return {"horizontal_rules": h_lines, "vertical_rules": v_lines, "intersection_pixels": intersections, "strong": h_lines >= 3 and v_lines >= 3 and intersections >= 8}
    except Exception as exc:
        return {"strong": False, "error": repr(exc)}


def region_grid_evidence(image, page_width, page_height, bbox, padding=10):
    """Measure ruled-table evidence in a PDF-coordinate proposal."""
    sx, sy = image.width / page_width, image.height / page_height
    pixel_box = [
        max(0, int(bbox[0] * sx) - padding),
        max(0, int(bbox[1] * sy) - padding),
        min(image.width, int(bbox[2] * sx) + padding),
        min(image.height, int(bbox[3] * sy) + padding),
    ]
    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
        return {"strong": False, "pixel_crop": pixel_box, "error": "empty proposal crop"}
    evidence = visual_grid_evidence(image.crop(pixel_box))
    evidence["pixel_crop"] = pixel_box
    return evidence


def fallback_evidence(candidate, tatr_table, crop, threshold=4):
    max_detection = max((item["score"] for item in candidate["tatr_detections"]), default=0.0)
    native_grid = any(table.get("nrows", 0) >= 2 and table.get("ncols", 0) >= 2 for table in candidate["native_tables"])
    partial_structure = bool(tatr_table and tatr_table.get("nrows", 0) >= 2 and tatr_table.get("ncols", 0) >= 2)
    visual = visual_grid_evidence(crop)
    score = (2 if max_detection >= 0.85 else 0) + (3 if native_grid else 0) + (3 if partial_structure else 0) + (3 if visual.get("strong") else 0)
    structural_source = native_grid or partial_structure or visual.get("strong", False)
    return {
        "sufficient": bool(structural_source and score >= threshold),
        "score": score,
        "max_tatr_detection_score": round(max_detection, 4),
        "native_grid": native_grid,
        "partial_tatr_structure": partial_structure,
        "visual_grid": visual,
        "policy": f"requires score>={threshold} and native/partial-TATR/visual-grid structural evidence",
    }


class PDFPipeline:
    def __init__(self, dpi=144, max_pages=None, pages=None, use_mineru=True, resume=False, config=None):
        self.dpi = dpi
        self.max_pages = max_pages
        self.pages = pages
        self.use_mineru = use_mineru
        self.resume = resume
        defaults = (config or {}).get("defaults", {})
        self.detection_threshold = float(defaults.get("tatr_detection_threshold", 0.6))
        self.structure_threshold = float(defaults.get("tatr_structure_threshold", 0.5))
        self.candidate_iou = float(defaults.get("candidate_match_iou", 0.25))
        self.evidence_threshold = int(defaults.get("mineru_table_evidence_threshold", 4))
        self._rapidocr = None
        self._tatr = None
        self._mineru = None

    @property
    def rapidocr(self):
        if self._rapidocr is None:
            self._rapidocr = RapidOCREngine()
        return self._rapidocr

    @property
    def tatr(self):
        if self._tatr is None:
            self._tatr = TATREngine(self.detection_threshold, self.structure_threshold)
        return self._tatr

    @property
    def mineru(self):
        if not self.use_mineru:
            return None
        if self._mineru is None:
            self._mineru = MinerUTableEngine()
        return self._mineru

    def process_candidate(self, image, page_record, candidate, words, page_dir):
        candidate_dir = page_dir / "raw" / candidate["id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        attempts = []
        good_native = [table for table in candidate["native_tables"] if table["quality"]["passed"]]
        if good_native:
            table = max(good_native, key=lambda item: (item["strategy"] == "lines", len(item["cells"])))
            table = json.loads(json.dumps(table))
            table["id"] = candidate["id"]
            table["decision"] = "accepted_pdfplumber"
            table["region_sources"] = {"pdfplumber": [item["id"] for item in candidate["native_tables"]], "tatr_detection": candidate["tatr_detections"]}
            attempts.append({"stage": "pdfplumber", "status": "accepted", "quality": table["quality"]})
            atomic_json(candidate_dir / "decision.json", {"candidate": candidate, "attempts": attempts, "selected": table})
            return table, None

        line_evidence = [
            item.get("visual_grid", {})
            for item in candidate["native_tables"] + candidate["tatr_detections"]
            if item.get("visual_grid", {}).get("strong")
        ]
        attempts.append(
            {
                "stage": "visual-grid-candidate-gate",
                "status": "passed" if line_evidence else "rejected",
                "evidence": line_evidence,
            }
        )
        if not line_evidence:
            rejected = {
                "candidate": candidate,
                "classification": "borderless_or_false_positive",
                "reason": "no strong horizontal/vertical ruled grid",
                "content_preserved_as_text": True,
                "attempts": attempts,
            }
            atomic_json(candidate_dir / "decision.json", rejected)
            return None, rejected

        tatr_table, raw, crop = self.tatr.structure(image, page_record, candidate, words)
        crop.save(candidate_dir / "crop.png")
        atomic_json(candidate_dir / "tatr.json", raw)
        source_words = words_in(words, (tatr_table or candidate)["bbox"])
        tatr_quality = quality_check(tatr_table, source_words, "tatr") if tatr_table else {"passed": False, "stage": "tatr", "issues": ["no_structure_returned"]}
        attempts.append({"stage": "tatr-v1.1-all", "status": "accepted" if tatr_quality["passed"] else "failed_qc", "quality": tatr_quality})
        if tatr_table and tatr_quality["passed"]:
            tatr_table["quality"] = tatr_quality
            tatr_table["decision"] = "accepted_tatr_repair"
            atomic_json(candidate_dir / "decision.json", {"candidate": candidate, "attempts": attempts, "selected": tatr_table})
            return tatr_table, None

        evidence = fallback_evidence(candidate, tatr_table, crop, self.evidence_threshold)
        attempts.append({"stage": "table_evidence_gate", "status": "passed" if evidence["sufficient"] else "rejected", "evidence": evidence})
        if not evidence["sufficient"]:
            rejected = {"candidate": candidate, "classification": "figure_or_false_positive", "content_skipped": True, "attempts": attempts}
            atomic_json(candidate_dir / "decision.json", rejected)
            return None, rejected
        if not self.use_mineru:
            unresolved = {"candidate": candidate, "classification": "unresolved_table", "reason": "MinerU disabled", "attempts": attempts}
            atomic_json(candidate_dir / "decision.json", unresolved)
            return None, unresolved
        try:
            mineru_table, mineru_raw = self.mineru.recognize(crop, page_record, candidate)
            atomic_json(candidate_dir / "mineru.json", mineru_raw)
            quality = quality_check(mineru_table, words_in(words, candidate["bbox"]), "mineru")
            attempts.append({"stage": "mineru-table-region", "status": "accepted" if quality["passed"] else "failed_qc", "quality": quality})
            mineru_table["quality"] = quality
            if quality["passed"]:
                mineru_table["decision"] = "accepted_mineru_fallback"
                atomic_json(candidate_dir / "decision.json", {"candidate": candidate, "attempts": attempts, "selected": mineru_table})
                return mineru_table, None
            unresolved = {"candidate": candidate, "classification": "unresolved_table", "reason": "MinerU result failed source/structure QC", "attempts": attempts, "mineru_result": mineru_table}
        except Exception:
            attempts.append({"stage": "mineru-table-region", "status": "error", "traceback": traceback.format_exc()})
            unresolved = {"candidate": candidate, "classification": "unresolved_table", "reason": "MinerU error", "attempts": attempts}
        atomic_json(candidate_dir / "decision.json", unresolved)
        return None, unresolved

    def process_page(self, fitz_page, plumber_page, page_number, page_dir):
        started = time.time()
        page_dir.mkdir(parents=True, exist_ok=True)
        image = render_page(fitz_page, self.dpi)
        image.save(page_dir / "page.png")
        page_words = native_words(fitz_page)
        layer = text_layer_quality(page_words)
        ocr = None
        if layer["reliable"]:
            words, text_source = page_words, "pymupdf"
        else:
            words, ocr = self.rapidocr.words(image, float(fitz_page.rect.width), float(fitz_page.rect.height))
            text_source = "rapidocr"
            atomic_json(page_dir / "raw" / "rapidocr.json", ocr)
        record = {
            "page": page_number,
            "width": float(fitz_page.rect.width),
            "height": float(fitz_page.rect.height),
            "rotation": int(fitz_page.rotation),
            "image": f"pages/page-{page_number:04d}/page.png",
            "image_width": image.width,
            "image_height": image.height,
            "coordinate_system": "PDF points, top-left origin, display orientation",
            "text_layer": layer,
            "text_source": text_source,
            "ocr_runtime": None if ocr is None else {key: value for key, value in ocr.items() if key != "raw"},
            "words": words,
        }
        native_all = pdfplumber_tables(plumber_page, words, page_number) if layer["reliable"] else []
        native = []
        prefilter_rejected = []
        for table in native_all:
            if table["quality"]["passed"]:
                table["candidate_eligible"] = True
                native.append(table)
                continue
            evidence = region_grid_evidence(image, record["width"], record["height"], table["bbox"])
            table["visual_grid"] = evidence
            table["candidate_eligible"] = bool(evidence.get("strong"))
            if table["candidate_eligible"]:
                native.append(table)
            else:
                prefilter_rejected.append(
                    {
                        "classification": "invalid_pdfplumber_lines",
                        "proposal": table,
                        "reason": "pdfplumber QC failed and no strong visual grid",
                        "content_preserved_as_text": True,
                    }
                )
        detections, detection_raw = self.tatr.detect(image, record["width"], record["height"])
        eligible_detections = []
        for detection in detections:
            evidence = region_grid_evidence(image, record["width"], record["height"], detection["bbox"])
            detection["visual_grid"] = evidence
            detection["candidate_eligible"] = bool(evidence.get("strong"))
            if detection["candidate_eligible"]:
                eligible_detections.append(detection)
            else:
                prefilter_rejected.append(
                    {
                        "classification": "borderless_or_false_positive",
                        "proposal": detection,
                        "reason": "TATR Detection proposal has no strong visual grid",
                        "content_preserved_as_text": True,
                    }
                )
        atomic_json(
            page_dir / "raw" / "tatr-detection.json",
            {"normalized": detections, "eligible": eligible_detections, "raw": detection_raw},
        )
        candidates = merge_candidates(native, eligible_detections, page_number, self.candidate_iou)
        tables, rejected = [], list(prefilter_rejected)
        for candidate in candidates:
            table, reject = self.process_candidate(image, record, candidate, words, page_dir)
            if table:
                tables.append(table)
            if reject:
                rejected.append(reject)
        excluded = [table["bbox"] for table in tables]
        record.update(
            {
                "blocks": text_blocks(words, excluded),
                "tables": tables,
                "table_candidates": candidates,
                "rejected_or_unresolved": rejected,
                "seconds": round(time.time() - started, 3),
            }
        )
        atomic_json(page_dir / "page.json", record)
        return record

    def run_document(self, input_path, output):
        input_path, output = pathlib.Path(input_path), pathlib.Path(output)
        output.mkdir(parents=True, exist_ok=True)
        errors, pages = [], []
        document = fitz.open(input_path)
        with pdfplumber.open(input_path) as plumber:
            if self.pages is not None:
                unavailable = [page for page in self.pages if page > len(document)]
                if unavailable:
                    requested = ",".join(str(page) for page in unavailable)
                    raise ValueError(
                        f"Requested page(s) {requested} exceed document page count {len(document)}"
                    )
                page_numbers = self.pages
            else:
                count = len(document) if self.max_pages is None else min(len(document), self.max_pages)
                page_numbers = range(1, count + 1)
            for page_number in page_numbers:
                index = page_number - 1
                page_dir = output / "pages" / f"page-{page_number:04d}"
                page_json = page_dir / "page.json"
                if self.resume and page_json.exists():
                    pages.append(json.loads(page_json.read_text()))
                    print(f"page {page_number}/{len(document)}: resumed", flush=True)
                    continue
                try:
                    page = self.process_page(document[index], plumber.pages[index], page_number, page_dir)
                    pages.append(page)
                    print(f"page {page_number}/{len(document)}: {len(page['tables'])} tables, {page['text_source']}", flush=True)
                except Exception:
                    error = {"page": page_number, "traceback": traceback.format_exc()}
                    errors.append(error)
                    atomic_json(page_dir / "error.json", error)
                    print(f"page {page_number}/{len(document)}: failed", flush=True)
        tables = [table for page in pages for table in page.get("tables", [])]
        result = {
            "schema_version": "1.0",
            "status": "partial" if errors else "ok",
            "input": str(input_path),
            "page_count": len(document),
            "processed_pages": len(pages),
            "pages": pages,
            "tables": tables,
            "unresolved_cross_page_tables": continuation_candidates(tables, pages),
            "errors": errors,
            "metrics": {"status": "not_computed", "reason": "No ground-truth annotations supplied; model agreement is not accuracy"},
        }
        atomic_json(output / "document.json", result)
        self.write_report(output, result)
        runtime = {
            "tatr": None if self._tatr is None else self._tatr.runtime,
            "rapidocr": None if self._rapidocr is None else {"provider": self._rapidocr.provider, "events": self._rapidocr.events},
            "mineru": None if self._mineru is None else self._mineru.runtime,
        }
        return result, runtime

    @staticmethod
    def write_report(output, document):
        style = "body{font:15px sans-serif;margin:20px}section{display:grid;grid-template-columns:minmax(320px,46%) 1fr;gap:20px;margin:20px 0 40px}img{width:100%;height:auto}table{border-collapse:collapse;margin:8px 0 24px}td,th{border:1px solid #888;padding:4px;vertical-align:top}.warn{color:#a33}pre{white-space:pre-wrap;max-height:240px;overflow:auto}"
        body = ["<!doctype html><html lang='zh'><meta charset='utf-8'><title>PDF 解析人工核验</title>", f"<style>{style}</style>", "<h1>PDF 解析人工核验</h1>", "<p class='warn'>无标注时不计算准确率；候选一致性也不等于准确率。红框为采用的表格，蓝框为单元格。</p>"]
        for page in document["pages"]:
            source = Image.open(output / page["image"]).convert("RGB")
            draw = ImageDraw.Draw(source)
            sx, sy = source.width / page["width"], source.height / page["height"]
            for table in page["tables"]:
                box = table["bbox"]
                draw.rectangle((box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy), outline="red", width=4)
                for cell in table["cells"]:
                    if cell.get("bbox"):
                        c = cell["bbox"]
                        draw.rectangle((c[0] * sx, c[1] * sy, c[2] * sx, c[3] * sy), outline="blue", width=1)
            overlay = output / "pages" / f"page-{page['page']:04d}" / "overlay.png"
            source.save(overlay)
            body.append(f"<h2>第 {page['page']} 页 · {html.escape(page['text_source'])}</h2><section><img src='pages/page-{page['page']:04d}/overlay.png'><article>")
            for table in page["tables"]:
                table_html = render(table)
                table_path = output / "pages" / f"page-{page['page']:04d}" / "tables" / f"{table['id']}.html"
                table_path.parent.mkdir(parents=True, exist_ok=True)
                table_path.write_text("<meta charset='utf-8'>" + table_html)
                body.append(f"<h3>{html.escape(table['id'])} · {html.escape(table['decision'])}</h3>{table_html}<pre>{html.escape(json.dumps(table.get('quality'), ensure_ascii=False, indent=2))}</pre>")
            if not page["tables"]:
                body.append("<p>本页无采用表格。</p>")
            body.append(f"<h3>正文</h3><pre>{html.escape(chr(10).join(block['text'] for block in page['blocks']))}</pre>")
            if page["rejected_or_unresolved"]:
                body.append(f"<h3>丢弃/未解决候选</h3><pre>{html.escape(json.dumps(page['rejected_or_unresolved'], ensure_ascii=False, indent=2))}</pre>")
            body.append("</article></section>")
        body.append("</html>")
        (output / "index.html").write_text("\n".join(body))


def document_output_name(path):
    path = pathlib.Path(path)
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    return f"{path.stem}-{digest[:8]}", digest
