"""Table normalization, candidate fusion, quality checks, and HTML output."""

from __future__ import annotations

import html
import re
from collections import Counter
from typing import Iterable

from lxml import etree


def bbox_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) if box else 0.0


def bbox_intersection(a, b):
    if not a or not b:
        return 0.0
    return bbox_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    intersection = bbox_intersection(a, b)
    return intersection / max(bbox_area(a) + bbox_area(b) - intersection, 1e-9)


def overlap_min(a, b):
    return bbox_intersection(a, b) / max(min(bbox_area(a), bbox_area(b)), 1e-9)


def center_in(box, outer, margin=0.0):
    if not box or not outer:
        return False
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return outer[0] - margin <= x <= outer[2] + margin and outer[1] - margin <= y <= outer[3] + margin


def words_in(words: Iterable[dict], box: list[float]):
    return [word for word in words if center_in(word.get("bbox"), box)]


def parse_html(source, page, bbox, parser, table_id):
    """Normalize HTML rows/spans without inventing missing cell geometry."""
    root = etree.HTML(source)
    if root is None or not root.xpath("//table"):
        raise ValueError("No HTML table")
    node = root.xpath("//table")[0]
    cells, occupied = [], set()
    rows = node.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    for row_index, tr in enumerate(rows):
        column_index = 0
        for td in tr.xpath("./td|./th"):
            while (row_index, column_index) in occupied:
                column_index += 1
            rowspan = int(td.get("rowspan", "1"))
            colspan = int(td.get("colspan", "1"))
            if not 1 <= rowspan <= 1000 or not 1 <= colspan <= 1000:
                raise ValueError("Invalid or excessive cell span")
            cell = {
                "row": row_index,
                "col": column_index,
                "rowspan": rowspan,
                "colspan": colspan,
                "text": "".join(td.itertext()).strip(),
                "header": td.tag == "th" or tr.getparent().tag == "thead",
                "bbox": None,
                "source": {
                    "parser": parser,
                    "page": page,
                    "region_bbox": bbox,
                    "geometry": "unavailable",
                },
            }
            cells.append(cell)
            for rr in range(row_index, row_index + rowspan):
                for cc in range(column_index, column_index + colspan):
                    occupied.add((rr, cc))
            column_index += colspan
    result = {
        "id": table_id,
        "page": page,
        "bbox": bbox,
        "parser": parser,
        "cells": cells,
        "nrows": max((c["row"] + c["rowspan"] for c in cells), default=0),
        "ncols": max((c["col"] + c["colspan"] for c in cells), default=0),
        "raw_html": source,
    }
    result["issues"] = check(result)
    return result


def render(table):
    rows = []
    for row_index in range(table.get("nrows", 0)):
        row = []
        cells = sorted(
            (cell for cell in table.get("cells", []) if cell["row"] == row_index),
            key=lambda cell: cell["col"],
        )
        for cell in cells:
            tag = "th" if cell.get("header") else "td"
            row.append(
                f'<{tag} rowspan="{cell["rowspan"]}" colspan="{cell["colspan"]}">'
                f'{html.escape(cell.get("text", ""))}</{tag}>'
            )
        rows.append("<tr>" + "".join(row) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _normalized_text(value):
    return "".join(ch.casefold() for ch in value if ch.isalnum())


def text_recall(reference, prediction):
    """Multiset character recall; robust to whitespace and cell ordering noise."""
    reference = Counter(_normalized_text(reference))
    prediction = Counter(_normalized_text(prediction))
    total = sum(reference.values())
    if not total:
        return None
    return sum((reference & prediction).values()) / total


def check(table):
    issues, occupied = [], set()
    cells = table.get("cells", [])
    nrows, ncols = table.get("nrows", 0), table.get("ncols", 0)
    if not cells:
        issues.append("empty_table")
    if nrows < 2:
        issues.append("too_few_rows")
    if ncols < 2:
        issues.append("too_few_columns")
    for cell in cells:
        if cell["row"] < 0 or cell["col"] < 0 or cell["rowspan"] < 1 or cell["colspan"] < 1:
            issues.append("invalid_cell_index_or_span")
            continue
        for row in range(cell["row"], cell["row"] + cell["rowspan"]):
            for col in range(cell["col"], cell["col"] + cell["colspan"]):
                if (row, col) in occupied:
                    issues.append("overlapping_cells")
                occupied.add((row, col))
        if cell.get("bbox") is None:
            issues.append("cell_geometry_unavailable")
        if "\ufffd" in cell.get("text", ""):
            issues.append("replacement_character")
    if nrows and ncols and len(occupied) != nrows * ncols:
        issues.append("nonrectangular_grid")
    if cells and sum(not cell.get("text", "").strip() for cell in cells) / len(cells) > 0.72:
        issues.append("mostly_empty_cells")
    return sorted(set(issues))


def quality_check(table, source_words=(), stage=None):
    """Return auditable QC instead of hiding a parser decision in one score."""
    issues = check(table)
    cells = table.get("cells", [])
    nonempty_ratio = (
        sum(bool(cell.get("text", "").strip()) for cell in cells) / len(cells) if cells else 0.0
    )
    source_words = list(source_words)
    reference = " ".join(word.get("text", "") for word in source_words)
    prediction = " ".join(cell.get("text", "") for cell in cells)
    recall = text_recall(reference, prediction)
    geometry_ratio = (
        sum(cell.get("bbox") is not None for cell in cells) / len(cells) if cells else 0.0
    )
    fatal = {
        "empty_table",
        "too_few_rows",
        "too_few_columns",
        "invalid_cell_index_or_span",
        "overlapping_cells",
        "nonrectangular_grid",
        "mostly_empty_cells",
        "replacement_character",
    }
    passed = not fatal.intersection(issues) and nonempty_ratio >= 0.28
    if recall is not None and len(_normalized_text(reference)) >= 8:
        required = 0.45 if stage == "mineru" else 0.35
        if recall < required:
            issues.append("low_source_text_recall")
            passed = False
    confidence = table.get("confidence")
    if stage == "tatr" and confidence is not None and confidence < 0.25:
        issues.append("low_structure_confidence")
        passed = False
    return {
        "passed": passed,
        "stage": stage or table.get("parser"),
        "issues": sorted(set(issues)),
        "nonempty_cell_ratio": round(nonempty_ratio, 4),
        "cell_geometry_ratio": round(geometry_ratio, 4),
        "source_text_recall": None if recall is None else round(recall, 4),
    }


def merge_candidates(native_tables, detections, page, iou_threshold=0.25):
    """Fuse proposals without letting a broad box join distinct physical tables."""
    candidates = []

    def create(box):
        candidate = {
            "id": "",
            "page": page,
            "bbox": list(box),
            "_anchor_bbox": list(box),
            "native_tables": [],
            "tatr_detections": [],
        }
        candidates.append(candidate)
        return candidate

    def extend(candidate, box):
        current = candidate["bbox"]
        candidate["bbox"] = [
            min(current[0], box[0]),
            min(current[1], box[1]),
            max(current[2], box[2]),
            max(current[3], box[3]),
        ]

    def best_match(box, pool):
        matches = []
        for candidate in pool:
            anchor = candidate["_anchor_bbox"]
            score = iou(box, anchor)
            overlap = overlap_min(box, anchor)
            if (score > 0 and score >= iou_threshold) or overlap >= 0.72:
                matches.append((candidate, score, overlap))
        return max(matches, key=lambda item: (item[1], item[2]))[0] if matches else None

    # A native result that already passed QC is strong physical-table evidence.
    # Build these anchors before considering broad/failed native proposals, so a
    # page-spanning text proposal cannot transitively join two valid line tables.
    reliable_native = [table for table in native_tables if table.get("quality", {}).get("passed")]
    weak_native = [table for table in native_tables if not table.get("quality", {}).get("passed")]
    for table in sorted(
        reliable_native,
        key=lambda item: (item.get("strategy") != "lines", item["bbox"][1], item["bbox"][0]),
    ):
        candidate = best_match(table["bbox"], candidates) or create(table["bbox"])
        candidate["native_tables"].append(table)
        extend(candidate, table["bbox"])

    # Each non-duplicate TATR detection is also an anchor. It may attach to one
    # reliable native anchor, but never collapse two anchors into one candidate.
    for detection in sorted(detections, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        duplicate = next(
            (
                candidate
                for candidate in candidates
                if any(iou(detection["bbox"], other["bbox"]) >= 0.80 for other in candidate["tatr_detections"])
            ),
            None,
        )
        available_native = [
            candidate
            for candidate in candidates
            if candidate["native_tables"] and not candidate["tatr_detections"]
        ]
        candidate = duplicate or best_match(detection["bbox"], available_native) or create(detection["bbox"])
        candidate["tatr_detections"].append(detection)
        extend(candidate, detection["bbox"])

    # Weak proposals are supporting evidence only. Match against immutable
    # anchor boxes and assign to one best candidate, preventing bridge merging.
    for table in sorted(weak_native, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        candidate = best_match(table["bbox"], candidates) or create(table["bbox"])
        candidate["native_tables"].append(table)
        extend(candidate, table["bbox"])

    candidates.sort(key=lambda item: (item["_anchor_bbox"][1], item["_anchor_bbox"][0]))
    for index, candidate in enumerate(candidates, 1):
        candidate["id"] = f"p{page:04d}-candidate-{index:03d}"
        candidate.pop("_anchor_bbox", None)
    return candidates


def signature(table, text=True):
    return [
        (
            cell["row"], cell["col"], cell["rowspan"], cell["colspan"],
            re.sub(r"\s+", "", cell.get("text", "")) if text else "",
        )
        for cell in sorted(table.get("cells", []), key=lambda cell: (cell["row"], cell["col"]))
    ]


def continuation_candidates(tables, pages):
    page_map = {page["page"]: page for page in pages}
    result = []
    for first in tables:
        for second in tables:
            if second["page"] != first["page"] + 1 or first.get("ncols") != second.get("ncols"):
                continue
            if not first.get("bbox") or not second.get("bbox"):
                continue
            first_page, second_page = page_map[first["page"]], page_map[second["page"]]
            if first["bbox"][3] > 0.78 * first_page["height"] and second["bbox"][1] < 0.25 * second_page["height"]:
                result.append({
                    "tables": [first["id"], second["id"]],
                    "status": "unresolved_cross_page_candidate",
                    "reason": "adjacent pages, equal column count, bottom/top placement; no automatic merge",
                })
    return result
