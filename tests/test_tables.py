import argparse
import json

import pytest

from pdfpipe.aggregate_parallel import aggregate
from pdfpipe.cli import page_selection, probability, shard_files
from pdfpipe.metrics import evaluate, teds
from pdfpipe.pipeline import text_blocks, text_layer_quality
from pdfpipe.tables import merge_candidates, parse_html, quality_check, render, signature


def test_probability_argument():
    assert probability("0.5") == 0.5
    for value in ("-0.1", "1.1", "nan", "not-a-number"):
        with pytest.raises(argparse.ArgumentTypeError):
            probability(value)


def test_page_selection_argument():
    assert page_selection("1, 3-5,5") == [1, 3, 4, 5]
    for value in ("", "0", "3-1", "1,,2", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            page_selection(value)


def test_round_robin_file_shards():
    files = list("abcdefg")
    assert shard_files(files, 0, 3) == ["a", "d", "g"]
    assert shard_files(files, 1, 3) == ["b", "e"]
    assert shard_files(files, 2, 3) == ["c", "f"]
    with pytest.raises(ValueError):
        shard_files(files, 3, 3)


def test_parallel_aggregation(tmp_path):
    worker = tmp_path / ".parallel" / "worker-000"
    document = worker / "example-deadbeef"
    document.mkdir(parents=True)
    (document / "index.html").write_text("ok")
    (worker / "summary.json").write_text(
        json.dumps(
            [
                {
                    "input": "example.pdf",
                    "output": document.name,
                    "status": "ok",
                    "pages": 2,
                    "tables": 1,
                }
            ]
        )
    )
    status = tmp_path / ".parallel-status"
    status.mkdir()
    (status / "worker-000.exit").write_text("0\n")
    assert aggregate(tmp_path, "0") == 0
    assert (tmp_path / document.name).is_symlink()
    metadata = json.loads((tmp_path / "parallel.json").read_text())
    assert metadata["documents"] == 1
    assert metadata["workers"][0]["gpu_id"] == "0"


def test_multilevel_merged_header_roundtrip():
    source = '<table><tr><th rowspan="2">Region</th><th colspan="2">2025</th></tr><tr><th>Q1</th><th>Q2</th></tr><tr><td>East</td><td>1</td><td>2</td></tr></table>'
    table = parse_html(source, 1, [1, 2, 10, 20], "test", "t")
    assert (table["nrows"], table["ncols"]) == (3, 3)
    assert signature(parse_html(render(table), 1, None, "test", "t")) == signature(table)
    assert "nonrectangular_grid" not in table["issues"]


def test_quality_check_rejects_nonrectangular_grid():
    table = parse_html('<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>', 1, None, "x", "t")
    quality = quality_check(table, [], "mineru")
    assert not quality["passed"]
    assert "nonrectangular_grid" in quality["issues"]


def test_candidate_fusion_preserves_both_proposals():
    native = {"id": "n", "bbox": [10, 10, 100, 100], "cells": [], "quality": {"passed": False}}
    detection = {"bbox": [12, 12, 101, 101], "score": 0.9, "label": "table"}
    result = merge_candidates([native], [detection], 1)
    assert len(result) == 1
    assert result[0]["native_tables"][0]["id"] == "n"
    assert result[0]["tatr_detections"][0]["score"] == 0.9


def test_broad_native_box_does_not_merge_distinct_tatr_tables():
    native = [
        {"id": "upper", "bbox": [0, 0, 100, 45], "cells": [], "quality": {"passed": True}},
        {"id": "bridge", "bbox": [0, 0, 100, 100], "cells": [], "quality": {"passed": False}},
        {"id": "lower", "bbox": [0, 55, 100, 100], "cells": [], "quality": {"passed": True}},
    ]
    detections = [
        {"bbox": [10, 5, 90, 40], "score": 0.95, "label": "table"},
        {"bbox": [10, 60, 90, 95], "score": 0.94, "label": "table"},
    ]
    result = merge_candidates(native, detections, 1)
    assert len(result) == 2
    assert [len(candidate["tatr_detections"]) for candidate in result] == [1, 1]
    assert "upper" in {table["id"] for table in result[0]["native_tables"]}
    assert "lower" in {table["id"] for table in result[1]["native_tables"]}


def test_failed_native_bridge_does_not_merge_reliable_native_tables():
    native = [
        {"id": "upper", "strategy": "lines", "bbox": [0, 20, 100, 40], "cells": [], "quality": {"passed": True}},
        {"id": "bridge", "strategy": "text", "bbox": [0, 0, 100, 100], "cells": [], "quality": {"passed": False}},
        {"id": "lower", "strategy": "lines", "bbox": [0, 60, 100, 100], "cells": [], "quality": {"passed": True}},
    ]
    result = merge_candidates(native, [], 1)
    assert len(result) == 2
    assert "upper" in {table["id"] for table in result[0]["native_tables"]}
    assert "lower" in {table["id"] for table in result[1]["native_tables"]}


def test_broad_tatr_box_does_not_merge_reliable_native_tables():
    native = [
        {"id": "upper", "strategy": "lines", "bbox": [0, 0, 100, 40], "cells": [], "quality": {"passed": True}},
        {"id": "lower", "strategy": "lines", "bbox": [0, 60, 100, 100], "cells": [], "quality": {"passed": True}},
    ]
    detections = [{"bbox": [5, 0, 95, 100], "score": 0.9, "label": "table"}]
    result = merge_candidates(native, detections, 1)
    assert len(result) == 2
    assert sum(len(candidate["tatr_detections"]) for candidate in result) == 1
    assert {table["id"] for candidate in result for table in candidate["native_tables"]} == {"upper", "lower"}


def test_text_layer_and_table_exclusion():
    words = [
        {"text": "This is enough native text for one page", "bbox": [0, 0, 100, 10], "block": 0, "line": 0, "source": "pymupdf"},
        {"text": "cell", "bbox": [0, 20, 20, 30], "block": 1, "line": 0, "source": "pymupdf"},
        {"text": "value", "bbox": [30, 20, 50, 30], "block": 1, "line": 0, "source": "pymupdf"},
    ]
    assert text_layer_quality(words)["reliable"]
    blocks = text_blocks(words, [[0, 15, 60, 35]])
    assert len(blocks) == 1
    assert "native text" in blocks[0]["text"]


def test_escaped_output():
    table = parse_html('<table><tr><td>&lt;script&gt;alert(1)&lt;/script&gt;</td><td>x</td></tr><tr><td>y</td><td>z</td></tr></table>', 1, None, "x", "x")
    assert "<script>" not in render(table)


def test_teds_and_structure_only():
    first = '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'
    second = '<table><tr><td rowspan="2">X</td><td>B</td></tr><tr><td>C</td></tr></table>'
    assert teds(first, first) == 1
    assert teds(first, second) < 1
    assert teds(first, second, structure_only=True) == 1


def test_official_grits_with_cell_boxes():
    source = '<table><tr><td>A</td></tr></table>'
    cell = {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "A", "bbox": [0, 0, 10, 10], "header": False, "source": {}}
    truth = {"tables": [{"id": "truth", "prediction_id": "pred", "html": source, "cells": [cell]}]}
    prediction = {"tables": [{"id": "pred", "page": 1, "bbox": [0, 0, 10, 10], "nrows": 1, "ncols": 1, "cells": [cell]}]}
    metrics = evaluate(truth, prediction)
    assert all(metrics["aggregate"][key] == 1 for key in ("teds", "teds_s", "grits_top", "grits_con", "grits_loc"))
