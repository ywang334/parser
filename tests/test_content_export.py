import json

from content_export.exporter import build_content_document, export_document, render_markdown


def _keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _keys(nested)


def test_pdf_content_is_ordered_and_parser_fields_are_removed(tmp_path):
    source = {
        "input": "/secret/source.pdf",
        "errors": [{"traceback": "ignored"}],
        "pages": [
            {
                "page": 3,
                "blocks": [
                    {"type": "text", "text": "below", "bbox": [0, 80, 10, 90], "source": "pymupdf"},
                    {"type": "text", "text": "above", "bbox": [0, 10, 10, 20], "source": "pymupdf"},
                ],
                "tables": [
                    {
                        "id": "candidate-1",
                        "bbox": [0, 40, 100, 60],
                        "nrows": 1,
                        "ncols": 2,
                        "parser": "tatr",
                        "quality": {"score": 1},
                        "cells": [
                            {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "A", "header": True, "bbox": [0, 0, 1, 1], "source": {}},
                            {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "text": "B", "header": False, "bbox": [1, 0, 2, 1], "source": {}},
                        ],
                    }
                ],
            }
        ],
    }
    content = build_content_document(source)
    items = content["content"][0]["content"]
    assert [item["type"] for item in items] == ["text", "table", "text"]
    assert items[0]["text"] == "above"
    assert items[1]["cells"][0]["text"] == "A"
    assert not ({"bbox", "source", "parser", "quality", "input", "errors"} & set(_keys(content)))

    export_document(source, tmp_path)
    assert json.loads((tmp_path / "content.json").read_text()) == content
    assert "rowspan" not in (tmp_path / "content.html").read_text()
    assert "<table>" in (tmp_path / "content.md").read_text()


def test_word_content_preserves_structure_and_inlines_nested_tables():
    source = {
        "source_format": "docx",
        "metadata": {"title": "Manual", "creator": "ignored"},
        "blocks": [
            {"type": "heading", "order": 1, "section": 0, "text": "Title", "heading_level": 2, "runs": [{"text": "Title", "bold": True}]},
            {"type": "table", "order": 2, "section": 0, "table_id": "outer", "text": "derived"},
            {"type": "list_item", "order": 3, "section": 1, "text": "Step", "list": {"level": 1, "format": "decimal", "text": "%2."}},
        ],
        "tables": [
            {
                "id": "outer",
                "nrows": 1,
                "ncols": 1,
                "cells": [{"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "Outer\n[嵌套表格 inner]", "header": False, "nested_tables": ["inner"]}],
            },
            {
                "id": "inner",
                "parent_table": "outer",
                "nrows": 1,
                "ncols": 1,
                "cells": [{"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "Inner", "header": False}],
            },
        ],
        "ancillary": {
            "headers": [{"part": "word/header1.xml", "blocks": [{"type": "paragraph", "order": 10, "text": "Header"}]}],
            "footers": [],
            "footnotes": [],
            "endnotes": [],
        },
    }
    content = build_content_document(source)
    assert content["title"] == "Manual"
    assert [unit["number"] for unit in content["content"]] == [1, 2]
    heading = content["content"][0]["content"][0]
    assert heading == {"type": "heading", "text": "Title", "level": 2, "runs": [{"text": "Title", "bold": True}]}
    table = content["content"][0]["content"][1]
    assert table["cells"][0]["text"] == "Outer"
    assert table["cells"][0]["nested_tables"][0]["cells"][0]["text"] == "Inner"
    assert content["supplementary"]["header"][0]["content"][0]["text"] == "Header"
    assert "part" not in set(_keys(content))


def test_markdown_keeps_ordinary_punctuation_and_escapes_leading_syntax():
    document = {
        "content": [
            {
                "type": "section",
                "number": 1,
                "content": [{"type": "paragraph", "text": "A-B.txt\n# literal\n---"}],
            }
        ]
    }
    markdown = render_markdown(document)
    assert "A-B.txt" in markdown
    assert "\\# literal" in markdown
    assert "\\---" in markdown
