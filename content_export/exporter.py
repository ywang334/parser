"""Create compact JSON, HTML, and Markdown views containing original content only.

The parser's ``document.json`` remains the auditable source of truth. This module
uses a strict allow-list so geometry, engines, confidence scores, candidates,
quality checks, and runtime diagnostics cannot leak into the content exports.
"""

from __future__ import annotations

import html
import json
import pathlib
import re
from collections import defaultdict


NESTED_MARKER = re.compile(r"(?:\n)?\[嵌套表格\s+[^\]]+\]")


def _clean_text(value):
    return NESTED_MARKER.sub("", str(value or ""))


def _content_runs(block):
    text = str(block.get("text", ""))
    runs = block.get("runs") or []
    if not runs or "".join(str(run.get("text", "")) for run in runs) != text:
        return None
    result = []
    for run in runs:
        item = {"text": str(run.get("text", ""))}
        for key in ("bold", "italic", "underline"):
            if run.get(key):
                item[key] = True
        if run.get("vertical_align") in {"superscript", "subscript"}:
            item["vertical_align"] = run["vertical_align"]
        if run.get("hyperlink"):
            item["hyperlink"] = str(run["hyperlink"])
        result.append(item)
    return None if all(set(run) == {"text"} for run in result) else result


def _text_item(block):
    kind = block.get("type", "paragraph")
    if kind not in {"text", "paragraph", "heading", "list_item"}:
        kind = "paragraph"
    item = {"type": kind, "text": str(block.get("text", ""))}
    if kind == "heading":
        item["level"] = max(1, min(6, int(block.get("heading_level") or 1)))
    if kind == "list_item":
        numbering = block.get("list") or {}
        item["level"] = max(0, int(numbering.get("level") or 0))
        item["format"] = str(numbering.get("format") or "bullet")
        if numbering.get("text"):
            item["template"] = str(numbering["text"])
    runs = _content_runs(block)
    if runs:
        item["runs"] = runs
    return item


def _content_table(table, table_by_id, ancestors=()):
    table_id = str(table.get("id", ""))
    if table_id and table_id in ancestors:
        return {"type": "table", "rows": 0, "columns": 0, "cells": []}
    next_ancestors = ancestors + ((table_id,) if table_id else ())
    cells = []
    for cell in sorted(table.get("cells", []), key=lambda value: (value.get("row", 0), value.get("col", 0))):
        item = {
            "row": int(cell.get("row", 0)),
            "col": int(cell.get("col", 0)),
            "rowspan": max(1, int(cell.get("rowspan", 1))),
            "colspan": max(1, int(cell.get("colspan", 1))),
            "text": _clean_text(cell.get("text", "")),
            "header": bool(cell.get("header", False)),
        }
        nested = []
        for nested_id in cell.get("nested_tables") or []:
            nested_table = table_by_id.get(str(nested_id))
            if nested_table is not None and str(nested_id) not in next_ancestors:
                nested.append(_content_table(nested_table, table_by_id, next_ancestors))
        if nested:
            item["nested_tables"] = nested
        cells.append(item)
    return {
        "type": "table",
        "rows": max(0, int(table.get("nrows", 0))),
        "columns": max(0, int(table.get("ncols", 0))),
        "cells": cells,
    }


def _pdf_content(document):
    units = []
    for page in document.get("pages", []):
        positioned = []
        for index, block in enumerate(page.get("blocks", [])):
            bbox = block.get("bbox") or [0, 0, 0, 0]
            positioned.append(((float(bbox[1]), float(bbox[0]), 1, index), _text_item(block)))
        for index, table in enumerate(page.get("tables", [])):
            bbox = table.get("bbox") or [0, 0, 0, 0]
            positioned.append(((float(bbox[1]), float(bbox[0]), 0, index), _content_table(table, {})))
        units.append(
            {
                "type": "page",
                "number": int(page.get("page", len(units) + 1)),
                "content": [item for _, item in sorted(positioned, key=lambda pair: pair[0])],
            }
        )
    return units


def _word_blocks(blocks, table_by_id):
    result = []
    for block in sorted(blocks, key=lambda item: item.get("order", 0)):
        if block.get("type") == "table":
            table = table_by_id.get(str(block.get("table_id", "")))
            if table is not None:
                result.append(_content_table(table, table_by_id))
        else:
            result.append(_text_item(block))
    return result


def _word_content(document):
    table_by_id = {str(table.get("id", "")): table for table in document.get("tables", [])}
    groups = defaultdict(list)
    for block in document.get("blocks", []):
        groups[int(block.get("section", 0))].append(block)
    units = [
        {"type": "section", "number": section + 1, "content": _word_blocks(groups[section], table_by_id)}
        for section in sorted(groups)
    ]
    supplementary = {}
    labels = {
        "headers": "header",
        "footers": "footer",
        "footnotes": "footnotes",
        "endnotes": "endnotes",
    }
    for source_name, output_name in labels.items():
        parts = []
        for part in document.get("ancillary", {}).get(source_name, []):
            content = _word_blocks(part.get("blocks", []), table_by_id)
            if content:
                parts.append({"content": content})
        if parts:
            supplementary[output_name] = parts
    return units, supplementary


def build_content_document(document):
    """Return a parser-neutral document made only from source content/structure."""
    source_format = str(document.get("source_format") or "pdf").lower()
    if isinstance(document.get("pages"), list):
        structure = "pages"
        content = _pdf_content(document)
        supplementary = {}
    elif isinstance(document.get("blocks"), list):
        structure = "sections"
        content, supplementary = _word_content(document)
    else:
        raise ValueError("unsupported document.json: expected pages or blocks")
    result = {
        "schema_version": "1.0",
        "source_format": source_format,
        "structure": structure,
        "content": content,
    }
    title = str((document.get("metadata") or {}).get("title") or "").strip()
    if title:
        result["title"] = title
    if supplementary:
        result["supplementary"] = supplementary
    return result


def _inline_html(item):
    runs = item.get("runs") or []
    if not runs:
        return html.escape(item.get("text", "")).replace("\n", "<br>\n")
    parts = []
    for run in runs:
        value = html.escape(run.get("text", "")).replace("\n", "<br>\n")
        if run.get("vertical_align") == "superscript":
            value = f"<sup>{value}</sup>"
        elif run.get("vertical_align") == "subscript":
            value = f"<sub>{value}</sub>"
        if run.get("underline"):
            value = f"<u>{value}</u>"
        if run.get("italic"):
            value = f"<em>{value}</em>"
        if run.get("bold"):
            value = f"<strong>{value}</strong>"
        if run.get("hyperlink"):
            value = f'<a href="{html.escape(run["hyperlink"], quote=True)}">{value}</a>'
        parts.append(value)
    return "".join(parts)


def _table_html(table):
    rows = defaultdict(list)
    for cell in table.get("cells", []):
        rows[int(cell.get("row", 0))].append(cell)
    body = ["<table>"]
    for row in range(int(table.get("rows", 0))):
        body.append("<tr>")
        for cell in sorted(rows.get(row, []), key=lambda value: value.get("col", 0)):
            tag = "th" if cell.get("header") else "td"
            attributes = [f'data-row="{row}"', f'data-col="{int(cell.get("col", 0))}"']
            if int(cell.get("rowspan", 1)) > 1:
                attributes.append(f'rowspan="{int(cell["rowspan"])}"')
            if int(cell.get("colspan", 1)) > 1:
                attributes.append(f'colspan="{int(cell["colspan"])}"')
            value = html.escape(cell.get("text", "")).replace("\n", "<br>\n")
            for nested in cell.get("nested_tables", []):
                value += _table_html(nested)
            body.append(f"<{tag} {' '.join(attributes)}>{value}</{tag}>")
        body.append("</tr>")
    body.append("</table>")
    return "\n".join(body)


def _items_html(items):
    body = []
    for item in items:
        kind = item.get("type")
        if kind == "table":
            body.append(_table_html(item))
        elif kind == "heading":
            level = int(item.get("level", 1))
            body.append(f"<h{level}>{_inline_html(item)}</h{level}>")
        elif kind == "list_item":
            level = int(item.get("level", 0))
            body.append(
                f'<div class="list-item" data-level="{level}" style="--level:{level}">'
                f'{_inline_html(item)}</div>'
            )
        else:
            body.append(f"<p>{_inline_html(item)}</p>")
    return "\n".join(body)


def render_html(document):
    title = html.escape(document.get("title") or "结构化原文")
    style = (
        "body{font:15px/1.6 sans-serif;margin:28px auto;max-width:1200px;padding:0 20px}"
        "section{margin:0 0 32px}p{white-space:pre-wrap}"
        "table{border-collapse:collapse;margin:12px 0 24px;max-width:100%}"
        "td,th{border:1px solid #888;padding:5px;vertical-align:top;white-space:pre-wrap}"
        ".list-item{margin-left:calc(var(--level) * 1.5em);padding-left:1.2em}"
        ".list-item:before{content:'•';display:inline-block;width:1.2em;margin-left:-1.2em}"
    )
    body = [
        "<!doctype html>",
        f'<html lang="zh"><head><meta charset="utf-8"><title>{title}</title><style>{style}</style></head><body>',
    ]
    for unit in document.get("content", []):
        attribute = "page" if unit.get("type") == "page" else "section"
        body.append(f'<section data-{attribute}="{int(unit.get("number", 0))}">')
        body.append(_items_html(unit.get("content", [])))
        body.append("</section>")
    for kind, parts in document.get("supplementary", {}).items():
        for part in parts:
            body.append(f'<aside data-content-kind="{html.escape(kind, quote=True)}">')
            body.append(_items_html(part.get("content", [])))
            body.append("</aside>")
    body.append("</body></html>")
    return "\n".join(body)


def _markdown_escape(value):
    value = str(value).replace("\\", "\\\\")
    value = re.sub(r"([`*_\[\]<>|~])", r"\\\1", value)
    value = re.sub(r"(?m)^(\s*)([#>+-])(?=\s)", r"\1\\\2", value)
    value = re.sub(r"(?m)^(\s*)(\d+)([.)])(?=\s)", r"\1\2\\\3", value)
    return re.sub(r"(?m)^(\s*)(---+)(\s*)$", r"\1\\\2\3", value)


def _inline_markdown(item):
    runs = item.get("runs") or []
    if not runs:
        return _markdown_escape(item.get("text", ""))
    parts = []
    for run in runs:
        value = _markdown_escape(run.get("text", ""))
        if run.get("vertical_align") == "superscript":
            value = f"<sup>{value}</sup>"
        elif run.get("vertical_align") == "subscript":
            value = f"<sub>{value}</sub>"
        if run.get("underline"):
            value = f"<u>{value}</u>"
        if run.get("italic"):
            value = f"*{value}*"
        if run.get("bold"):
            value = f"**{value}**"
        if run.get("hyperlink"):
            target = str(run["hyperlink"]).replace(">", "%3E")
            value = f"[{value}](<{target}>)"
        parts.append(value)
    return "".join(parts)


def _items_markdown(items):
    body = []
    for item in items:
        kind = item.get("type")
        if kind == "table":
            body.append(_table_html(item))
        elif kind == "heading":
            body.append(f'{"#" * int(item.get("level", 1))} {_inline_markdown(item)}')
        elif kind == "list_item":
            marker = "1." if item.get("format") not in {"bullet", "none"} else "-"
            body.append(f'{"  " * int(item.get("level", 0))}{marker} {_inline_markdown(item)}')
        else:
            body.append(_inline_markdown(item))
    return "\n\n".join(body)


def render_markdown(document):
    body = []
    for unit in document.get("content", []):
        body.append(f'<!-- {unit.get("type")}: {int(unit.get("number", 0))} -->')
        rendered = _items_markdown(unit.get("content", []))
        if rendered:
            body.append(rendered)
    for kind, parts in document.get("supplementary", {}).items():
        for part in parts:
            body.append(f"<!-- {kind} -->")
            rendered = _items_markdown(part.get("content", []))
            if rendered:
                body.append(rendered)
    return "\n\n".join(body).rstrip() + "\n"


def _atomic_write(path, value):
    path = pathlib.Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def export_document(document, output):
    """Write content.json/content.html/content.md beside a parser result."""
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    content = build_content_document(document)
    _atomic_write(output / "content.json", json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(output / "content.html", render_html(content))
    _atomic_write(output / "content.md", render_markdown(content))
    return content
