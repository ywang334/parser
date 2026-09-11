"""Small, auditable WordprocessingML reader focused on text and tables."""

from __future__ import annotations

import collections
import posixpath
import zipfile
from pathlib import Path

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W_NS, "r": R_NS, "m": M_NS}


def qn(namespace, name):
    return f"{{{namespace}}}{name}"


def _int_attr(element, name, default=0):
    if element is None:
        return default
    try:
        return int(element.get(qn(W_NS, name), default))
    except (TypeError, ValueError):
        return default


def _on(element):
    if element is None:
        return False
    return element.get(qn(W_NS, "val"), "true").lower() not in {"0", "false", "off", "none"}


class OOXMLReader:
    """Read a DOCX package without rendering or inventing page coordinates."""

    max_entries = 10000
    max_uncompressed_bytes = 2 * 1024**3

    def __init__(self, path):
        self.path = Path(path)
        self.parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
        self.tables = []
        self.unresolved = []
        self._table_number = 0
        self._block_number = 0
        self._part_block_numbers = collections.defaultdict(int)

    def _read(self, archive, name, required=False):
        try:
            return archive.read(name)
        except KeyError:
            if required:
                raise ValueError(f"DOCX package is missing {name}") from None
            return None

    def _xml(self, archive, name, required=False):
        data = self._read(archive, name, required)
        return etree.fromstring(data, self.parser) if data is not None else None

    @staticmethod
    def _relationship_part(part):
        directory, filename = posixpath.split(part)
        return posixpath.join(directory, "_rels", filename + ".rels")

    def _relationships(self, archive, part):
        root = self._xml(archive, self._relationship_part(part))
        result = {}
        if root is None:
            return result
        for rel in root.findall(qn(PKG_REL_NS, "Relationship")):
            rel_id = rel.get("Id")
            target = rel.get("Target")
            if not rel_id or not target:
                continue
            if rel.get("TargetMode") == "External":
                result[rel_id] = {"target": target, "external": True, "type": rel.get("Type")}
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(part), target))
            if resolved.startswith("../") or resolved.startswith("/"):
                self.unresolved.append({"type": "unsafe_relationship", "part": part, "target": target})
                continue
            result[rel_id] = {"target": resolved, "external": False, "type": rel.get("Type")}
        return result

    def _styles(self, archive):
        root = self._xml(archive, "word/styles.xml")
        styles = {}
        if root is None:
            return styles
        for style in root.findall("w:style", NS):
            style_id = style.get(qn(W_NS, "styleId"))
            name = style.find("w:name", NS)
            based_on = style.find("w:basedOn", NS)
            outline = style.find("w:pPr/w:outlineLvl", NS)
            styles[style_id] = {
                "name": name.get(qn(W_NS, "val")) if name is not None else style_id,
                "based_on": based_on.get(qn(W_NS, "val")) if based_on is not None else None,
                "outline": _int_attr(outline, "val", -1),
            }
        return styles

    def _numbering(self, archive):
        root = self._xml(archive, "word/numbering.xml")
        if root is None:
            return {}
        abstracts = {}
        for abstract in root.findall("w:abstractNum", NS):
            abstract_id = abstract.get(qn(W_NS, "abstractNumId"))
            levels = {}
            for level in abstract.findall("w:lvl", NS):
                ilvl = _int_attr(level, "ilvl", 0)
                num_fmt = level.find("w:numFmt", NS)
                lvl_text = level.find("w:lvlText", NS)
                levels[ilvl] = {
                    "format": num_fmt.get(qn(W_NS, "val")) if num_fmt is not None else None,
                    "text": lvl_text.get(qn(W_NS, "val")) if lvl_text is not None else None,
                }
            abstracts[abstract_id] = levels
        result = {}
        for num in root.findall("w:num", NS):
            num_id = num.get(qn(W_NS, "numId"))
            abstract = num.find("w:abstractNumId", NS)
            result[num_id] = abstracts.get(abstract.get(qn(W_NS, "val")), {}) if abstract is not None else {}
        return result

    def _style_info(self, style_id, styles):
        result = {"id": style_id, "name": None, "outline": -1}
        seen = set()
        while style_id and style_id not in seen:
            seen.add(style_id)
            style = styles.get(style_id, {})
            if result["name"] is None:
                result["name"] = style.get("name")
            if style.get("outline", -1) >= 0:
                result["outline"] = style["outline"]
                break
            style_id = style.get("based_on")
        return result

    @staticmethod
    def _inline_text(element):
        pieces = []
        for node in element.iter():
            if node.tag in {qn(W_NS, "t"), qn(M_NS, "t"), qn(W_NS, "delText")}:
                if node.tag != qn(W_NS, "delText"):
                    pieces.append(node.text or "")
            elif node.tag == qn(W_NS, "tab"):
                pieces.append("\t")
            elif node.tag in {qn(W_NS, "br"), qn(W_NS, "cr")}:
                pieces.append("\n")
            elif node.tag == qn(W_NS, "noBreakHyphen"):
                pieces.append("‑")
        return "".join(pieces)

    def _paragraph(self, paragraph, part, styles, numbering, relationships, section_index):
        self._part_block_numbers[part] += 1
        ppr = paragraph.find("w:pPr", NS)
        style_node = ppr.find("w:pStyle", NS) if ppr is not None else None
        style_id = style_node.get(qn(W_NS, "val")) if style_node is not None else None
        style = self._style_info(style_id, styles)
        direct_outline = ppr.find("w:outlineLvl", NS) if ppr is not None else None
        outline = _int_attr(direct_outline, "val", style["outline"])
        heading_level = outline + 1 if 0 <= outline <= 8 else None
        if heading_level is None and style.get("name"):
            normalized = style["name"].lower().replace(" ", "")
            for prefix in ("heading", "标题"):
                if normalized.startswith(prefix) and normalized[len(prefix):].isdigit():
                    heading_level = min(9, int(normalized[len(prefix):]))
                    break

        num_pr = ppr.find("w:numPr", NS) if ppr is not None else None
        num_id_node = num_pr.find("w:numId", NS) if num_pr is not None else None
        level_node = num_pr.find("w:ilvl", NS) if num_pr is not None else None
        num_id = num_id_node.get(qn(W_NS, "val")) if num_id_node is not None else None
        level = _int_attr(level_node, "val", 0)
        list_info = None
        if num_id is not None:
            list_info = {"num_id": num_id, "level": level, **numbering.get(num_id, {}).get(level, {})}

        runs = []
        for run in paragraph.xpath("./w:r|./w:hyperlink/w:r", namespaces=NS):
            text = self._inline_text(run)
            if not text:
                continue
            rpr = run.find("w:rPr", NS)
            parent = run.getparent()
            hyperlink = None
            if parent is not None and parent.tag == qn(W_NS, "hyperlink"):
                rel_id = parent.get(qn(R_NS, "id"))
                hyperlink = relationships.get(rel_id, {}).get("target")
            runs.append(
                {
                    "text": text,
                    "bold": _on(rpr.find("w:b", NS)) if rpr is not None else False,
                    "italic": _on(rpr.find("w:i", NS)) if rpr is not None else False,
                    "underline": _on(rpr.find("w:u", NS)) if rpr is not None else False,
                    "vertical_align": (
                        rpr.find("w:vertAlign", NS).get(qn(W_NS, "val"))
                        if rpr is not None and rpr.find("w:vertAlign", NS) is not None
                        else None
                    ),
                    "hyperlink": hyperlink,
                }
            )
        text = self._inline_text(paragraph).strip()
        page_breaks = len(paragraph.findall(".//w:br[@w:type='page']", NS))
        if ppr is not None and _on(ppr.find("w:pageBreakBefore", NS)):
            page_breaks += 1
        return {
            "id": f"block-{self._block_number:05d}",
            "type": "heading" if heading_level else ("list_item" if list_info else "paragraph"),
            "order": self._block_number,
            "section": section_index,
            "text": text,
            "heading_level": heading_level,
            "style": style,
            "list": list_info,
            "runs": runs,
            "page_breaks": page_breaks,
            "source": {
                "parser": "ooxml",
                "part": part,
                "block_index": self._part_block_numbers[part],
                "xml_path": paragraph.getroottree().getpath(paragraph),
            },
        }

    @staticmethod
    def _block_children(container):
        for child in container:
            if child.tag in {qn(W_NS, "p"), qn(W_NS, "tbl")}:
                yield child
            elif child.tag in {qn(W_NS, "sdt"), qn(W_NS, "customXml"), qn(W_NS, "smartTag")}:
                nested = child.find("w:sdtContent", NS) if child.tag == qn(W_NS, "sdt") else child
                if nested is not None:
                    yield from OOXMLReader._block_children(nested)

    def _table(self, node, part, styles, numbering, relationships, section_index, parent_table=None):
        self._table_number += 1
        table_id = f"table-{self._table_number:04d}"
        raw_xml = etree.tostring(node, encoding="unicode")
        table = {
            "id": table_id,
            "page": None,
            "bbox": None,
            "parser": "ooxml",
            "decision": "accepted_ooxml",
            "nrows": 0,
            "ncols": 0,
            "cells": [],
            "parent_table": parent_table,
            "source": {
                "parser": "ooxml",
                "part": part,
                "section": section_index,
                "xml_path": node.getroottree().getpath(node),
                "geometry": "unavailable_in_flow_document",
            },
            "_raw_xml": raw_xml,
        }
        self.tables.append(table)
        rows = node.findall("w:tr", NS)
        grid_columns = len(node.findall("w:tblGrid/w:gridCol", NS))
        estimated_columns = grid_columns
        for row in rows:
            before = _int_attr(row.find("w:trPr/w:gridBefore", NS), "val", 0)
            after = _int_attr(row.find("w:trPr/w:gridAfter", NS), "val", 0)
            width = before + after + sum(max(1, _int_attr(tc.find("w:tcPr/w:gridSpan", NS), "val", 1)) for tc in row.findall("w:tc", NS))
            estimated_columns = max(estimated_columns, width)

        cells = []
        active_vertical = {}
        issues = []
        for row_index, row in enumerate(rows):
            header = _on(row.find("w:trPr/w:tblHeader", NS))
            column = _int_attr(row.find("w:trPr/w:gridBefore", NS), "val", 0)
            next_vertical = {}
            previous_horizontal = None
            for cell_index, tc in enumerate(row.findall("w:tc", NS)):
                tcpr = tc.find("w:tcPr", NS)
                colspan = max(1, _int_attr(tcpr.find("w:gridSpan", NS) if tcpr is not None else None, "val", 1))
                vmerge = tcpr.find("w:vMerge", NS) if tcpr is not None else None
                vmerge_value = vmerge.get(qn(W_NS, "val"), "continue") if vmerge is not None else None
                hmerge = tcpr.find("w:hMerge", NS) if tcpr is not None else None
                hmerge_value = hmerge.get(qn(W_NS, "val"), "continue") if hmerge is not None else None
                direct_paragraphs = [self._inline_text(p).strip() for p in tc.findall("w:p", NS)]
                text = "\n".join(value for value in direct_paragraphs if value)
                nested_ids = []
                for nested in tc.findall("w:tbl", NS):
                    nested_ids.append(
                        self._table(nested, part, styles, numbering, relationships, section_index, table_id)["id"]
                    )
                if nested_ids:
                    marker = " ".join(f"[嵌套表格 {item}]" for item in nested_ids)
                    text = "\n".join(value for value in (text, marker) if value)
                key = (column, colspan)

                if hmerge_value == "continue" and previous_horizontal is not None:
                    previous_horizontal["colspan"] += colspan
                    if text:
                        previous_horizontal["text"] = "\n".join(
                            value for value in (previous_horizontal["text"], text) if value
                        )
                    issues.append("legacy_hmerge_normalized")
                    column += colspan
                    continue

                if vmerge_value == "continue":
                    anchor = active_vertical.get(key)
                    if anchor is None:
                        issues.append("orphan_vertical_merge")
                    else:
                        anchor["rowspan"] += 1
                        if text:
                            anchor["text"] = "\n".join(value for value in (anchor["text"], text) if value)
                        next_vertical[key] = anchor
                        column += colspan
                        previous_horizontal = None
                        continue

                cell = {
                    "row": row_index,
                    "col": column,
                    "rowspan": 1,
                    "colspan": colspan,
                    "text": text,
                    "header": header,
                    "bbox": None,
                    "nested_tables": nested_ids,
                    "source": {
                        "parser": "ooxml",
                        "part": part,
                        "table": table_id,
                        "row_index": row_index,
                        "cell_index": cell_index,
                        "xml_path": tc.getroottree().getpath(tc),
                        "geometry": "unavailable_in_flow_document",
                    },
                }
                cells.append(cell)
                if vmerge_value == "restart":
                    next_vertical[key] = cell
                if hmerge_value == "restart":
                    previous_horizontal = cell
                else:
                    previous_horizontal = None
                column += colspan
            active_vertical = next_vertical

        occupied = set()
        for cell in cells:
            for row in range(cell["row"], cell["row"] + cell["rowspan"]):
                for col in range(cell["col"], cell["col"] + cell["colspan"]):
                    if (row, col) in occupied:
                        issues.append("overlapping_cells")
                    occupied.add((row, col))
        for row in range(len(rows)):
            for col in range(estimated_columns):
                if (row, col) not in occupied:
                    cells.append(
                        {
                            "row": row,
                            "col": col,
                            "rowspan": 1,
                            "colspan": 1,
                            "text": "",
                            "header": _on(rows[row].find("w:trPr/w:tblHeader", NS)),
                            "bbox": None,
                            "nested_tables": [],
                            "source": {
                                "parser": "ooxml",
                                "part": part,
                                "table": table_id,
                                "geometry": "omitted_grid_cell",
                            },
                        }
                    )
        table["nrows"] = len(rows)
        table["ncols"] = estimated_columns
        table["cells"] = sorted(cells, key=lambda item: (item["row"], item["col"]))
        fatal = {"orphan_vertical_merge", "overlapping_cells"}
        table["quality"] = {
            "passed": not fatal.intersection(issues) and bool(rows) and estimated_columns > 0,
            "stage": "ooxml",
            "issues": sorted(set(issues)),
            "nonempty_cell_ratio": round(
                sum(bool(cell["text"].strip()) for cell in cells) / max(len(cells), 1), 4
            ),
            "source_text_recall": 1.0,
        }
        return table

    def _container_blocks(self, container, part, styles, numbering, relationships, section_index=0, main=False):
        blocks = []
        for child in self._block_children(container):
            self._block_number += 1
            if child.tag == qn(W_NS, "p"):
                block = self._paragraph(child, part, styles, numbering, relationships, section_index)
                if block["text"] or block["page_breaks"]:
                    blocks.append(block)
                if main and child.find("w:pPr/w:sectPr", NS) is not None:
                    section_index += 1
            else:
                table = self._table(child, part, styles, numbering, relationships, section_index)
                blocks.append(
                    {
                        "id": f"block-{self._block_number:05d}",
                        "type": "table",
                        "order": self._block_number,
                        "section": section_index,
                        "table_id": table["id"],
                        "text": "\n".join(cell["text"] for cell in table["cells"] if cell["text"]),
                        "source": table["source"],
                    }
                )
        return blocks, section_index

    def _metadata(self, archive):
        root = self._xml(archive, "docProps/core.xml")
        if root is None:
            return {}
        result = {}
        for child in root:
            name = etree.QName(child).localname
            if child.text:
                result[name] = child.text
        return result

    def read(self):
        if not zipfile.is_zipfile(self.path):
            raise ValueError("Input is not a valid DOCX/ZIP package")
        with zipfile.ZipFile(self.path) as archive:
            entries = archive.infolist()
            if len(entries) > self.max_entries:
                raise ValueError(f"DOCX package has too many entries: {len(entries)}")
            total_size = sum(entry.file_size for entry in entries)
            if total_size > self.max_uncompressed_bytes:
                raise ValueError(f"DOCX uncompressed size is excessive: {total_size}")
            names = set(archive.namelist())
            root = self._xml(archive, "word/document.xml", required=True)
            body = root.find("w:body", NS)
            if body is None:
                raise ValueError("DOCX has no word/document.xml body")
            styles = self._styles(archive)
            numbering = self._numbering(archive)
            relationships = self._relationships(archive, "word/document.xml")
            blocks, last_section = self._container_blocks(
                body, "word/document.xml", styles, numbering, relationships, main=True
            )

            ancillary = {"headers": [], "footers": [], "footnotes": [], "endnotes": []}
            prefixes = {
                "headers": "word/header",
                "footers": "word/footer",
                "footnotes": "word/footnotes.xml",
                "endnotes": "word/endnotes.xml",
            }
            for kind, prefix in prefixes.items():
                matching = sorted(name for name in names if name.startswith(prefix) and name.endswith(".xml"))
                for part in matching:
                    part_root = self._xml(archive, part)
                    if part_root is None:
                        continue
                    part_blocks, _ = self._container_blocks(
                        part_root, part, styles, numbering, self._relationships(archive, part)
                    )
                    ancillary[kind].append({"part": part, "blocks": part_blocks})

            drawing_count = len(root.findall(".//w:drawing", NS)) + len(root.findall(".//w:pict", NS))
            equation_count = len(root.findall(".//m:oMath", NS)) + len(root.findall(".//m:oMathPara", NS))
            textbox_count = len(root.findall(".//w:txbxContent", NS))
            inserted_count = len(root.findall(".//w:ins", NS))
            deleted_count = len(root.findall(".//w:del", NS))
            if drawing_count:
                self.unresolved.append({"type": "images_skipped", "count": drawing_count})
            if textbox_count:
                self.unresolved.append(
                    {"type": "floating_text_order", "count": textbox_count, "handling": "kept_with_anchor_paragraph"}
                )
            native_text = "\n".join(block.get("text", "") for block in blocks)
            return {
                "blocks": blocks,
                "tables": self.tables,
                "sections": last_section + 1,
                "ancillary": ancillary,
                "metadata": self._metadata(archive),
                "unresolved": self.unresolved,
                "package": {
                    "entry_count": len(entries),
                    "uncompressed_bytes": total_size,
                    "drawing_count": drawing_count,
                    "equation_count": equation_count,
                    "textbox_count": textbox_count,
                    "inserted_revision_count": inserted_count,
                    "deleted_revision_count": deleted_count,
                },
                "native_character_count": sum(not char.isspace() for char in native_text),
            }
