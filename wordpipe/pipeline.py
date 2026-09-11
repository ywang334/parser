"""Native Word pipeline with LibreOffice conversion and PDF fallback."""

from __future__ import annotations

import hashlib
import html
import json
import pathlib
import shutil
import subprocess
import tempfile
import time

from pdfpipe.tables import render

from .ooxml import OOXMLReader


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def document_output_name(path):
    path = pathlib.Path(path)
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    return f"{path.stem}-{digest[:8]}", digest


class LibreOfficeConverter:
    def __init__(self):
        self.executable = shutil.which("libreoffice") or shutil.which("soffice")
        self.events = []
        self._version = None

    def version(self):
        if not self.executable:
            return None
        if self._version is not None:
            return self._version
        completed = subprocess.run(
            [self.executable, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
        self._version = (completed.stdout or completed.stderr).strip() or None
        return self._version

    def convert(self, source, output_dir, extension):
        if not self.executable:
            raise RuntimeError("LibreOffice is not installed in the runtime image")
        source = pathlib.Path(source).resolve()
        output_dir = pathlib.Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        filter_name = "Office Open XML Text" if extension == "docx" else "writer_pdf_Export"
        before = set(output_dir.glob(f"*.{extension}"))
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="lo-profile-") as profile:
            command = [
                self.executable,
                f"-env:UserInstallation={pathlib.Path(profile).resolve().as_uri()}",
                "--headless",
                "--nologo",
                "--nodefault",
                "--nofirststartwizard",
                "--norestore",
                "--convert-to",
                f"{extension}:{filter_name}",
                "--outdir",
                str(output_dir),
                str(source),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
        created = [path for path in output_dir.glob(f"*.{extension}") if path not in before]
        expected = output_dir / f"{source.stem}.{extension}"
        if completed.returncode != 0 or (not created and not expected.exists()):
            raise RuntimeError(
                f"LibreOffice conversion to {extension} failed: "
                f"{(completed.stderr or completed.stdout).strip()}"
            )
        result = expected if expected.exists() else created[0]
        event = {
            "source": str(source),
            "output": str(result),
            "format": extension,
            "seconds": round(time.time() - started, 3),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        self.events.append(event)
        return result


class WordPipeline:
    def __init__(self, *, resume=False, pdf_config=None, dpi=144, use_mineru=True):
        self.resume = resume
        self.pdf_config = pdf_config
        self.dpi = dpi
        self.use_mineru = use_mineru
        self.converter = LibreOfficeConverter()

    @staticmethod
    def _write_table_files(output, tables):
        table_dir = output / "tables"
        raw_dir = output / "raw" / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for table in tables:
            raw_xml = table.pop("_raw_xml", None)
            table["raw_xml"] = f"raw/tables/{table['id']}.xml" if raw_xml is not None else None
            if raw_xml is not None:
                (raw_dir / f"{table['id']}.xml").write_text(raw_xml)
            (table_dir / f"{table['id']}.html").write_text(
                "<meta charset='utf-8'>" + render(table)
            )

    @staticmethod
    def _run_html(block):
        runs = block.get("runs", [])
        if not runs or "".join(run["text"] for run in runs).strip() != block.get("text", "").strip():
            return html.escape(block.get("text", "")).replace("\n", "<br>")
        parts = []
        for run in runs:
            value = html.escape(run["text"]).replace("\n", "<br>")
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
                value = f"<a href='{html.escape(run['hyperlink'], quote=True)}'>{value}</a>"
            parts.append(value)
        return "".join(parts)

    @classmethod
    def write_report(cls, output, document):
        table_by_id = {table["id"]: table for table in document["tables"]}
        style = (
            "body{font:15px sans-serif;line-height:1.55;margin:24px;max-width:1200px}"
            "table{border-collapse:collapse;margin:10px 0 28px;width:auto;max-width:100%}"
            "td,th{border:1px solid #888;padding:5px;vertical-align:top}"
            ".meta,.warn{color:#666}.list{margin-left:var(--indent)}"
            ".source{font-size:12px;color:#777}section{margin:24px 0}"
        )
        body = [
            "<!doctype html><html lang='zh'><meta charset='utf-8'><title>Word 解析人工核验</title>",
            f"<style>{style}</style><h1>Word 解析人工核验</h1>",
            f"<p class='meta'>文件：{html.escape(document['input'])} · "
            f"模式：{html.escape(document['parse_mode'])} · "
            f"正文块：{len(document.get('blocks', []))} · 表格：{len(document.get('tables', []))}</p>",
            "<section><h2>正文与表格顺序</h2>",
        ]
        for block in document.get("blocks", []):
            if block["type"] == "table":
                table = table_by_id.get(block["table_id"])
                if table:
                    body.append(
                        f"<h3>{html.escape(table['id'])} · {table['nrows']}×{table['ncols']}</h3>"
                        f"{render(table)}<p class='source'>{html.escape(table['source']['xml_path'])}</p>"
                    )
                continue
            content = cls._run_html(block)
            if block["type"] == "heading":
                level = min(6, block.get("heading_level") or 2)
                body.append(f"<h{level}>{content}</h{level}>")
            elif block["type"] == "list_item":
                indent = 24 * (1 + block.get("list", {}).get("level", 0))
                body.append(f"<p class='list' style='--indent:{indent}px'>• {content}</p>")
            else:
                body.append(f"<p>{content}</p>")
        body.append("</section>")
        top_level_tables = {
            block["table_id"] for block in document.get("blocks", []) if block["type"] == "table"
        }
        nested_tables = [table for table in document["tables"] if table["id"] not in top_level_tables]
        if nested_tables:
            body.append("<section><h2>嵌套表格</h2>")
            for table in nested_tables:
                body.append(
                    f"<h3>{html.escape(table['id'])} · 父表 {html.escape(table.get('parent_table') or '')} · "
                    f"{table['nrows']}×{table['ncols']}</h3>{render(table)}"
                )
            body.append("</section>")
        if document.get("ancillary"):
            body.append("<section><h2>页眉、页脚与注释性内容</h2>")
            for kind, parts in document["ancillary"].items():
                for part in parts:
                    text = "\n".join(block.get("text", "") for block in part["blocks"])
                    if text.strip():
                        body.append(
                            f"<h3>{html.escape(kind)} · {html.escape(part['part'])}</h3>"
                            f"<pre>{html.escape(text)}</pre>"
                        )
            body.append("</section>")
        if document.get("unresolved") or document.get("errors"):
            body.append(
                "<section><h2>未解决项与错误</h2><pre>"
                + html.escape(
                    json.dumps(
                        {"unresolved": document.get("unresolved", []), "errors": document.get("errors", [])},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre></section>"
            )
        body.append("</html>")
        (output / "index.html").write_text("\n".join(body))

    def _pdf_fallback(self, input_path, docx_path, output, native_probe):
        converted_dir = output / "converted"
        converted_pdf = self.converter.convert(docx_path, converted_dir, "pdf")
        stable_pdf = converted_dir / "source.pdf"
        if converted_pdf != stable_pdf:
            converted_pdf.replace(stable_pdf)
        from pdfpipe.cli import cuda_preflight
        from pdfpipe.pipeline import PDFPipeline

        gpu_preflight = cuda_preflight()
        config = json.loads(pathlib.Path(self.pdf_config).read_text())
        pipeline = PDFPipeline(
            dpi=self.dpi,
            use_mineru=self.use_mineru,
            resume=self.resume,
            config=config,
        )
        result, pdf_runtime = pipeline.run_document(stable_pdf, output)
        result.pop("metrics", None)
        result.update(
            {
                "input": str(input_path),
                "source_format": input_path.suffix.lower().lstrip("."),
                "parse_mode": "word_image_pdf_fallback",
                "converted_pdf": "converted/source.pdf",
                "word_native_probe": native_probe,
            }
        )
        atomic_json(output / "document.json", result)
        report = (output / "index.html").read_text()
        report = report.replace("PDF 解析人工核验", "图像型 Word 解析人工核验")
        (output / "index.html").write_text(report)
        return result, {"gpu_preflight": gpu_preflight, "pdf": pdf_runtime}

    def run_document(self, input_path, output):
        started = time.time()
        input_path, output = pathlib.Path(input_path), pathlib.Path(output)
        output.mkdir(parents=True, exist_ok=True)
        source_format = input_path.suffix.lower().lstrip(".")
        docx_path = input_path
        if source_format == "doc":
            converted_dir = output / "converted"
            converted = self.converter.convert(input_path, converted_dir, "docx")
            docx_path = converted_dir / "source.docx"
            if converted != docx_path:
                converted.replace(docx_path)
        parsed = OOXMLReader(docx_path).read()
        native_probe = {
            "native_character_count": parsed["native_character_count"],
            "drawing_count": parsed["package"]["drawing_count"],
            "image_only": parsed["native_character_count"] < 20 and parsed["package"]["drawing_count"] > 0,
        }
        atomic_json(
            output / "raw" / "ooxml.json",
            {"package": parsed["package"], "native_probe": native_probe, "unresolved": parsed["unresolved"]},
        )
        if native_probe["image_only"]:
            result, fallback_runtime = self._pdf_fallback(input_path, docx_path, output, native_probe)
            fallback_runtime["seconds"] = round(time.time() - started, 3)
            return result, fallback_runtime

        self._write_table_files(output, parsed["tables"])
        structural_failures = [
            {"table": table["id"], "quality": table["quality"]}
            for table in parsed["tables"]
            if not table["quality"]["passed"]
        ]
        result = {
            "schema_version": "1.0",
            "status": "partial" if structural_failures else "ok",
            "input": str(input_path),
            "source_format": source_format,
            "parse_mode": "native_ooxml" if source_format == "docx" else "libreoffice_doc_to_docx_ooxml",
            "page_count": None,
            "processed_pages": None,
            "section_count": parsed["sections"],
            "metadata": parsed["metadata"],
            "blocks": parsed["blocks"],
            "tables": parsed["tables"],
            "ancillary": parsed["ancillary"],
            "unresolved": parsed["unresolved"],
            "errors": structural_failures,
        }
        atomic_json(output / "document.json", result)
        self.write_report(output, result)
        return result, {
            "libreoffice_events": self.converter.events,
            "seconds": round(time.time() - started, 3),
        }
