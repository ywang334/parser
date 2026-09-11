import zipfile

from wordpipe.ooxml import OOXMLReader


DOCUMENT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Title</w:t></w:r></w:p>
    <w:tbl>
      <w:tblGrid><w:gridCol/><w:gridCol/><w:gridCol/></w:tblGrid>
      <w:tr>
        <w:trPr><w:tblHeader/></w:trPr>
        <w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>C</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>D</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>E</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc>
        <w:tc><w:p><w:r><w:t>F</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>G</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:r><w:t>After</w:t></w:r></w:p>
    <w:sectPr/>
  </w:body>
</w:document>
"""

STYLES = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr>
  </w:style>
</w:styles>
"""


def make_docx(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", DOCUMENT)
        archive.writestr("word/styles.xml", STYLES)


def test_ooxml_preserves_order_and_merged_cells(tmp_path):
    path = tmp_path / "fixture.docx"
    make_docx(path)
    result = OOXMLReader(path).read()
    assert [block["type"] for block in result["blocks"]] == ["heading", "table", "paragraph"]
    table = result["tables"][0]
    assert (table["nrows"], table["ncols"]) == (3, 3)
    cells = {cell["text"]: cell for cell in table["cells"] if cell["text"]}
    assert cells["A"]["colspan"] == 2
    assert cells["A"]["header"]
    assert cells["C"]["rowspan"] == 2
    assert table["quality"]["passed"]


def test_rejects_non_docx_zip(tmp_path):
    path = tmp_path / "bad.docx"
    path.write_text("not a zip")
    try:
        OOXMLReader(path).read()
    except ValueError as exc:
        assert "valid DOCX" in str(exc)
    else:
        raise AssertionError("invalid package was accepted")
