"""Public real PDFs plus a labeled synthetic regression fixture (kept distinct)."""
import hashlib,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from download import fetch

ROOT=pathlib.Path(__file__).resolve().parents[1]
out=ROOT/'data/samples';out.mkdir(parents=True,exist_ok=True)
sources={
 'background-checks.pdf':'https://raw.githubusercontent.com/jsvine/pdfplumber/stable/examples/pdfs/background-checks.pdf',
 'eu-energy.pdf':'https://raw.githubusercontent.com/jsvine/pdfplumber/stable/examples/pdfs/ca-warn-report.pdf',
 'attention.pdf':'https://arxiv.org/pdf/1706.03762',
}
manifest=[]
for name,url in sources.items():
    try:
        dest=out/name
        if not dest.exists():fetch(url,dest)
        if not dest.read_bytes().startswith(b'%PDF'):raise ValueError('Not PDF')
        manifest.append(dict(file=name,url=url,sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),ground_truth=None,type='real_public_pdf'))
    except Exception as e:manifest.append(dict(file=name,url=url,status='download_failed',error=str(e)))
(out/'sources.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
