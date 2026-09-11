"""Download immutable official model snapshots; all bytes live in the mount."""
import concurrent.futures, hashlib, json, os, pathlib, sys, time
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = {
    'mineru': 'opendatalab/MinerU2.5-Pro-2605-1.2B',
    'tatr': 'microsoft/table-transformer-structure-recognition-v1.1-all',
    'tatr-detection': 'microsoft/table-transformer-detection',
}

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    for attempt in range(3):
        try:
            offset = tmp.stat().st_size if tmp.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(20, 180)) as r:
                r.raise_for_status()
                append = offset and r.status_code == 206
                with tmp.open('ab' if append else 'wb') as f:
                    for chunk in r.iter_content(1024*1024):
                        f.write(chunk)
            tmp.replace(path)
            return
        except Exception:
            if attempt == 2: raise
            time.sleep(2)

def model(name):
    repo = MODELS[name]
    endpoints = ['https://huggingface.co'] if os.environ.get('https_proxy') else ['https://hf-mirror.com', 'https://huggingface.co']
    for endpoint in endpoints:
        try:
            r = requests.get(f'{endpoint}/api/models/{repo}', timeout=25)
            r.raise_for_status()
            info = r.json()
            dest = ROOT/'models'/name
            dest.mkdir(parents=True, exist_ok=True)
            manifest = ROOT/'manifests'/f'model-{name}.json'
            old = json.loads(manifest.read_text()) if manifest.exists() else {}
            sha = old.get('revision', info['sha'])
            if sha != info['sha']:
                info = requests.get(f'{endpoint}/api/models/{repo}/revision/{sha}', timeout=25).json()
            names = [s['rfilename'] for s in info['siblings'] if s['rfilename'].endswith(('.json','.safetensors','.py','.txt','.model','.jinja','.md'))]
            if not any(n.endswith('.safetensors') for n in names):
                names += [s['rfilename'] for s in info['siblings'] if s['rfilename'].endswith('pytorch_model.bin')]
            def one(n):
                p = dest/n
                if not p.exists(): fetch(f'{endpoint}/{repo}/resolve/{sha}/{n}', p)
                h = hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                print(f'{name}/{n} {p.stat().st_size}', flush=True)
                return {'file':n,'bytes':p.stat().st_size,'sha256':h}
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                files = list(pool.map(one,names))
            manifest.write_text(json.dumps({'repository':repo,'revision':sha,'endpoint':endpoint,'files':files},indent=2))
            return
        except Exception as e:
            print(f'{name} endpoint {endpoint} failed: {e}',flush=True)
    raise RuntimeError(f'Download failed for {name}; retry after sourcing the configured proxy')

if __name__ == '__main__':
    for name in sys.argv[1:] or MODELS: model(name)
