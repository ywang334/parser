"""Fetch immutable official GitHub source snapshots into the mounted workspace."""
import hashlib
import io
import json
import pathlib
import tarfile

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOS = {
    'opendatalab/MinerU': '4fe4bde114a23ee5dd637eae99b767f4669bf58c',
    'microsoft/table-transformer': '16d124f616109746b7785f03085100f1f6247575',
    'opendatalab/mineru-vl-utils': '1e56064864559d6fc643f423a14e69053c06eee5',
}


def download(repo, commit):
    name = repo.rsplit('/', 1)[1]
    target = ROOT/'vendor/sources'/name
    marker = target/'.snapshot.json'
    if marker.exists() and json.loads(marker.read_text()).get('commit') == commit:
        return json.loads(marker.read_text())
    response = requests.get(f'https://codeload.github.com/{repo}/tar.gz/{commit}', timeout=(30, 300))
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    staging = ROOT/'vendor/sources'/f'.{name}-{commit}.part'
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as archive:
        members = archive.getmembers()
        prefix = members[0].name.split('/', 1)[0] + '/'
        for member in members:
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix):]
            if member.name:
                archive.extract(member, staging, filter='data')
    if target.exists():
        target.rename(target.with_name(target.name + f'.old-{commit[:8]}'))
    staging.rename(target)
    record = {'repository': repo, 'commit': commit, 'archive_sha256': digest}
    marker.write_text(json.dumps(record, indent=2))
    (ROOT/'manifests'/f'source-{name}.json').write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    for repo, commit in REPOS.items():
        print(json.dumps(download(repo, commit)), flush=True)
