"""Download MinerU into the host-mounted model directory via ModelScope."""

from modelscope.hub.snapshot_download import snapshot_download

print(snapshot_download(
    "OpenDataLab/MinerU2.5-Pro-2605-1.2B",
    local_dir="/workspace/models/mineru",
    max_workers=4,
), flush=True)
