# GPU environment

- Runtime image: `pdf-structure-parser:gpu`, built from `Dockerfile`.
- Host-mounted Python environment: `envs/pipeline` only.
- PyTorch source: `https://download.pytorch.org/whl/cu128` by default.
- TATR Detection and TATR-v1.1-All require CUDA and run in FP32.
- MinerU requires CUDA. `auto` uses BF16 when `torch.cuda.is_bf16_supported()` is true, otherwise FP32.
- RapidOCR prefers its Torch CUDA engine and is the only component allowed to fall back to ONNX Runtime CPU.
- Authoritative dependency record: `manifests/pipeline.freeze.txt`.
- Model weights and SHA-256 manifests stay under `models/` and `manifests/model-*.json`, never inside the image.
