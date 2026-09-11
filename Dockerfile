FROM docker.m.daocloud.io/library/python:3.12-slim-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends libreoffice-writer fonts-noto-cjk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENV HF_HOME=/workspace/cache/huggingface XDG_CACHE_HOME=/workspace/cache \
    MODELSCOPE_CACHE=/workspace/cache/modelscope PYTHONPATH=/workspace \
    USER=parser LOGNAME=parser TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor
CMD ["/bin/bash"]
