# Use multi-stage build with caching optimizations
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 AS base

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
   PIP_PREFER_BINARY=1 \
   PYTHONUNBUFFERED=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=8

ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip curl ffmpeg ninja-build git git-lfs wget vim libgl1 libglib2.0-0 \
    python3-dev build-essential gcc \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# PIP caching with BuildKit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# More pip installations with cache
# Install basic Python tools first in a separate step
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install packaging setuptools wheel

# Install other Python packages in a second step
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install gdown runpod triton comfy-cli jupyterlab jupyterlab-lsp \
    jupyter-server jupyter-server-terminals \
    ipykernel jupyterlab_code_formatter

# SageAttention installation in a third step
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install git+https://github.com/thu-ml/SageAttention.git

# ComfyUI installation with cache
RUN --mount=type=cache,target=/root/.cache/pip \
    /usr/bin/yes | comfy --workspace /ComfyUI install \
    --cuda-version 12.6 --nvidia

FROM base AS final
RUN python -m pip install opencv-python

RUN for repo in \
    https://github.com/kijai/ComfyUI-KJNodes.git \
    https://github.com/rgthree/rgthree-comfy.git \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
    https://github.com/cubiq/ComfyUI_essentials.git \
    https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    https://github.com/tsogzark/ComfyUI-load-image-from-url.git; \
    do \
        cd /ComfyUI/custom_nodes; \
        repo_dir=$(basename "$repo" .git); \
        if [ "$repo" = "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git" ]; then \
            git clone --recursive "$repo"; \
        else \
            git clone "$repo"; \
        fi; \
        if [ -f "/ComfyUI/custom_nodes/$repo_dir/requirements.txt" ]; then \
            pip install -r "/ComfyUI/custom_nodes/$repo_dir/requirements.txt"; \
        fi; \
        if [ -f "/ComfyUI/custom_nodes/$repo_dir/install.py" ]; then \
            python "/ComfyUI/custom_nodes/$repo_dir/install.py"; \
        fi; \
    done

COPY src/start_script.sh /start_script.sh
COPY 4xLSDIR.pth /4xLSDIR.pth

CMD ["/start_script.sh"]