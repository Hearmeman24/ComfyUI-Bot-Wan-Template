# Use multi-stage build with caching optimizations
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
   PIP_PREFER_BINARY=1 \
   PYTHONUNBUFFERED=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=8

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
        curl ffmpeg ninja-build git git-lfs wget vim \
        libgl1 libglib2.0-0 build-essential gcc && \
    \
    # make Python3.12 the default python & pip
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    \
    # Create and activate virtual environment
    python3.12 -m venv /opt/venv && \
    \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128

# Core Python tooling
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging setuptools wheel

# Runtime libraries
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyyaml gdown runpod triton comfy-cli jupyterlab jupyterlab-lsp \
        jupyter-server jupyter-server-terminals \
        ipykernel jupyterlab_code_formatter

# ------------------------------------------------------------
# ComfyUI install
# ------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    /usr/bin/yes | comfy --workspace /ComfyUI install

FROM base AS final
# Make sure to use the virtual environment here too
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install opencv-python

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
RUN chmod +x /start_script.sh
COPY 4xLSDIR.pth /4xLSDIR.pth

CMD ["/start_script.sh"]