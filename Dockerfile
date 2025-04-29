# Use multi-stage build with caching optimizations
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
   PIP_PREFER_BINARY=1 \
   PYTHONUNBUFFERED=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=8

# Consolidated installation to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
   python3.11 python3-pip curl ffmpeg ninja-build git git-lfs wget vim libgl1 libglib2.0-0 \
   python3-dev build-essential gcc \
   && ln -sf /usr/bin/python3.11 /usr/bin/python \
   && ln -sf /usr/bin/pip3 /usr/bin/pip \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# First install the SPECIFIC torch version you need
# This establishes it as the base version that other packages will use
RUN pip install --no-cache-dir torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Use build cache for pip installations - but do NOT install any conflicting torch versions
RUN pip install --no-cache-dir gdown runpod packaging setuptools wheel comfy-cli jupyterlab jupyterlab-lsp \
    jupyter-server jupyter-server-terminals \
    ipykernel jupyterlab_code_formatter

# Prevent comfy from installing its own torch version by setting environment variables
ENV TORCH_CUDA_ARCH_LIST="8.9" \
    COMFYUI_SKIP_TORCH_INSTALL=1

# Install ComfyUI but skip torch installation
RUN /usr/bin/yes | comfy --workspace /ComfyUI install \
   --skip-torch --cuda-version 12.4 --nvidia

FROM base AS final
RUN python -m pip install opencv-python

# Install custom nodes
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
            # Modify requirements files to prevent torch installations
            sed -i '/torch/d' "/ComfyUI/custom_nodes/$repo_dir/requirements.txt"; \
            pip install -r "/ComfyUI/custom_nodes/$repo_dir/requirements.txt"; \
        fi; \
        if [ -f "/ComfyUI/custom_nodes/$repo_dir/install.py" ]; then \
            python "/ComfyUI/custom_nodes/$repo_dir/install.py"; \
        fi; \
    done

# Re-install torch at the end to ensure it's the final version
RUN pip uninstall -y torch torchvision torchaudio
RUN pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps

# Install SageAttention after ensuring the correct torch version
COPY sageattention-2.1.1-cp310-cp310-linux_x86_64.whl /tmp/
RUN pip install /tmp/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl

# Verify torch version
RUN python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

COPY src/start_script.sh /start_script.sh
RUN chmod +x /start_script.sh
COPY 4xLSDIR.pth /4xLSDIR.pth

EXPOSE 8888
CMD ["/start_script.sh"]