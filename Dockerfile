# Use multi-stage build with caching optimizations
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
   PIP_PREFER_BINARY=1 \
   PYTHONUNBUFFERED=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python 3.10 specifically and make it the default
RUN apt-get update && apt-get install -y --no-install-recommends \
   python3.10 python3.10-dev python3.10-distutils python3-pip curl ffmpeg ninja-build \
   git git-lfs wget vim libgl1 libglib2.0-0 build-essential gcc \
   && ln -sf /usr/bin/python3.10 /usr/bin/python \
   && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
   && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
   && ln -sf /usr/local/bin/pip /usr/bin/pip \
   && ln -sf /usr/local/bin/pip /usr/bin/pip3 \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# Verify Python version
RUN python --version && pip --version

# Install the specific torch version first
RUN pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps
RUN pip install torchvision==0.17.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Create a constraint file to prevent torch upgrades
RUN echo "torch==2.6.0+cu124" > /torch-constraint.txt

# Install other packages with the constraint
RUN pip install --no-cache-dir gdown runpod packaging setuptools wheel --constraint /torch-constraint.txt

# Install ComfyUI with specific flags to avoid torch conflicts
RUN pip install --no-cache-dir comfy-cli --constraint /torch-constraint.txt
RUN /usr/bin/yes | comfy --workspace /ComfyUI install --cuda-version 12.4 --nvidia

FROM base AS final
RUN pip install opencv-python --constraint /torch-constraint.txt

# Install custom nodes with constraint to prevent torch upgrades
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
            # Install requirements with the torch constraint
            pip install -r "/ComfyUI/custom_nodes/$repo_dir/requirements.txt" --constraint /torch-constraint.txt; \
        fi; \
        if [ -f "/ComfyUI/custom_nodes/$repo_dir/install.py" ]; then \
            python "/ComfyUI/custom_nodes/$repo_dir/install.py"; \
        fi; \
    done

# Ensure torch version is correct at the end by force reinstalling
RUN pip uninstall -y torch torchvision torchaudio
RUN pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps
RUN pip install torchvision==0.17.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install SageAttention after ensuring the correct torch version
COPY sageattention-2.1.1-cp310-cp310-linux_x86_64.whl /tmp/
RUN pip install /tmp/sageattention-2.1.1-cp310-cp310-linux_x86_64.whl

# Install jupyterlab AFTER SageAttention to avoid package conflicts
RUN pip install jupyterlab jupyterlab-lsp jupyter-server jupyter-server-terminals \
    ipykernel jupyterlab_code_formatter --constraint /torch-constraint.txt

# Verify Python and PyTorch version
RUN python -c "import sys; print('Python version:', sys.version); import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

COPY src/start_script.sh /start_script.sh
RUN chmod +x /start_script.sh
COPY 4xLSDIR.pth /4xLSDIR.pth

EXPOSE 8888
CMD ["/start_script.sh"]