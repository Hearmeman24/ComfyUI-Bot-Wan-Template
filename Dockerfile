# Use multi-stage build with caching optimizations
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

# ------------------------------------------------------------
# Consolidated environment variables
# ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# ------------------------------------------------------------
# System packages + Python 3.12 venv
# ------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        aria2 \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
        curl ffmpeg ninja-build git git-lfs vim \
        libgl1 libglib2.0-0 build-essential gcc && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python3.12 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# ------------------------------------------------------------
# PyTorch (CUDA 12.8) & core tooling (no pip cache mounts)
# ------------------------------------------------------------
RUN pip install --upgrade pip && \
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip freeze | grep -E "^(torch|torchvision|torchaudio)" > /tmp/torch-constraint.txt && \
    pip install packaging setuptools wheel pyyaml gdown triton runpod opencv-python

# Clone ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /ComfyUI

# Install ComfyUI requirements using torch constraint file
RUN cd /ComfyUI && \
    pip install -r requirements.txt --constraint /tmp/torch-constraint.txt

# ------------------------------------------------------------
# Final stage
# ------------------------------------------------------------
FROM base AS final
ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir -p /models/diffusion_models /models/text_encoders /models/vae /models/clip_vision /models/loras

# Download models with aria2
RUN aria2c -x16 -s16 -d /models/diffusion_models -o wan2.1_i2v_720p_14B_fp16.safetensors \
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors
RUN aria2c -x16 -s16 -d /models/diffusion_models -o wan2.1_t2v_14B_bf16.safetensors \
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors
RUN aria2c -x16 -s16 -d /models/diffusion_models -o wan2.1_vace_1.3B_preview_fp16.safetensors \
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_preview_fp16.safetensors
RUN aria2c -x16 -s16 -d /models/diffusion_models -o wan2.1_t2v_1.3B_bf16.safetensors \
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors

# Text encoders
RUN aria2c -x16 -s16 -d /models/text_encoders -o umt5-xxl-enc-bf16.safetensors \
    https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors
RUN aria2c -x16 -s16 -d /models/text_encoders -o open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors \
    https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors
RUN aria2c -x16 -s16 -d /models/text_encoders -o umt5_xxl_fp8_e4m3fn_scaled.safetensors \
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

# Variational Autoencoders (VAE)
RUN aria2c -x16 -s16 -d /models/vae -o Wan2_1_VAE_bf16.safetensors \
    https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors
RUN aria2c -x16 -s16 -d /models/vae -o wan_2.1_vae.safetensors \
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors

# CLIP vision model
RUN aria2c -x16 -s16 -d /models/clip_vision -o clip_vision_h.safetensors \
    https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors

RUN aria2c -x16 -s16 -d /models/loras -o Wan21_CausVid_14B_T2V_lora_rank32.safetensors \
    https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors

# Upscalers
RUN git clone https://github.com/Hearmeman24/upscalers.git /tmp/upscalers && \
    cp /tmp/upscalers/4xLSDIR.pth /4xLSDIR.pth && \
    rm -rf /tmp/upscalers

RUN mkdir -p /models/loras
COPY download_loras.sh /tmp/
RUN chmod +x /tmp/download_loras.sh && /tmp/download_loras.sh

RUN echo "torch==2.8.0.dev20250511+cu128" > /torch-constraint.txt && \
    echo "torchaudio==2.6.0.dev20250511+cu128" >> /torch-constraint.txt && \
    echo "torchsde==0.2.6" >> /torch-constraint.txt && \
    echo "torchvision==0.22.0.dev20250511+cu128" >> /torch-constraint.txt

# Clone and install all your custom nodes
RUN for repo in \
    https://github.com/kijai/ComfyUI-KJNodes.git \
    https://github.com/Comfy-Org/ComfyUI-Manager.git \
    https://github.com/nonnonstop/comfyui-faster-loading.git \
    https://github.com/rgthree/rgthree-comfy.git \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    https://github.com/cubiq/ComfyUI_essentials.git \
    https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    https://github.com/chrisgoringe/cg-use-everywhere.git \
    https://github.com/tsogzark/ComfyUI-load-image-from-url.git; \
  do \
    cd /ComfyUI/custom_nodes; \
    repo_dir=$(basename "$repo" .git); \
    git clone "$repo"; \
    if [ -f "/ComfyUI/custom_nodes/$repo_dir/requirements.txt" ]; then \
      pip install -r "/ComfyUI/custom_nodes/$repo_dir/requirements.txt" --constraint /torch-constraint.txt; \
    fi; \
    if [ -f "/ComfyUI/custom_nodes/$repo_dir/install.py" ]; then \
      python "/ComfyUI/custom_nodes/$repo_dir/install.py"; \
    fi; \
  done

# SageAttention and other Python deps
RUN pip install --no-cache-dir \
    https://raw.githubusercontent.com/Hearmeman24/upscalers/master/sageattention-2.1.1-cp312-cp312-linux_x86_64.whl

RUN pip install --no-cache-dir discord.py==2.5.2 \
                              python-dotenv==1.1.0 \
                              Requests==2.32.3 \
                              websocket_client==1.8.0 \
                              "httpx[http2]"

# Frame Interp checkpoint
RUN mkdir -p /ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/film/ && \
    aria2c -x16 -s16 -d /ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/film -o film_net_fp32.pt \
    https://d1s3da0dcaf6kx.cloudfront.net/film_net_fp32.pt

# Entrypoint
COPY src/start_script.sh /start_script.sh
RUN chmod +x /start_script.sh
EXPOSE 8888
CMD ["/start_script.sh"]