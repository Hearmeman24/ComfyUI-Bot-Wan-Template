#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

set -eo pipefail
set +u
# Set the network volume path
NETWORK_VOLUME="/workspace"

# Check if NETWORK_VOLUME exists; if not, use root directory instead
if [ ! -d "$NETWORK_VOLUME" ]; then
    NETWORK_VOLUME="/"
    echo "Settings network volume to $NETWORK_VOLUME"
fi


FLAG_FILE="$NETWORK_VOLUME/.comfyui_initialized"
COMFYUI_DIR="$NETWORK_VOLUME/ComfyUI"
REPO_DIR="$NETWORK_VOLUME/comfyui-discord-bot"
URL="http://127.0.0.1:8188"

sync_bot_repo() {
  # pick branch based on IS_DEV
  if [ "${IS_DEV:-false}" = "true" ]; then
    BRANCH="dev"
  else
    BRANCH="master"
  fi

  echo "Syncing bot repo (branch: $BRANCH)…"
  if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning '$BRANCH' into $REPO_DIR"
    git clone --branch "$BRANCH" \
      "https://${GITHUB_PAT}@github.com/Hearmeman24/comfyui-discord-bot.git" \
      "$REPO_DIR"
    echo "Clone complete"

    echo "Installing Python deps…"
    cd "$REPO_DIR"
    pip install --upgrade -r requirements.txt
    echo "Dependencies installed"
    cd /
  else
    echo "Updating existing repo in $REPO_DIR"
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"

    echo "🐍 Re‑installing any updated deps…"
    pip install --upgrade -r requirements.txt
    cd /
  fi
}

if [ -f "$FLAG_FILE" ]; then
  echo "FLAG FILE FOUND"

  # Add cd $NETWORK_VOLUME to shell startup if not already present
  grep -qxF "cd $NETWORK_VOLUME" ~/.bashrc || echo "cd $NETWORK_VOLUME" >> ~/.bashrc
  grep -qxF "cd $NETWORK_VOLUME" ~/.bash_profile || echo "cd $NETWORK_VOLUME" >> ~/.bash_profile

  sync_bot_repo

  echo "▶️  Starting ComfyUI"
  # group both the main and fallback commands so they share the same log
  nohup python3 "$NETWORK_VOLUME"/ComfyUI/main.py --listen > "$NETWORK_VOLUME"/comfyui_nohup.log 2>&1 &

  echo "⏳  Waiting for ComfyUI to be up at $URL…"
  if ! command -v curl >/dev/null 2>&1; then
    echo "🔧 curl not found. Installing..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update && apt-get install -y curl
    elif command -v yum >/dev/null 2>&1; then
      yum install -y curl
    else
      echo "❌ No supported package manager found. Please install curl manually."
      exit 1
    fi
  fi
  until curl --silent --fail "$URL" --output /dev/null; do
    echo "🔄  Still waiting…"
    sleep 2
  done

  echo "✅  ComfyUI is up! Starting worker!"
  nohup python3 "$NETWORK_VOLUME/comfyui-discord-bot/worker.py" \
    > "$NETWORK_VOLUME/worker.log" 2>&1 &

  # Wait on background jobs forever
  wait

else
  echo "NO FLAG FILE FOUND – starting initial setup"
fi

sync_bot_repo
# Set the target directory
CUSTOM_NODES_DIR="$NETWORK_VOLUME/ComfyUI/custom_nodes"

if [ ! -d "$COMFYUI_DIR" ]; then
    mv /ComfyUI "$COMFYUI_DIR"
else
    echo "Directory already exists, skipping move."
fi

echo "Downloading CivitAI download script to /usr/local/bin"
git clone "https://github.com/Hearmeman24/CivitAI_Downloader.git" || { echo "Git clone failed"; exit 1; }
mv CivitAI_Downloader/download.py "/usr/local/bin/" || { echo "Move failed"; exit 1; }
chmod +x "/usr/local/bin/download.py" || { echo "Chmod failed"; exit 1; }
rm -rf CivitAI_Downloader  # Clean up the cloned repo
pip install huggingface_hub
pip install onnxruntime-gpu



if [ "$enable_optimizations" == "true" ]; then
echo "Downloading Triton"
pip install triton
fi

# Determine which branch to use


# Change to the directory
cd "$CUSTOM_NODES_DIR" || exit 1

# Function to download a model using huggingface-cli
download_model() {
  local destination_dir="$1"
  local destination_file="$2"
  local repo_id="$3"
  local file_path="$4"

  mkdir -p "$destination_dir"

  if [ ! -f "$destination_dir/$destination_file" ]; then
    echo "Downloading $destination_file..."

    # First, download to a temporary directory
    local temp_dir=$(mktemp -d)
    huggingface-cli download "$repo_id" "$file_path" --local-dir "$temp_dir" --resume-download

    # Find the downloaded file in the temp directory (may be in subdirectories)
    local downloaded_file=$(find "$temp_dir" -type f -name "$(basename "$file_path")")

    # Move it to the destination directory with the correct name
    if [ -n "$downloaded_file" ]; then
      mv "$downloaded_file" "$destination_dir/$destination_file"
      echo "Successfully downloaded to $destination_dir/$destination_file"
    else
      echo "Error: File not found after download"
    fi

    # Clean up temporary directory
    rm -rf "$temp_dir"
  else
    echo "$destination_file already exists, skipping download."
  fi
}

# Define base paths
DIFFUSION_MODELS_DIR="$NETWORK_VOLUME/ComfyUI/models/diffusion_models"
TEXT_ENCODERS_DIR="$NETWORK_VOLUME/ComfyUI/models/text_encoders"
CLIP_VISION_DIR="$NETWORK_VOLUME/ComfyUI/models/clip_vision"
VAE_DIR="$NETWORK_VOLUME/ComfyUI/models/vae"

# Download 480p native models
echo "Downloading 480p native models..."

download_model "$DIFFUSION_MODELS_DIR" "wan2.1_i2v_480p_14B_bf16.safetensors" \
  "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors"

download_model "$DIFFUSION_MODELS_DIR" "wan2.1_t2v_14B_bf16.safetensors" \
  "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors"

download_model "$DIFFUSION_MODELS_DIR" "wan2.1_t2v_1.3B_fp16.safetensors" \
  "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"

# Download text encoders
echo "Downloading text encoders..."

download_model "$TEXT_ENCODERS_DIR" "umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

download_model "$TEXT_ENCODERS_DIR" "open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors" \
  "Kijai/WanVideo_comfy" "open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors"

# Create CLIP vision directory and download models
mkdir -p "$CLIP_VISION_DIR"
download_model "$CLIP_VISION_DIR" "clip_vision_h.safetensors" \
  "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/clip_vision/clip_vision_h.safetensors"

# Download VAE
echo "Downloading VAE..."
download_model "$VAE_DIR" "Wan2_1_VAE_bf16.safetensors" \
  "Kijai/WanVideo_comfy" "Wan2_1_VAE_bf16.safetensors"

download_model "$VAE_DIR" "wan_2.1_vae.safetensors" \
  "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/vae/wan_2.1_vae.safetensors"

# Download upscale model
echo "Downloading upscale models"
mkdir -p "$NETWORK_VOLUME/ComfyUI/models/upscale_models"
if [ ! -f "$NETWORK_VOLUME/ComfyUI/models/upscale_models/4xLSDIR.pth" ]; then
    if [ -f "/4xLSDIR.pth" ]; then
        mv "/4xLSDIR.pth" "$NETWORK_VOLUME/ComfyUI/models/upscale_models/4xLSDIR.pth"
        echo "Moved 4xLSDIR.pth to the correct location."
    else
        echo "4xLSDIR.pth not found in the root directory."
    fi
else
    echo "4xLSDIR.pth already exists. Skipping."
fi

# Download film network model
echo "Downloading film network model"
if [ ! -f "$NETWORK_VOLUME/ComfyUI/models/upscale_models/film_net_fp32.pt" ]; then
    wget -O "$NETWORK_VOLUME/ComfyUI/models/upscale_models/film_net_fp32.pt" \
    https://huggingface.co/nguu/film-pytorch/resolve/887b2c42bebcb323baf6c3b6d59304135699b575/film_net_fp32.pt
fi

echo "Finished downloading models!"

declare -A MODEL_CATEGORIES=(
    ["$NETWORK_VOLUME/ComfyUI/models/checkpoints"]="CHECKPOINT_IDS_TO_DOWNLOAD"
    ["$NETWORK_VOLUME/ComfyUI/models/loras"]="LORAS_IDS_TO_DOWNLOAD"
)

# Ensure directories exist and download models
for TARGET_DIR in "${!MODEL_CATEGORIES[@]}"; do
    ENV_VAR_NAME="${MODEL_CATEGORIES[$TARGET_DIR]}"
    MODEL_IDS_STRING="${!ENV_VAR_NAME}"  # Get the value of the environment variable

    # Skip if the environment variable is set to "ids_here"
    if [ "$MODEL_IDS_STRING" == "replace_with_ids" ]; then
        echo "Skipping downloads for $TARGET_DIR ($ENV_VAR_NAME is 'ids_here')"
        continue
    fi

    mkdir -p "$TARGET_DIR"
    IFS=',' read -ra MODEL_IDS <<< "$MODEL_IDS_STRING"

    for MODEL_ID in "${MODEL_IDS[@]}"; do
        echo "Downloading model: $MODEL_ID to $TARGET_DIR"
        (cd "$TARGET_DIR" && download.py --model "$MODEL_ID") || {
            echo "ERROR: Failed to download model $MODEL_ID to $TARGET_DIR, continuing with next model..."
        }
    done
done

# Workspace as main working directory
echo "cd $NETWORK_VOLUME" >> ~/.bashrc
echo "cd $NETWORK_VOLUME" >> ~/.bash_profile

if [ ! -d "$NETWORK_VOLUME/ComfyUI/custom_nodes/ComfyUI-KJNodes" ]; then
    cd $NETWORK_VOLUME/ComfyUI/custom_nodes
    git clone https://github.com/kijai/ComfyUI-KJNodes.git
else
    echo "Updating KJ Nodes"
    cd $NETWORK_VOLUME/ComfyUI/custom_nodes/ComfyUI-KJNodes
    git pull
fi

# Install dependencies
pip install --no-cache-dir -r $NETWORK_VOLUME/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt

echo "Starting ComfyUI"
touch "$FLAG_FILE"
nohup python3 "$NETWORK_VOLUME"/ComfyUI/main.py --listen > "$NETWORK_VOLUME"/comfyui_nohup.log 2>&1 &

until curl --silent --fail "$URL" --output /dev/null; do
    echo "🔄  Still waiting…"
    sleep 2
done
echo "ComfyUI is UP Starting worker"
nohup python3 "$NETWORK_VOLUME"/comfyui-discord-bot/worker.py > "$NETWORK_VOLUME"/worker.log 2>&1 &

wait