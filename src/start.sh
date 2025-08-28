#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

set -eo pipefail
set +u

echo "🔧 Installing KJNodes packages..."
pip install -r /ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt &
KJ_PID=$!

echo "🔧 Installing WanVideoWrapper packages..."
pip install -r /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt &
WAN_PID=$!


if [[ "${IS_DEV,,}" =~ ^(true|1|t|yes)$ ]]; then
    API_URL="https://comfyui-job-api-dev.fly.dev"  # Replace with your development API URL
    echo "Using development API endpoint"
else
    API_URL="https://comfyui-job-api-prod.fly.dev"  # Replace with your production API URL
    echo "Using production API endpoint"
fi

URL="http://127.0.0.1:8188"

# Function to report pod status
  report_status() {
    local status=$1
    local details=$2

    echo "Reporting status: $details"

    curl -X POST "${API_URL}/pods/$RUNPOD_POD_ID/status" \
      -H "Content-Type: application/json" \
      -H "x-api-key: ${API_KEY}" \
      -d "{\"initialized\": $status, \"details\": \"$details\"}" \
      --silent

    echo "Status reported: $status - $details"
}
report_status false "Starting initialization"
# Set the network volume path
# Determine the network volume based on environment
# Check if /workspace exists
if [ -d "/workspace" ]; then
    NETWORK_VOLUME="/workspace"
# If not, check if /runpod-volume exists
elif [ -d "/runpod-volume" ]; then
    NETWORK_VOLUME="/runpod-volume"
# Fallback to root if neither directory exists
else
    echo "Warning: Neither /workspace nor /runpod-volume exists, falling back to root directory"
    NETWORK_VOLUME="/"
fi


echo "Using NETWORK_VOLUME: $NETWORK_VOLUME"
FLAG_FILE="$NETWORK_VOLUME/.comfyui_initialized"
COMFYUI_DIR="$NETWORK_VOLUME/ComfyUI"
if [ "${IS_DEV:-false}" = "true" ]; then
    REPO_DIR="$NETWORK_VOLUME/comfyui-discord-bot-dev"
    BRANCH="dev"
  else
    REPO_DIR="$NETWORK_VOLUME/comfyui-discord-bot-master"
    BRANCH="master"
fi


sync_bot_repo() {
  echo "Syncing bot repo (branch: $BRANCH)..."
  if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning '$BRANCH' into $REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --branch "$BRANCH" \
      "https://${GITHUB_PAT}@github.com/Hearmeman24/comfyui-discord-bot.git" \
      "$REPO_DIR"
    echo "Clone complete"
  fi
}


if [ -f "$FLAG_FILE" ] || [ "$new_config" = "true" ]; then
  echo "FLAG FILE FOUND"
  pip install --no-cache-dir -r $NETWORK_VOLUME/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt
  mv "/4xLSDIR.pth" "$NETWORK_VOLUME/ComfyUI/models/upscale_models" || echo "Move operation failed, continuing..."
  rm -rf "$NETWORK_VOLUME/ComfyUI/custom_nodes/ComfyUI-Manager" || echo "Remove operation failed, continuing..."
  sync_bot_repo

  wait $KJ_PID
  KJ_STATUS=$?

  wait $WAN_PID
  WAN_STATUS=$?

  # Check results
  if [ $KJ_STATUS -ne 0 ]; then
    echo "❌ KJNodes install failed."
    exit 1
  fi
  echo "✅ KJNodes install complete" 

  if [ $WAN_STATUS -ne 0 ]; then
    echo "❌ WanVideoWrapper install failed."
    exit 1
  fi
  echo "✅ WanVideoWrapper install complete" 


  echo "▶️  Starting ComfyUI"
  # group both the main and fallback commands so they share the same log
  mkdir -p "$NETWORK_VOLUME/${RUNPOD_POD_ID}"
  nohup bash -c "python3 \"$NETWORK_VOLUME\"/ComfyUI/main.py --listen --use-sage-attention --extra-model-paths-config '/ComfyUI-Bot-Wan-Template/extra_model_paths.yaml' 2>&1 | tee \"$NETWORK_VOLUME\"/comfyui_\"$RUNPOD_POD_ID\"_nohup.log" &

  until curl --silent --fail "$URL" --output /dev/null; do
      echo "🔄  Still waiting…"
      sleep 2
  done

  echo "ComfyUI is UP Starting worker"
  nohup bash -c "python3 \"$REPO_DIR\"/worker.py 2>&1 | tee \"$NETWORK_VOLUME\"/\"$RUNPOD_POD_ID\"/worker.log" &

  report_status true "Pod fully initialized and ready for processing"
  echo "Initialization complete! Pod is ready to process jobs."

  # Wait on background jobs forever
  wait

else
  echo "NO FLAG FILE FOUND – starting initial setup"
fi