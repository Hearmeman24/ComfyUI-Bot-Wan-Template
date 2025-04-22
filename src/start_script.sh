#!/usr/bin/env bash
set -euo pipefail

# Decide which branch to clone
if [ "${IS_DEV:-false}" = "true" ]; then
  BRANCH="dev"
else
  BRANCH="master"
fi

# Check if directory exists and remove it or update it
if [ -d "ComfyUI-Bot-Wan-Template" ]; then
  echo "📂 Directory already exists. Removing it first..."
  rm -rf ComfyUI-Bot-Wan-Template
fi

echo "📥 Cloning branch '$BRANCH' of ComfyUI-Bot-Wan-Template…"
git clone --branch "$BRANCH" https://github.com/Hearmeman24/ComfyUI-Bot-Wan-Template.git

echo "📂 Moving start.sh into place…"
mv ComfyUI-Bot-Wan-Template/src/start.sh /

echo "▶️ Running start.sh"
bash /start.sh