#!/bin/bash

# Define checkpoint variables
CKPT_URL="https://www.dropbox.com/scl/fi/gigq3jln53alg8aalhsdn/v2-1_512-ema-pruned.ckpt?rlkey=absdglgiw5strfkdociwqdcw9&st=hx4tqbyb&dl=1"
CKPT_NAME="v2-1_512-ema-pruned.ckpt"
TARGET_DIR="./cobl/LDM/SD2p1/"
CKPT_PATH="$TARGET_DIR/$CKPT_NAME"

# Ensure the target directory exists
mkdir -p "$TARGET_DIR"

# Check if checkpoint already exists
if [ -f "$CKPT_PATH" ]; then
    echo "Checkpoint already exists at $CKPT_PATH. Skipping download."
else
    echo "Downloading Stable Diffusion checkpoint..."
    curl -L "$CKPT_URL" -o "$CKPT_PATH"

    # Verify
    if [ -f "$CKPT_PATH" ]; then
        echo "Checkpoint successfully downloaded to $CKPT_PATH"
    else
        echo "Download failed!"
        exit 1
    fi
fi

# Install torch
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install xformers --index-url https://download.pytorch.org/whl/cu126


# Install the repo
echo "Installing the current repository..."
pip install -e .
