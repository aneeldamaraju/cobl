#!/bin/bash

# Download the SD base checkpoint used to initialize our model
CKPT_URL="https://www.dropbox.com/scl/fi/gigq3jln53alg8aalhsdn/v2-1_512-ema-pruned.ckpt?rlkey=absdglgiw5strfkdociwqdcw9&st=hx4tqbyb&dl=1"
CKPT_NAME="v2-1_512-ema-pruned.ckpt"
TARGET_DIR="./cobl/LDM/SD2p1/"
CKPT_PATH="$TARGET_DIR/$CKPT_NAME"
mkdir -p "$TARGET_DIR"
if [ -f "$CKPT_PATH" ]; then
    echo "Checkpoint already exists at $CKPT_PATH. Skipping download."
else
    echo "Downloading Stable Diffusion checkpoint..."
    curl -L "$CKPT_URL" -o "$CKPT_PATH"
    if [ -f "$CKPT_PATH" ]; then
        echo "Checkpoint successfully downloaded to $CKPT_PATH"
    else
        echo "Download failed!"
        exit 1
    fi
fi

# Download the Cobl parameters
CKPT_URL="https://www.dropbox.com/scl/fi/cirx1dh4hst6ar3o2w33h/cobl_added_params.ckpt?rlkey=2p9lqvqz9ae0djg9uthtari3p&st=sq05l8kn&dl=1"
CKPT_NAME="cobl_added_params.ckpt"
TARGET_DIR="./cobl/model/"
CKPT_PATH="$TARGET_DIR/$CKPT_NAME"
mkdir -p "$TARGET_DIR"
if [ -f "$CKPT_PATH" ]; then
    echo "Checkpoint already exists at $CKPT_PATH. Skipping download."
else
    echo "Downloading Cobl checkpoint..."
    curl -L "$CKPT_URL" -o "$CKPT_PATH"
    if [ -f "$CKPT_PATH" ]; then
        echo "Checkpoint successfully downloaded to $CKPT_PATH"
    else
        echo "Download failed!"
        exit 1
    fi
fi

# Download the U2Net checkpoint
CKPT_URL="https://www.dropbox.com/scl/fi/obpwapfgjl6gwmmwpc1le/u2net.pth?rlkey=0q0dqx17fk9yve3ve98ws9zy9&st=z6uuzf1c&dl=1"
CKPT_NAME="u2net.pth"
TARGET_DIR="./cobl/U2Net/"
CKPT_PATH="$TARGET_DIR/$CKPT_NAME"
mkdir -p "$TARGET_DIR"
if [ -f "$CKPT_PATH" ]; then
    echo "Checkpoint already exists at $CKPT_PATH. Skipping download."
else
    echo "Downloading U2Net checkpoint..."
    curl -L "$CKPT_URL" -o "$CKPT_PATH"
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
