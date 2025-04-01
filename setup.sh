#!/bin/bash
set -e

echo "Setting up DOT (Dense Optical Tracking)"
echo "========================================"

# Create directories
mkdir -p checkpoints
mkdir -p datasets

# Install dependencies
pip install -r requirements.txt

# Note about PyTorch3D
echo "Note: PyTorch3D modules not installed (optional, code works without them)"

# Download model checkpoints
echo "Downloading model checkpoints..."
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker_patch_4_wind_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth
wget -O checkpoints/movi_f_cotracker3_wind_60.pth https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_tapir.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_plus_bootstapir.pth

# Download demo data
echo "Downloading demo data..."
wget -P datasets https://huggingface.co/16lemoing/dot/resolve/main/demo.zip
unzip datasets/demo.zip -d datasets/

echo "Setup complete! You can now run the demo with:"
echo "python demo.py --visualization_modes spaghetti_last_static --video_path orange.mp4"
