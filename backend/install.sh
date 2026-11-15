#!/usr/bin/env bash
set -e
echo "Installing FFmpeg and Python dependencies for HuntSense (Linux + AMD ROCm)..."
sudo apt update && sudo apt install -y ffmpeg python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# Install PyTorch with AMD ROCm support
pip install torch torchvision torchaudio --index-url https://pytorch.org
pip install -r backend/requirements.txt
echo "Setup complete! Add your keys to backend/.env and run: source venv/bin/activate && python backend/scanner.py"
