#!/bin/bash

# Vast.ai Deployment Script
# Ejecuta este script en la instancia de vast.ai

set -e  # Exit on error

echo "=========================================="
echo "ML BLACKJACK - VAST.AI DEPLOYMENT"
echo "=========================================="

# Update system
echo "[1/6] Updating system..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv git

# Create virtual environment
echo "[2/6] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "[3/6] Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo "[4/6] Installing Python dependencies..."
pip install -q torch gymnasium numpy tensorboard matplotlib seaborn pandas tqdm

# Verificar GPU
echo "[5/6] Checking GPU..."
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Start training
echo "[6/6] Starting training..."
echo "=========================================="
echo ""

python3 train_vast.py

echo ""
echo "=========================================="
echo "TRAINING COMPLETED!"
echo "=========================================="
