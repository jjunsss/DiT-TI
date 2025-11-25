#!/usr/bin/env bash
set -e

########################################
# DiT-TI Environment Setup Script
# Creates a conda environment for FLUX Textual Inversion training
########################################

ENV_NAME="DiT-TI"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "DiT-TI Environment Setup"
echo "=========================================="
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Warning: Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Aborting setup."
        exit 0
    fi
fi

echo ""
echo "Step 1/4: Creating conda environment..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo ""
echo "Step 2/4: Installing PyTorch (CUDA 11.8)..."
conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 3/4: Installing diffusers from source..."
conda run -n $ENV_NAME pip install git+https://github.com/huggingface/diffusers.git

echo ""
echo "Step 4/4: Installing other dependencies..."
conda run -n $ENV_NAME pip install transformers accelerate safetensors sentencepiece protobuf peft

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "    conda activate $ENV_NAME"
echo ""
echo "To start training, run:"
echo "    bash run_ti_fscil.sh [START] [END] [GPUS] [CLASSES_PER_GPU]"
echo ""
echo "Example:"
echo "    bash run_ti_fscil.sh 65 104 \"0,1,2,3\" 10"
echo ""
echo "=========================================="
