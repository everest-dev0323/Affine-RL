#!/bin/bash
# ==============================================================================
# Kimi K2.5 SFT Training Script with NeMo Automodel
# Hardware: 8x H100 80GB GPUs
# ==============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/nemo_kimi_k25_sft.yaml"

# Number of GPUs
NUM_GPUS=${NUM_GPUS:-8}

# Environment variables for optimal performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Enable TF32 for better performance on H100
export NVIDIA_TF32_OVERRIDE=1

# Flash Attention environment
export FLASH_ATTENTION_DETERMINISTIC=0

echo "=============================================="
echo "Kimi K2.5 SFT Training with NeMo Automodel"
echo "=============================================="
echo "Config: ${CONFIG_FILE}"
echo "GPUs: ${NUM_GPUS}"
echo "Project Root: ${PROJECT_ROOT}"
echo "=============================================="

# Check if NeMo Automodel is installed
if ! python -c "import nemo_automodel" 2>/dev/null; then
    echo "ERROR: NeMo Automodel is not installed."
    echo "Please install it with:"
    echo "  pip install nemo-automodel"
    echo "Or from source:"
    echo "  git clone https://github.com/NVIDIA-NeMo/Automodel.git"
    echo "  cd Automodel && pip install -e ."
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Run training with torchrun
echo ""
echo "Starting training..."
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    -m nemo_automodel.training.train \
    --config "${CONFIG_FILE}"

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="

