#!/bin/bash
# Training script for Qwen 30B on OpenSpiel dataset with 2xH100

set -e

# Preprocess data first
echo "========== Preprocessing Data =========="
python scripts/sft/preprocess_openspiel.py \
    --input_dir data/openspiel \
    --output_path data/openspiel_processed/train.jsonl

# Run training with DeepSpeed ZeRO-3
echo "========== Starting Training =========="
deepspeed --num_gpus=2 \
    scripts/sft/train_openspiel.py \
    --deepspeed scripts/sft/ds_config_zero3.json

