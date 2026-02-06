#!/bin/bash
# GRPO Training with vLLM colocate mode + DDP
# In colocate mode, vLLM runs within the training process and weights are synced automatically
# DDP is used for distributed training (better for models that fit on single GPU)

export WANDB_PROJECT="affine-rl"

# Run training (all GPUs used for training + colocated vLLM)
accelerate launch --num_processes 8 \
    --config_file config/accelerate_ddp.yaml \
    scripts/rl/r1_grpo.py --config config/r1_grpo.yaml