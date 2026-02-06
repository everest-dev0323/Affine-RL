#!/bin/bash
# GRPO Training with vLLM colocate mode
# In colocate mode, vLLM runs within the training process and weights are synced automatically

# Run training (all GPUs used for training + colocated vLLM)
accelerate launch --num_processes 8 \
    --config_file config/accelerate_deepspeed_zero3.yaml \
    scripts/rl/r1_grpo.py --config config/r1_grpo.yaml