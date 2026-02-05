#!/bin/bash
# Start vLLM server in background
CUDA_VISIBLE_DEVICES=7 trl vllm-serve --model Qwen/Qwen2.5-3B-Instruct --gpu_memory_utilization 0.9 &
VLLM_PID=$!

# Wait for server to be ready
sleep 60

# Run training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes 7 \
    --config_file config/deepspeed_zero3.yaml \
    scripts/rl/r1_grpo.py --config config/r1_grpo.yaml

# Cleanup
kill $VLLM_PID