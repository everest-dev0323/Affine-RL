#!/bin/bash
# TiG Training with OpenRLHF (GRPO)
# Generated from config: config/tig.yaml
#
# Prerequisites:
#   1. Install OpenRLHF: pip install openrlhf[vllm]
#   2. Start Ray cluster: ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

set -x

# Submit job to Ray cluster
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "."}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --pretrain deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
   --advantage_estimator group_norm \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.5 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 8 \
   --max_epochs 2 \
   --prompt_max_len 8192 \
   --generate_max_len 8196 \
   --use_dynamic_batch \
   --packing_samples \
   --gradient_checkpointing \
   --actor_learning_rate 1e-05 \
   --init_kl_coef 0.001 \
   --zero_stage 3 \
   --param_dtype bf16 \
   --lora_rank 16 \
   --lora_alpha 32 \
   --prompt_data checkpoints/tig/dataset.jsonl \
   --input_key prompt \
   --label_key label \
   --apply_chat_template \
   --extra_keys env task extra_data \
   --remote_rm_url scripts/rl/openrlhf/reward_func.py \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0.0 1.0 \
   --save_path ./checkpoints/tig/final \
   --ckpt_path ./checkpoints/tig/ckpt \
   --save_steps 10 \
   --save_hf_ckpt \
   --attn_implementation flash_attention_2 \
   --use_wandb true \
   --wandb_project tig-training \
   --wandb_run_name tig-qwen-8b
