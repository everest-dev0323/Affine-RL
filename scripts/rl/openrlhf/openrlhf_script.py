"""
Think in Games (TiG) Trainer using OpenRLHF

This script implements the TiG training algorithm using the OpenRLHF framework
with Ray + vLLM distributed architecture for high-performance GRPO training.

Key features:
- OpenRLHF's Ray + vLLM distributed architecture for scalable training
- Custom reward function for game action matching
- GRPO algorithm (--advantage_estimator group_norm)
- Support for both single-turn and multi-turn agent execution

Usage:
    # Start Ray cluster first
    ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
    
    # Run training
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{"working_dir": "."}' \
        -- python scripts/rl/openrlhf.py --config config/tig.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.utils import load_config_from_yaml


def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    return load_config_from_yaml(path)


def load_dataset(dataset_path: str, max_data_per_task: int = 10000) -> List[Dict]:
    """
    Load TiG dataset from JSONL files organized by env/task.
    
    Returns a list of prompts in the format expected by OpenRLHF.
    """
    dataset = []
    path_obj = Path(dataset_path)
    
    if not path_obj.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    envs = [x for x in path_obj.iterdir() if x.is_dir() and not x.name.startswith(".")]
    
    print(f"\n{'='*60}")
    print("📂 LOADING TiG DATASET")
    print(f"{'='*60}")
    print(f"Dataset path: {dataset_path}")
    print(f"Environments found: {len(envs)}")
    
    for env in envs:
        task_files = list(env.glob("*.jsonl"))
        env_count = 0
        
        for task_file in task_files:
            task_count = 0
            with open(task_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if task_count >= max_data_per_task:
                        break
                        
                    trajectory = json.loads(line)
                    conversation = trajectory.get('conversation', [])

                    # Extract extra data for verifier-based reward computation
                    extra_data = {}
                    if 'info' in trajectory:
                        # lgc format with info containing task metadata
                        extra_data['info'] = trajectory['info']
                    if 'game_data' in trajectory:
                        # lgc-v2 format with game_data
                        extra_data['game_data'] = trajectory['game_data']
                    if 'task_type' in trajectory:
                        extra_data['task_type'] = trajectory['task_type']
                    if 'metadata' in trajectory:
                        extra_data['metadata'] = trajectory['metadata']

                    # Extract each assistant turn as a training sample
                    for i in range(len(conversation)):
                        if conversation[i].get('role') == 'assistant':
                            # Format for OpenRLHF: list of message dicts
                            prompt = conversation[:i]
                            ground_truth = conversation[i].get('content', '')

                            dataset.append({
                                'prompt': prompt,  # List of {role, content} dicts
                                'label': ground_truth,  # Ground truth for reward func
                                'env': env.stem,
                                'task': task_file.stem,
                                'task_id': trajectory.get('task_id', 0),
                                'extra_data': extra_data,  # Additional metadata for verifiers
                            })
                            task_count += 1

                            if task_count >= max_data_per_task:
                                break
            
            env_count += task_count
            print(f"  • {env.stem}/{task_file.stem}: {task_count} samples")
        
        print(f"  Total for {env.stem}: {env_count} samples")
    
    print(f"\n✅ Total samples loaded: {len(dataset)}")
    print(f"{'='*60}\n")
    
    return dataset


def get_reward_func_path() -> str:
    """Get the path to the reward function file.

    The reward function file (reward_func.py) contains detailed reward
    computation for various environments including:
    - lgc-v2: Logic tasks with verifiers (sudoku, game_of_24, boolean_expressions, etc.)
    - lgc: Legacy logic tasks with i3_logic verifiers
    - game/print: Simple equality check environments

    Returns:
        str: Absolute path to the reward function file
    """
    reward_func_path = Path(__file__).parent / "reward_func.py"

    if not reward_func_path.exists():
        raise FileNotFoundError(
            f"Reward function file not found at {reward_func_path}. "
            "Please ensure scripts/rl/openrlhf/reward_func.py exists."
        )

    print(f"✅ Using reward function file: {reward_func_path}")
    return str(reward_func_path.resolve())


def build_openrlhf_command(cfg: dict, dataset_path: str, reward_func_path: str) -> str:
    """Build the OpenRLHF training command."""

    # Base command
    cmd_parts = ["python3 -m openrlhf.cli.train_ppo_ray"]

    # === Model Configuration ===
    cmd_parts.append(f"--pretrain {cfg.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')}")

    # === GRPO Algorithm ===
    # Use group_norm for GRPO advantage estimation
    cmd_parts.append("--advantage_estimator group_norm")

    # === GPU Configuration ===
    num_gpus = cfg.get('num_gpus', 8)
    cmd_parts.extend([
        f"--ref_num_nodes 1",
        f"--ref_num_gpus_per_node {num_gpus}",
        f"--actor_num_nodes 1",
        f"--actor_num_gpus_per_node {num_gpus}",
        # No critic for GRPO
    ])

    # === vLLM Configuration ===
    vllm_engines = cfg.get('vllm_num_engines', 4)
    vllm_tp = cfg.get('vllm_tensor_parallel_size', 1)
    vllm_mem = cfg.get('vllm_gpu_memory_utilization', 0.5)
    cmd_parts.extend([
        f"--vllm_num_engines {vllm_engines}",
        f"--vllm_tensor_parallel_size {vllm_tp}",
        f"--vllm_gpu_memory_utilization {vllm_mem}",
        "--colocate_all_models",  # Hybrid engine mode
        "--vllm_enable_sleep",
        "--deepspeed_enable_sleep",
        "--vllm_sync_backend nccl",
        "--enforce_eager",
    ])

    # === Training Configuration ===
    cmd_parts.extend([
        f"--micro_train_batch_size {cfg.get('micro_train_batch_size', 2)}",
        f"--train_batch_size {cfg.get('train_batch_size', 128)}",
        f"--micro_rollout_batch_size {cfg.get('micro_rollout_batch_size', 8)}",
        f"--rollout_batch_size {cfg.get('rollout_batch_size', 1024)}",
        f"--n_samples_per_prompt {cfg.get('num_generations', 8)}",
        f"--max_epochs {cfg.get('num_epochs', 2)}",
        f"--prompt_max_len {cfg.get('max_seq_length', 8192) // 2}",
        f"--generate_max_len {cfg.get('max_new_tokens', 8196)}",
        "--use_dynamic_batch",
        "--packing_samples",
        "--gradient_checkpointing",
    ])

    # === Learning Rate & Optimization ===
    cmd_parts.extend([
        f"--actor_learning_rate {cfg.get('learning_rate', 1e-6)}",
        f"--init_kl_coef {cfg.get('kl_coef', 0.001)}",
        "--zero_stage 3",
        "--param_dtype bf16",
    ])

    # === LoRA Configuration ===
    if cfg.get('lora_r', 0) > 0:
        cmd_parts.extend([
            f"--lora_rank {cfg.get('lora_r', 16)}",
            f"--lora_alpha {cfg.get('lora_alpha', 32)}",
        ])

    # === Dataset Configuration ===
    cmd_parts.extend([
        f"--prompt_data {dataset_path}",
        "--input_key prompt",
        "--label_key label",
        "--apply_chat_template",
        # Pass extra keys for detailed reward computation
        "--extra_keys env task task_id extra_data",
    ])

    # === Custom Reward Function ===
    cmd_parts.append(f"--remote_rm_url {reward_func_path}")

    # === Dynamic Filtering (DAPO-style) ===
    cmd_parts.extend([
        "--dynamic_filtering",
        "--dynamic_filtering_reward_range 0.0 1.0",
    ])

    # === Checkpointing ===
    checkpoint_dir = cfg.get('checkpoint_dir', './checkpoints/openrlhf')
    cmd_parts.extend([
        f"--save_path {checkpoint_dir}/final",
        f"--ckpt_path {checkpoint_dir}/ckpt",
        f"--save_steps {cfg.get('save_steps', 10)}",
        "--save_hf_ckpt",
    ])

    # === Logging ===
    if cfg.get('use_wandb', False):
        cmd_parts.extend([
            f"--use_wandb {cfg.get('wandb_token', 'true')}",
            f"--wandb_project {cfg.get('wandb_project', 'openrlhf')}",
            f"--wandb_run_name {cfg.get('wandb_run_name', 'grpo')}",
        ])

    return " \\\n   ".join(cmd_parts)


def save_dataset_for_openrlhf(dataset: List[Dict], output_path: str) -> str:
    """Save dataset in format expected by OpenRLHF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Saved dataset to: {output_path}")
    return str(output_path)


def main():
    """Main entry point for TiG training with OpenRLHF."""
    parser = argparse.ArgumentParser(description="TiG Training with OpenRLHF")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--generate-script", type=str, help="Generate shell script to this path")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    print("\n" + "="*60)
    print("🎮 TiG Training with OpenRLHF (GRPO)")
    print("="*60)
    print(f"Model: {cfg.get('model_name')}")
    print(f"Algorithm: GRPO (group_norm)")
    print(f"Generations per prompt: {cfg.get('num_generations', 8)}")
    print(f"KL coefficient: {cfg.get('kl_coef', 0.001)}")
    print("="*60 + "\n")

    # Load and prepare dataset
    dataset = load_dataset(
        cfg.get('dataset_path', 'dataset/openrlhf'),
        cfg.get('max_data_per_task', 100_000)
    )

    # Save dataset for OpenRLHF
    dataset_output = save_dataset_for_openrlhf(
        dataset,
        f"{cfg.get('checkpoint_dir', './checkpoints/openrlhf')}/dataset.jsonl"
    )

    # Get reward function file path
    reward_func_path = get_reward_func_path()

    # Build training command
    cmd = build_openrlhf_command(cfg, dataset_output, reward_func_path)

    if args.generate_script:
        script_content = f'''#!/bin/bash
# TiG Training with OpenRLHF (GRPO)
# Generated from config: {args.config}
#
# Prerequisites:
#   1. Install OpenRLHF: pip install openrlhf[vllm]
#   2. Start Ray cluster: ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

set -x

# Submit job to Ray cluster
ray job submit --address="http://127.0.0.1:8265" \\
   --runtime-env-json='{{"working_dir": "."}}' \\
   -- {cmd}
'''
        with open(args.generate_script, 'w') as f:
            f.write(script_content)
        print(f"\n✅ Generated training script: {args.generate_script}")
        print(f"   Run with: bash {args.generate_script}")
        return

    print("\n📋 OpenRLHF Training Command:")
    print("-" * 60)
    print(cmd)
    print("-" * 60)

    if args.dry_run:
        print("\n⚠️  Dry run mode - command not executed")
        return

    print("\n🚀 To execute, run this command after starting Ray cluster:")
    print(f"   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8")
    print(f"   ray job submit --address='http://127.0.0.1:8265' -- {cmd.split()[0]} ...")


if __name__ == "__main__":
    main()

