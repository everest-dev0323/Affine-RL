"""
TiG Trainer using TRL's GRPOTrainer with vLLM colocate mode.

Single-GPU:
    python tig_trainer_vllm.py --config config/tig.yaml

Multi-GPU (with DeepSpeed ZeRO-3):
    accelerate launch --config_file config/accelerate_deepspeed_zero3.yaml \
        scripts/rl/tig_trainer_vllm.py --config config/tig.yaml

Features:
- vLLM-accelerated inference for fast generation
- Per-task reward tracking with wandb logging
- TRL's GRPOTrainer for stable training
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from src.evaluation import reward_response


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


class PerTaskRewardTracker:
    """
    Tracks rewards per (env, task) combination.

    This allows monitoring training progress on individual tasks,
    similar to the tracking in tig_trainer_v2.py.
    """

    def __init__(self):
        # {(env, task): [rewards]}
        self.rewards: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    def add_reward(self, env: str, task: str, reward: float):
        """Add a reward for a specific env/task combination."""
        self.rewards[(env, task)].append(reward)

    def add_rewards_batch(self, envs: List[str], tasks: List[str], rewards: List[float]):
        """Add rewards for a batch of samples."""
        for env, task, reward in zip(envs, tasks, rewards):
            self.add_reward(env, task, reward)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked tasks."""
        stats = {}
        for (env, task), reward_list in self.rewards.items():
            if reward_list:
                stats[f"{env}/{task}"] = {
                    "latest": reward_list[-1],
                    "avg": sum(reward_list) / len(reward_list),
                    "count": len(reward_list),
                }
        return stats

    def clear(self):
        """Clear all tracked rewards."""
        self.rewards.clear()


class PerTaskRewardCallback(TrainerCallback):
    """
    Callback to log per-task rewards to wandb and CLI.

    Logs reward/{env}/{task} and reward/{env}/{task}_avg for each
    tracked env/task combination.
    """

    def __init__(self, tracker: PerTaskRewardTracker, use_wandb: bool = True):
        self.tracker = tracker
        self.use_wandb = use_wandb
        self._last_logged_step = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log per-task rewards when trainer logs."""
        # Avoid duplicate logging for the same step
        if state.global_step == self._last_logged_step:
            return
        self._last_logged_step = state.global_step

        stats = self.tracker.get_stats()
        if not stats:
            return

        # Build log dict for wandb
        task_logs = {}
        cli_lines = []

        for task_key, task_stats in stats.items():
            task_logs[f"reward/{task_key}"] = task_stats["latest"]
            cli_lines.append(
                f"    {task_key}: {task_stats['latest']:.3f} (n={task_stats['count']})"
            )

        # Print to CLI
        if cli_lines:
            print(f"\n  Per-task rewards (step {state.global_step}):")
            for line in sorted(cli_lines):
                print(line)

        # Log to wandb
        if self.use_wandb and task_logs:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(task_logs, step=state.global_step)
            except ImportError:
                pass


def load_dataset_from_path(dataset_path: str, max_per_task: int = 10000) -> Dataset:
    """
    Load dataset from JSONL files into HuggingFace Dataset format.

    Returns dataset with columns: prompt, ground_truth, env, task, task_id
    """
    items = []
    path_obj = Path(dataset_path)
    task_counts = defaultdict(int)

    for env_dir in path_obj.iterdir():
        if not env_dir.is_dir() or env_dir.name.startswith("."):
            continue

        for task_file in env_dir.glob("*.jsonl"):
            count = 0
            with open(task_file) as f:
                for line in f:
                    if not line.strip() or count >= max_per_task:
                        continue
                    traj = json.loads(line)
                    conv = traj.get("conversation", [])
                    for i, msg in enumerate(conv):
                        if msg.get("role") == "assistant":
                            items.append({
                                "prompt": conv[:i],
                                "ground_truth": msg.get("content", ""),
                                "env": env_dir.stem,
                                "task": task_file.stem,  # Task name for tracking
                                "task_id": traj.get("task_id", 0),
                            })
                            count += 1

            task_counts[(env_dir.stem, task_file.stem)] = count

    # Print dataset summary
    print("\n" + "=" * 60)
    print("📂 DATASET LOADED")
    print("=" * 60)
    print(f"Path: {dataset_path}")
    print(f"Total samples: {len(items)}")
    print(f"Tasks: {len(task_counts)}")
    for (env, task), count in sorted(task_counts.items()):
        print(f"  • {env}/{task}: {count} samples")
    print("=" * 60 + "\n")

    return Dataset.from_list(items)


def make_reward_func(loop: asyncio.AbstractEventLoop, tracker: PerTaskRewardTracker):
    """
    Create reward function compatible with TRL's GRPOTrainer.

    Also tracks rewards per (env, task) for monitoring.
    """

    def reward_func(completions, env, task, task_id, ground_truth, **kwargs):
        async def compute():
            tasks = [
                reward_response(e, tid, c, gt)
                for c, e, tid, gt in zip(completions, env, task_id, ground_truth)
            ]
            return await asyncio.gather(*tasks)

        rewards = list(loop.run_until_complete(compute()))

        # Track per-task rewards
        tracker.add_rewards_batch(env, task, rewards)

        return rewards

    return reward_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_wandb = cfg.get("use_wandb", False)

    # Initialize wandb early if enabled
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "tig-grpo"),
                name=cfg.get("wandb_run_name", "tig-grpo"),
                config=cfg,
            )
            print(f"WandB initialized: {wandb.run.name}")
        except ImportError:
            print("Warning: wandb not installed, disabling wandb logging")
            use_wandb = False

    # Load model with Unsloth + vLLM fast inference
    print(f"\nLoading model with vLLM fast inference: {cfg['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg.get("max_seq_length", 8192),
        load_in_4bit=cfg.get("load_in_4bit", False),
        fast_inference=True,
        gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.5),
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    dataset = load_dataset_from_path(
        cfg["dataset_path"],
        cfg.get("max_data_per_task", 10000),
    )

    # Create async event loop for reward computation
    loop = asyncio.new_event_loop()

    # Create per-task reward tracker
    reward_tracker = PerTaskRewardTracker()

    # Training config with vLLM colocate mode (multi-GPU supported)
    training_args = GRPOConfig(
        output_dir=cfg.get("checkpoint_dir", "outputs/tig"),
        num_train_epochs=cfg.get("num_epochs", 1),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 1e-5),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        # GRPO params
        num_generations=cfg.get("num_generations", 4),
        max_completion_length=cfg.get("max_completion_length", 8192),
        temperature=cfg.get("temperature", 1.0),
        beta=cfg.get("kl_coef", 0.0),
        # vLLM colocate mode with multi-GPU support
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.5),
        vllm_tensor_parallel_size=cfg.get("vllm_tensor_parallel_size", 1),
        # Misc
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        report_to="wandb" if use_wandb else "none",
        run_name=cfg.get("wandb_run_name", "tig-grpo"),
        bf16=True,
        remove_unused_columns=False,
    )

    # Print training config summary
    print("\n" + "=" * 60)
    print("🚀 TRAINING CONFIG")
    print("=" * 60)
    print(f"Model: {cfg['model_name']}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Num generations: {training_args.num_generations}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"vLLM mode: {training_args.vllm_mode}")
    print(f"vLLM GPU memory: {training_args.vllm_gpu_memory_utilization}")
    print("=" * 60 + "\n")

    # Create trainer with per-task reward callback
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=make_reward_func(loop, reward_tracker),
        callbacks=[PerTaskRewardCallback(reward_tracker, use_wandb=use_wandb)],
    )

    # Train
    print("\n🏋️ Starting training...")
    trainer.train()

    # Save final model
    final_path = cfg.get("checkpoint_dir", "outputs/tig") + "/final"
    print(f"\n💾 Saving final model to {final_path}")
    trainer.save_model(final_path)

    # Print final per-task statistics
    print("\n" + "=" * 60)
    print("📊 FINAL PER-TASK REWARD STATISTICS")
    print("=" * 60)
    stats = reward_tracker.get_stats()
    for task_key in sorted(stats.keys()):
        task_stats = stats[task_key]
        print(f"  {task_key}: avg={task_stats['avg']:.3f}, n={task_stats['count']}")
    print("=" * 60)

    # Cleanup
    loop.close()
    if use_wandb:
        try:
            import wandb
            wandb.finish()
            print("\n✅ WandB run finished")
        except ImportError:
            pass

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
