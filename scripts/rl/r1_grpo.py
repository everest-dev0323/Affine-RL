"""
GRPO Training Script for Game Tasks (Othello, etc.)

This script implements GRPO (Group Relative Policy Optimization) training
with custom reward functions for game-playing tasks.

Usage:
    accelerate launch --num_processes 7 --config_file config/deepspeed_zero3.yaml \
        scripts/rl/r1_grpo.py --config config/r1_grpo.yaml
"""

import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import Dataset, load_dataset as hf_load_dataset, interleave_datasets
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config

# ============================================================================
# Logging Setup
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


# ============================================================================
# Script Arguments
# ============================================================================
@dataclass
class ScriptArguments:
    """Arguments for the GRPO training script."""

    dataset_path: str = field(
        default="dataset/grpo",
        metadata={"help": "Directory containing dataset files (othello.jsonl, hex.jsonl, etc.)"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use (None for all)"}
    )
    othello_weight: float = field(default=1.0, metadata={"help": "Weight for othello dataset"})
    hex_weight: float = field(default=1.0, metadata={"help": "Weight for hex dataset"})


# ============================================================================
# Reward Functions
# ============================================================================

def parse_response(text: str) -> str:
    """Parse response, removing <think> tags if present."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()


def compute_game_reward(
    parsed_response: str,
    label: str,
    task: str,
    extra_data: dict
) -> float:
    """
    Compute reward for game tasks (othello, etc.).

    For othello: uses probability distribution from MCTS as reward.
    For other games: exact match with label.
    """
    if task == "othello":
        probabilities = extra_data.get("probabilities", {})
        if parsed_response in probabilities:
            return probabilities[parsed_response]
        return 0.0

    # Default: exact match
    return 1.0 if parsed_response.strip() == label.strip() else 0.0


def reward_func(completions, _prompts, labels, extra_data, tasks, **_kwargs):
    """
    Main reward function for GRPO training.

    Args:
        completions: List of model completions
        prompts: List of prompts (unused, for compatibility)
        labels: List of ground truth labels
        extra_data: List of extra data dicts (contains probabilities for othello)
        tasks: List of task names
        **kwargs: Additional arguments

    Returns:
        List of reward scores
    """
    rewards = []

    for completion, label, extra, task in zip(completions, labels, extra_data, tasks):
        try:
            parsed = parse_response(completion)
            reward = compute_game_reward(parsed, label, task, extra)
            rewards.append(reward)

            # Log successful completions occasionally
            if reward > 0.5 and random.random() < 0.05:
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Task: {task}, Label: {label}, Reward: {reward}\n")
                    f.write(f"Response: {parsed}\n")
        except Exception as e:
            logger.warning(f"Error computing reward: {e}")
            rewards.append(0.0)

    return rewards

# ============================================================================
# Dataset Loading
# ============================================================================

def load_datasets(
    dataset_path: str,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load and merge datasets from directory using HuggingFace datasets.

    Loads othello.jsonl and hex.jsonl from dataset_path directory
    and merges them with specified weights.

    Args:
        dataset_path: Directory containing othello.jsonl and hex.jsonl
        max_samples: Maximum total samples (None for all)

    Returns:
        Merged Dataset
    """
    othello_weight = 1.0
    hex_weight = 1.0
    
    datasets_list = []
    weights = []

    # Load othello dataset
    othello_path = os.path.join(dataset_path, "othello.jsonl")
    if os.path.exists(othello_path):
        othello_ds = hf_load_dataset("json", data_files=othello_path, split="train")
        datasets_list.append(othello_ds)
        weights.append(othello_weight)
        logger.info(f"Loaded othello dataset: {len(othello_ds)} samples (weight={othello_weight})")

    # Load hex dataset
    hex_path = os.path.join(dataset_path, "hex.jsonl")
    if os.path.exists(hex_path):
        hex_ds = hf_load_dataset("json", data_files=hex_path, split="train")
        datasets_list.append(hex_ds)
        weights.append(hex_weight)
        logger.info(f"Loaded hex dataset: {len(hex_ds)} samples (weight={hex_weight})")

    if not datasets_list:
        raise ValueError(f"No datasets found in {dataset_path}")

    # Merge datasets with weights
    if len(datasets_list) == 1:
        merged = datasets_list[0]
    else:
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        merged = interleave_datasets(datasets_list, probabilities=probs, seed=42)
        logger.info(f"Merged datasets with probabilities: {probs}")

    # Limit samples if specified
    if max_samples is not None:
        merged = merged.shuffle(seed=42).select(range(min(max_samples, len(merged))))
        logger.info(f"Limited to {len(merged)} samples")

    return merged


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def get_checkpoint(training_args: GRPOConfig):
    """Get last checkpoint if exists."""
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


# ============================================================================
# Main Training Function
# ============================================================================

def grpo_function(
    model_args: ModelConfig,
    script_args: ScriptArguments,
    training_args: GRPOConfig
):
    """Main GRPO training function."""

    # Log parameters
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")

    # -------------------------
    # Load dataset (pre-formatted)
    # -------------------------
    logger.info(f"Loading datasets from: {script_args.dataset_path}")

    train_dataset = load_datasets(
        dataset_path=script_args.dataset_path,
        max_samples=script_args.max_samples,
    )

    logger.info(f"Train samples: {len(train_dataset)}")

    # -------------------------
    # Initialize trainer
    # -------------------------
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
    )

    # -------------------------
    # Training loop
    # -------------------------
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ***'
    )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # -------------------------
    # Save model
    # -------------------------
    logger.info("*** Saving model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    # Create model card on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "game"]})

    logger.info("*** Done! ***")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()