"""
GRPO Training Script for Game Tasks (Othello, etc.)

This script implements GRPO (Group Relative Policy Optimization) training
with custom reward functions for game-playing tasks.

Supports:
- lgc-v2: Logic tasks with verifiers (sudoku, game_of_24, boolean_expressions, etc.)
- game/openspiel: Custom reward functions for game environments
- print: Simple equality check environments

Usage:
    accelerate launch --num_processes 7 --config_file config/deepspeed_zero3.yaml \
        scripts/rl/r1_grpo.py --config config/r1_grpo.yaml
"""

import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, load_dataset as hf_load_dataset, interleave_datasets
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add lgc-v2 environment to path for verifiers
lgc_v2_path = project_root / "AffineFoundation" / "affinetes" / "environments" / "primeintellect" / "lgc-v2"
sys.path.insert(0, str(lgc_v2_path))

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
        default="dataset/Upload",
        metadata={"help": "Directory containing GAME/, LGC/, PRINT/ subdirectories with parquet files"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use (None for all)"}
    )


# ============================================================================
# Reward Functions
# ============================================================================

def parse_response(text: str) -> str:
    """Parse response, removing <think> tags if present."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()


# ============================================================================
# Sudoku Reward Functions
# ============================================================================

def extract_sudoku_answer(response: str) -> str:
    """Extract sudoku grid from model response."""
    lines = response.strip().split('\n')
    grid_lines = []
    for line in lines:
        clean_line = line.replace('`', '').strip()
        digits = re.findall(r'[1-9]', clean_line)
        if len(digits) == 9:
            grid_lines.append(''.join(digits))

    if len(grid_lines) >= 9:
        return '\n'.join(grid_lines[:9])

    # Fallback: try code block
    pattern = r"```([\s\S]*?)```"
    matches = re.findall(pattern, response)
    if matches:
        grid_str = matches[-1].strip()
        if grid_str and '\n' in grid_str:
            first_line = grid_str.split('\n')[0]
            if not re.search(r'[1-9]', first_line):
                grid_str = '\n'.join(grid_str.split('\n')[1:])
        return grid_str

    return response.strip()


def parse_sudoku_grid(grid_str: str) -> Optional[List[List[int]]]:
    """Parse grid string into 9x9 list of integers."""
    if not grid_str:
        return None

    lines = grid_str.strip().split('\n')
    grid = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        row = []
        for char in line:
            if char.isdigit():
                row.append(int(char))
            elif char == '.':
                row.append(0)

        if len(row) == 9:
            grid.append(row)

    if len(grid) != 9:
        return None

    return grid


def sudoku_reward(answer_grid: List[List[int]], puzzle_grid: List[List[int]]) -> float:
    """
    Calculate granular reward for sudoku answer.

    Returns:
        float: Reward between 0 and 1
            - 0 if doesn't respect original puzzle
            - Otherwise based on row/col/box validity
    """
    # Check if answer respects original puzzle
    for i in range(9):
        for j in range(9):
            if puzzle_grid[i][j] != 0:
                if answer_grid[i][j] != puzzle_grid[i][j]:
                    return 0.0

    def is_valid_group(group: List[int]) -> bool:
        return sorted(group) == list(range(1, 10))

    # Count valid rows
    valid_rows = sum(1 for row in answer_grid if is_valid_group(row))
    row_score = valid_rows / 9

    # Count valid columns
    valid_cols = 0
    for j in range(9):
        col = [answer_grid[i][j] for i in range(9)]
        if is_valid_group(col):
            valid_cols += 1
    col_score = valid_cols / 9

    # Count valid 3x3 boxes
    valid_boxes = 0
    for box_row in range(3):
        for box_col in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(answer_grid[box_row * 3 + i][box_col * 3 + j])
            if is_valid_group(box):
                valid_boxes += 1
    box_score = valid_boxes / 9

    # Weighted average: boxes slightly more important
    return (3 * row_score + 3 * col_score + 4 * box_score) / 10


# ============================================================================
# Verifier-based Reward Functions (lgc-v2)
# ============================================================================

def compute_lgc_v2_reward(
    parsed_response: str,
    task_type: str,
    game_data: dict,
    label: str
) -> float:
    """
    Compute reward for lgc-v2 environment tasks using verifiers.

    Args:
        parsed_response: Parsed model response (without <think> tags)
        task_type: Type of task (sudoku, game_of_24, boolean_expressions, etc.)
        game_data: Game data dict containing metadata
        label: Expected answer

    Returns:
        float: Reward score (0.0 to 1.0)
    """
    try:
        # Import Data class and verifiers
        from base.data import Data
        from games.verifiers import verifier_classes

        # Create Data object from game_data
        data = Data.from_json(game_data) if isinstance(game_data, dict) else Data.from_json_str(game_data)

        # Get verifier for this task type
        verifier_cls = verifier_classes.get(task_type)
        if verifier_cls is None:
            # Fallback to exact match if no verifier
            return 1.0 if parsed_response.strip() == label.strip() else 0.0

        verifier = verifier_cls()

        # Special handling for sudoku with granular rewards
        if task_type == "sudoku":
            # First check perfect solution
            if verifier.verify(data, parsed_response):
                return 1.0

            # Calculate granular reward for partial solutions
            answer_str = extract_sudoku_answer(parsed_response)
            answer_grid = parse_sudoku_grid(answer_str)
            puzzle_str = data.metadata.get("puzzle", "")
            puzzle_grid = parse_sudoku_grid(puzzle_str)

            if answer_grid is None or puzzle_grid is None:
                return 0.0

            return sudoku_reward(answer_grid, puzzle_grid)

        # Standard verification for other tasks
        return 1.0 if verifier.verify(data, parsed_response) else 0.0

    except ImportError as e:
        # Verifier not available, fallback to exact match
        logger.warning(f"Could not import verifier for {task_type}: {e}")
        return 1.0 if parsed_response.strip() == label.strip() else 0.0
    except Exception as e:
        # Error during verification
        logger.warning(f"Error during verification for {task_type}: {e}")
        return 0.0


# ============================================================================
# Game/Openspiel Reward Functions
# ============================================================================

def compute_game_reward(
    parsed_response: str,
    label: str,
    task: str,
    extra_data: dict
) -> float:
    """
    Compute reward for openspiel environment tasks.

    Args:
        parsed_response: Parsed model response
        label: Expected answer/action
        task: Task name
        extra_data: Additional data (game_data, metadata, etc.)

    Returns:
        float: Reward score (0.0 to 1.0)
    """
    if task == "othello" or task == "hex":
        probabilities = extra_data.get("probabilities", {})
        if parsed_response in probabilities:
            return probabilities[parsed_response] / max(probabilities.values())
        return 0.0

    # Default: exact match
    return 1.0 if parsed_response.strip() == label.strip() else 0.0


# ============================================================================
# Main Reward Computation
# ============================================================================

def compute_reward(
    response: str,
    label: str,
    env: str = "",
    task: str = "",
    extra_data: Optional[dict] = None
) -> float:
    """
    Compute reward for a single response based on environment and task type.

    Args:
        response: Model's full response
        label: Expected answer/action
        env: Environment name (lgc-v2, game, print, etc.)
        task: Task name within environment
        extra_data: Additional data (game_data, metadata, etc.)

    Returns:
        float: Reward score between 0.0 and 1.0
    """
    parsed_response = parse_response(response)
    env_lower = env.lower() if env else ""
    extra_data = extra_data or {}

    # lgc-v2 environment with verifiers
    if env_lower in ("lgc-v2", "lgc_v2", "logic-v2"):
        task_type = extra_data.get("task_type", task)
        game_data = extra_data.get("game_data", {})
        return compute_lgc_v2_reward(parsed_response, task_type, game_data, label)

    # openspiel environment
    if env_lower in ("game", "openspiel"):
        return compute_game_reward(parsed_response, label, task, extra_data)

    # Default: simple equality check for print and other environments
    return 1.0 if parsed_response.strip() == label.strip() else 0.0


def reward_func(completions, prompts, labels, **kwargs):
    """
    Main reward function for GRPO training (OpenRLHF-style).

    This function computes rewards and returns a dict with rewards tensor
    and extra logging metrics, matching the OpenRLHF interface.

    Args:
        completions: List of model completions (full text: prompt + response)
        prompts: List of original prompts
        labels: List of ground truth labels
        **kwargs: Additional arguments including:
            - envs: List[str] - Environment names (lgc-v2, game, print, etc.)
            - tasks: List[str] - Task names
            - extra_data: List[dict] - Additional metadata

    Returns:
        dict with:
            - rewards: torch.Tensor of reward scores
            - scores: torch.Tensor of scores (same as rewards, for dynamic filtering)
            - extra_logs: dict with logging metrics
    """
    batch_size = len(completions)
    rewards = []

    # Extract extra data from kwargs
    envs = kwargs.get("envs", [""] * batch_size)
    tasks = kwargs.get("tasks", [""] * batch_size)
    extra_data_list = kwargs.get("extra_data", [{}] * batch_size)

    # Ensure lists match batch size
    if len(envs) != batch_size:
        envs = [envs[0] if envs else ""] * batch_size
    if len(tasks) != batch_size:
        tasks = [tasks[0] if tasks else ""] * batch_size
    if len(extra_data_list) != batch_size:
        extra_data_list = [extra_data_list[0] if extra_data_list else {}] * batch_size

    # Track rewards by (env, task) for logging
    env_task_rewards = {}

    for i in range(batch_size):
        try:
            # Extract response (remove prompt from full completion if present)
            completion = completions[i]
            prompt = prompts[i] if prompts else ""
            if prompt and completion.startswith(prompt):
                response = completion[len(prompt):].strip()
            else:
                response = completion

            label = labels[i] if labels else ""
            env = envs[i] if isinstance(envs[i], str) else str(envs[i])
            task = tasks[i] if isinstance(tasks[i], str) else str(tasks[i])
            extra = extra_data_list[i] if isinstance(extra_data_list[i], dict) else {}

            # Compute reward using the unified compute_reward function
            reward = compute_reward(
                response=response,
                label=label,
                env=env,
                task=task,
                extra_data=extra
            )
            rewards.append(reward)

            # Track by (env, task)
            key = (env, task)
            if key not in env_task_rewards:
                env_task_rewards[key] = []
            env_task_rewards[key].append(reward)

            # Log successful completions occasionally
            if reward > 0.5 and random.random() < 0.05:
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Env: {env}, Task: {task}, Label: {label}, Reward: {reward}\n")
                    f.write(f"Response: {parse_response(response)}\n")

        except Exception as e:
            logger.warning(f"Error computing reward: {e}")
            rewards.append(0.0)

    # Convert to tensor
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    # Build extra logs
    extra_logs = {
        "mean_reward": reward_tensor.mean().item(),
        "reward_std": reward_tensor.std().item() if len(rewards) > 1 else 0.0,
        "num_correct": sum(1 for r in rewards if r == 1.0),
        "accuracy": sum(rewards) / len(rewards) if rewards else 0.0,
    }

    # Add per (env, task) stats
    for (env_name, task_name), reward_list in env_task_rewards.items():
        # Create key: "env/task" or just "env" or "task" if one is empty
        if env_name and task_name:
            key = f"{env_name}/{task_name}"
        elif env_name:
            key = env_name
        elif task_name:
            key = task_name
        else:
            key = "unknown"

        extra_logs[f"{key}_accuracy"] = sum(reward_list) / len(reward_list)
        extra_logs[f"{key}_count"] = len(reward_list)

    return {
        "rewards": reward_tensor,  # Used for advantage calculation
        "scores": reward_tensor,   # Used for dynamic filtering (DAPO)
        "extra_logs": extra_logs,
    }

# ============================================================================
# Dataset Loading
# ============================================================================

# Dataset weights configuration (placeholder values - update as needed)
DATASET_WEIGHTS = {
    # GAME datasets
    "GAME/gin_rummy": 0.05,
    "GAME/goofspiel": 0.05,
    "GAME/leduc_poker": 0.1,
    "GAME/liars_dice_1": 0.1,
    "GAME/liars_dice_2": 0.03,
    "GAME/othello": 1.5,
    "GAME/hex_5": 0.3,
    "GAME/hex_7": 0.35,
    "GAME/hex_9": 0.4,
    "GAME/hex_11": 0.45,
    # LGC datasets
    "LGC/boolean_expressions": 0.1,
    "LGC/cryptarithm": 0.1,
    "LGC/dyck_language": 0.1,
    "LGC/dyck_language2": 0.15,
    "LGC/dyck_language_reasoning_errors": 0.1,
    "LGC/game_of_24": 0.1,
    "LGC/operation": 0.1,
    "LGC/sudoku": 0.3,
    # PRINT datasets
    "PRINT/print": 0.3,
}


def load_datasets(
    dataset_path: str,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load and merge all parquet datasets from dataset/Upload directory.

    Loads all parquet files from GAME/, LGC/, and PRINT/ subdirectories
    and merges them with specified weights.

    Args:
        dataset_path: Directory containing GAME/, LGC/, PRINT/ subdirectories
        max_samples: Maximum total samples (None for all)

    Returns:
        Merged Dataset
    """
    datasets_list = []
    weights = []
    dataset_names = []

    # Iterate through all configured datasets
    for dataset_key, weight in DATASET_WEIGHTS.items():
        parquet_path = os.path.join(dataset_path, f"{dataset_key}.parquet")

        if os.path.exists(parquet_path):
            ds = hf_load_dataset("parquet", data_files=parquet_path, split="train")
            datasets_list.append(ds)
            weights.append(weight)
            dataset_names.append(dataset_key)
            logger.info(f"Loaded {dataset_key}: {len(ds)} samples (weight={weight})")
        else:
            logger.warning(f"Dataset not found: {parquet_path}")

    if not datasets_list:
        raise ValueError(f"No datasets found in {dataset_path}")

    logger.info(f"Total datasets loaded: {len(datasets_list)}")

    # Merge datasets with weights
    if len(datasets_list) == 1:
        merged = datasets_list[0]
    else:
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        merged = interleave_datasets(datasets_list, probabilities=probs, seed=42)
        logger.info(f"Merged {len(datasets_list)} datasets with probabilities:")
        for name, prob in zip(dataset_names, probs):
            logger.info(f"  {name}: {prob:.4f}")

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