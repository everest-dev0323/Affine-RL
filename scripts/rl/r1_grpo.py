"""
GRPO Training Script for Game Tasks (Othello, etc.)

This script implements GRPO (Group Relative Policy Optimization) training
with custom reward functions for game-playing tasks.

Supports:
- lgc-v2: Logic tasks with verifiers (sudoku, game_of_24, boolean_expressions, etc.)
- game/openspiel: Custom reward functions for game environments
- print: Simple equality check environments

Usage:
    accelerate launch --num_processes 2 --config_file config/accelerate_ddp.yaml \
        scripts/rl/r1_grpo.py --config config/r1_grpo.yaml --debug_mode
"""

import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import Dataset, load_dataset as hf_load_dataset, interleave_datasets
from transformers import TrainerCallback
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
# Per-Task Reward Tracking
# ============================================================================

class PerTaskRewardTracker:
    """
    Tracks rewards and completion lengths per (env, task) combination for wandb logging.
    Uses cumulative sample count as x-axis for each task.
    """

    def __init__(self):
        # {(env, task): [rewards]} - stores rewards since last log
        self.rewards: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        # {(env, task): [lengths]} - stores completion lengths since last log
        self.lengths: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        # {(env, task): cumulative_count} - total samples seen per task (x-axis)
        self.cumulative_samples: Dict[Tuple[str, str], int] = defaultdict(int)
        # {(env, task): sum_rewards} - for running average
        self.sum_rewards: Dict[Tuple[str, str], float] = defaultdict(float)
        # {(env, task): sum_lengths} - for running average
        self.sum_lengths: Dict[Tuple[str, str], int] = defaultdict(int)
        # Track which metrics have been defined in wandb
        self._defined_metrics: set = set()
        self._wandb = None
        # Try to import wandb
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            pass
        # {(env, task): last_sample_details} - stores last sample for debug logging
        self.last_samples: Dict[Tuple[str, str], Dict] = {}

    def _get_task_key(self, env: str, task: str) -> str:
        """Get task key (without env prefix)."""
        if task:
            return task
        elif env:
            return env
        return "unknown"

    def _define_wandb_metric(self, task_key: str):
        """Define custom x-axis metric in wandb for a task."""
        if self._wandb is None or self._wandb.run is None:
            return
        if task_key in self._defined_metrics:
            return

        # Define metrics with custom x-axis (samples count)
        # Format: {task_key}/samples, {task_key}/reward, {task_key}/length
        # This groups metrics by task in wandb (e.g., game/hex/reward, game/hex/length)
        samples_metric = f"{task_key}/samples"
        self._wandb.define_metric(samples_metric)
        self._wandb.define_metric(f"{task_key}/reward", step_metric=samples_metric)
        self._wandb.define_metric(f"{task_key}/reward_avg", step_metric=samples_metric)
        self._wandb.define_metric(f"{task_key}/length", step_metric=samples_metric)
        self._wandb.define_metric(f"{task_key}/length_avg", step_metric=samples_metric)
        self._defined_metrics.add(task_key)
        logger.info(f"Defined wandb metrics for {task_key} with x-axis: {samples_metric}")

    def add_reward(
        self,
        env: str,
        task: str,
        reward: float,
        completion_length: int = 0,
        sample_details: Optional[Dict] = None
    ):
        """Add a reward and completion length for a specific env/task combination."""
        key = (env, task)
        self.rewards[key].append(reward)
        self.sum_rewards[key] += reward
        if completion_length > 0:
            self.lengths[key].append(completion_length)
            self.sum_lengths[key] += completion_length
        # Store last sample details for debug logging
        if sample_details is not None:
            self.last_samples[key] = sample_details

    def get_stats_and_clear(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int], Dict[str, Dict]]:
        """
        Get statistics for all tracked tasks and clear the batch rewards/lengths.
        Returns:
            - task_stats: {task_key: {metric_name: value}}
            - sample_counts: {task_key: cumulative_sample_count}
            - last_samples: {task_key: sample_details} for debug logging
        """
        task_stats = {}
        sample_counts = {}
        last_samples = {}

        for (env, task), reward_list in self.rewards.items():
            if reward_list:
                key = self._get_task_key(env, task)

                # Update cumulative sample count
                batch_count = len(reward_list)
                self.cumulative_samples[(env, task)] += batch_count
                cumulative = self.cumulative_samples[(env, task)]

                # Define wandb metric if not already defined
                self._define_wandb_metric(key)

                # Current batch reward stats
                batch_avg_reward = sum(reward_list) / len(reward_list)

                # Running average reward
                running_avg_reward = self.sum_rewards[(env, task)] / cumulative

                # Completion length stats
                length_list = self.lengths.get((env, task), [])
                batch_avg_length = sum(length_list) / len(length_list) if length_list else 0
                running_avg_length = self.sum_lengths[(env, task)] / cumulative if cumulative > 0 else 0

                task_stats[key] = {
                    "reward": batch_avg_reward,
                    "reward_avg": running_avg_reward,
                    "length": batch_avg_length,
                    "length_avg": running_avg_length,
                }
                sample_counts[key] = cumulative

                # Get last sample for debug logging
                if (env, task) in self.last_samples:
                    last_samples[key] = self.last_samples[(env, task)]

        # Clear batch data
        self.rewards.clear()
        self.lengths.clear()
        self.last_samples.clear()
        return task_stats, sample_counts, last_samples


class PerTaskRewardCallback(TrainerCallback):
    """
    Callback to log per-task rewards and completion lengths to wandb.
    Uses cumulative sample count as x-axis for each task.

    In multi-GPU training, gathers rewards from all GPUs and logs:
    - {task_key}/reward_gpu0, {task_key}/reward_gpu1, ... (per-GPU rewards, 8 lines in one panel)
    - {task_key}/reward_mean (average across all GPUs)
    - {task_key}/reward_avg (running average)

    In debug mode, also logs full sample details (prompt, label, completion, etc.)
    """

    def __init__(self, tracker: PerTaskRewardTracker, debug_mode: bool = False):
        self.tracker = tracker
        self.debug_mode = debug_mode
        self._last_logged_step = -1
        self._wandb = None
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            logger.warning("wandb not available, per-task metrics will only be logged to CLI")

    def _gather_rewards_from_all_gpus(
        self,
        task_stats: Dict[str, Dict[str, float]],
        sample_counts: Dict[str, int]
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, int]]:
        """
        Gather rewards and lengths from all GPUs to rank 0.

        Returns:
            - all_rewards: {task_key: [reward_gpu0, reward_gpu1, ...]}
            - all_lengths: {task_key: [length_gpu0, length_gpu1, ...]}
            - total_samples: {task_key: total_samples_across_all_gpus}
        """
        if not dist.is_initialized():
            # Single GPU - just return local stats
            all_rewards = {k: [v["reward"]] for k, v in task_stats.items()}
            all_lengths = {k: [v["length"]] for k, v in task_stats.items()}
            return all_rewards, all_lengths, sample_counts

        world_size = dist.get_world_size()

        # Get all unique task keys across all processes
        local_task_keys = list(task_stats.keys())
        # Gather task keys from all processes
        all_task_keys_list = [None] * world_size
        dist.all_gather_object(all_task_keys_list, local_task_keys)
        # Flatten and deduplicate
        all_task_keys = sorted(set(k for keys in all_task_keys_list for k in keys))

        all_rewards = {}
        all_lengths = {}
        total_samples = {}

        for task_key in all_task_keys:
            # Get local values (0 if this GPU didn't see this task)
            local_reward = task_stats.get(task_key, {}).get("reward", 0.0)
            local_length = task_stats.get(task_key, {}).get("length", 0.0)
            local_samples = sample_counts.get(task_key, 0)
            # Flag to indicate if this GPU has data for this task
            has_data = 1.0 if task_key in task_stats else 0.0

            # Create tensors for gathering
            local_tensor = torch.tensor(
                [local_reward, local_length, float(local_samples), has_data],
                device="cuda"
            )

            # Gather from all GPUs
            gathered_tensors = [torch.zeros(4, device="cuda") for _ in range(world_size)]
            dist.all_gather(gathered_tensors, local_tensor)

            # Extract rewards and lengths only from GPUs that have data
            rewards_list = []
            lengths_list = []
            total_sample_count = 0

            for tensor in gathered_tensors:
                reward_val, length_val, sample_val, has_data_val = tensor.tolist()
                if has_data_val > 0.5:  # This GPU has data for this task
                    rewards_list.append(reward_val)
                    lengths_list.append(length_val)
                    total_sample_count += int(sample_val)

            if rewards_list:  # Only include if at least one GPU has data
                all_rewards[task_key] = rewards_list
                all_lengths[task_key] = lengths_list
                total_samples[task_key] = total_sample_count

        return all_rewards, all_lengths, total_samples

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log per-task rewards and lengths when trainer logs."""
        # Avoid duplicate logging for the same step
        if state.global_step == self._last_logged_step:
            return
        self._last_logged_step = state.global_step

        # Get local stats and clear batch
        task_stats, sample_counts, last_samples = self.tracker.get_stats_and_clear()

        # Gather rewards from all GPUs
        all_rewards, all_lengths, total_samples = self._gather_rewards_from_all_gpus(
            task_stats, sample_counts
        )

        if not all_rewards:
            return

        # Determine rank for logging
        rank = dist.get_rank() if dist.is_initialized() else 0
        is_main_process = rank == 0

        # Log to wandb (only on main process, but with all GPU data)
        if is_main_process and self._wandb is not None and self._wandb.run is not None:
            for task_key in all_rewards:
                rewards = all_rewards[task_key]
                lengths = all_lengths.get(task_key, [])
                samples = total_samples.get(task_key, 0)

                log_dict = {
                    f"{task_key}/samples": samples,
                }

                # Log per-GPU rewards (will show as multiple lines in same wandb panel)
                for gpu_idx, reward in enumerate(rewards):
                    log_dict[f"{task_key}/reward_gpu{gpu_idx}"] = reward

                # Log mean reward across all GPUs
                if rewards:
                    log_dict[f"{task_key}/reward_mean"] = sum(rewards) / len(rewards)

                # Log per-GPU lengths
                for gpu_idx, length in enumerate(lengths):
                    if length > 0:
                        log_dict[f"{task_key}/length_gpu{gpu_idx}"] = length

                # Log mean length across all GPUs
                valid_lengths = [l for l in lengths if l > 0]
                if valid_lengths:
                    log_dict[f"{task_key}/length_mean"] = sum(valid_lengths) / len(valid_lengths)

                # Also log running averages from local tracker (approximate)
                if task_key in task_stats:
                    log_dict[f"{task_key}/reward_avg"] = task_stats[task_key]["reward_avg"]
                    if task_stats[task_key]["length_avg"] > 0:
                        log_dict[f"{task_key}/length_avg"] = task_stats[task_key]["length_avg"]

                self._wandb.log(log_dict)

        # Print summary to CLI (only on main process)
        if is_main_process:
            logger.info(f"Per-task stats (step {state.global_step}):")
            for task_key in sorted(all_rewards.keys()):
                rewards = all_rewards[task_key]
                samples = total_samples.get(task_key, 0)
                mean_reward = sum(rewards) / len(rewards) if rewards else 0
                rewards_str = ", ".join(f"{r:.3f}" for r in rewards)
                logger.info(
                    f"  {task_key}: rewards=[{rewards_str}], mean={mean_reward:.4f}, "
                    f"samples={samples}, n_gpus={len(rewards)}"
                )

        # Debug mode: log full sample details for each task (only on main process)
        if is_main_process and self.debug_mode and last_samples:
            logger.info(f"\n{'='*80}")
            logger.info(f"DEBUG: Sample details per task (step {state.global_step})")
            logger.info(f"{'='*80}")
            for task_key in sorted(last_samples.keys()):
                sample = last_samples[task_key]
                logger.info(f"\n--- {task_key} ---")
                logger.info(f"Reward: {sample.get('reward', 'N/A')}")
                logger.info(f"Label: {sample.get('label', 'N/A')}")

                # Log extra_data keys and values (truncated if too long)
                extra_data = sample.get('extra_data', {})
                if extra_data:
                    extra_str = json.dumps(extra_data, indent=2, default=str)
                    if len(extra_str) > 1000:
                        extra_str = extra_str[-1000:] + "... (truncated)"
                    logger.info(f"Extra data: {extra_str}")

                # Log prompt (truncated)
                prompt = sample.get('prompt', '')
                if len(prompt) > 1000:
                    prompt = prompt[-1000:] + "... (truncated)"
                logger.info(f"Prompt:\n{prompt}")

                # Log completion (truncated)
                completion = sample.get('completion', '')
                if len(completion) > 1000:
                    completion = completion[-1000:] + "... (truncated)"
                logger.info(f"Completion:\n{completion}")
            logger.info(f"{'='*80}\n")


# Global tracker instance (will be used by reward_func and callback)
reward_tracker = PerTaskRewardTracker()

# Debug mode flag (set via environment variable or script argument)
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0").lower() in ("1", "true", "yes")


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
        metadata={"help": "Maximum total samples after merging (None for all)"}
    )
    max_samples_per_file: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum samples to load per parquet file (None for all)"}
    )
    debug_mode: bool = field(
        default=False,
        metadata={"help": "Enable debug mode to log sample details per task"}
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
    # Handle othello and hex (including hex_5, hex_7, hex_9, hex_11)
    task_lower = task.lower() if task else ""
    if task_lower == "othello" or task_lower.startswith("hex"):
        probabilities = extra_data.get("probabilities", {})

        if not probabilities:
            logger.debug(f"No probabilities found for {task}, using exact match")
            return 1.0 if parsed_response.strip() == label.strip() else 0.0

        # Filter out None values from probabilities
        valid_probs = {k: v for k, v in probabilities.items() if v is not None}

        if not valid_probs:
            logger.debug(f"All probabilities are None for {task}, using exact match")
            return 1.0 if parsed_response.strip() == label.strip() else 0.0

        # Try to find the response in probabilities (handle both string and int keys)
        prob_value = None
        if parsed_response in valid_probs:
            prob_value = valid_probs[parsed_response]
        elif parsed_response.isdigit() and int(parsed_response) in valid_probs:
            prob_value = valid_probs[int(parsed_response)]

        if prob_value is not None:
            max_prob = max(valid_probs.values())
            if max_prob > 0:
                return float(prob_value) / float(max_prob)
            return 0.0
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


def reward_func(prompts, completions, label=None, env=None, task=None, extra_data=None, **kwargs):
    """
    Main reward function for GRPO training (TRL-style).

    TRL's GRPOTrainer calls reward functions with:
    - prompts: List of original prompts
    - completions: List of model completions (response only, not including prompt)
    - Additional columns from the dataset are passed as keyword arguments

    Args:
        prompts: List of original prompts
        completions: List of model completions (response text)
        label: List of ground truth labels (from dataset column)
        env: List of environment names (from dataset column)
        task: List of task names (from dataset column)
        extra_data: List of extra data JSON strings (from dataset column)
        **kwargs: Additional arguments

    Returns:
        List of float rewards (TRL expects a list, not a tensor)
    """
    batch_size = len(completions)
    rewards = []

    # Handle None values with defaults
    labels = label if label is not None else [""] * batch_size
    envs = env if env is not None else [""] * batch_size
    tasks = task if task is not None else [""] * batch_size
    extra_data_list = extra_data if extra_data is not None else ["{}"] * batch_size

    # Ensure lists match batch size
    if len(labels) != batch_size:
        labels = [labels[0] if labels else ""] * batch_size
    if len(envs) != batch_size:
        envs = [envs[0] if envs else ""] * batch_size
    if len(tasks) != batch_size:
        tasks = [tasks[0] if tasks else ""] * batch_size
    if len(extra_data_list) != batch_size:
        extra_data_list = [extra_data_list[0] if extra_data_list else "{}"] * batch_size

    for i in range(batch_size):
        try:
            # TRL passes completions as response only (not including prompt)
            response = completions[i]

            label_i = labels[i] if labels[i] else ""
            env_i = envs[i] if isinstance(envs[i], str) else str(envs[i])
            task_i = tasks[i] if isinstance(tasks[i], str) else str(tasks[i])

            # Parse extra_data from JSON string if needed
            extra_raw = extra_data_list[i]
            if isinstance(extra_raw, dict):
                extra = extra_raw
            elif isinstance(extra_raw, str):
                try:
                    extra = json.loads(extra_raw) if extra_raw else {}
                except json.JSONDecodeError:
                    extra = {}
            else:
                extra = {}

            # Compute reward using the unified compute_reward function
            reward = compute_reward(
                response=response,
                label=label_i,
                env=env_i,
                task=task_i,
                extra_data=extra
            )

            # Handle None reward (shouldn't happen, but defensive)
            if reward is None:
                logger.warning(
                    f"compute_reward returned None - env: {env_i}, task: {task_i}, "
                    f"label: {label_i[:50] if label_i else 'None'}..., "
                    f"response: {response[:50] if response else 'None'}..."
                )
                reward = 0.0

            rewards.append(float(reward))

            # Track reward and completion length per task for wandb logging
            completion_length = len(response) if response else 0

            # Build sample details for debug logging
            sample_details = None
            if DEBUG_MODE:
                prompt_i = prompts[i] if i < len(prompts) else ""
                sample_details = {
                    "prompt": prompt_i,
                    "label": label_i,
                    "completion": response,
                    "reward": float(reward),
                    "extra_data": extra,
                }

            reward_tracker.add_reward(env_i, task_i, float(reward), completion_length, sample_details)

            # Log successful completions occasionally
            if reward > 0.5 and random.random() < 0.01:
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Env: {env_i}, Task: {task_i}, Label: {label_i}, Reward: {reward}\n")
                    f.write(f"Response: {parse_response(response)}\n")

        except Exception as e:
            logger.warning(
                f"Error computing reward: {e} - "
                f"env: {env_i if 'env_i' in dir() else 'unknown'}, "
                f"task: {task_i if 'task_i' in dir() else 'unknown'}, "
                f"label: {(label_i[:50] + '...') if 'label_i' in dir() and label_i else 'unknown'}, "
                f"response: {(response[:50] + '...') if 'response' in dir() and response else 'unknown'}"
            )
            rewards.append(0.0)
            # Also track failed rewards (with 0 length since we may not have valid response)
            if 'env_i' in dir() and 'task_i' in dir():
                reward_tracker.add_reward(env_i, task_i, 0.0, 0, None)

    return rewards

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
    max_samples_per_file: Optional[int] = None,
) -> Dataset:
    """
    Load and merge all parquet datasets from dataset/Upload directory.

    Loads all parquet files from GAME/, LGC/, and PRINT/ subdirectories
    and merges them with specified weights.

    Args:
        dataset_path: Directory containing GAME/, LGC/, PRINT/ subdirectories
        max_samples: Maximum total samples after merging (None for all)
        max_samples_per_file: Maximum samples to load per parquet file (None for all)

    Returns:
        Merged Dataset
    """
    datasets_list = []
    weights = []
    dataset_names = []

    def normalize_extra_data(extra):
        """Convert extra_data to JSON string, handling numpy arrays."""
        if extra is None:
            return "{}"
        elif isinstance(extra, str):
            return extra
        else:
            # Convert numpy arrays to lists recursively
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()  # Convert numpy scalars to Python types
                else:
                    return obj

            converted = convert_numpy(extra)
            return json.dumps(converted)

    # Iterate through all configured datasets
    for dataset_key, weight in DATASET_WEIGHTS.items():
        parquet_path = os.path.join(dataset_path, f"{dataset_key}.parquet")

        if os.path.exists(parquet_path):
            try:
                # Read parquet with PyArrow
                table = pq.read_table(parquet_path)
                total_rows = len(table)

                # Convert to pandas first (handles string_view conversion automatically)
                df = table.to_pandas()

                # Limit samples per file if specified (sample after converting to pandas)
                if max_samples_per_file is not None and total_rows > max_samples_per_file:
                    df = df.sample(n=max_samples_per_file, random_state=42)
                    logger.info(f"  Sampled {max_samples_per_file} from {total_rows} rows")

                # Vectorized normalization (much faster than iterrows)
                normalized_df = {
                    "prompt": df["prompt"].astype(str) if "prompt" in df.columns else [""] * len(df),
                    "label": df["label"].astype(str) if "label" in df.columns else [""] * len(df),
                    "env": df["env"].astype(str) if "env" in df.columns else [""] * len(df),
                    "task": df["task"].astype(str) if "task" in df.columns else [""] * len(df),
                }

                # Handle extra_data column
                if "extra_data" in df.columns:
                    normalized_df["extra_data"] = df["extra_data"].apply(normalize_extra_data)
                else:
                    normalized_df["extra_data"] = ["{}"] * len(df)

                # Create dataset directly from dict (faster than from_list)
                ds = Dataset.from_dict(normalized_df)
                datasets_list.append(ds)
                weights.append(weight)
                dataset_names.append(dataset_key)
                logger.info(f"Loaded {dataset_key}: {len(ds)} samples (weight={weight})")
            except Exception as e:
                logger.warning(f"Failed to load {dataset_key}: {e}")
                import traceback
                traceback.print_exc()
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
        # Use stopping_strategy="all_exhausted" to repeat smaller datasets
        # This ensures datasets with fewer samples are upsampled to match the weighted distribution
        merged = interleave_datasets(
            datasets_list,
            probabilities=probs,
            seed=42,
            stopping_strategy="all_exhausted"
        )
        logger.info(f"Merged {len(datasets_list)} datasets with probabilities (all_exhausted):")
        for name, prob, ds in zip(dataset_names, probs, datasets_list):
            logger.info(f"  {name}: {prob:.4f} ({len(ds)} samples)")

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
        max_samples_per_file=script_args.max_samples_per_file,
    )

    logger.info(f"Train samples: {len(train_dataset)}")

    # -------------------------
    # Initialize trainer with per-task reward callback
    # -------------------------
    # Enable debug mode from script args or environment variable
    global DEBUG_MODE
    DEBUG_MODE = script_args.debug_mode or DEBUG_MODE
    if DEBUG_MODE:
        logger.info("DEBUG MODE ENABLED: Will log sample details per task")

    per_task_callback = PerTaskRewardCallback(reward_tracker, debug_mode=DEBUG_MODE)

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=[per_task_callback],
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