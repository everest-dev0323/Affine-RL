"""
Dataset Preparation Script for Training Data

This script:
1. Normalizes rewards (sciworld: -100 to 100 → 0 to 1, others already 0-1)
2. Applies round-based penalties to rewards
3. Classifies data by task_id (env_name + task_number)
4. Splits trajectories with multiple reasoning_content into separate histories
5. Outputs files like train_alfworld_1.jsonl
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from pathlib import Path


# 8 environments
ENVS = ["abd", "ded", "sat", "alfworld", "sciworld", "babyai", "textcraft", "webshop"]


# ============================================================================
# Reward Functions
# ============================================================================

def normalize_sciworld_reward(reward: float) -> float:
    """
    Normalize sciworld reward from [-100, 100] to [0, 1]
    """
    # Clamp to valid range
    reward = max(-100, min(100, reward))
    # Normalize: (reward + 100) / 200
    return (reward + 100) / 200


def get_base_reward(data: Dict) -> float:
    """
    Get base reward from data, normalizing if necessary.
    Uses 'reward' if available, otherwise uses 'score'.
    """
    env = data.get("env", "")

    # Try score first, then reward
    reward = data.get("score")
    if reward is None:
        reward = data.get("reward", 0)

    if reward is None:
        reward = 0

    # Normalize sciworld rewards
    if env == "sciworld":
        reward = normalize_sciworld_reward(reward)
    else:
        # Others should already be 0-1, but clamp just in case
        reward = max(0, min(1, float(reward)))

    return reward


def calculate_round_penalty(
    num_rounds: int,
    max_rounds: int = 10,
    penalty_weight: float = 0.05
) -> float:
    """
    Calculate penalty based on number of conversation rounds.

    Args:
        num_rounds: Number of conversation rounds (user-assistant pairs)
        max_rounds: Maximum expected rounds
        penalty_weight: Weight for the penalty

    Returns:
        Penalty value (positive number to be subtracted from reward)
    """
    if num_rounds <= 1:
        return 0.0

    # Penalty increases with number of rounds
    # Exponential penalty for exceeding max rounds
    if num_rounds > max_rounds + 1:
        penalty = 1
    elif num_rounds < 6:
        penalty = 0
    else:
        penalty = penalty_weight * (num_rounds - 5)

    return penalty


def calculate_final_reward(
    base_reward: float,
    num_rounds: int,
    max_rounds: int = 10,
    round_penalty_weight: float = 0.05
) -> float:
    """
    Calculate final reward with round penalty.

    Args:
        base_reward: Base reward (0-1)
        num_rounds: Number of conversation rounds
        max_rounds: Maximum expected rounds
        round_penalty_weight: Weight for round penalty

    Returns:
        Final reward (clamped to 0-1)
    """
    penalty = calculate_round_penalty(num_rounds, max_rounds, round_penalty_weight)
    final_reward = base_reward - penalty

    # Clamp to valid range
    return max(0.0, min(1.0, final_reward))


# ============================================================================
# Trajectory Splitting
# ============================================================================

def count_reasoning_content(conversations: List[Dict]) -> int:
    """Count number of assistant messages with reasoning_content."""
    count = 0
    for conv in conversations:
        if conv.get("role") == "assistant" and conv.get("reasoning_content"):
            count += 1
    return count


def split_trajectory_by_reasoning(
    conversations: List[Dict],
    base_reward: float,
    max_rounds: int = 10,
    round_penalty_weight: float = 0.05
) -> List[Tuple[List[Dict], float]]:
    """
    Split a trajectory with multiple reasoning_content into separate trajectories.

    Each resulting trajectory will have at most one reasoning_content,
    which will be at the last assistant response.

    Args:
        conversations: List of conversation messages
        base_reward: Base reward for the complete trajectory
        max_rounds: Max rounds for penalty calculation
        round_penalty_weight: Weight for round penalty

    Returns:
        List of (conversations, reward) tuples
    """
    # Find indices of assistant messages with reasoning_content
    reasoning_indices = []
    for i, conv in enumerate(conversations):
        if conv.get("role") == "assistant" and conv.get("reasoning_content"):
            reasoning_indices.append(i)

    num_rounds = len([c for c in conversations if c.get("role") == "assistant"])
    final_reward = calculate_final_reward(
        base_reward, num_rounds, max_rounds, round_penalty_weight
    )
    
    # If 0 or 1 reasoning_content, no splitting needed
    if len(reasoning_indices) <= 1:
        return [(conversations, final_reward)]

    # Split at each reasoning_content point
    results = []

    for reasoning_idx in reasoning_indices:
        # Include conversations up to and including this assistant message
        split_convs = conversations[:reasoning_idx + 1]
        results.append((split_convs, final_reward))

    return results


# ============================================================================
# Task ID Processing
# ============================================================================

def extract_task_number(task_id: Any) -> str:
    """
    Extract a numeric identifier from task_id.
    If task_id is already a number, use it directly.
    If it's a string/hash, use a hash-based number.
    """
    if task_id is None:
        return "unknown"

    if isinstance(task_id, (int, float)):
        return str(int(task_id))

    # For string task_ids, extract numbers or use hash
    task_str = str(task_id)

    # Try to extract leading numbers
    match = re.match(r'^(\d+)', task_str)
    if match:
        return match.group(1)

    # Use hash for complex IDs (take modulo to keep reasonable)
    return str(abs(hash(task_str)) % 100000)


def get_output_key(env: str, task_id: Any) -> str:
    """
    Generate output file key from env and task_id.
    Format: train_{env}_{task_number}
    """
    task_number = extract_task_number(task_id)
    return f"train_{env}_{task_number}"


# ============================================================================
# Main Processing
# ============================================================================

def process_single_item(
    data: Dict,
    max_rounds: int = 10,
    round_penalty_weight: float = 0.05
) -> List[Dict]:
    """
    Process a single rollout item.

    Returns:
        List of processed items (may be multiple if trajectory was split)
    """
    env = data.get("env")
    task_id = data.get("task_id")
    conversations = data.get("conversations", [])

    if not env or not conversations:
        return []

    # Get base reward
    base_reward = get_base_reward(data)

    # Split trajectory if multiple reasoning_content
    split_trajectories = split_trajectory_by_reasoning(
        conversations, base_reward, max_rounds, round_penalty_weight
    )
    # Count rounds for this split
    num_rounds = len([c for c in conversations if c.get("role") == "assistant"])

    results = []
    for split_convs, final_reward in split_trajectories:

        result = {
            "env": env,
            "task_id": task_id,
            "conversations": split_convs,
            "reward": final_reward,
            "base_reward": base_reward,
            "num_rounds": num_rounds,
            "original_score": data.get("score"),
            "original_reward": data.get("reward"),
        }
        results.append(result)

    return results


def process_rollout_files(
    input_dir: str,
    output_dir: str,
    max_rounds: int = 10,
    round_penalty_weight: float = 0.05
) -> Dict[str, int]:
    """
    Process all rollout files and output by task_id.

    Args:
        input_dir: Directory containing rollout .jsonl files
        output_dir: Directory to write output files
        max_rounds: Max rounds for penalty calculation
        round_penalty_weight: Weight for round penalty

    Returns:
        Statistics dictionary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all processed items by output key
    output_data: Dict[str, List[Dict]] = defaultdict(list)

    stats = {
        "total_input": 0,
        "total_output": 0,
        "split_count": 0,
        "env_counts": defaultdict(int),
        "task_counts": defaultdict(int),
    }

    # Process all .jsonl files in input directory
    jsonl_files = list(input_path.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} .jsonl files in {input_dir}")

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stats["total_input"] += 1

                # Process item (may return multiple if split)
                processed_items = process_single_item(
                    data, max_rounds, round_penalty_weight
                )

                if len(processed_items) > 1:
                    stats["split_count"] += 1

                for item in processed_items:
                    env = item.get("env")
                    task_id = item.get("task_id")

                    output_key = get_output_key(env, task_id)
                    output_data[output_key].append(item)

                    stats["total_output"] += 1
                    stats["env_counts"][env] += 1
                    stats["task_counts"][output_key] += 1

    # Write output files
    print(f"\nWriting output files to {output_dir}")
    for output_key, items in output_data.items():
        output_file = output_path / f"{output_key}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"  {output_key}.jsonl: {len(items)} items")

    return stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./datasets/rollouts",
        help="Input directory containing rollout .jsonl files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/train",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=10,
        help="Maximum expected conversation rounds"
    )
    parser.add_argument(
        "--round_penalty_weight",
        type=float,
        default=0.05,
        help="Weight for round penalty"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Preparation")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Round penalty weight: {args.round_penalty_weight}")
    print()

    stats = process_rollout_files(
        args.input_dir,
        args.output_dir,
        args.max_rounds,
        args.round_penalty_weight
    )

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"Total input items: {stats['total_input']}")
    print(f"Total output items: {stats['total_output']}")
    print(f"Trajectories split: {stats['split_count']}")
    print()
    print("Items per environment:")
    for env, count in sorted(stats["env_counts"].items()):
        print(f"  {env}: {count}")
    print()
    print(f"Total unique task files: {len(stats['task_counts'])}")


if __name__ == "__main__":
    main()
