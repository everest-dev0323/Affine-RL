"""
TiG Reward Function for OpenRLHF

This reward function computes detailed rewards for various environments
including logic games, sudoku, and other reasoning tasks.

Supports:
- lgc-v2: Logic tasks with verifiers (sudoku, game_of_24, boolean_expressions, etc.)
- game/print: Simple equality check environments
- openspiel: Custom reward functions for openspiel environments
"""

import re
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add lgc-v2 environment to path for verifiers
lgc_v2_path = project_root / "AffineFoundation" / "affinetes" / "environments" / "primeintellect" / "lgc-v2"
sys.path.insert(0, str(lgc_v2_path))


def parse_response(text: str) -> str:
    """Parse response, removing <think> tags if present."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def extract_code_block(text: str) -> Optional[str]:
    """Extract code from ```...``` blocks."""
    pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


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
# Verifier-based Reward Functions
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
        print(f"Warning: Could not import verifier for {task_type}: {e}")
        return 1.0 if parsed_response.strip() == label.strip() else 0.0
    except Exception as e:
        # Error during verification
        print(f"Warning: Error during verification for {task_type}: {e}")
        return 0.0

# ============================================================================
# Custom Openspiel Reward Functions
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
    if task == "othello":
        probabilities = extra_data.get("probabilities", {})
        if parsed_response in probabilities:
            return probabilities[parsed_response]
        else:
            return 0.0
    # TODO: Implement custom reward logic for openspiel tasks
    return 1.0 if parsed_response.strip() == label.strip() else 0.0

# ============================================================================
# Main Reward Function
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

    # Default: simple equality check for game, print, and other environments
    return 1.0 if parsed_response.strip() == label.strip() else 0.0


def reward_func(queries, prompts, labels, **kwargs):
    """
    Compute TiG rewards for generated responses.

    This is the main entry point called by OpenRLHF.

    Args:
        queries: List[str] - Full text (prompt + response)
        prompts: List[str] - Original prompts only
        labels: List[str] - Ground truth actions (from --label_key)
        **kwargs: Additional info passed via --extra_keys:
            - envs: List[str] - Environment names
            - tasks: List[str] - Task names
            - extra_data: List[dict] - Additional metadata

    Returns:
        dict with rewards, scores, and extra_logs
    """
    batch_size = len(queries)
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

    # Track rewards by environment for logging
    env_rewards = {}

    for i in range(batch_size):
        # Extract response (remove prompt from full query)
        full_text = queries[i]
        prompt_text = prompts[i]
        response = full_text[len(prompt_text):].strip()

        label = labels[i] if labels else ""
        env = envs[i] if isinstance(envs[i], str) else str(envs[i])
        task = tasks[i] if isinstance(tasks[i], str) else str(tasks[i])
        extra = extra_data_list[i] if isinstance(extra_data_list[i], dict) else {}

        # Compute reward
        reward = compute_reward(
            response=response,
            label=label,
            env=env,
            task=task,
            extra_data=extra
        )

        rewards.append(reward)

        # Track by environment
        if env not in env_rewards:
            env_rewards[env] = []
        env_rewards[env].append(reward)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    # Build extra logs
    extra_logs = {
        "mean_reward": reward_tensor.mean().item(),
        "reward_std": reward_tensor.std().item() if len(rewards) > 1 else 0.0,
        "num_correct": sum(1 for r in rewards if r == 1.0),
        "accuracy": sum(rewards) / len(rewards) if rewards else 0.0,
    }

    # Add per-environment stats
    for env_name, env_reward_list in env_rewards.items():
        if env_name:  # Skip empty env names
            extra_logs[f"{env_name}_accuracy"] = sum(env_reward_list) / len(env_reward_list)
            extra_logs[f"{env_name}_count"] = len(env_reward_list)

    return {
        "rewards": reward_tensor,  # Used for advantage calculation
        "scores": reward_tensor,   # Used for dynamic filtering (DAPO)
        "extra_logs": extra_logs,
    }