"""Response parsing and reward computation for affine environments.

This module provides the core reward signal functions used during RL training.
The primary entry-point is :func:`reward_response`, which dispatches to
environment-specific logic (e.g. Sudoku partial credit) or falls back to a
simple string-equality check.
"""

from typing import Optional


def parse_response(text: str) -> str:
    """Strip chain-of-thought ``<think>`` blocks from a model response.

    Args:
        text: Raw model response, potentially containing ``<think>…</think>``
            reasoning wrapped around the final answer.

    Returns:
        The final answer text with any leading ``<think>`` block removed and
        surrounding whitespace stripped.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()


async def reward_response(env: str, task_id: int, response: str, ground_truth: str) -> float:
    """Calculate the reward for a model response given an environment and task.

    Supports:
    - ``lgc-v2`` — dispatches to the LogicTaskV2 verifier; awards partial
      credit for Sudoku tasks using :func:`~src.evaluation.sudoku.sudoku_reward`.
    - ``game``, ``print``, and all other environments — simple string equality.

    Args:
        env: Environment identifier (e.g. ``"lgc-v2"``, ``"game"``, ``"print"``).
        task_id: Numeric task ID used to load ground-truth metadata.
        response: Raw model output (may contain ``<think>`` blocks).
        ground_truth: Expected answer string (used for non-lgc-v2 environments).

    Returns:
        Scalar reward in ``[0, 1]``.
    """
    from .sudoku import extract_sudoku_answer, parse_sudoku_grid, sudoku_reward
    from ._constants import ACTOR_MAPPING  # noqa: F401 – ensures sys.path is set

    env_lower = env.lower()
    parsed = parse_response(response)

    if env_lower == "lgc-v2":
        from logic_task_v2 import LogicTaskV2  # available via sys.path in _constants

        logic_task = LogicTaskV2()
        challenge = await logic_task.generate_dataset_by_id(task_id)

        if challenge.extra.get("task_type") == "sudoku":
            # Try for a perfect solution first
            if await logic_task.evaluate(parsed, challenge):
                return 1.0

            # Fall back to partial-credit scoring
            answer_str = extract_sudoku_answer(parsed)
            answer_grid = parse_sudoku_grid(answer_str)
            puzzle_str = challenge.extra.get("metadata", {}).get("puzzle", "")
            puzzle_grid = parse_sudoku_grid(puzzle_str)

            if answer_grid is None or puzzle_grid is None:
                return 0.0

            return sudoku_reward(answer_grid, puzzle_grid)

        return float(await logic_task.evaluate(parsed, challenge))

    # Default: exact string match for game / print / other environments
    return 1.0 if parsed.strip() == ground_truth.strip() else 0.0

