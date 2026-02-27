"""Sudoku-specific reward computation and grid-parsing utilities.

These helpers are used by the reward module when the lgc-v2 environment
returns a task whose ``task_type`` is ``"sudoku"``.
"""

import re
from typing import List, Optional


def sudoku_reward(answer_grid: List[List[int]], puzzle_grid: List[List[int]]) -> float:
    """Calculate a partial reward for a Sudoku answer.

    The reward is 0 immediately if the answer violates any pre-filled cell from
    the original puzzle.  Otherwise it is the weighted average of the fraction
    of valid rows, columns, and 3×3 boxes:

        reward = (3·row_score + 3·col_score + 4·box_score) / 10

    Args:
        answer_grid: 9×9 grid of integers representing the model's answer.
        puzzle_grid: 9×9 grid with the original puzzle (0 = empty cell).

    Returns:
        Float reward in [0, 1].
    """
    # Pre-filled cells must be preserved
    for i in range(9):
        for j in range(9):
            if puzzle_grid[i][j] != 0 and answer_grid[i][j] != puzzle_grid[i][j]:
                return 0.0

    def _is_valid_group(group: List[int]) -> bool:
        return sorted(group) == list(range(1, 10))

    row_score = sum(1 for row in answer_grid if _is_valid_group(row)) / 9

    col_score = sum(
        1 for j in range(9) if _is_valid_group([answer_grid[i][j] for i in range(9)])
    ) / 9

    valid_boxes = 0
    for br in range(3):
        for bc in range(3):
            box = [
                answer_grid[br * 3 + i][bc * 3 + j]
                for i in range(3)
                for j in range(3)
            ]
            if _is_valid_group(box):
                valid_boxes += 1
    box_score = valid_boxes / 9

    return (3 * row_score + 3 * col_score + 4 * box_score) / 10


def extract_sudoku_answer(response: str) -> str:
    """Extract a Sudoku grid string from a model response.

    First tries to find 9 consecutive lines that each contain exactly 9 digits.
    Falls back to the last fenced code block in the response.

    Args:
        response: Raw model response text.

    Returns:
        Extracted grid string (may be empty if nothing matched).
    """
    lines = response.strip().split("\n")
    grid_lines: List[str] = []
    for line in lines:
        digits = re.findall(r"[1-9]", line.replace("`", "").strip())
        if len(digits) == 9:
            grid_lines.append("".join(digits))

    if len(grid_lines) >= 9:
        return "\n".join(grid_lines[:9])

    # Fallback: last fenced code block
    matches = re.findall(r"```([\s\S]*?)```", response)
    if matches:
        grid_str = matches[-1].strip()
        # Strip optional language identifier on the first line
        if "\n" in grid_str:
            first_line, rest = grid_str.split("\n", 1)
            if not re.search(r"[1-9]", first_line):
                grid_str = rest
        return grid_str

    return response.strip()


def parse_sudoku_grid(grid_str: str) -> Optional[List[List[int]]]:
    """Parse a string representation of a Sudoku grid into a 9×9 integer list.

    Accepts digits (1–9) and dots/zeros as empty cells.  Ignores whitespace,
    separators, and other non-digit characters within each line.

    Args:
        grid_str: Multi-line string representing the grid.

    Returns:
        9×9 list of integers, or ``None`` if the string cannot be parsed into
        exactly 9 rows of 9 cells.
    """
    if not grid_str:
        return None

    grid: List[List[int]] = []
    for line in grid_str.strip().split("\n"):
        row: List[int] = []
        for ch in line.strip():
            if ch.isdigit():
                row.append(int(ch))
            elif ch == ".":
                row.append(0)
        if len(row) == 9:
            grid.append(row)

    return grid if len(grid) == 9 else None

