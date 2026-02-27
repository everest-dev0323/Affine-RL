"""Smoke tests for src/evaluation/sudoku.py.

Loads ``sudoku.py`` via :mod:`importlib` to bypass ``src/evaluation/__init__.py``
and avoid triggering the heavy AffineFoundation actor imports in ``_constants.py``.

All three public functions are pure Python (stdlib only) and can be tested
without any external dependencies or environment servers.
"""

import importlib.util
from pathlib import Path

import pytest

# Load the submodule directly — src/evaluation/__init__.py (and _constants.py)
# are intentionally NOT executed so this test has zero external dependencies.
_ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "src.evaluation.sudoku", _ROOT / "src" / "evaluation" / "sudoku.py"
)
_sudoku_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sudoku_mod)

extract_sudoku_answer = _sudoku_mod.extract_sudoku_answer
parse_sudoku_grid = _sudoku_mod.parse_sudoku_grid
sudoku_reward = _sudoku_mod.sudoku_reward

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# A known-valid completed Sudoku (classic Wikipedia example).
_SOLVED = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]
_EMPTY_PUZZLE = [[0] * 9 for _ in range(9)]  # no pre-filled cells


# ---------------------------------------------------------------------------
# parse_sudoku_grid
# ---------------------------------------------------------------------------

class TestParseSudokuGrid:
    def test_valid_digit_grid(self):
        grid_str = "\n".join("".join(str(c) for c in row) for row in _SOLVED)
        result = parse_sudoku_grid(grid_str)
        assert result == _SOLVED

    def test_dots_as_zeros(self):
        grid_str = "\n".join("." * 9 for _ in range(9))
        result = parse_sudoku_grid(grid_str)
        assert result == _EMPTY_PUZZLE

    def test_none_input_returns_none(self):
        assert parse_sudoku_grid(None) is None

    def test_empty_string_returns_none(self):
        assert parse_sudoku_grid("") is None

    def test_too_few_rows_returns_none(self):
        grid_str = "\n".join("123456789" for _ in range(5))
        assert parse_sudoku_grid(grid_str) is None

    def test_ignores_separator_chars(self):
        # Pipes and dashes are common in formatted grids; they should be ignored.
        row = "5 3 4 | 6 7 8 | 9 1 2"
        grid_str = "\n".join([row] * 9)
        result = parse_sudoku_grid(grid_str)
        assert result is not None
        assert result[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]


# ---------------------------------------------------------------------------
# extract_sudoku_answer
# ---------------------------------------------------------------------------

class TestExtractSudokuAnswer:
    def test_extracts_nine_digit_lines(self):
        lines = "\n".join("".join(str(c) for c in row) for row in _SOLVED)
        extracted = extract_sudoku_answer(lines)
        # Must recover a parseable 9×9 grid
        assert parse_sudoku_grid(extracted) == _SOLVED

    def test_falls_back_to_code_block(self):
        body = "\n".join("123456789" for _ in range(9))
        response = f"Here is my answer:\n```\n{body}\n```"
        extracted = extract_sudoku_answer(response)
        assert "123456789" in extracted

    def test_returns_raw_on_no_match(self):
        response = "I have no idea."
        assert extract_sudoku_answer(response) == "I have no idea."


# ---------------------------------------------------------------------------
# sudoku_reward
# ---------------------------------------------------------------------------

class TestSudokuReward:
    def test_perfect_solution_scores_one(self):
        score = sudoku_reward(_SOLVED, _EMPTY_PUZZLE)
        assert score == pytest.approx(1.0)

    def test_prefilled_cell_violation_scores_zero(self):
        # Puzzle has cell [0][0] = 5; answer changes it to 3 → immediate 0.
        puzzle = [row[:] for row in _EMPTY_PUZZLE]
        puzzle[0][0] = 5
        answer = [row[:] for row in _SOLVED]
        answer[0][0] = 3  # violates pre-filled cell
        assert sudoku_reward(answer, puzzle) == 0.0

    def test_all_zeros_scores_zero(self):
        all_zeros = [[0] * 9 for _ in range(9)]
        assert sudoku_reward(all_zeros, _EMPTY_PUZZLE) == 0.0

    def test_partial_solution_between_zero_and_one(self):
        # Use the solved grid but corrupt the last row → still passes pre-fill check.
        partial = [row[:] for row in _SOLVED]
        partial[8] = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # invalid last row
        score = sudoku_reward(partial, _EMPTY_PUZZLE)
        assert 0.0 < score < 1.0

    def test_reward_formula_weights(self):
        # reward = (3*row + 3*col + 4*box) / 10  → perfect = 1.0
        score = sudoku_reward(_SOLVED, _EMPTY_PUZZLE)
        assert score == pytest.approx((3 * 1.0 + 3 * 1.0 + 4 * 1.0) / 10)

