"""Smoke tests for src/evaluation/rewards.py.

Loads ``rewards.py`` via :mod:`importlib` to bypass ``src/evaluation/__init__.py``
and avoid triggering the heavy AffineFoundation actor imports in ``_constants.py``.

Only :func:`parse_response` is tested here; :func:`reward_response` requires
live environment servers and is covered by integration tests.
"""

import importlib.util
from pathlib import Path

import pytest

# Load the submodule directly — src/evaluation/__init__.py (and _constants.py)
# are intentionally NOT executed so this test has zero external dependencies.
_ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "src.evaluation.rewards", _ROOT / "src" / "evaluation" / "rewards.py"
)
_rewards_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rewards_mod)

parse_response = _rewards_mod.parse_response


class TestParseResponse:
    """Unit tests for :func:`parse_response`."""

    def test_plain_text_passthrough(self):
        """Text without a </think> tag is returned stripped."""
        assert parse_response("hello world") == "hello world"

    def test_strips_think_block(self):
        """Everything up to and including </think> is removed."""
        text = "<think>some reasoning</think>final answer"
        assert parse_response(text) == "final answer"

    def test_strips_surrounding_whitespace(self):
        """Leading/trailing whitespace is always stripped."""
        assert parse_response("  hello  ") == "hello"

    def test_strips_whitespace_after_think(self):
        """Whitespace between </think> and the answer is stripped."""
        text = "<think>reasoning</think>   my answer   "
        assert parse_response(text) == "my answer"

    def test_uses_last_think_close_tag(self):
        """When multiple </think> tags are present the last one wins."""
        text = "<think>first</think>middle<think>second</think>final"
        assert parse_response(text) == "final"

    def test_empty_string(self):
        """An empty input returns an empty string."""
        assert parse_response("") == ""

    def test_no_closing_think_tag(self):
        """Without </think> the entire string is returned (stripped)."""
        assert parse_response("<think>still thinking") == "<think>still thinking"

    def test_think_at_end_leaves_empty_answer(self):
        """A response that ends with </think> and no answer returns ''."""
        text = "<think>reasoning</think>"
        assert parse_response(text) == ""

