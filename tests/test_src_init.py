"""Smoke tests for src/__init__.py.

Covers :func:`src.get_base_task_name`, which is a pure-Python utility with no
external dependencies.
"""

import pytest

from src import get_base_task_name


class TestGetBaseTaskName:
    """Unit tests for :func:`get_base_task_name`."""

    def test_strips_colon_prefix(self):
        """'affine:sat' should return 'sat'."""
        assert get_base_task_name("affine:sat") == "sat"

    def test_strips_version_suffix(self):
        """'-v2' suffix should be removed after the colon prefix is stripped."""
        assert get_base_task_name("affine:ded-v2") == "ded"
        assert get_base_task_name("affine:abd-v2") == "abd"

    def test_bare_name_no_prefix(self):
        """A plain name with no colon prefix is returned unchanged."""
        assert get_base_task_name("sat") == "sat"

    def test_bare_name_no_version(self):
        """A bare name with no version suffix is returned unchanged."""
        assert get_base_task_name("game") == "game"

    def test_bare_name_with_version_suffix(self):
        """Version suffix is stripped even without a colon prefix."""
        assert get_base_task_name("lgc-v2") == "lgc"

    def test_preserves_hyphen_not_version(self):
        """Hyphens that are not '-v<N>' suffixes must be preserved."""
        assert get_base_task_name("affine:arc-gen") == "arc-gen"

    def test_numeric_version_variants(self):
        """Version stripping should work for any '-v<N>' suffix."""
        assert get_base_task_name("affine:env-v3") == "env"
        assert get_base_task_name("affine:env-v10") == "env"

