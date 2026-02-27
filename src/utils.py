"""Shared utility functions for the Affine-RL project.

This module provides lightweight, dependency-free helpers that are reused
across training, evaluation, and data-processing scripts.  Import from here
instead of defining the same function in each individual script.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a plain dictionary.

    Args:
        config_path: Path to the YAML file.  Relative paths are resolved
            against the current working directory.

    Returns:
        A :class:`dict` containing the configuration values, or an empty
        dict if the file is empty or cannot be parsed.

    Raises:
        FileNotFoundError: If *config_path* does not exist.

    Example::

        config = load_config_from_yaml("config/tig.yaml")
        for key, value in config.items():
            setattr(my_config_obj, key, value)
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r") as fh:
        data = yaml.safe_load(fh)
    return data if data else {}

