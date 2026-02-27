"""Affine-RL evaluation package.

This package provides evaluation utilities for all supported environments.
It is structured as a collection of focused sub-modules, each covering one
inference back-end or concern:

Sub-modules
-----------
``_constants``
    Shared ``sys.path`` setup, Actor imports, and environment-level constants
    (``ACTOR_MAPPING``, ``AGENTGYM_PORTS``, ``AGENTGYM_MAX_ROUNDS``,
    ``ENVS_WITH_TASK_TYPE``).
``sudoku``
    Sudoku-specific grid parsing and partial-credit reward logic.
``rewards``
    :func:`parse_response` and the main :func:`reward_response` dispatcher.
``hf``
    :func:`evaluate_one` / :func:`evaluate_env` for HuggingFace models.
``api``
    :func:`evaluate_one_api` / :func:`evaluate_env_api` for OpenAI-compatible
    API endpoints.
``vllm``
    :func:`evaluate_one_vllm` / :func:`evaluate_env_vllm` for vLLM inference,
    plus :class:`VLLMModelWrapper` and :func:`_generate_with_vllm`.

Backward Compatibility
----------------------
All symbols that were previously exported from the monolithic
``src/evaluation.py`` are re-exported here, so existing ``from src.evaluation
import …`` statements continue to work without modification.
"""

# -- Constants & actor mapping -----------------------------------------------
from ._constants import (
    ACTOR_MAPPING,
    AGENTGYM_MAX_ROUNDS,
    AGENTGYM_PORTS,
    ENVS_WITH_TASK_TYPE,
)

# -- Sudoku utilities --------------------------------------------------------
from .sudoku import (
    extract_sudoku_answer,
    parse_sudoku_grid,
    sudoku_reward,
)

# -- Reward helpers ----------------------------------------------------------
from .rewards import (
    parse_response,
    reward_response,
)

# -- HuggingFace model evaluation --------------------------------------------
from .hf import (
    evaluate_env,
    evaluate_one,
)

# -- OpenAI-compatible API evaluation ----------------------------------------
from .api import (
    evaluate_env_api,
    evaluate_one_api,
)

# -- vLLM evaluation ---------------------------------------------------------
from .vllm import (
    VLLMModelWrapper,
    _generate_with_vllm,
    evaluate_env_vllm,
    evaluate_one_vllm,
)

__all__ = [
    # Constants
    "ACTOR_MAPPING",
    "AGENTGYM_MAX_ROUNDS",
    "AGENTGYM_PORTS",
    "ENVS_WITH_TASK_TYPE",
    # Sudoku
    "extract_sudoku_answer",
    "parse_sudoku_grid",
    "sudoku_reward",
    # Rewards
    "parse_response",
    "reward_response",
    # HuggingFace
    "evaluate_one",
    "evaluate_env",
    # API
    "evaluate_one_api",
    "evaluate_env_api",
    # vLLM
    "VLLMModelWrapper",
    "_generate_with_vllm",
    "evaluate_one_vllm",
    "evaluate_env_vllm",
]

