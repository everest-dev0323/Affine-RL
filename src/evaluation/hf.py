"""HuggingFace model evaluation for affine environments.

Provides :func:`evaluate_one` and :func:`evaluate_env` for evaluating models
loaded with ``transformers.AutoModelForCausalLM`` against any environment
registered in :data:`~src.evaluation._constants.ACTOR_MAPPING`.
"""

from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ._constants import ACTOR_MAPPING, ENVS_WITH_TASK_TYPE
from src import get_base_task_name


async def evaluate_one(
    model: AutoModelForCausalLM,
    env: str,
    task_id: int,
    tokenizer: Optional[AutoTokenizer] = None,
    seed: Optional[int] = None,
    max_rounds: Optional[int] = None,
    env_server_host: str = "localhost",
) -> Dict[str, Any]:
    """Evaluate a HuggingFace model on a single environment task.

    Args:
        model: A loaded ``AutoModelForCausalLM`` instance.
        env: Environment name, e.g. ``"lgc-v2"``, ``"game"``, ``"print"``,
            ``"arc-gen"``.
        task_id: Numeric task ID within the environment.
        tokenizer: Tokenizer for the model.  Loaded from ``model.config``
            automatically when omitted.
        seed: Optional random seed for reproducibility.
        max_rounds: Maximum interaction rounds (game environments only).
        env_server_host: Hostname for environment servers (default
            ``"localhost"``).

    Returns:
        Mapping with keys:

        - ``score`` (float): Task score in [0, 1].
        - ``success`` (bool): Whether the task was completed successfully.
        - ``time_taken`` (float): Wall-clock seconds elapsed.
        - ``rollout`` (list[dict]): Conversation history as
          ``[{"role": ..., "content": ...}, ...]``.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    env_lower = env.lower()
    actor_cls = ACTOR_MAPPING.get(env_lower)
    if actor_cls is None:
        raise ValueError(
            f"Unknown environment: {env!r}. Available: {sorted(ACTOR_MAPPING)}"
        )

    actor = actor_cls()

    eval_kwargs: Dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
        "task_id": task_id,
    }

    if env_lower in ENVS_WITH_TASK_TYPE:
        eval_kwargs["task_type"] = get_base_task_name(env)
    if seed is not None:
        eval_kwargs["seed"] = seed
    if env_lower == "swe-pro":
        eval_kwargs["max_iterations"] = 1

    result = await actor.local_evaluate(**eval_kwargs)

    rollout = []
    if "extra" in result and "conversation" in result["extra"]:
        rollout = result["extra"]["conversation"]

    return {
        "score": result.get("score", 0.0),
        "success": result.get("success", False),
        "time_taken": result.get("time_taken", 0.0),
        "rollout": rollout,
    }


async def evaluate_env(
    model: AutoModelForCausalLM,
    env: str,
    task_ids: List[int],
    tokenizer: Optional[AutoTokenizer] = None,
    seed: Optional[int] = None,
    max_rounds: Optional[int] = None,
    env_server_host: str = "localhost",
) -> Dict[str, Any]:
    """Evaluate a HuggingFace model on multiple tasks within one environment.

    Iterates over ``task_ids`` sequentially, logging each result, and returns
    the average score together with per-task detail records.

    Args:
        model: A loaded ``AutoModelForCausalLM`` instance.
        env: Environment name (see :func:`evaluate_one`).
        task_ids: Ordered list of task IDs to evaluate.
        tokenizer: Tokenizer for the model (auto-loaded when omitted).
        seed: Optional random seed forwarded to each :func:`evaluate_one` call.
        max_rounds: Maximum interaction rounds (game environments only).
        env_server_host: Hostname for environment servers.

    Returns:
        Mapping with keys:

        - ``score`` (float): Mean score across all tasks.
        - ``extra`` (list[dict]): Per-task :func:`evaluate_one` result dicts.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    results = []
    for task_id in task_ids:
        result = await evaluate_one(
            model=model,
            env=env,
            task_id=task_id,
            tokenizer=tokenizer,
            seed=seed,
            max_rounds=max_rounds,
            env_server_host=env_server_host,
        )
        results.append(result)
        print("-" * 60)
        print(f"task_id={task_id}  score={result['score']:.4f}")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {"score": avg_score, "extra": results}

