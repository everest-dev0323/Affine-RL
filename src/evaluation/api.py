"""OpenAI-compatible API evaluation for affine environments.

Provides :func:`evaluate_one_api` and :func:`evaluate_env_api` for evaluating
models served behind any OpenAI-compatible endpoint (vLLM server, Together AI,
Chutes, etc.) against environments registered in
:data:`~src.evaluation._constants.ACTOR_MAPPING`.
"""

from typing import Any, Dict, List, Optional

from ._constants import ACTOR_MAPPING, ENVS_WITH_TASK_TYPE
from src import get_base_task_name


async def evaluate_one_api(
    model: str,
    base_url: str,
    env: str,
    task_id: int,
    api_key: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate an API-hosted model on a single environment task.

    Args:
        model: Model identifier as expected by the API (e.g.
            ``"deepseek-ai/DeepSeek-V3"``).
        base_url: Base URL for the OpenAI-compatible API, e.g.
            ``"https://llm.chutes.ai/v1"``.
        env: Environment name, e.g. ``"lgc-v2"``, ``"game"``, ``"print"``,
            ``"arc-gen"``.
        task_id: Numeric task ID within the environment.
        api_key: API authentication key.
        seed: Optional random seed forwarded to the environment actor.

    Returns:
        Mapping with keys:

        - ``score`` (float): Task score in [0, 1].
        - ``success`` (bool): Whether the task was completed successfully.
        - ``time_taken`` (float): Wall-clock seconds elapsed.
        - ``rollout`` (list[dict]): Conversation history as
          ``[{"role": ..., "content": ...}, ...]``.

    Raises:
        ValueError: If ``env`` is not registered in ``ACTOR_MAPPING``.
    """
    env_lower = env.lower()
    actor_cls = ACTOR_MAPPING.get(env_lower)
    if actor_cls is None:
        raise ValueError(
            f"Unknown environment: {env!r}. Available: {sorted(ACTOR_MAPPING)}"
        )

    actor = actor_cls()

    eval_kwargs: Dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "task_id": task_id,
    }

    if env_lower in ENVS_WITH_TASK_TYPE:
        eval_kwargs["task_type"] = get_base_task_name(env)
    if seed is not None:
        eval_kwargs["seed"] = seed
    if env_lower == "swe-pro":
        eval_kwargs["max_iterations"] = 1

    result = await actor.evaluate(**eval_kwargs)

    rollout = []
    if "extra" in result and "conversation" in result["extra"]:
        rollout = result["extra"]["conversation"]

    return {
        "score": result.get("score", 0.0),
        "success": result.get("success", False),
        "time_taken": result.get("time_taken", 0.0),
        "rollout": rollout,
    }


async def evaluate_env_api(
    model: str,
    base_url: str,
    api_key: str,
    env: str,
    task_ids: List[int],
    seed: Optional[int] = None,
    max_rounds: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate an API-hosted model on multiple tasks within one environment.

    Iterates over ``task_ids`` sequentially, logging each result, and returns
    the average score together with per-task detail records.

    Args:
        model: Model identifier (see :func:`evaluate_one_api`).
        base_url: API base URL.
        api_key: API authentication key.
        env: Environment name (see :func:`evaluate_one_api`).
        task_ids: Ordered list of task IDs to evaluate.
        seed: Optional random seed forwarded to each call.
        max_rounds: Kept for API consistency; not currently used.

    Returns:
        Mapping with keys:

        - ``score`` (float): Mean score across all tasks.
        - ``extra`` (list[dict]): Per-task :func:`evaluate_one_api` result dicts.
    """
    results = []
    for task_id in task_ids:
        result = await evaluate_one_api(
            model=model,
            base_url=base_url,
            api_key=api_key,
            env=env,
            task_id=task_id,
            seed=seed,
        )
        results.append(result)
        print("-" * 60)
        print(f"task_id={task_id}  score={result['score']:.4f}")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {"score": avg_score, "extra": results}

