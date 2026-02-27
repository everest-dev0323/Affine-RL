"""vLLM-based evaluation for affine environments.

Uses a locally loaded `vllm.LLM` instance for high-throughput, low-latency
inference during evaluation.  Drop-in replacement for :mod:`src.evaluation.hf`
when GPU memory is shared between training and inference.

Public API
----------
- :class:`VLLMModelWrapper` — thin HF-compatible shim around a vLLM LLM.
- :func:`_generate_with_vllm` — low-level text generation helper.
- :func:`evaluate_one_vllm` — evaluate a single task with vLLM.
- :func:`evaluate_env_vllm` — evaluate multiple tasks with vLLM.
"""

import random
import time
import traceback
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from ._constants import ACTOR_MAPPING


# ---------------------------------------------------------------------------
# Low-level generation helper
# ---------------------------------------------------------------------------


def _generate_with_vllm(
    llm,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 16384,
    temperature: float = 0.7,
) -> str:
    """Generate a single response using a vLLM ``LLM`` instance.

    Applies the tokenizer's chat template before sampling so that model-specific
    special tokens (e.g. ``<|im_start|>``) are inserted correctly.

    Args:
        llm: A ``vllm.LLM`` instance.
        tokenizer: Matching tokenizer for applying the chat template.
        prompt: User-turn prompt text.
        max_new_tokens: Token budget for the generated response.
        temperature: Sampling temperature (higher → more diverse).

    Returns:
        Raw generated text (not including the prompt).
    """
    from vllm import SamplingParams

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    outputs = llm.generate([text], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text


# ---------------------------------------------------------------------------
# HF-compatible model wrapper
# ---------------------------------------------------------------------------


class VLLMModelWrapper:
    """Thin wrapper making a ``vllm.LLM`` compatible with the HF ``generate`` API.

    Some environment actors (e.g. ``game``) call ``model.generate(input_ids, …)``
    directly.  This wrapper decodes those token IDs back to text, runs vLLM, and
    re-encodes the combined sequence so the actor receives a tensor in the
    expected format.

    Args:
        llm: A ``vllm.LLM`` instance.
        tokenizer: Matching tokenizer for encode/decode round-trips.
    """

    def __init__(self, llm, tokenizer: AutoTokenizer) -> None:
        self._llm = llm
        self._tokenizer = tokenizer
        self.device = "cuda"

    def generate(
        self,
        input_ids,
        eos_token_id=None,
        max_new_tokens: int = 16384,
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        """Generate tokens, returning a tensor shaped like HF ``model.generate``."""
        import torch
        from vllm import SamplingParams

        text = self._tokenizer.decode(input_ids[0], skip_special_tokens=False)
        stop_ids = [eos_token_id] if eos_token_id else [self._tokenizer.eos_token_id]

        sampling_params = SamplingParams(
            temperature=max(temperature, 1e-3),
            max_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
        )

        outputs = self._llm.generate([text], sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        generated_ids = self._tokenizer.encode(generated_text, add_special_tokens=False)
        full_ids = input_ids[0].tolist() + generated_ids
        return torch.tensor([full_ids])


# ---------------------------------------------------------------------------
# Single-task vLLM evaluation
# ---------------------------------------------------------------------------


async def evaluate_one_vllm(
    llm,
    env: str,
    task_id: int,
    tokenizer: Optional[AutoTokenizer] = None,
    seed: Optional[int] = None,
    max_rounds: Optional[int] = None,
    env_server_host: str = "localhost",
    max_new_tokens: int = 16384,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Evaluate a vLLM model on a single environment task.

    Supports **single-turn** environments (``lgc-v2``, ``print``, ``arc-gen``)
    and the **multi-turn** ``game`` environment.  Single-turn environments
    generate one response and evaluate it directly.  The ``game`` environment
    uses :class:`VLLMModelWrapper` to interface with the actor's
    ``local_evaluate`` method.

    Args:
        llm: A ``vllm.LLM`` instance.
        env: Environment name (``"lgc-v2"``, ``"print"``, ``"arc-gen"``,
            ``"game"``).
        task_id: Numeric task ID within the environment.
        tokenizer: Tokenizer (required — no auto-load for vLLM).
        seed: Optional random seed (used by ``arc-gen``).
        max_rounds: Maximum interaction rounds (``game`` environment only).
        env_server_host: Hostname for environment servers.
        max_new_tokens: Token budget for generation (default 16 384).
        temperature: Sampling temperature (default 0.7).

    Returns:
        Mapping with keys ``score``, ``success``, ``time_taken``, ``rollout``,
        and ``error`` (``None`` on success).

    Raises:
        ValueError: If ``tokenizer`` is ``None`` or ``env`` is unsupported.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for vLLM evaluation")

    env_lower = env.lower()
    actor_cls = ACTOR_MAPPING.get(env_lower)
    if actor_cls is None:
        raise ValueError(
            f"Unknown environment: {env!r}. Available: {sorted(ACTOR_MAPPING)}"
        )

    actor = actor_cls()
    start = time.time()
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Single-turn environments
    # ------------------------------------------------------------------
    if env_lower in ("lgc-v2", "print", "arc-gen"):
        if env_lower == "lgc-v2":
            challenge = await actor.logic_task.generate(task_id=task_id)
        elif env_lower == "print":
            challenge = await actor.trace_task.generate(task_id=task_id)
        else:  # arc-gen
            actual_seed = seed if seed is not None else random.randint(0, 99_999_999)
            challenge = await actor.arc_task.generate(task_id=task_id, seed=actual_seed)

        try:
            resp = _generate_with_vllm(
                llm, tokenizer, challenge.prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            resp = None
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        score = 0.0
        if resp is not None:
            try:
                if env_lower == "lgc-v2":
                    score = float(await actor.logic_task.evaluate(resp, challenge))
                elif env_lower == "print":
                    score = float(await actor.trace_task.evaluate(resp, challenge))
                else:
                    score = float(await actor.arc_task.evaluate(resp, challenge))
            except Exception as exc:
                error = f"Evaluation error: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        rollout = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp},
        ]
        return {
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "rollout": rollout,
            "error": error,
        }

    # ------------------------------------------------------------------
    # Multi-turn game environment
    # ------------------------------------------------------------------
    if env_lower == "game":
        wrapper = VLLMModelWrapper(llm, tokenizer)
        result = await actor.local_evaluate(
            model=wrapper,
            tokenizer=tokenizer,
            task_id=task_id,
            seed=seed,
            temperature=temperature,
        )
        rollout = []
        if "extra" in result and "conversation" in result["extra"]:
            rollout = result["extra"]["conversation"]
        return {
            "score": result.get("score", 0.0),
            "success": result.get("success", False),
            "time_taken": result.get("time_taken", 0.0),
            "rollout": rollout,
            "error": None,
        }

    raise ValueError(f"Environment {env!r} is not supported for vLLM evaluation")


# ---------------------------------------------------------------------------
# Multi-task vLLM evaluation
# ---------------------------------------------------------------------------


async def evaluate_env_vllm(
    llm,
    env: str,
    task_ids: List[int],
    tokenizer: Optional[AutoTokenizer] = None,
    seed: Optional[int] = None,
    max_rounds: Optional[int] = None,
    env_server_host: str = "localhost",
    max_new_tokens: int = 16384,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Evaluate a vLLM model on multiple tasks within one environment.

    Args:
        llm: A ``vllm.LLM`` instance.
        env: Environment name (see :func:`evaluate_one_vllm`).
        task_ids: Ordered list of task IDs to evaluate.
        tokenizer: Tokenizer (required).
        seed: Optional random seed forwarded to each call.
        max_rounds: Maximum rounds for ``game`` environment.
        env_server_host: Hostname for environment servers.
        max_new_tokens: Token budget for generation.
        temperature: Sampling temperature.

    Returns:
        Mapping with keys:

        - ``score`` (float): Mean score across all tasks.
        - ``extra`` (list[dict]): Per-task :func:`evaluate_one_vllm` results.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for vLLM evaluation")

    results = []
    for task_id in task_ids:
        result = await evaluate_one_vllm(
            llm=llm,
            env=env,
            task_id=task_id,
            tokenizer=tokenizer,
            seed=seed,
            max_rounds=max_rounds,
            env_server_host=env_server_host,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        results.append(result)
        print("-" * 60)
        print(f"task_id={task_id}  score={result['score']:.4f}")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {"score": avg_score, "extra": results}

