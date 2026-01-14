#!/usr/bin/env python3
"""
Evaluation module for affine environments.

Provides evaluate_one and evaluate_env functions for evaluating models on various tasks.
"""

import time
import sys
import os
import importlib
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Load environment variables
load_dotenv()

# Add submodule paths for environment imports
submodule_base = os.path.join(
    os.path.dirname(__file__), "../AffineFoundation/affinetes"
)
sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, "affinetes/templates")))
# sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, 'affine')))
sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, "environments/trace")))
sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, "environments/openspiel")))
# sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, 'primeintellect/cde')))
# sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, 'primeintellect/lgc')))
# sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, 'primeintellect/mth')))
# sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, 'primeintellect/sci')))
sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, "environments/primeintellect/lgc-v2")))
sys.path.insert(0, os.path.abspath(os.path.join(submodule_base, "environments/arc_gen")))

# from AffineFoundation.affinetes.environments.affine.env import Actor as AffineActor
# from AffineFoundation.affinetes.environments.primeintellect.cde.env import Actor as CDEActor
# from AffineFoundation.affinetes.environments.primeintellect.lgc.env import Actor as LGCActor
# from AffineFoundation.affinetes.environments.primeintellect.mth.env import Actor as MTHActor
# from AffineFoundation.affinetes.environments.primeintellect.sci.env import Actor as SCIActor
lgc2module = importlib.import_module(
    "AffineFoundation.affinetes.environments.primeintellect.lgc-v2.env"
)
LGC2Actor = lgc2module.Actor

# AgentGym imports
# from agentenv.envs import (
#     AlfWorldTask,
#     BabyAITask,
#     SciworldTask,
#     TextCraftTask,
#     WebshopTask
# )
# from agentenv.controller import Agent, ChatMLTemplate

# swe_module = importlib.import_module('AffineFoundation.affinetes.environments.SWE-bench_Pro-os.env')
# SWEActor = swe_module.Actor


from AffineFoundation.affinetes.environments.trace.env import Actor as PrintActor
from AffineFoundation.affinetes.environments.openspiel.env import Actor as GameActor
from AffineFoundation.affinetes.environments.arc_gen.env import Actor as ArcGenActor

from src import get_base_task_name


# Environment name to Actor class mapping (for Affine/PrimeIntellect/SWE)
ACTOR_MAPPING = {
    # Affine environments
    # 'sat': AffineActor,
    # 'abd': AffineActor,
    # 'abd-v2': AffineActor,
    # 'ded': AffineActor,
    # 'ded-v2': AffineActor,
    # Prime Intellect environments
    # 'sci': SCIActor,
    # 'mth': MTHActor,
    # 'lgc': LGCActor,
    "lgc-v2": LGC2Actor,
    # 'cde': CDEActor,
    # # SWE-benchmark
    # 'swe-pro': SWEActor,
    "print": PrintActor,
    "game": GameActor,
    "arc-gen": ArcGenActor,
}

# AgentGym task class mapping
# AGENTGYM_TASK_MAPPING = {
#     'webshop': WebshopTask,
#     'alfworld': AlfWorldTask,
#     'babyai': BabyAITask,
#     'sciworld': SciworldTask,
#     'textcraft': TextCraftTask,
# }

# AgentGym environment port mapping (from .env)
AGENTGYM_PORTS = {
    "webshop": int(os.getenv("AGENTGYM_WEBSHOP_PORT", 36001)),
    "alfworld": int(os.getenv("AGENTGYM_ALFWORLD_PORT", 36002)),
    "sciworld": int(os.getenv("AGENTGYM_SCIWORLD_PORT", 36003)),
    "textcraft": int(os.getenv("AGENTGYM_TEXTCRAFT_PORT", 36004)),
    "babyai": int(os.getenv("AGENTGYM_BABYAI_PORT", 36005)),
}

# AgentGym default max rounds per environment
AGENTGYM_MAX_ROUNDS = {
    "webshop": 10,
    "alfworld": 30,
    "babyai": 20,
    "sciworld": 20,
    "textcraft": 20,
}

# Environments that require task_type parameter
ENVS_WITH_TASK_TYPE = {"abd-v2", "ded-v2", "sat", "abd", "ded"}


# def _evaluate_agentgym_one(
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     env: str,
#     task_id: int,
#     max_rounds: Optional[int] = None,
#     timeout: int = 2400,
#     env_server_host: str = "localhost",
# ) -> Dict[str, Any]:
#     """
#     Evaluate a model on a single AgentGym task.

#     Args:
#         model: HuggingFace AutoModelForCausalLM model
#         tokenizer: Tokenizer for the model
#         env: AgentGym environment name (alfworld, sciworld, textcraft, webshop, babyai)
#         task_id: Task ID to evaluate
#         max_rounds: Maximum interaction rounds (defaults to environment-specific value)
#         timeout: Timeout for environment server calls
#         env_server_host: Host for environment server (default: localhost)

#     Returns:
#         dict with evaluation results
#     """
#     start_time = time.time()

#     env_lower = env.lower()

#     # Get task class and port
#     task_cls = AGENTGYM_TASK_MAPPING.get(env_lower)
#     port = AGENTGYM_PORTS.get(env_lower)

#     if task_cls is None or port is None:
#         raise ValueError(f"Unknown AgentGym environment: {env}. Available: {list(AGENTGYM_TASK_MAPPING.keys())}")

#     # Use default max_rounds if not specified
#     if max_rounds is None:
#         max_rounds = AGENTGYM_MAX_ROUNDS.get(env_lower, 30)

#     # Set up environment args
#     env_server_base = f"http://{env_server_host}:{port}"
#     env_args = {
#         "env_server_base": env_server_base,
#         "data_len": 200,
#         "timeout": timeout,
#     }

#     # Create task and agent
#     task = task_cls(client_args=env_args, n_clients=1)
#     agent = Agent(model=model, tokenizer=tokenizer, chat_template=ChatMLTemplate())

#     # Generate experience for single task
#     generation_config = GenerationConfig(
#         max_length=4096,
#         do_sample=False,
#         temperature=1.0,
#     )

#     experiences = task.generate_experience(
#         agent=agent,
#         idxs=[task_id],
#         generation_config=generation_config,
#         max_rounds=max_rounds,
#     )

#     if not experiences:
#         return {
#             "score": 0.0,
#             "success": False,
#             "time_taken": time.time() - start_time,
#             "rollout": [],
#         }

#     exp = experiences[0]
#     reward = exp.reward

#     # Convert conversation to rollout format
#     rollout = []
#     for msg in exp.conversation:
#         role = "assistant" if msg["from"] == "gpt" else "user"
#         rollout.append({
#             "role": role,
#             "content": msg["value"]
#         })

#     # Determine success (reward == 1 or reward == 100 depending on env)
#     success = reward == 1.0 or reward == 100.0

#     return {
#         "score": float(reward),
#         "success": success,
#         "time_taken": time.time() - start_time,
#         "rollout": rollout,
#     }


async def evaluate_one(
    model: AutoModelForCausalLM,
    env: str,
    task_id: int,
    tokenizer: Optional[AutoTokenizer] = None,
    seed: Optional[int] = None,
    max_rounds: Optional[int] = None,
    env_server_host: str = "localhost",
) -> Dict[str, Any]:
    """
    Evaluate a model on a single task.

    Args:
        model: HuggingFace AutoModelForCausalLM model
        env: Environment name. Supported:
            - Affine: 'sat', 'abd', 'abd-v2', 'ded', 'ded-v2'
            - PrimeIntellect: 'cde', 'mth', 'lgc', 'sci'
            - SWE-bench: 'swe-pro'
            - AgentGym: 'alfworld', 'sciworld', 'textcraft', 'webshop', 'babyai'
        task_id: Task ID to evaluate
        tokenizer: Optional tokenizer. If not provided, will be loaded from model config.
        seed: Optional random seed for reproducibility (not used for AgentGym)
        max_rounds: Maximum interaction rounds for AgentGym environments
        env_server_host: Host for AgentGym environment servers (default: localhost)

    Returns:
        dict with keys:
            - score: float (evaluation score)
            - success: bool (whether task was successful)
            - time_taken: float (seconds taken)
            - rollout: list of dict with 'role' and 'content' keys (conversation history)
    """
    # Get tokenizer from model if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    env_lower = env.lower()

    # # Check if this is an AgentGym environment
    # if env_lower in AGENTGYM_TASK_MAPPING:
    #     return _evaluate_agentgym_one(
    #         model=model,
    #         tokenizer=tokenizer,
    #         env=env,
    #         task_id=task_id,
    #         max_rounds=max_rounds,
    #         env_server_host=env_server_host,
    #     )

    # Handle Affine/PrimeIntellect/SWE environments
    actor_cls = ACTOR_MAPPING.get(env_lower)
    # if actor_cls is None:
    #     all_envs = list(ACTOR_MAPPING.keys()) + list(AGENTGYM_TASK_MAPPING.keys())
    #     raise ValueError(f"Unknown environment: {env}. Available: {all_envs}")

    # Initialize actor
    actor = actor_cls()

    # Build evaluation kwargs
    eval_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "task_id": task_id,
    }

    # Add task_type for specific environments
    if env_lower in ENVS_WITH_TASK_TYPE:
        eval_kwargs["task_type"] = get_base_task_name(env)

    # Add seed if provided
    if seed is not None:
        eval_kwargs["seed"] = seed

    if env_lower == "swe-pro":
        eval_kwargs["max_iterations"] = 1

    # Run evaluation
    result = await actor.local_evaluate(**eval_kwargs)

    # Extract rollout from result
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
    """
    Evaluate a model on multiple tasks in an environment.

    Args:
        model: HuggingFace AutoModelForCausalLM model
        env: Environment name. Supported:
            - Affine: 'sat', 'abd', 'abd-v2', 'ded', 'ded-v2'
            - PrimeIntellect: 'cde', 'mth', 'lgc', 'sci'
            - SWE-bench: 'swe-pro'
            - AgentGym: 'alfworld', 'sciworld', 'textcraft', 'webshop', 'babyai'
        task_ids: List of task IDs to evaluate
        tokenizer: Optional tokenizer. If not provided, will be loaded from model config.
        seed: Optional random seed for reproducibility (not used for AgentGym)
        max_rounds: Maximum interaction rounds for AgentGym environments
        env_server_host: Host for AgentGym environment servers (default: localhost)

    Returns:
        dict with keys:
            - score: float (average score across all tasks)
            - extra: list of evaluate_one results for each task
    """
    # Get tokenizer from model if not provided
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

    # Calculate average score
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    return {
        "score": avg_score,
        "extra": results,
    }


async def reward_response(env: str, task_id: int, response: str, ground_truth: str) -> float:
    """
    Calculate reward for a response based on the environment type.

    Args:
        env: Environment name ('game', 'lgc-v2', 'print')
        task_id: Task ID for the evaluation
        response: Model's response
        ground_truth: Expected answer (used for 'game' and 'print')

    Returns:
        Reward score (1.0 for correct, 0.0 for incorrect)
    """
    env_lower = env.lower()
    

    if env_lower == "lgc-v2":
        # Use LogicTaskV2's evaluate method with verifier
        # logic_task_v2 module is in sys.path (added at top of file)
        from logic_task_v2 import LogicTaskV2  # noqa: E402

        logic_task = LogicTaskV2()

        # Generate challenge to get task metadata (task_type and game_data)
        challenge = await logic_task.generate_dataset_by_id(task_id)

        # Use the evaluate method which handles verifier logic
        return await logic_task.evaluate(response, challenge)

    else:
        parsed_response = parse_response(response)
        # Simple equality check for game and print environments
        return 1.0 if parsed_response.strip() == ground_truth.strip() else 0.0
    

def parse_response(text: str) -> str:
    """Parse response, removing <think> tags if present"""
    if "<think>" in text:
        if "</think>" not in text:
            return ""
        text = text.split("</think>")[-1].strip()
    return text.strip()
