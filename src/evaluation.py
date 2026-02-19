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


async def evaluate_one_api(
    model: str,
    base_url: str,
    env: str,
    task_id: int,
    api_key: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model on a single task.

    Args:
        model: Model name
        base_url: Base URL for the API
        api_key: API key for the model
        env: Environment name. Supported:
            - Affine: 'sat', 'abd', 'abd-v2', 'ded', 'ded-v2'
            - PrimeIntellect: 'cde', 'mth', 'lgc', 'sci'
            - SWE-bench: 'swe-pro'
        task_id: Task ID to evaluate
        seed: Optional random seed for reproducibility (not used for AgentGym)

    Returns:
        dict with keys:
            - score: float (evaluation score)
            - success: bool (whether task was successful)
            - time_taken: float (seconds taken)
            - rollout: list of dict with 'role' and 'content' keys (conversation history)
    """
    env_lower = env.lower()

    actor_cls = ACTOR_MAPPING.get(env_lower)

    # Initialize actor
    actor = actor_cls()

    # Build evaluation kwargs
    eval_kwargs = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
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
    result = await actor.evaluate(**eval_kwargs)

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
        print("-"*60)
        print(f"task id: {task_id}, score: {result['score']}")

    # Calculate average score
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    return {
        "score": avg_score,
        "extra": results,
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
    """
    Evaluate a model on multiple tasks in an environment.

    Args:
        model: Model name
        base_url: Base URL for the API
        api_key: API key for the model
        env: Environment name. Supported:
            - Affine: 'sat', 'abd', 'abd-v2', 'ded', 'ded-v2'
            - PrimeIntellect: 'cde', 'mth', 'lgc', 'sci'
            - SWE-bench: 'swe-pro'
            - AgentGym: 'alfworld', 'sciworld', 'textcraft', 'webshop', 'babyai'
        task_ids: List of task IDs to evaluate
        seed: Optional random seed for reproducibility (not used for AgentGym)
        max_rounds: Maximum interaction rounds for AgentGym environments

    Returns:
        dict with keys:
            - score: float (average score across all tasks)
            - extra: list of evaluate_one results for each task
    """
    results = []
    for task_id in task_ids:
        result = await evaluate_one_api(
            model=model,  # Model name
            base_url=base_url,  # Base URL for the API
            api_key=api_key,  # API key for the model
            task_id=task_id,  # Task ID to evaluate
            seed=seed,  # Optional random seed for reproducibility
            max_rounds=max_rounds,  # Maximum interaction rounds for AgentGym environments
        )
        results.append(result)
        print("-"*60)
        print(f"task id: {task_id}, score: {result['score']}")

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
    parsed_response = parse_response(response)
    
    if env_lower == "lgc-v2":
        # Use LogicTaskV2's evaluate method with verifier
        # logic_task_v2 module is in sys.path (added at top of file)
        from logic_task_v2 import LogicTaskV2  # noqa: E402

        logic_task = LogicTaskV2()

        # Generate challenge to get task metadata (task_type and game_data)
        challenge = await logic_task.generate_dataset_by_id(task_id)

        if challenge.extra.get("task_type") == "sudoku":
            # First check if it's a perfect solution
            score = await logic_task.evaluate(parsed_response, challenge)
            if score:
                return 1.0  # Perfect solution

            # Not perfect - calculate granular reward
            # Extract answer from response
            answer_str = extract_sudoku_answer(parsed_response)

            # Parse answer and puzzle grids
            answer_grid = parse_sudoku_grid(answer_str)
            puzzle_str = challenge.extra.get("metadata", {}).get("puzzle", "")
            puzzle_grid = parse_sudoku_grid(puzzle_str)

            if answer_grid is None or puzzle_grid is None:
                return 0.0  # Cannot parse grids

            return sudoku_reward(answer_grid, puzzle_grid)

        # Use the evaluate method which handles verifier logic
        return await logic_task.evaluate(parsed_response, challenge)

    else:
        # Simple equality check for game and print environments
        return 1.0 if parsed_response.strip() == ground_truth.strip() else 0.0
    

def parse_response(text: str) -> str:
    """Parse response, removing <think> tags if present"""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()


def sudoku_reward(answer_grid: list, puzzle_grid: list) -> float:
    """
    Calculate reward for a sudoku answer based on validity checks.

    Args:
        answer_grid: 9x9 grid of integers (the model's answer)
        puzzle_grid: 9x9 grid with original puzzle (0 for empty cells)

    Returns:
        float: Reward between 0 and 1
            - 0 if doesn't respect original puzzle
            - Otherwise based on row/col/box validity
    """
    # Check if answer respects original puzzle (pre-filled cells must match)
    for i in range(9):
        for j in range(9):
            if puzzle_grid[i][j] != 0:
                if answer_grid[i][j] != puzzle_grid[i][j]:
                    return 0.0  # Violates original puzzle

    def is_valid_group(group: list) -> bool:
        """Check if a group contains exactly 1-9 with no duplicates"""
        return sorted(group) == list(range(1, 10))

    # Count valid rows
    valid_rows = sum(1 for row in answer_grid if is_valid_group(row))
    row_score = valid_rows / 9

    # Count valid columns
    valid_cols = 0
    for j in range(9):
        col = [answer_grid[i][j] for i in range(9)]
        if is_valid_group(col):
            valid_cols += 1
    col_score = valid_cols / 9

    # Count valid 3x3 boxes
    valid_boxes = 0
    for box_row in range(3):
        for box_col in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(answer_grid[box_row * 3 + i][box_col * 3 + j])
            if is_valid_group(box):
                valid_boxes += 1
    box_score = valid_boxes / 9

    # Final reward: average of row, col, box validity scores
    return (3*row_score + 3*col_score + 4*box_score) / 10


def extract_sudoku_answer(response: str) -> str:
    """
    Extract the sudoku grid from model response.

    Looks for ```...``` block containing the grid, or 9 lines of 9 digits.

    Args:
        response: Model's response text

    Returns:
        Extracted grid string
    """
    import re

    # First try: find 9 lines of 9 digits anywhere in the response
    lines = response.strip().split('\n')
    grid_lines = []
    for line in lines:
        # Remove backticks and other common formatting
        clean_line = line.replace('`', '').strip()
        # Extract digits from line
        digits = re.findall(r'[1-9]', clean_line)
        if len(digits) == 9:
            grid_lines.append(''.join(digits))

    if len(grid_lines) >= 9:
        return '\n'.join(grid_lines[:9])

    # Fallback: try to find code block
    pattern = r"```([\s\S]*?)```"
    matches = re.findall(pattern, response)

    if matches:
        # Get the last code block
        grid_str = matches[-1].strip()
        # Remove language identifier if present
        if grid_str and '\n' in grid_str:
            first_line = grid_str.split('\n')[0]
            if not re.search(r'[1-9]', first_line):
                grid_str = '\n'.join(grid_str.split('\n')[1:])
        return grid_str

    return response.strip()


def parse_sudoku_grid(grid_str: str) -> list:
    """
    Parse grid string into 9x9 list of integers.

    Args:
        grid_str: String representation of grid

    Returns:
        9x9 list of integers, or None if invalid
    """
    if not grid_str:
        return None

    lines = grid_str.strip().split('\n')
    grid = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Extract digits (1-9) and dots/zeros
        row = []
        for char in line:
            if char.isdigit():
                row.append(int(char))
            elif char == '.':
                row.append(0)
            # Skip other characters (spaces, separators, etc.)

        if len(row) == 9:
            grid.append(row)

    if len(grid) != 9:
        return None

    return grid


# ============================================================================
# vLLM-based evaluation functions for faster inference
# ============================================================================

def _generate_with_vllm(
    llm,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 16384,
    temperature: float = 0.7,
) -> str:
    """
    Generate text using vLLM LLM instance.

    Args:
        llm: vLLM LLM instance
        tokenizer: Tokenizer for chat template
        prompt: User prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text response
    """
    from vllm import SamplingParams

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    # Configure sampling
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Generate
    outputs = llm.generate([text], sampling_params, use_tqdm=False)

    return outputs[0].outputs[0].text


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
    """
    Evaluate using vLLM LLM instance on a single task.

    This is a faster alternative to evaluate_one() that uses vLLM for inference.

    Args:
        llm: vLLM LLM instance (from vllm.LLM)
        env: Environment name. Supported:
            - 'lgc-v2', 'print', 'game', 'arc-gen'
        task_id: Task ID to evaluate
        tokenizer: Optional tokenizer. Required for vLLM.
        seed: Optional random seed for reproducibility
        max_rounds: Maximum interaction rounds (for game environment)
        env_server_host: Host for environment servers (default: localhost)
        max_new_tokens: Maximum tokens to generate (default: 16384)
        temperature: Sampling temperature (default: 0.7)

    Returns:
        dict with keys:
            - score: float (evaluation score)
            - success: bool (whether task was successful)
            - time_taken: float (seconds taken)
            - rollout: list of dict with 'role' and 'content' keys
    """
    import time

    if tokenizer is None:
        raise ValueError("tokenizer is required for vLLM evaluation")

    env_lower = env.lower()

    # Get actor class
    actor_cls = ACTOR_MAPPING.get(env_lower)
    if actor_cls is None:
        raise ValueError(f"Unknown environment: {env}. Available: {list(ACTOR_MAPPING.keys())}")

    # Initialize actor
    actor = actor_cls()

    start = time.time()

    # Handle single-turn environments (lgc-v2, print, arc-gen)
    if env_lower in ["lgc-v2", "print", "arc-gen"]:
        # Generate challenge
        if env_lower == "lgc-v2":
            challenge = await actor.logic_task.generate(task_id=task_id)
        elif env_lower == "print":
            challenge = await actor.trace_task.generate(task_id=task_id)
        elif env_lower == "arc-gen":
            import random
            actual_seed = seed if seed is not None else random.randint(0, 99_999_999)
            challenge = await actor.arc_task.generate(task_id=task_id, seed=actual_seed)

        # Generate response using vLLM
        try:
            resp = _generate_with_vllm(
                llm, tokenizer, challenge.prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            error = None
        except Exception as e:
            import traceback
            resp = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        # Evaluate
        score = 0.0
        if resp:
            try:
                if env_lower == "lgc-v2":
                    score = await actor.logic_task.evaluate(resp, challenge)
                elif env_lower == "print":
                    score = await actor.trace_task.evaluate(resp, challenge)
                elif env_lower == "arc-gen":
                    score = await actor.arc_task.evaluate(resp, challenge)
            except Exception as e:
                import traceback
                error = f"Evaluation error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        rollout = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp}
        ]

        return {
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "rollout": rollout,
            "error": error,
        }

    # Handle game environment (multi-turn)
    elif env_lower == "game":
        # For game environment, use local_evaluate with a wrapper
        # that provides vLLM generation

        # Create a wrapper class that uses vLLM for generation
        class VLLMModelWrapper:
            """Wrapper to make vLLM LLM instance compatible with HF model interface"""
            def __init__(self, llm, tokenizer):
                self._llm = llm
                self._tokenizer = tokenizer
                self.device = "cuda"

            def generate(self, input_ids, eos_token_id=None, max_new_tokens=16384,
                        max_length=None, temperature=0.7, **kwargs):
                from vllm import SamplingParams
                import torch

                # Decode input_ids to text
                text = self._tokenizer.decode(input_ids[0], skip_special_tokens=False)

                sampling_params = SamplingParams(
                    temperature=temperature if temperature > 0 else 0.01,
                    max_tokens=max_new_tokens,
                    stop_token_ids=[eos_token_id] if eos_token_id else [self._tokenizer.eos_token_id],
                )

                outputs = self._llm.generate([text], sampling_params, use_tqdm=False)

                # Return as tensor matching HF generate output format
                generated_text = outputs[0].outputs[0].text
                generated_ids = self._tokenizer.encode(generated_text, add_special_tokens=False)
                full_ids = input_ids[0].tolist() + generated_ids

                return torch.tensor([full_ids])

        vllm_wrapper = VLLMModelWrapper(llm, tokenizer)

        # Use the existing local_evaluate with wrapper
        result = await actor.local_evaluate(
            model=vllm_wrapper,
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
        }

    else:
        raise ValueError(f"Environment {env} not supported for vLLM evaluation")


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
    """
    Evaluate using vLLM LLM instance on multiple tasks.

    This is a faster alternative to evaluate_env() that uses vLLM for inference.

    Args:
        llm: vLLM LLM instance (from vllm.LLM)
        env: Environment name. Supported:
            - 'lgc-v2', 'print', 'game', 'arc-gen'
        task_ids: List of task IDs to evaluate
        tokenizer: Tokenizer (required for vLLM)
        seed: Optional random seed for reproducibility
        max_rounds: Maximum interaction rounds for game environment
        env_server_host: Host for environment servers (default: localhost)
        max_new_tokens: Maximum tokens to generate (default: 16384)
        temperature: Sampling temperature (default: 0.7)

    Returns:
        dict with keys:
            - score: float (average score across all tasks)
            - extra: list of evaluate_one_vllm results for each task
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
        print(f"task id: {task_id}, score: {result['score']}")

    # Calculate average score
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    return {
        "score": avg_score,
        "extra": results,
    }
