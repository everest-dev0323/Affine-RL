"""
TiG Reward Function for OpenRLHF

This reward function computes binary rewards based on action matching
for the Think in Games (TiG) training approach.
"""

import asyncio
import sys
from pathlib import Path

import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import reward_response, parse_response


def reward_func(queries, prompts, labels, **kwargs):
    """
    Compute TiG rewards for generated responses.
    
    Args:
        queries: List[str] - Full text (prompt + response)
        prompts: List[str] - Original prompts only
        labels: List[str] - Ground truth actions (from --label_key)
        **kwargs: Additional info (env, task_id, etc.)
    
    Returns:
        dict with rewards, scores, and extra_logs
    """
    batch_size = len(queries)
    rewards = []
    
    for i in range(batch_size):
        # Extract the response (remove prompt from full query)
        full_text = queries[i]
        prompt_text = prompts[i]
        
        # Response is everything after the prompt
        response = full_text[len(prompt_text):].strip()
        ground_truth = labels[i] if labels else ""
        
        # Parse response to extract action (handles <think> tags)
        parsed_response = parse_response(response)
        
        # Simple equality check for reward
        # For more complex environments, use reward_response async function
        if parsed_response.strip() == ground_truth.strip():
            reward = 1.0
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    return {
        "rewards": reward_tensor,  # Used for advantage calculation
        "scores": reward_tensor,   # Used for dynamic filtering (DAPO)
        "extra_logs": {
            "mean_reward": reward_tensor.mean().item(),
            "reward_std": reward_tensor.std().item() if len(rewards) > 1 else 0.0,
            "num_correct": sum(rewards),
            "accuracy": sum(rewards) / len(rewards) if rewards else 0.0,
        },
    }
