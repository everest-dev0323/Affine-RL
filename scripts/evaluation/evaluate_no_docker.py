#!/usr/bin/env python3
"""
Universal evaluation script for affine environments using affinetes

Connects to locally deployed environment services via URL mode.

Usage:
    ./evaluate_local_model.py --env sat --model your-model --base-url http://host.docker.internal:30000/v1 --samples 10
    ./evaluate_local_model.py --env alfworld --task-id 2 --model deepseek-ai/DeepSeek-V3 --base-url https://llm.chutes.ai/v1 --samples 5
"""
import asyncio
import argparse
import sys
import os
import importlib
import json
from typing import Optional
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Add the submodule root to PYTHONPATH
submodule_path = os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/affine')
sys.path.insert(0, os.path.abspath(submodule_path))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/primeintellect/cde')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/primeintellect/lgc')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/primeintellect/mth')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/primeintellect/sci')))


# Load environment variables
load_dotenv()

from src.config import ENVIRONMENT_NAMES
from src import get_base_task_name

from AffineFoundation.affinetes.environments.affine.env import Actor as AffineActor
from AffineFoundation.affinetes.environments.primeintellect.cde.env import Actor as CDEActor
from AffineFoundation.affinetes.environments.primeintellect.lgc.env import Actor as LGCActor
from AffineFoundation.affinetes.environments.primeintellect.mth.env import Actor as MTHActor
from AffineFoundation.affinetes.environments.primeintellect.sci.env import Actor as SCIActor

swe_module = importlib.import_module('AffineFoundation.affinetes.environments.SWE-bench_Pro-os.env')
SWEActor = swe_module.Actor

# Environment name to port mapping (from .env)
Actor_Mapping = {
    # Affine environments
    'sat': AffineActor,
    'abd': AffineActor,
    'abd-v2': AffineActor,
    'ded': AffineActor,
    'ded-v2': AffineActor,
    # Affine new environments
    'sci': SCIActor,
    'mth': MTHActor,
    'lgc': LGCActor,
    'cde': CDEActor,
    # SWE-benchmark
    'swe-pro': SWEActor
}

ENVS_WITH_TASK_TYPE = {'abd-v2', 'ded-v2', 'sat'}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate models on affine environments using affinetes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on SAT environment
  %(prog)s --env sat --model your-model --base-url http://host.docker.internal:30000/v1

  # Evaluate on ALFWORLD with specific task
  %(prog)s --env alfworld --task-id 2 --model deepseek-ai/DeepSeek-V3 --base-url https://llm.chutes.ai/v1 --samples 5

Supported environments: """ + ', '.join(ENVIRONMENT_NAMES)
    )

    parser.add_argument('--task-id', type=int,
                        help='Task ID for environment')
    parser.add_argument('--samples', type=int, default=1,
                        help='Number of evaluation samples (default: 1)')
    parser.add_argument('--env-host', default='localhost',
                        help='Environment service host (default: localhost)')
    parser.add_argument('--output', '-o', default='result/result.json',
                        help='Output file path for JSON results (optional)')

    args = parser.parse_args()

    return args


async def main():
    """Main evaluation function"""
    args = parse_args()

    tasks = [
        # {"env": "sat", "task_id": 10},
        # {"env": "abd-v2", "task_id": 20010},
        # {"env": "ded-v2", "task_id": 20010},
        # {"env": "sci", "task_id": 10},
        # {"env": "mth", "task_id": 10},
        # {"env": "lgc", "task_id": 10},
        # {"env": "cde", "task_id": 10},
        # {"env": "alfworld", "task_id": 10},
        # {"env": "sciworld", "task_id": 10},
        # {"env": "textcraft", "task_id": 10},
        # {"env": "babyai", "task_id": 10},
        # {"env": "webshop", "task_id": 10}
        {"env": "swe-pro", "task_id": 0}
    ]
    
    model_name = os.getenv('MODEL_NAME')
    
    print("=" * 60)
    print("Evaluation Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Number of Tasks: {len(tasks)}")
    print("=" * 60)
    
    model_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    results = []
    
    for task in tasks:
        agent = Actor_Mapping[task['env']]()
        # Build evaluate kwargs - only include task_type for specific envs
        eval_kwargs = {
            "model": model,
            "tokenizer": model_tokenizer,
            "task_id": task['task_id'],
        }
        if task['env'] in ENVS_WITH_TASK_TYPE:
            eval_kwargs["task_type"] = get_base_task_name(task['env'])
            
        result = await agent.local_evaluate(**eval_kwargs)
    
        results.append(result)
    
    # Save to JSON file if specified
    if args.output:
        output_path = args.output
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to: {output_path}")
        except Exception as e:
            print(f"\n✗ Failed to save results: {e}")
    

if __name__ == "__main__":
    asyncio.run(main())