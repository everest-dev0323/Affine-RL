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
import json
from typing import Optional
from dotenv import load_dotenv
import affinetes as af_env

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

from src.config import ENVIRONMENT_NAMES
from src import get_base_task_name

# Environment name to port mapping (from .env)
ENVIRONMENT_PORTS = {
    # Affine environments
    'sat': int(os.getenv('AFFINE_PORT', 37001)),
    'abd': int(os.getenv('AFFINE_PORT', 37001)),
    'abd-v2': int(os.getenv('AFFINE_PORT', 37001)),
    'ded': int(os.getenv('AFFINE_PORT', 37001)),
    'ded-v2': int(os.getenv('AFFINE_PORT', 37001)),
    # Affine new environments
    'sci': int(os.getenv('AFFINE_SCI_PORT', 37002)),
    'mth': int(os.getenv('AFFINE_MTH_PORT', 37003)),
    'lgc': int(os.getenv('AFFINE_LGC_PORT', 37004)),
    'cde': int(os.getenv('AFFINE_CDE_PORT', 37005)),
    # AgentGym environments
    'webshop': int(os.getenv('AGENTGYM_WEBSHOP_PORT', 36001)),
    'alfworld': int(os.getenv('AGENTGYM_ALFWORLD_PORT', 36002)),
    'sciworld': int(os.getenv('AGENTGYM_SCIWORLD_PORT', 36003)),
    'textcraft': int(os.getenv('AGENTGYM_TEXTCRAFT_PORT', 36004)),
    'babyai': int(os.getenv('AGENTGYM_BABYAI_PORT', 36005)),
}

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

    parser.add_argument('--api-key', default='random',
                        help='API key for model service (default: random)')
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
        {"env": "abd-v2", "task_id": 20010},
        # {"env": "ded-v2", "task_id": 20010},
        # {"env": "alfworld", "task_id": 10},
        # {"env": "sciworld", "task_id": 10},
        # {"env": "textcraft", "task_id": 10},
        # {"env": "babyai", "task_id": 10},
        # {"env": "webshop", "task_id": 10}
    ]
    
    model_name = os.getenv('MODEL_NAME')
    model_base_url = os.getenv('MODEL_BASE_URL')
    model_port = os.getenv('MODEL_PORT')
    
    print("=" * 60)
    print("Evaluation Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Model Base URL: {model_base_url}:{model_port}/v1")
    print(f"  Number of Tasks: {len(tasks)}")
    print("=" * 60)

    results = []
    
    for task in tasks:
        env = af_env.load_env(
            mode="url",
            base_url=f"http://localhost:{ENVIRONMENT_PORTS[task['env']]}"
        )
        
        result = await env.evaluate(
            model=model_name,
            base_url=f"{model_base_url}:{model_port}/v1",
            task_type=get_base_task_name(task['env']),
            api_key='No API key',
            task_id=task['task_id'],
        )
    
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