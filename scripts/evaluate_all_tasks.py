#!/usr/bin/env python3
"""
Comprehensive evaluation script for all affine environments.

Samples random tasks from each environment and generates detailed reports
with per-environment and total average scores.

Usage:
    python scripts/evaluate_all_tasks.py
    python scripts/evaluate_all_tasks.py --samples 100
    python scripts/evaluate_all_tasks.py --config config/eval_sampling.yaml
"""
import asyncio
import argparse
import sys
import os
import json
import random
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import affinetes as af_env

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

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

# Default config path
DEFAULT_CONFIG_PATH = "config/eval_sampling.yaml"

# Environments that require task_type parameter (only affine v2 envs)
ENVS_WITH_TASK_TYPE = {'abd-v2', 'ded-v2', 'sat'}


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def sample_task_ids(task_range: Dict, num_samples: int, seed: int) -> List[int]:
    """Sample random task IDs from the given range."""
    min_id = task_range['min']
    max_id = task_range['max']
    all_ids = list(range(min_id, max_id + 1))
    
    # Set seed for reproducibility
    rng = random.Random(seed)
    
    # Sample without replacement
    num_samples = min(num_samples, len(all_ids))
    return sorted(rng.sample(all_ids, num_samples))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models on all affine environments with random sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH,
                        help=f'Config file path (default: {DEFAULT_CONFIG_PATH})')
    parser.add_argument('--samples', type=int,
                        help='Override samples per environment from config')
    parser.add_argument('--seed', type=int,
                        help='Override random seed from config')
    parser.add_argument('--output', '-o',
                        help='Override output file path')
    parser.add_argument('--envs', nargs='+',
                        help='Specific environments to evaluate (default: all)')
    return parser.parse_args()


async def evaluate_environment(
    env_name: str,
    task_ids: List[int],
    model_name: str,
    model_base_url: str,
) -> Dict[str, Any]:
    """Evaluate a single environment with given task IDs."""
    env = af_env.load_env(
        mode="url",
        base_url=f"http://localhost:{ENVIRONMENT_PORTS[env_name]}"
    )
    
    results = []
    scores = []
    
    for task_id in task_ids:
        try:
            # Build evaluate kwargs - only include task_type for specific envs
            eval_kwargs = {
                "model": model_name,
                "base_url": model_base_url,
                "api_key": 'No API key',
                "task_id": task_id,
            }
            if env_name in ENVS_WITH_TASK_TYPE:
                eval_kwargs["task_type"] = get_base_task_name(env_name)

            result = await env.evaluate(**eval_kwargs)
            results.append({
                "task_id": task_id,
                "result": result,
                "score": result.get("score", 0) if isinstance(result, dict) else 0,
                "success": True,
                "error": None
            })
            scores.append(result.get("score", 0) if isinstance(result, dict) else 0)
            print(f"  ✓ {env_name} task {task_id}: score={scores[-1]:.3f}")
        except Exception as e:
            results.append({
                "task_id": task_id,
                "result": None,
                "score": 0,
                "success": False,
                "error": str(e)
            })
            scores.append(0)
            print(f"  ✗ {env_name} task {task_id}: error={str(e)[:50]}")
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    return {
        "env": env_name,
        "num_tasks": len(task_ids),
        "sampled_task_ids": task_ids,
        "average_score": avg_score,
        "results": results,
        "successful_tasks": sum(1 for r in results if r["success"]),
        "failed_tasks": sum(1 for r in results if not r["success"]),
    }


def generate_report(
    env_results: List[Dict],
    config: Dict,
    start_time: datetime,
    end_time: datetime,
    model_name: str,
) -> Dict[str, Any]:
    """Generate comprehensive evaluation report."""
    # Calculate total statistics
    total_tasks = sum(r["num_tasks"] for r in env_results)
    total_successful = sum(r["successful_tasks"] for r in env_results)
    total_failed = sum(r["failed_tasks"] for r in env_results)

    # Calculate weighted average (by number of tasks)
    all_scores = []
    for r in env_results:
        all_scores.extend([res["score"] for res in r["results"]])
    total_avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

    # Per-environment summary
    env_summary = {}
    for r in env_results:
        env_summary[r["env"]] = {
            "average_score": r["average_score"],
            "num_tasks": r["num_tasks"],
            "successful_tasks": r["successful_tasks"],
            "failed_tasks": r["failed_tasks"],
            "sampled_task_ids": r["sampled_task_ids"],
        }

    return {
        "report_metadata": {
            "generated_at": end_time.isoformat(),
            "evaluation_start": start_time.isoformat(),
            "evaluation_end": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "model_name": model_name,
            "config_used": config,
        },
        "summary": {
            "total_average_score": total_avg_score,
            "total_tasks_evaluated": total_tasks,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "num_environments": len(env_results),
        },
        "per_environment": env_summary,
        "detailed_results": env_results,
    }


def print_report_summary(report: Dict):
    """Print a formatted summary of the evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT SUMMARY")
    print("=" * 70)

    meta = report["report_metadata"]
    print(f"\nModel: {meta['model_name']}")
    print(f"Duration: {meta['duration_seconds']:.1f} seconds")

    summary = report["summary"]
    print(f"\n{'Environment':<15} {'Avg Score':>10} {'Tasks':>8} {'Success':>8} {'Failed':>8}")
    print("-" * 55)

    for env_name, env_data in report["per_environment"].items():
        print(f"{env_name:<15} {env_data['average_score']:>10.4f} "
              f"{env_data['num_tasks']:>8} {env_data['successful_tasks']:>8} "
              f"{env_data['failed_tasks']:>8}")

    print("-" * 55)
    print(f"{'TOTAL':<15} {summary['total_average_score']:>10.4f} "
          f"{summary['total_tasks_evaluated']:>8} {summary['total_successful']:>8} "
          f"{summary['total_failed']:>8}")
    print("=" * 70)


async def main():
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI arguments
    samples_per_env = args.samples or config.get('samples_per_env', 50)
    random_seed = args.seed or config.get('random_seed', 42)
    task_ranges = config.get('task_ranges', {})

    # Get model configuration from environment
    model_name = os.getenv('MODEL_NAME', 'unknown-model')
    model_base_url = os.getenv('MODEL_BASE_URL', 'http://localhost')
    model_port = os.getenv('MODEL_PORT', '30000')
    full_model_url = f"{model_base_url}:{model_port}/v1"

    # Determine which environments to evaluate
    envs_to_evaluate = args.envs or list(task_ranges.keys())

    print("=" * 70)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Model URL: {full_model_url}")
    print(f"Samples per environment: {samples_per_env}")
    print(f"Random seed: {random_seed}")
    print(f"Environments: {', '.join(envs_to_evaluate)}")
    print("=" * 70)

    start_time = datetime.now()
    env_results = []

    for env_name in envs_to_evaluate:
        if env_name not in task_ranges:
            print(f"\n⚠ Skipping {env_name}: not in config")
            continue

        print(f"\n>>> Evaluating {env_name}...")

        # Sample task IDs
        # Use different seed per environment for diversity but reproducibility
        env_seed = random_seed + hash(env_name) % 10000
        task_ids = sample_task_ids(task_ranges[env_name], samples_per_env, env_seed)
        print(f"    Sampled {len(task_ids)} tasks: {task_ids[:5]}...{task_ids[-5:]}")

        # Evaluate
        result = await evaluate_environment(
            env_name=env_name,
            task_ids=task_ids,
            model_name=model_name,
            model_base_url=full_model_url,
        )
        env_results.append(result)
        print(f"    Average score: {result['average_score']:.4f}")

    end_time = datetime.now()

    # Generate report
    report = generate_report(
        env_results=env_results,
        config=config,
        start_time=start_time,
        end_time=end_time,
        model_name=model_name,
    )

    # Print summary
    print_report_summary(report)

    # Save report
    output_config = config.get('output', {})
    report_dir = output_config.get('report_dir', 'result/eval_reports')
    report_filename = output_config.get('report_filename', 'eval_report_{timestamp}.json')

    # Create output directory
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp and model name
    timestamp = end_time.strftime('%Y%m%d_%H%M%S')
    # Sanitize model name for filename (replace special chars)
    safe_model_name = model_name.replace('/', '_').replace(':', '_').replace(' ', '_')
    output_path = args.output or os.path.join(
        report_dir,
        report_filename.format(timestamp=timestamp, model=safe_model_name)
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Report saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

