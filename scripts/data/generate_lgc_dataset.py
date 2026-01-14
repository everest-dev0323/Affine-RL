#!/usr/bin/env python3
"""
Generate datasets for LGC-v2 using trajectory generation with multiprocessing.

"""

import os
import sys
import json
import random
import argparse
import asyncio
from tqdm import tqdm
import concurrent.futures
import threading

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/environments/primeintellect/lgc-v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/affinetes/templates'))

from logic_task_v2 import LogicTaskV2

SUPPORTED_TASKS = {
    "dyck_language": {
        "task_type_id": 0,  # Range: 0-99,999,999
        "module": "games.dyck_language.generator",
        "class": "DyckLanguageGenerator",
        "default_config": {
            "n_types": 3,
            "total_length": 0,      # 0 = random (40-60)
            "to_fill_length": 0,    # 0 = random (20%-50% of total)
            "nesting_depth": 0      # 0 = no minimum depth requirement
        }
    },
    "game_of_24": {
        "task_type_id": 1,  # Range: 100,000,000-199,999,999
        "module": "games.game_of_24.generator",
        "class": "GameOf24Generator",
        "default_config": {}  # All parameters derived from seed
    },
    "operation": {
        "task_type_id": 2,  # Range: 200,000,000-299,999,999
        "module": "games.operation.generator",
        "class": "OperationGenerator",
        "default_config": {}  # All parameters derived from seed
    },
    "cryptarithm": {
        "task_type_id": 3,  # Range: 300,000,000-399,999,999
        "module": "games.cryptarithm.generator",
        "class": "CryptarithmGenerator",
        "default_config": {}  # All parameters derived from seed
    },
    "dyck_language2": {
        "task_type_id": [4, 5],  # Range: 400,000,000-599,999,999 (200M seeds)
        "module": "games.dyck_language2.generator",
        "class": "DyckLanguage2Generator",
        "default_config": {}  # n_types and length derived from seed
    },
    "boolean_expressions": {
        "task_type_id": 6,  # Range: 600,000,000-699,999,999
        "module": "games.boolean_expressions.generator",
        "class": "BooleanExpressionsGenerator",
        "default_config": {}  # All parameters derived from seed
    },
    "sudoku": {
        "task_type_id": 7,  # Range: 700,000,000-799,999,999
        "module": "games.sudoku.generator",
        "class": "SudokuGenerator",
        "default_config": {}  # All parameters derived from seed
    },
}


def generate_trajectory(args):
    """Generate a single trajectory with a random task_id"""
    task_type_id, task_id_range = args

    task_id = random.randint(0, task_id_range - 1) + task_type_id * 100_000_000

    try:
        logic_task = LogicTaskV2()
        challenge = asyncio.run(logic_task.generate_dataset_by_id(task_id))

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": challenge.extra["answer"]}
        ]

        return {
            "env": "lgc-v2",
            "task_id": task_id,
            "conversation": conversation,
        }
    except Exception as e:
        return {"error": str(e), "task_id": task_id}


def count_existing_trajectories(output_file: str) -> int:
    """Count existing trajectories in the output file"""
    if not os.path.exists(output_file):
        return 0

    count = 0
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    return count


def generate_dataset_for_task(task_type: str, num_trajectories: int, workers: int,
                               output_dir: str):
    """Generate dataset for a single task type with streaming writes and resume support"""
    task_config = SUPPORTED_TASKS[task_type]
    task_type_id = task_config.get("task_type_id")

    task_id_range = 100_000_000
    if isinstance(task_type_id, list):
        task_id_range *= len(task_type_id)
        task_type_id = task_type_id[0]  # Use the first ID in the range for generation

    output_file = os.path.join(output_dir, f'{task_type}.jsonl')

    # Count existing trajectories for resume
    existing_count = count_existing_trajectories(output_file)
    remaining = max(0, num_trajectories - existing_count)

    print(f"\n{'=' * 50}")
    print(f"Generating {task_type} dataset")
    print(f"{'=' * 50}")
    print(f"Task type ID: {task_type_id}")
    print(f"Task ID range: {task_id_range:,}")
    print(f"Target trajectories: {num_trajectories:,}")
    print(f"Already generated: {existing_count:,}")
    print(f"Remaining to generate: {remaining:,}")
    print(f"Workers: {workers}")

    if remaining <= 0:
        print(f"Already have {existing_count} trajectories, skipping.")
        return existing_count

    total_written = 0
    errors = 0
    file_lock = threading.Lock()

    pbar = tqdm(total=remaining, desc=f"{task_type} trajectories")

    # Open file for APPEND (resume support)
    with open(output_file, 'a') as f:
        while total_written < remaining:
            # Calculate batch size
            still_needed = remaining - total_written
            batch_size = min(max(100, still_needed), 1000)

            # Generate batch of task args
            task_args = [
                (task_type_id, task_id_range)
                for _ in range(batch_size)
            ]

            # Process batch
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
            try:
                futures = {executor.submit(generate_trajectory, arg): arg
                          for arg in task_args}

                for future in concurrent.futures.as_completed(futures):
                    if total_written >= remaining:
                        break

                    result = future.result()

                    if "error" in result:
                        errors += 1
                        continue

                    # Write immediately to file
                    with file_lock:
                        f.write(json.dumps(result) + '\n')
                        f.flush()

                    total_written += 1
                    pbar.update(1)
            finally:
                # Cancel remaining futures and shutdown immediately
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

    pbar.close()

    total_in_file = existing_count + total_written
    print(f"\nResults for {task_type}:")
    print(f"  New trajectories written: {total_written}")
    print(f"  Previously existing: {existing_count}")
    print(f"  Total in file: {total_in_file}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    return total_in_file


def main():
    parser = argparse.ArgumentParser(description='Generate LGC-v2 datasets')
    parser.add_argument('--num_trajectories', type=int, default=100_000,
                        help='Number of trajectories per task')
    parser.add_argument('--task_type', type=str, default="dyck_language2",
                        choices=list(SUPPORTED_TASKS.keys()) + ['all'],
                        help='Task type to generate')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='data/lgc-v2',
                        help='Output directory for datasets')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which tasks to process
    tasks_to_process = list(SUPPORTED_TASKS.keys()) if args.task_type == 'all' else [args.task_type]

    print(f"LGC-v2 Dataset Generation")
    print(f"Tasks: {', '.join(tasks_to_process)}")
    print(f"Target trajectories per task: {args.num_trajectories:,}")
    print(f"Workers: {args.workers}")

    # Generate datasets
    total_trajectories = 0
    for task_type in tasks_to_process:
        count = generate_dataset_for_task(
            task_type=task_type,
            num_trajectories=args.num_trajectories,
            workers=args.workers,
            output_dir=args.output_dir,
        )
        total_trajectories += count

    print(f"\n{'=' * 50}")
    print(f"All datasets complete!")
    print(f"Total trajectories: {total_trajectories:,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()

