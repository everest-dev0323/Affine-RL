#!/usr/bin/env python3
"""
Generate datasets for print using trajectory generation.

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/primeintellect/lgc-v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/affinetes/templates'))

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


def main():
    parser = argparse.ArgumentParser(description='Generate PRINT datasets')
    parser.add_argument('--num_trajectories', type=int, default=0,
                        help='Number of trajectories per task')
    parser.add_argument('--task_type', type=str, default="dyck_language",
                        help='Number of trajectories per task')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='data/logic',
                        help='Output directory for datasets')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, f"{args.task_type}.jsonl")

    task_type_id = SUPPORTED_TASKS[args.task_type].get("task_type_id")

    task_id_range = 100000000
    if isinstance(task_type_id, list):
        task_id_range *= len(task_type_id)
        task_type_id = task_type_id[0]  # Use the first ID in the range for generation
        
    logic_task = LogicTaskV2()
    dataset = []

    existing_trajectories = 0
    
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                existing_trajectories += 1
        print(f"Resuming from {existing_trajectories} existing trajectories.")

    for _ in range(args.num_trajectories - existing_trajectories):
        task_id = random.randint(0, task_id_range - 1) + task_type_id * 100000000

        try:
            challenge = asyncio.run(logic_task.generate_dataset_by_id(task_id))
            # log_event("challenge_generated")
        except ValueError as e:
            print(f"Error generating challenge for task_id {task_id}: {e}")
            continue

        conversation = [{"role": "user", "content": challenge.prompt}, {"role": "assistant", "content": challenge.extra["answer"]}]

        trajectory = {
            "conversation": conversation,
        }
        
        dataset.append(trajectory)
        with open(output_file, 'a') as f:
            f.write(json.dumps(trajectory) + '\n')

    print(f"\n{'=' * 50}")
    print(f"All datasets complete!")
    print(f"Total trajectories: {len(dataset):,}")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()

