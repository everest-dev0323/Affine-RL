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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/trace'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/affinetes/templates'))

from trace_task import TraceTask

DATASET_RANGE = 23303

def main():
    parser = argparse.ArgumentParser(description='Generate PRINT datasets')
    parser.add_argument('--num_trajectories', type=int, default=0,
                        help='Number of trajectories per task')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='data/print',
                        help='Output directory for datasets')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, 'print_dataset.jsonl')

    trace_task = TraceTask()
    dataset = []

    existing_trajectories = 0

    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                existing_trajectories += 1
        print(f"Resuming from {existing_trajectories} existing trajectories.")


    for _ in tqdm(range(args.num_trajectories - existing_trajectories), desc="Generating trajectories"):
        task_id = random.randint(0, DATASET_RANGE - 1)
        challenge = asyncio.run(trace_task.generate(task_id))
        conversation = [{"role": "user", "content": challenge.prompt}, {"role": "assistant", "content": challenge.extra["ground_truth"]}]
        trajectory = {
            "conversation": conversation
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

