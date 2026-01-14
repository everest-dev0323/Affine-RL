#!/usr/bin/env python3
"""
Generate datasets for print using trajectory generation with multiprocessing.

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/environments/trace'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/affinetes/templates'))

from trace_task import TraceTask

DATASET_RANGE = 23303


def generate_trajectory(args):
    """Generate a single trajectory with a random task_id"""
    dataset_range = args

    task_id = random.randint(0, dataset_range - 1)

    try:
        trace_task = TraceTask()
        challenge = asyncio.run(trace_task.generate(task_id))

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": challenge.extra["ground_truth"]}
        ]

        return {
            "env": "print",
            "task_id": task_id,
            "seed": challenge.extra['seed'],
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


def generate_dataset(num_trajectories: int, workers: int, output_dir: str):
    """Generate dataset with streaming writes and resume support"""
    output_file = os.path.join(output_dir, 'print.jsonl')

    # Count existing trajectories for resume
    existing_count = count_existing_trajectories(output_file)
    remaining = max(0, num_trajectories - existing_count)

    print(f"\n{'=' * 50}")
    print(f"Generating PRINT dataset")
    print(f"{'=' * 50}")
    print(f"Dataset range: {DATASET_RANGE:,}")
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

    pbar = tqdm(total=remaining, desc="print trajectories")

    # Open file for APPEND (resume support)
    with open(output_file, 'a') as f:
        while total_written < remaining:
            # Calculate batch size
            still_needed = remaining - total_written
            batch_size = min(max(100, still_needed), 1000)

            # Generate batch of task args
            task_args = [DATASET_RANGE for _ in range(batch_size)]

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
    print(f"\nResults:")
    print(f"  New trajectories written: {total_written}")
    print(f"  Previously existing: {existing_count}")
    print(f"  Total in file: {total_in_file}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    return total_in_file


def main():
    parser = argparse.ArgumentParser(description='Generate PRINT datasets')
    parser.add_argument('--num_trajectories', type=int, default=100_000,
                        help='Number of trajectories per task')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='data/print',
                        help='Output directory for datasets')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"PRINT Dataset Generation")
    print(f"Target trajectories: {args.num_trajectories:,}")
    print(f"Workers: {args.workers}")

    # Generate dataset
    total_trajectories = generate_dataset(
        num_trajectories=args.num_trajectories,
        workers=args.workers,
        output_dir=args.output_dir,
    )

    print(f"\n{'=' * 50}")
    print(f"All datasets complete!")
    print(f"Total trajectories: {total_trajectories:,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()

