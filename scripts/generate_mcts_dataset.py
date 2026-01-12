#!/usr/bin/env python3
"""
Generate datasets for MCTS-based games using dual trajectory generation.

Games:
- liars_dice: 100,000,000 ~ 199,999,999
- leduc_poker: 200,000,000 ~ 299,999,999
- gin_rummy: 300,000,000 ~ 399,999,999
- othello: 400,000,000 ~ 499,999,999
# - backgammon: 500,000,000 ~ 599,999,999
- hex: 600,000,000 ~ 699,999,999
- clobber: 700,000,000 ~ 799,999,999

Strategy: MCTS self-play (both players use MCTS)
Save BOTH winning and losing trajectories
"""

import os
import sys
import json
import random
import argparse
from tqdm import tqdm
import concurrent.futures
import threading

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/openspiel'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/affinetes/templates'))

from env import play_game_dual


# Game configurations: (name, min_task_id, max_task_id)
GAMES = {
    "liars_dice": (100_000_000, 199_999_999),
    "leduc_poker": (200_000_000, 299_999_999),
    "gin_rummy": (300_000_000, 399_999_999),
    "othello": (400_000_000, 499_999_999),
    # "backgammon": (500_000_000, 599_999_999),
    "hex": (600_000_000, 699_999_999),
    "clobber": (700_000_000, 799_999_999),
}

TARGET_TRAJECTORIES = 1_000


def generate_dual_trajectory(args):
    """Generate dual trajectory (returns 2 trajectories) with random task_id and seed"""
    _, task_id_min, task_id_max, mcts_sims = args

    # Generate truly random task_id and seed
    task_id = random.randint(task_id_min, task_id_max)
    seed = random.randint(0, 2**31 - 1)

    try:
        trajectories = play_game_dual(
            task_id=task_id,
            seed=seed,
            algorithm="mcts",
            mcts_simulations=mcts_sims,
        )
        return trajectories
    except Exception as e:
        return [{"error": str(e), "task_id": task_id, "seed": seed}]


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


def generate_dataset_for_game(game_name: str, num_trajectories: int, workers: int,
                               output_dir: str, mcts_sims: int = None):
    """Generate dataset for a single game with streaming writes and resume support"""
    task_id_min, task_id_max = GAMES[game_name]

    output_file = os.path.join(output_dir, f'{game_name}.jsonl')

    # Count existing trajectories for resume
    existing_count = count_existing_trajectories(output_file)
    remaining = max(0, num_trajectories - existing_count)

    print(f"\n{'=' * 50}")
    print(f"Generating {game_name} dataset")
    print(f"{'=' * 50}")
    print(f"Task ID range: {task_id_min:,} ~ {task_id_max:,}")
    print(f"Target trajectories: {num_trajectories:,}")
    print(f"Already generated: {existing_count:,}")
    print(f"Remaining to generate: {remaining:,}")
    print(f"MCTS simulations: {mcts_sims or 'default'}")
    print(f"Mode: Fully random (no fixed seed)")

    if remaining <= 0:
        print(f"Already have {existing_count} trajectories, skipping.")
        return existing_count

    total_games = 0
    total_written = 0
    errors = 0
    file_lock = threading.Lock()

    pbar = tqdm(total=remaining, desc=f"{game_name} trajectories")

    # Open file for APPEND (resume support)
    with open(output_file, 'a') as f:
        while total_written < remaining:
            # Calculate batch size - dual trajectory means 2 trajectories per game
            still_needed = remaining - total_written
            batch_size = min(max(100, (still_needed + 1) // 2), 1000)

            # Generate batch of task args (fully random)
            task_args = [
                (game_name, task_id_min, task_id_max, mcts_sims)
                for _ in range(batch_size)
            ]

            # Process batch
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
            try:
                futures = {executor.submit(generate_dual_trajectory, arg): arg
                          for arg in task_args}

                for future in concurrent.futures.as_completed(futures):
                    if total_written >= remaining:
                        break

                    dual_result = future.result()
                    total_games += 1

                    for traj in dual_result:
                        if total_written >= remaining:
                            break

                        if "error" in traj:
                            errors += 1
                            continue

                        # Save both winning and losing trajectories
                        trajectory_data = {
                            "conversation": traj['conversation'],
                            "score": traj['score'],
                            "task_id": traj['task_id'],
                            "seed": traj['seed'],
                            "game_name": traj['game_name'],
                            "player_id": traj['player_id'],
                            "returns": traj['returns'],
                            "success": traj.get('success', False),
                        }

                        # Write immediately to file
                        with file_lock:
                            f.write(json.dumps(trajectory_data) + '\n')
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
    print(f"\nResults for {game_name}:")
    print(f"  Games played this run: {total_games}")
    print(f"  New trajectories written: {total_written}")
    print(f"  Previously existing: {existing_count}")
    print(f"  Total in file: {total_in_file}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    return total_in_file


def main():
    parser = argparse.ArgumentParser(description='Generate MCTS game datasets')
    parser.add_argument('--game', type=str, choices=list(GAMES.keys()) + ['all'], default='hex',
                        help='Game to generate dataset for (default: all)')
    parser.add_argument('--num_trajectories', type=int, default=TARGET_TRAJECTORIES,
                        help='Number of trajectories per game (both winning and losing)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='data/openspiel',
                        help='Output directory for datasets')
    parser.add_argument('--mcts_sims', type=int, default=None,
                        help='MCTS simulations per move')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which games to process
    games_to_process = list(GAMES.keys()) if args.game == 'all' else [args.game]

    print(f"MCTS Dataset Generation")
    print(f"Games: {', '.join(games_to_process)}")
    print(f"Target trajectories per game: {args.num_trajectories:,}")
    print(f"Workers: {args.workers}")
    print(f"MCTS simulations: {args.mcts_sims or 'default'}")
    print(f"Mode: Fully random (no fixed seed)")

    # Generate datasets
    total_trajectories = 0
    for game_name in games_to_process:
        count = generate_dataset_for_game(
            game_name=game_name,
            num_trajectories=args.num_trajectories,
            workers=args.workers,
            output_dir=args.output_dir,
            mcts_sims=args.mcts_sims,
        )
        total_trajectories += count

    print(f"\n{'=' * 50}")
    print(f"All datasets complete!")
    print(f"Total trajectories: {total_trajectories:,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()

