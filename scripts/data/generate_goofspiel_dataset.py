#!/usr/bin/env python3
"""
Generate Goofspiel dataset using Matching algorithm vs Random opponent.

Goofspiel task_id range: 0 ~ 99,999,999
Strategy: Match the prize card value (deterministic optimal strategy)
Save ALL trajectories (regardless of win/loss)
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

from env import play_game
from game_config import create_game
from agents import GAME_AGENTS
from matching_bot import MatchingBot


GAME_NAME = "goofspiel"
TASK_ID_MIN = 0
TASK_ID_MAX = 99_999_999
TARGET_TRAJECTORIES = 10_000


def generate_single_trajectory(_):
    """Generate a single trajectory with random task_id and seed"""
    # Generate truly random task_id and seed
    task_id = random.randint(TASK_ID_MIN, TASK_ID_MAX)
    seed = random.randint(0, 2**31 - 1)

    try:
        # Create game and agent
        game, game_config = create_game(task_id)
        agent = GAME_AGENTS[game_config['game_name']]()

        # Create MatchingBot
        bot = MatchingBot(
            game=game,
            agent=agent,
            player_id=seed % 2,  # Alternate player positions
            seed=seed,
        )

        # Play game with random opponent
        result = play_game(
            bot=bot,
            task_id=task_id,
            seed=seed,
            opponent="random",
        )

        # Return trajectory data
        return {
            "conversation": result['conversation'],
            "score": result['score'],
            "success": result['success'],
            "task_id": task_id,
            "seed": seed,
            "game_name": result['game_name'],
            "player_id": result['player_id'],
            "returns": result['returns'],
        }

    except Exception as e:
        return {"error": str(e), "task_id": task_id, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description='Generate Goofspiel dataset')
    parser.add_argument('--num_trajectories', type=int, default=TARGET_TRAJECTORIES,
                        help='Number of trajectories to generate')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='data/openspiel',
                        help='Output directory for dataset')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{GAME_NAME}.jsonl')

    print(f"Generating {args.num_trajectories} {GAME_NAME} trajectories...")
    print(f"Strategy: Matching (bid prize card value)")
    print(f"Opponent: Random")
    print(f"Output: {output_file}")
    print(f"Mode: Fully random (no fixed seed)")
    print()

    # Stats
    total_written = 0
    errors = 0
    wins = 0
    file_lock = threading.Lock()

    # Open file for streaming writes
    with open(output_file, 'w') as f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            futures = {executor.submit(generate_single_trajectory, i): i
                      for i in range(args.num_trajectories)}

            # Process as they complete and write immediately
            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=args.num_trajectories, desc="Generating"):
                result = future.result()

                if "error" in result:
                    errors += 1
                else:
                    # Write immediately to file
                    with file_lock:
                        f.write(json.dumps(result) + '\n')
                        f.flush()  # Ensure it's written to disk
                    total_written += 1
                    if result['success']:
                        wins += 1

    # Print statistics
    print()
    print(f"=" * 50)
    print(f"Dataset Generation Complete: {GAME_NAME}")
    print(f"=" * 50)
    print(f"Total trajectories: {total_written}")
    if total_written > 0:
        print(f"Wins: {wins} ({100*wins/total_written:.1f}%)")
        print(f"Losses: {total_written - wins}")
    print(f"Errors: {errors}")
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()

