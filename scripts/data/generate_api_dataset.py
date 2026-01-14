#!/usr/bin/env python3
"""
Generate datasets using APIBot (OpenAI-compatible API) self-play.

Two APIBots play against each other and both trajectories are saved.
Only sends system + last user message to API to save cost.
Includes reasoning in responses.

Supports resume from interrupted runs.
"""

import os
import sys
import json
import random
import argparse
import time
from tqdm import tqdm

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/openspiel'))

from game_config import create_game
from agents import GAME_AGENTS


# Game configurations: (name, min_task_id, max_task_id)
GAMES = {
    "goofspiel": (0, 99_999_999),
    "liars_dice": (100_000_000, 199_999_999),
    "leduc_poker": (200_000_000, 299_999_999),
    "gin_rummy": (300_000_000, 399_999_999),
    "othello": (400_000_000, 499_999_999),
    # "backgammon": (500_000_000, 599_999_999),
    "hex": (600_000_000, 699_999_999),
    "clobber": (700_000_000, 799_999_999),
}

TARGET_TRAJECTORIES = 10_000


def play_game_api_dual(
    task_id: int,
    seed: int,
    api_url: str,
    api_key: str,
    model: str,
    # temperature: float,
) -> list:
    """
    Play a game with 2 APIBots and return both trajectories.
    """
    import numpy as np
    from open_spiel.python.algorithms import evaluate_bots
    from api_bot import APIBot
    
    start_time = time.time()
    
    try:
        # Create game
        game, game_config = create_game(task_id)
        game_name = game_config["game_name"]
        num_players = game.num_players()
        
        if num_players != 2:
            raise ValueError(f"Only supports 2-player games, got {num_players}")
        
        # Get agent
        agent_class = GAME_AGENTS.get(game_name)
        if not agent_class:
            raise ValueError(f"No agent found for game: {game_name}")
        agent = agent_class()
        
        # Create 2 APIBots
        bots = []
        for player_id in range(2):
            bot = APIBot(
                game=game,
                agent=agent,
                player_id=player_id,
                api_url=api_url,
                api_key=api_key,
                model=model,
                # temperature=temperature,
                seed=seed + player_id,
            )
            bots.append(bot)
        
        # Play game
        returns = evaluate_bots.evaluate_bots(
            game.new_initial_state(),
            bots,
            np.random.RandomState(seed),
        )
        
        # Compute scores
        def compute_score(returns, player_id, game):
            player_return = returns[player_id]
            min_utility = game.min_utility()
            max_utility = game.max_utility()
            if max_utility > min_utility:
                return float((player_return - min_utility) / (max_utility - min_utility))
            return 0.5
        
        # Build trajectories for both players
        trajectories = []
        for player_id in range(2):
            bot = bots[player_id]
            score = compute_score(returns, player_id, game)
            
            trajectory = {
                "task_name": f"openspiel:{game_name}",
                "score": score,
                "success": score > 0.5,
                "time_taken": time.time() - start_time,
                "conversation": bot.get_conversation(),
                "action_history": bot.get_action_history(),
                "game_name": game_name,
                "task_id": task_id,
                "seed": seed,
                "player_id": player_id,
                "returns": list(returns),
                "model": model,
            }
            trajectories.append(trajectory)
        
        return trajectories
        
    except Exception as e:
        import traceback
        error_result = {
            "task_name": "openspiel:unknown",
            "score": 0.0,
            "success": False,
            "time_taken": time.time() - start_time,
            "error": f"[{type(e).__name__}] {str(e)}\n{traceback.format_exc()}",
            "task_id": task_id,
            "seed": seed,
        }
        return [error_result, error_result]


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


def generate_dataset_for_game(
    game_name: str,
    num_trajectories: int,
    output_dir: str,
    api_url: str,
    api_key: str,
    model: str,
    # temperature: float,
    delay: float,
):
    """Generate dataset for a single game with streaming writes and resume support"""
    task_id_min, task_id_max = GAMES[game_name]

    output_file = os.path.join(output_dir, f'{game_name}_api.jsonl')

    # Count existing trajectories for resume
    existing_count = count_existing_trajectories(output_file)
    # Each game produces 2 trajectories, so we need num_trajectories / 2 games
    target_games = (num_trajectories + 1) // 2
    existing_games = existing_count // 2
    remaining_games = max(0, target_games - existing_games)

    print(f"\n{'=' * 50}")
    print(f"Generating {game_name} dataset (API)")
    print(f"{'=' * 50}")
    print(f"Task ID range: {task_id_min:,} ~ {task_id_max:,}")
    print(f"Target trajectories: {num_trajectories:,} ({target_games} games)")
    print(f"Already generated: {existing_count:,} trajectories ({existing_games} games)")
    print(f"Remaining: {remaining_games} games")
    print(f"Model: {model}")
    print(f"API delay: {delay}s between games")

    if remaining_games <= 0:
        print(f"Already have enough trajectories, skipping.")
        return existing_count

    total_written = 0
    errors = 0

    pbar = tqdm(total=remaining_games, desc=f"{game_name}")

    # Open file for APPEND (resume support)
    with open(output_file, 'a') as f:
        for _ in range(remaining_games):
            # Generate random task_id and seed
            task_id = random.randint(task_id_min, task_id_max)
            seed = random.randint(0, 2**31 - 1)

            # Play game
            trajectories = play_game_api_dual(
                task_id=task_id,
                seed=seed,
                api_url=api_url,
                api_key=api_key,
                model=model,
                # temperature=temperature,
            )

            # Write both trajectories
            for traj in trajectories:
                if "error" in traj:
                    errors += 1
                else:
                    f.write(json.dumps(traj) + '\n')
                    f.flush()
                    total_written += 1

            pbar.update(1)

            # Delay between API calls to avoid rate limiting
            if delay > 0:
                time.sleep(delay)

    pbar.close()

    total_in_file = existing_count + total_written
    print(f"\nResults for {game_name}:")
    print(f"  New trajectories written: {total_written}")
    print(f"  Previously existing: {existing_count}")
    print(f"  Total in file: {total_in_file}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_file}")
    if os.path.exists(output_file):
        print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    return total_in_file


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    parser = argparse.ArgumentParser(description='Generate game datasets using API')
    parser.add_argument('--game', type=str, choices=list(GAMES.keys()) + ['all'], default='othello',
                        help='Game to generate dataset for')
    parser.add_argument('--num_trajectories', type=int, default=TARGET_TRAJECTORIES,
                        help='Number of trajectories per game')
    parser.add_argument('--output_dir', type=str, default='data/openspiel',
                        help='Output directory for datasets')
    parser.add_argument('--api_url', type=str, default='https://api.openai.com/v1/chat/completions',
                        help='OpenAI-compatible API URL')
    parser.add_argument('--model', type=str, default='gpt-5.2',
                        help='Model name')
    # parser.add_argument('--temperature', type=float, default=0.7,
    #                     help='Sampling temperature')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between API calls (seconds)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which games to process
    games_to_process = list(GAMES.keys()) if args.game == 'all' else [args.game]

    print(f"API Dataset Generation")
    print(f"Games: {', '.join(games_to_process)}")
    print(f"Target trajectories per game: {args.num_trajectories:,}")
    print(f"Model: {args.model}")
    print(f"API URL: {args.api_url}")

    # Generate datasets
    total_trajectories = 0
    for game_name in games_to_process:
        count = generate_dataset_for_game(
            game_name=game_name,
            num_trajectories=args.num_trajectories,
            output_dir=args.output_dir,
            api_url=args.api_url,
            api_key=api_key,
            model=args.model,
            # temperature=args.temperature,
            delay=args.delay,
        )
        total_trajectories += count

    print(f"\n{'=' * 50}")
    print(f"All datasets complete!")
    print(f"Total trajectories: {total_trajectories:,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()

