#!/usr/bin/env python3
"""
Generate Clobber golden datasets using strong MCTS for winning trajectories.

This script:
1. Takes task_id and seed as input (same as evaluation)
2. Strong MCTS (high simulations) plays as the "learner" (same player_id as LLM in evaluation)
3. Standard MCTS bot plays as opponent (exact same as evaluation, same seed)
4. Only saves trajectories where strong MCTS wins (golden trajectories)

Unlike KataHex for Hex, there's no specialized Clobber engine.
We use a stronger MCTS (more simulations) to generate winning trajectories.


python scripts/data/generate_clobber_golden_dataset.py \
    --output_file data/openspiel/clobber_golden.jsonl \
    --processed_log_file log/data/openspiel/clobber_processed.log \
    --workers 8

python scripts/data/generate_clobber_golden_dataset.py \
    --output_file data/openspiel/clobber_determine.jsonl \
    --processed_log_file log/data/openspiel/clobber_determine.log \
    --workers 100 \
    --random \
    --num_random 10000
"""

import os
import sys
import json
import argparse
from typing import Optional, List, Dict, Any
import numpy as np
import random
import concurrent.futures
import threading

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/affinetes/templates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/environments/openspiel'))

import pyspiel
from open_spiel.python.algorithms import mcts
from game_config import decode_task_id
from agents import GAME_AGENTS
from env import Actor, SafeRandomRolloutEvaluator


def _generate_single_trajectory(args: tuple) -> Optional[Dict[str, Any]]:
    """Wrapper function for multiprocessing - must be at module level to be picklable."""
    task_id, seed, strong_mcts_simulations = args
    try:
        return generate_clobber_golden_trajectory(task_id, seed, strong_mcts_simulations)
    except Exception as e:
        print(f"Error generating trajectory for task_id={task_id}, seed={seed}: {e}")
        return None


def generate_clobber_golden_trajectory(
    task_id: int,
    seed: int,
    strong_mcts_simulations: int = 3000,
) -> Optional[Dict[str, Any]]:
    """
    Generate winning trajectory using strong MCTS vs standard MCTS opponent.

    Args:
        task_id: Task ID (determines board size for clobber)
        seed: Random seed (same as evaluation seed)
        strong_mcts_simulations: Number of simulations for strong MCTS player

    Returns:
        Trajectory dict if strong MCTS wins, None otherwise
    """
    # Decode task_id to get game params
    config = decode_task_id(task_id)
    if config['game_name'] != 'clobber':
        raise ValueError(f"Task ID {task_id} is not a clobber game (got {config['game_name']})")

    game_params = config['game_params']
    agent = GAME_AGENTS['clobber']()

    # Determine which player strong MCTS plays as (same logic as evaluation)
    # In evaluation: llm_player_id = seed % num_players
    num_players = 2
    strong_player_id = seed % num_players
    opponent_player_id = 1 - strong_player_id

    # Create OpenSpiel game
    game = pyspiel.load_game("clobber", game_params)
    state = game.new_initial_state()

    # Create Actor instance for this process (each process needs its own)
    actor = Actor()

    # Create standard MCTS opponent using Actor._create_opponent_bot (exact same as evaluation)
    mcts_opponent = actor._create_opponent_bot(
        "mcts", opponent_player_id, seed + 2 + opponent_player_id, game, agent
    )

    # Create strong MCTS player (higher simulations)
    strong_evaluator = SafeRandomRolloutEvaluator(
        n_rollouts=100, random_state=np.random.RandomState(12345)
    )
    strong_mcts = mcts.MCTSBot(
        game=game,
        uct_c=1.414,
        max_simulations=strong_mcts_simulations,
        evaluator=strong_evaluator,
        random_state=np.random.RandomState(12345),
    )

    # Track conversation for strong MCTS player only
    conversation = []
    system_prompt_generated = False
    system_prompt = agent.generate_system_prompt()

    # Play the game
    moves = []

    while not state.is_terminal():
        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)

        if current_player == strong_player_id:
            # Strong MCTS's turn - generate conversation and get move

            # Generate system prompt if not done
            if not system_prompt_generated:
                conversation.append({
                    "role": "system",
                    "content": system_prompt
                })
                system_prompt_generated = True

            # Generate user prompt
            user_prompt = agent.generate_user_prompt(
                state=state,
                player_id=current_player,
                legal_actions=legal_actions
            )
            conversation.append({
                "role": "user",
                "content": user_prompt
            })

            # Get move from strong MCTS
            action = strong_mcts.step(state)

            # Record assistant response
            conversation.append({
                "role": "assistant",
                "content": str(action)
            })
        else:
            # Standard MCTS opponent's turn
            action = mcts_opponent.step(state)

        # Apply to state
        state.apply_action(action)
        moves.append((current_player, action))

    # Get final returns
    returns = state.returns()
    strong_return = returns[strong_player_id]

    # Normalize score
    if strong_return > 0:
        score = 1.0
    elif strong_return < 0:
        score = 0.0
    else:
        score = 0.5

    # Only return trajectory if strong MCTS won
    if score <= 0.5:
        return None

    # Build trajectory for strong MCTS player
    trajectory = {
        'conversation': conversation,
        'score': score,
        'success': True,
        'task_id': task_id,
        'seed': seed,
        'game_name': 'clobber',
        'player_id': strong_player_id,
        'returns': list(returns),
        'board_rows': game_params.get('rows', 5),
        'board_cols': game_params.get('columns', 5),
        'total_moves': len(moves),
        'generator': 'strong_mcts',
        'strong_mcts_simulations': strong_mcts_simulations,
    }

    return trajectory


def generate_trajectories_batch(
    task_ids: List[int],
    seeds: List[int],
    strong_mcts_simulations: int = 3000,
    output_file: Optional[str] = None,
    processed_log_file: Optional[str] = None,
    workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Generate multiple golden trajectories for Clobber using multiprocessing.

    Args:
        task_ids: List of task IDs
        seeds: List of seeds (same length as task_ids)
        strong_mcts_simulations: Simulations for strong MCTS
        output_file: Optional file to write winning trajectories
        processed_log_file: Optional file to log all processed task_ids (wins and losses)
        workers: Number of parallel workers

    Returns:
        List of winning trajectories only
    """
    if len(task_ids) != len(seeds):
        raise ValueError("task_ids and seeds must have same length")

    all_trajectories = []
    wins = 0
    losses = 0
    file_lock = threading.Lock()

    # Prepare task arguments
    task_args = [
        (task_id, seed, strong_mcts_simulations)
        for task_id, seed in zip(task_ids, seeds)
    ]

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(task_args), desc="Generating")
    except ImportError:
        pbar = None

    # Process in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_generate_single_trajectory, arg): arg for arg in task_args}

        for future in concurrent.futures.as_completed(futures):
            arg = futures[future]
            task_id, seed, _ = arg
            trajectory = future.result()

            # Log processed task_id (both wins and losses)
            if processed_log_file:
                with file_lock:
                    with open(processed_log_file, 'a') as f:
                        f.write(f"{task_id}\n")

            if trajectory is not None:
                wins += 1
                all_trajectories.append(trajectory)

                # Write winning trajectory to output file
                if output_file:
                    with file_lock:
                        with open(output_file, 'a') as f:
                            f.write(json.dumps(trajectory) + '\n')
            else:
                losses += 1

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    print(f"\nBatch complete: {wins} wins, {losses} losses (win rate: {win_rate:.1f}%)")
    return all_trajectories


def main():
    parser = argparse.ArgumentParser(description='Generate Clobber golden dataset using strong MCTS')
    parser.add_argument('--merged_file', type=str, default='data/merged.json',
                        help='JSON file with merged evaluation data')
    parser.add_argument('--strong_mcts_sims', type=int, default=3000,
                        help='Strong MCTS simulations per move')
    parser.add_argument('--output_file', type=str, default='data/openspiel/clobber_golden.jsonl',
                        help='Output file path')
    parser.add_argument('--processed_log_file', type=str, default='log/data/openspiel/clobber_processed.log',
                        help='Log file to track all processed task_ids (wins and losses)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tasks to process')
    parser.add_argument('--latest_n', type=int, default=200,
                        help='Number of latest tasks to collect per env')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--random', action='store_true', default=False,
                        help='Generate random task_ids and seeds instead of using merged.json')
    parser.add_argument('--num_random', type=int, default=1000,
                        help='Number of random tasks to generate (only used with --random)')
    args = parser.parse_args()


    if args.random:
        # Generate random task_ids and seeds
        task_ids = [random.randint(700_000_000, 799_999_999) for _ in range(args.num_random)]
        seeds = [random.randint(0, 2**32 - 1) for _ in range(args.num_random)]
        print(f"Generated {len(task_ids)} random hex tasks")
    else:
        # Import get_sampling_list
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
        from src.data import get_sampling_list

        # Load processed task_ids from log file (includes both wins and losses)
        processed_task_ids = set()
        if os.path.exists(args.processed_log_file):
            with open(args.processed_log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed_task_ids.add(int(line))
            print(f"Loaded {len(processed_task_ids)} processed task_ids from {args.processed_log_file}")

        # Load merged.json and get sampling list
        with open(args.merged_file, 'r') as f:
            merged_data = json.load(f)

        sampling_list = get_sampling_list(merged_data, latest_n=args.latest_n)

        # Get clobber tasks from sampling list
        clobber_tasks = sampling_list.get('GAME', {}).get('clobber', [])

        # Filter out already processed tasks
        task_ids = []
        seeds = []
        skipped_existing = 0

        for task in clobber_tasks:
            task_id = task['task_id']
            seed = task['seed']
            if task_id in processed_task_ids:
                skipped_existing += 1
                continue
            task_ids.append(task_id)
            seeds.append(seed)

        print(f"Found {len(clobber_tasks)} clobber tasks in merged.json")
        print(f"Skipped {skipped_existing} already processed tasks")
        print(f"Remaining {len(task_ids)} new clobber tasks to process")

        if args.limit:
            task_ids = task_ids[:args.limit]
            seeds = seeds[:args.limit]

    if len(task_ids) == 0:
        print("No tasks to process. Exiting.")
        return

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.processed_log_file) or '.', exist_ok=True)

    print(f"\nClobber Golden Dataset Generation")
    print(f"=" * 50)
    print(f"Merged file: {args.merged_file}")
    print(f"Number of tasks: {len(task_ids)}")
    print(f"Strong MCTS simulations: {args.strong_mcts_sims}")
    print(f"Workers: {args.workers}")
    print(f"Output file: {args.output_file}")
    print(f"Processed log file: {args.processed_log_file}")
    print()

    trajectories = generate_trajectories_batch(
        task_ids=task_ids,
        seeds=seeds,
        strong_mcts_simulations=args.strong_mcts_sims,
        output_file=args.output_file,
        processed_log_file=args.processed_log_file,
        workers=args.workers,
    )

    print(f"\nGeneration complete!")
    print(f"Golden trajectories saved: {len(trajectories)}")
    print(f"Output file: {args.output_file}")
    print(f"Processed log file: {args.processed_log_file}")


if __name__ == "__main__":
    main()

