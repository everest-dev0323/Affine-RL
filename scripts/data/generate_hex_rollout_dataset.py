#!/usr/bin/env python3
"""
Generate Hex datasets from existing rollout data (SGF format).

This script:
1. Parses SGF rollout files (KataHex format)
2. Replays moves in OpenSpiel hex game
3. When rollout ends (incomplete game), uses MCTS to complete the game
4. Generates conversation format data for both players

SGF Format Notes:
- startTurnIdx=0: White plays first
- startTurnIdx=1: Black plays first
- SZ[N]: Board size (5, 7, 9, 11)
- RE[B+] or RE[W+]: Game result
- Moves: B[coord] or W[coord] where coord is like "a1", "k11"
"""

import os
import sys
import re
import json
import random
import argparse
from glob import glob
from tqdm import tqdm
import concurrent.futures
import threading

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/environments/openspiel'))

import pyspiel
import numpy as np
from open_spiel.python.algorithms import mcts, evaluate_bots
from algorithm_bot import AlgorithmBot, SafeRandomRolloutEvaluator
from agents import GAME_AGENTS

# Hex coordinate conversion
def sgf_coord_to_action(coord: str, board_size: int) -> int:
    """
    Convert SGF coordinate (e.g., "a1", "k11") to OpenSpiel action index.

    SGF Hex coordinates: column (a-k) + row (1-11)
    OpenSpiel hex action: row * board_size + col
    """
    col_char = coord[0].lower()
    row_str = coord[1:]

    col = ord(col_char) - ord('a')
    row = int(row_str) - 1

    # OpenSpiel hex uses row * board_size + col
    action = row * board_size + col
    return action


def parse_sgf_game(sgf_line: str) -> dict:
    """
    Parse a single SGF game line.

    Returns:
        dict with: board_size, first_player (0=black, 1=white),
                   moves list, result ('B' or 'W')
        Returns None if game has initial setup stones (AB/AW properties)
    """
    # Check for initial setup stones (AB = Add Black, AW = Add White)
    # These games have pre-placed handicap stones - skip them
    if re.search(r'AB\[', sgf_line) or re.search(r'AW\[', sgf_line):
        return None

    # Extract who plays first from comment
    # startTurnIdx=0 means White first, startTurnIdx=1 means Black first
    # Only accept games where Black plays first (startTurnIdx=1)
    start_turn_match = re.search(r'startTurnIdx=(\d+)', sgf_line)
    if start_turn_match:
        start_turn_idx = int(start_turn_match.group(1))
        if start_turn_idx != 1:
            # White plays first - skip this game
            return None
        first_player = 0  # Black plays first
    else:
        first_player = 0  # Default: Black first

    # Extract board size
    sz_match = re.search(r'SZ\[(\d+)\]', sgf_line)
    board_size = int(sz_match.group(1)) if sz_match else 11

    # Extract result
    re_match = re.search(r'RE\[([BW])\+\]', sgf_line)
    result = re_match.group(1) if re_match else None

    # Extract moves: B[coord] or W[coord]
    # Note: moves may have comments C[...] after them
    move_pattern = r';([BW])\[([a-z]+\d+)\]'
    moves = []
    for match in re.finditer(move_pattern, sgf_line):
        player_char = match.group(1)  # 'B' or 'W'
        coord = match.group(2)
        # B = Black = OpenSpiel player 0
        # W = White = OpenSpiel player 1
        player = 0 if player_char == 'B' else 1
        moves.append({
            'player': player,
            'coord': coord,
        })

    # Extract game hash for unique ID
    hash_match = re.search(r'gameHash=([A-F0-9]+)', sgf_line)
    game_hash = hash_match.group(1) if hash_match else None

    return {
        'board_size': board_size,
        'first_player': first_player,
        'moves': moves,
        'result': result,
        'game_hash': game_hash,
    }


def load_sgf_files(rollout_dir: str) -> list:
    """Load all SGF games from rollout directory (only Black-first games)."""
    games = []
    skipped = 0
    sgf_files = glob(os.path.join(rollout_dir, '**/*.sgfs'), recursive=True)

    for sgf_file in sgf_files:
        with open(sgf_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('(;'):
                    try:
                        game_data = parse_sgf_game(line)
                        if game_data is None:
                            # Skipped due to AB/AW setup stones or White-first
                            skipped += 1
                            continue
                        game_data['source_file'] = sgf_file
                        games.append(game_data)
                    except Exception as e:
                        print(f"Warning: Failed to parse line in {sgf_file}: {e}")

    if skipped > 0:
        print(f"Skipped {skipped} games (setup stones or White-first)")

    return games


def create_hex_game(board_size: int):
    """Create OpenSpiel hex game with given board size."""
    game = pyspiel.load_game("hex", {"board_size": board_size})
    return game


def create_mcts_bot(game, player_id: int, seed: int, mcts_sims: int = 1000):
    """Create MCTS bot for completing games."""
    evaluator = SafeRandomRolloutEvaluator(
        n_rollouts=50,
        random_state=np.random.RandomState(seed)
    )
    return mcts.MCTSBot(
        game=game,
        uct_c=1.414,
        max_simulations=mcts_sims,
        evaluator=evaluator,
        random_state=np.random.RandomState(seed + 1),
    )


def generate_trajectory_from_rollout(game_data: dict, seed: int = None, mcts_sims: int = 1000) -> list:
    """
    Generate dual trajectories from a rollout game.

    1. Replay existing moves from rollout
    2. If game not finished, use MCTS to complete
    3. Generate conversation format for both players

    Returns:
        List of 2 trajectory dicts (one for each player)
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    board_size = game_data['board_size']
    moves = game_data['moves']

    # Create game and agent
    game = create_hex_game(board_size)
    agent = GAME_AGENTS['hex']()

    # Create algorithm bots for both players (for conversation recording)
    bots = []
    for player_id in range(2):
        bot = AlgorithmBot(
            game=game,
            player_id=player_id,
            agent=agent,
            algorithm="mcts",
            seed=seed + player_id,
            mcts_simulations=mcts_sims,
        )
        bots.append(bot)

    # Create MCTS bots for completing the game (separate from recording bots)
    mcts_bots = [create_mcts_bot(game, pid, seed + 100 + pid, mcts_sims) for pid in range(2)]

    state = game.new_initial_state()

    # Restart bots
    for bot in bots:
        bot.restart_at(state)

    # Phase 1: Replay moves from rollout data
    rollout_move_count = 0
    for move_data in moves:
        if state.is_terminal():
            break

        current_player = state.current_player()
        coord = move_data['coord']

        try:
            action = sgf_coord_to_action(coord, board_size)
        except Exception as e:
            print(f"Warning: Failed to convert coord {coord}: {e}")
            break

        # Validate action is legal
        legal_actions = state.legal_actions(current_player)
        if action not in legal_actions:
            print(f"Warning: Action {action} (coord {coord}) not legal. Legal: {legal_actions[:5]}...")
            break

        # Record action via bot's step method (generates conversation)
        # But we override with our predetermined action
        bot = bots[current_player]

        # Generate prompts and record conversation
        if not bot._system_prompt_generated:
            system_prompt = agent.generate_system_prompt()
            bot._conversation.append({"role": "system", "content": system_prompt})
            bot._system_prompt_generated = True

        user_prompt = agent.generate_user_prompt(
            state=state,
            player_id=current_player,
            legal_actions=legal_actions
        )
        bot._conversation.append({"role": "user", "content": user_prompt})
        bot._conversation.append({"role": "assistant", "content": str(action)})

        # Inform both bots about the action
        for b in bots:
            b.inform_action(state, current_player, action)

        # Apply action
        state.apply_action(action)
        rollout_move_count += 1

    # Phase 2: Complete game with MCTS if not terminal
    mcts_move_count = 0
    while not state.is_terminal():
        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)

        # Get action from MCTS bot
        action = mcts_bots[current_player].step(state)

        # Record in conversation
        bot = bots[current_player]

        if not bot._system_prompt_generated:
            system_prompt = agent.generate_system_prompt()
            bot._conversation.append({"role": "system", "content": system_prompt})
            bot._system_prompt_generated = True

        user_prompt = agent.generate_user_prompt(
            state=state,
            player_id=current_player,
            legal_actions=legal_actions
        )
        bot._conversation.append({"role": "user", "content": user_prompt})
        bot._conversation.append({"role": "assistant", "content": str(action)})

        # Inform both bots
        for b in bots:
            b.inform_action(state, current_player, action)

        state.apply_action(action)
        mcts_move_count += 1

    # Get final returns
    returns = state.returns()

    # Build trajectories
    # - Black (player 0): Only collect if Black wins (first player advantage)
    # - White (player 1): Always collect (second player, learn from both outcomes)
    trajectories = []
    for player_id in range(2):
        bot = bots[player_id]
        player_return = returns[player_id]

        # Normalize score: 1.0 for win, 0.0 for loss, 0.5 for draw
        if player_return > 0:
            score = 1.0
        elif player_return < 0:
            score = 0.0
        else:
            score = 0.5

        # Skip Black's trajectory if Black loses
        if player_id == 0 and score < 0.5:
            continue

        trajectory = {
            'conversation': bot._conversation,
            'score': score,
            'success': score > 0.5,
            'task_id': 600_000_000 + (board_size - 5) // 2,  # hex task_id range
            'seed': seed,
            'game_name': 'hex',
            'player_id': player_id,
            'returns': list(returns),
            'board_size': board_size,
            'rollout_moves': rollout_move_count,
            'mcts_moves': mcts_move_count,
            'game_hash': game_data.get('game_hash'),
        }
        trajectories.append(trajectory)

    return trajectories


def process_single_game(args):
    """Process a single game (for parallel execution)."""
    game_data, base_seed, mcts_sims = args
    try:
        return generate_trajectory_from_rollout(game_data, seed=base_seed, mcts_sims=mcts_sims)
    except Exception as e:
        return [{"error": str(e), "game_hash": game_data.get('game_hash')}]


def generate_dataset(
    rollout_dir: str,
    output_file: str,
    num_trajectories: int = None,
    workers: int = 4,
    mcts_sims: int = 1000,
):
    """
    Generate dataset from rollout files.

    Args:
        rollout_dir: Directory containing .sgfs files
        output_file: Output JSONL file path
        num_trajectories: Maximum number of trajectories (None = all)
        workers: Number of parallel workers
        mcts_sims: MCTS simulations for completing games
    """
    print(f"Loading SGF files from: {rollout_dir}")
    games = load_sgf_files(rollout_dir)
    print(f"Loaded {len(games)} games")

    if len(games) == 0:
        print("No games found!")
        return 0

    # Limit games if needed (each game produces 2 trajectories)
    if num_trajectories is not None:
        max_games = (num_trajectories + 1) // 2
        if len(games) > max_games:
            games = games[:max_games]

    print(f"Processing {len(games)} games (1-2 trajectories per game)")

    # Prepare arguments
    base_seeds = [random.randint(0, 2**31 - 1) for _ in range(len(games))]
    args_list = [(game, seed, mcts_sims) for game, seed in zip(games, base_seeds)]

    total_written = 0
    errors = 0
    file_lock = threading.Lock()

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    with open(output_file, 'a') as f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_game, arg): arg for arg in args_list}

            with tqdm(total=len(games), desc="Processing games") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        trajectories = future.result()

                        for traj in trajectories:
                            if "error" in traj:
                                errors += 1
                                continue

                            # Write trajectory
                            with file_lock:
                                f.write(json.dumps(traj) + '\n')
                                f.flush()

                            total_written += 1

                    except Exception as e:
                        errors += 1
                        print(f"Error processing game: {e}")

                    pbar.update(1)

    print(f"\nGeneration complete!")
    print(f"Total trajectories written: {total_written}")
    print(f"Errors: {errors}")

    return total_written


def main():
    parser = argparse.ArgumentParser(description='Generate Hex dataset from rollout files')
    parser.add_argument('--rollout_dir', type=str, default='dataset/rollouts_output',
                        help='Directory containing SGF rollout files')
    parser.add_argument('--output_dir', type=str, default='data/openspiel',
                        help='Output directory for datasets')
    parser.add_argument('--num_trajectories', type=int, default=None,
                        help='Maximum number of trajectories (default: all)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--mcts_sims', type=int, default=1000,
                        help='MCTS simulations for completing games')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename
    output_file = os.path.join(args.output_dir, 'hex_rollout_dataset.jsonl')

    print(f"Hex Rollout Dataset Generation")
    print(f"=" * 50)
    print(f"Rollout directory: {args.rollout_dir}")
    print(f"Output file: {output_file}")
    print(f"Workers: {args.workers}")
    print(f"MCTS simulations: {args.mcts_sims}")
    if args.num_trajectories:
        print(f"Max trajectories: {args.num_trajectories}")
    print()

    count = generate_dataset(
        rollout_dir=args.rollout_dir,
        output_file=output_file,
        num_trajectories=args.num_trajectories,
        workers=args.workers,
        mcts_sims=args.mcts_sims,
    )

    print(f"\nDataset saved to: {output_file}")
    print(f"Total trajectories: {count}")


if __name__ == "__main__":
    main()

