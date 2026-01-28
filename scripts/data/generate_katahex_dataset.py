#!/usr/bin/env python3
"""
Generate Hex datasets using KataHex for high-quality winning trajectories.

This script:
1. Takes task_id and seed as input (same as evaluation)
2. KataHex plays as the "learner" (same player_id as LLM in evaluation)
3. MCTS bot plays as opponent (exact same as evaluation, same seed)
4. Generates conversation format data for the KataHex player

This ensures the trajectory teaches how to beat the exact MCTS opponent
that will be used during evaluation.

KataHex GTP commands used:
- boardsize N: Set board size
- clear_board: Reset the board
- genmove COLOR: Generate and play a move for COLOR (black/white)
- play COLOR VERTEX: Play a move at the given vertex

python scripts/data/generate_katahex_dataset.py --config_path external/katahex/config_deterministic.cfg
python scripts/data/generate_katahex_dataset.py \
    --output_file data/openspiel/hex_golden.jsonl \
    --processed_log_file log/data/openspiel/hex_processed.log \
    --workers 8
python scripts/data/generate_katahex_dataset.py \
    --output_file data/openspiel/hex_determine.jsonl \
    --processed_log_file log/data/openspiel/hex_processed_determine.log \
    --workers 100 \
    --random \
    --num_random 10000
"""

import os
import sys
import subprocess
import json
import argparse
import random
from typing import Optional, List, Dict, Any
import concurrent.futures
import threading

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/affinetes/templates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/environments/openspiel'))

import pyspiel
from game_config import decode_task_id
from agents import GAME_AGENTS
from env import Actor


class KataHexGTP:
    """KataHex GTP interface for communicating with the KataHex engine."""

    def __init__(
        self,
        katahex_path: str = "external/katahex/build/katahex",
        model_path: str = "external/katahex/katahex_model_20220618.bin.gz",
        config_path: str = "external/katahex/config.cfg",
        max_visits: int = 500,
    ):
        self.katahex_path = katahex_path
        self.model_path = model_path
        self.config_path = config_path
        self.max_visits = max_visits
        self.process = None

    def start(self) -> bool:
        """Start the KataHex GTP process."""
        try:
            cmd = [
                self.katahex_path,
                "gtp",
                "-config", self.config_path,
                "-model", self.model_path,
            ]
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Set max visits
            self._send_command(f"kata-set-param maxVisits {self.max_visits}")
            return True
        except Exception as e:
            print(f"Failed to start KataHex: {e}")
            return False

    def stop(self):
        """Stop the KataHex process."""
        if self.process:
            self._send_command("quit")
            self.process.terminate()
            self.process.wait()
            self.process = None

    def _send_command(self, command: str) -> str:
        """Send a GTP command and return the response."""
        if not self.process:
            raise RuntimeError("KataHex process not started")

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

        response_lines = []
        while True:
            line = self.process.stdout.readline()
            if line == "\n" and response_lines:
                break
            if line.startswith("=") or line.startswith("?"):
                response_lines.append(line.strip())
            elif line.strip():
                response_lines.append(line.strip())

        return "\n".join(response_lines)

    def set_board_size(self, size: int) -> bool:
        """Set the board size."""
        response = self._send_command(f"boardsize {size}")
        return response.startswith("=")

    def clear_board(self) -> bool:
        """Clear the board."""
        response = self._send_command("clear_board")
        return response.startswith("=")

    def genmove(self, color: str) -> Optional[str]:
        """Generate a move for the given color. Returns vertex (e.g., 'A1', 'K11') or None."""
        response = self._send_command(f"genmove {color}")
        if response.startswith("="):
            # Extract move from response like "= A1" or "=A1"
            move = response[1:].strip()
            if move.lower() == "resign":
                return "resign"
            if move.lower() == "pass":
                return "pass"
            return move
        return None

    def play(self, color: str, vertex: str) -> bool:
        """Play a move at the given vertex."""
        response = self._send_command(f"play {color} {vertex}")
        return response.startswith("=")

    def showboard(self) -> str:
        """Get the current board state."""
        return self._send_command("showboard")


def gtp_vertex_to_openspiel_action(vertex: str, board_size: int) -> int:
    """
    Convert GTP vertex (e.g., 'A1', 'K11') to OpenSpiel action index.

    GTP uses: column (A-K) + row (1-11), where A1 is bottom-left
    OpenSpiel hex action: row * board_size + col
    """
    vertex = vertex.upper()
    col_char = vertex[0]
    row_str = vertex[1:]

    col = ord(col_char) - ord('A')
    row = int(row_str) - 1

    return row * board_size + col


def openspiel_action_to_gtp_vertex(action: int, board_size: int) -> str:
    """Convert OpenSpiel action to GTP vertex."""
    row = action // board_size
    col = action % board_size
    col_char = chr(ord('A') + col)
    return f"{col_char}{row + 1}"


def _generate_single_trajectory(args: tuple) -> Optional[Dict[str, Any]]:
    """Wrapper function for multiprocessing - creates fresh KataHex instance per task."""
    task_id, seed, katahex_path, model_path, config_path, max_visits = args
    try:
        return generate_katahex_trajectory(
            task_id=task_id,
            seed=seed,
            katahex_path=katahex_path,
            model_path=model_path,
            config_path=config_path,
            max_visits=max_visits,
        )
    except Exception as e:
        print(f"Error generating trajectory for task_id={task_id}, seed={seed}: {e}")
        return None


def generate_katahex_trajectory(
    task_id: int,
    seed: int,
    katahex_path: str,
    model_path: str,
    config_path: str,
    max_visits: int = 500,
) -> Optional[Dict[str, Any]]:
    """
    Generate winning trajectory using KataHex vs MCTS opponent.

    Creates a fresh KataHex instance for each task to ensure determinism.

    - KataHex plays as the "learner" (same player_id as LLM in evaluation)
    - MCTS bot plays as opponent (exact same seed as evaluation)

    This ensures the trajectory teaches how to beat the exact MCTS opponent
    that will be used during evaluation.

    Args:
        task_id: Task ID (determines board size for hex)
        seed: Random seed (same as evaluation seed)
        katahex_path: Path to katahex binary
        model_path: Path to neural network model
        config_path: Path to config file
        max_visits: MCTS visits per move for KataHex

    Returns:
        Trajectory dict if successful, None otherwise
    """
    # Decode task_id to get board_size
    config = decode_task_id(task_id)
    if config['game_name'] != 'hex':
        raise ValueError(f"Task ID {task_id} is not a hex game (got {config['game_name']})")

    board_size = config['game_params']['board_size']
    agent = GAME_AGENTS['hex']()

    # Determine which player KataHex plays as (same logic as evaluation)
    # In evaluation: llm_player_id = seed % num_players
    num_players = 2
    katahex_player_id = seed % num_players
    opponent_player_id = 1 - katahex_player_id

    # Create fresh KataHex instance for this task (ensures determinism)
    katahex = KataHexGTP(
        katahex_path=katahex_path,
        model_path=model_path,
        config_path=config_path,
        max_visits=max_visits,
    )

    if not katahex.start():
        print(f"Failed to start KataHex for task_id={task_id}")
        return None

    try:
        # Setup KataHex
        katahex.set_board_size(board_size)
        katahex.clear_board()

        # Create OpenSpiel game for state tracking
        game = pyspiel.load_game("hex", {"board_size": board_size})
        state = game.new_initial_state()

        # Create Actor instance for this task
        actor = Actor()

        # Create MCTS opponent using Actor._create_opponent_bot (exact same as evaluation)
        # In evaluation: _create_opponent_bot(opponent, player_id, seed + 2 + player_id, game, agent)
        mcts_opponent = actor._create_opponent_bot(
            "mcts", opponent_player_id, seed + 2 + opponent_player_id, game, agent
        )

        # Track conversation for KataHex player only
        conversation = []
        system_prompt_generated = False
        system_prompt = agent.generate_system_prompt()

        # Play the game
        moves = []  # [(player, action), ...]
        colors = ["black", "white"]

        while not state.is_terminal():
            current_player = state.current_player()
            legal_actions = state.legal_actions(current_player)

            if current_player == katahex_player_id:
                # KataHex's turn - generate conversation and get move

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

                # Get move from KataHex
                color = colors[current_player]
                vertex = katahex.genmove(color)

                if vertex is None or vertex == "resign":
                    print(f"Warning: KataHex returned {vertex} at move {len(moves)}")
                    return None

                # Convert to OpenSpiel action
                action = gtp_vertex_to_openspiel_action(vertex, board_size)

                # Record assistant response
                conversation.append({
                    "role": "assistant",
                    "content": str(action)
                })
            else:
                # MCTS opponent's turn
                action = mcts_opponent.step(state)

                # Tell KataHex about opponent's move
                vertex = openspiel_action_to_gtp_vertex(action, board_size)
                color = colors[current_player]
                katahex.play(color, vertex)

            # Apply to OpenSpiel state
            state.apply_action(action)
            moves.append((current_player, action))

        # Get final returns
        returns = state.returns()

        # Get KataHex player's result
        katahex_return = returns[katahex_player_id]

        # Normalize score
        if katahex_return > 0:
            score = 1.0
        elif katahex_return < 0:
            score = 0.0
        else:
            score = 0.5

        # Only return trajectory if KataHex won (golden trajectory)
        if score <= 0.5:
            return None

        # Build trajectory for KataHex player only
        trajectory = {
            'conversation': conversation,
            'score': score,
            'success': True,
            'task_id': task_id,
            'seed': seed,
            'game_name': 'hex',
            'player_id': katahex_player_id,
            'returns': list(returns),
            'board_size': board_size,
            'total_moves': len(moves),
            'generator': 'katahex',
            'max_visits': max_visits,
            'katahex_won': True,
        }

        return trajectory

    finally:
        katahex.stop()


def generate_trajectories_batch(
    task_ids: List[int],
    seeds: List[int],
    katahex_path: str = "external/katahex/build/katahex",
    model_path: str = "external/katahex/katahex_model_20220618.bin.gz",
    config_path: str = "external/katahex/config_deterministic.cfg",
    max_visits: int = 500,
    output_file: Optional[str] = None,
    processed_log_file: Optional[str] = None,
    workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Generate multiple trajectories using KataHex with parallel processing.

    Each task creates a fresh KataHex instance to ensure determinism.

    Args:
        task_ids: List of task IDs
        seeds: List of seeds (same length as task_ids)
        katahex_path: Path to katahex binary
        model_path: Path to neural network model
        config_path: Path to config file
        max_visits: MCTS visits per move
        output_file: Optional file to write winning trajectories
        processed_log_file: Optional file to log all processed task_ids
        workers: Number of parallel workers

    Returns:
        List of all generated trajectories
    """
    if len(task_ids) != len(seeds):
        raise ValueError("task_ids and seeds must have same length")

    all_trajectories = []
    wins = 0
    losses = 0
    file_lock = threading.Lock()

    # Prepare task arguments
    task_args = [
        (task_id, seed, katahex_path, model_path, config_path, max_visits)
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
            task_id, seed, _, _, _, _ = arg
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
    parser = argparse.ArgumentParser(description='Generate Hex dataset using KataHex')
    parser.add_argument('--merged_file', type=str, default='data/merged.json',
                        help='JSON file with merged evaluation data')
    parser.add_argument('--max_visits', type=int, default=500,
                        help='KataHex MCTS visits per move')
    parser.add_argument('--output_file', type=str, default='data/openspiel/hex_golden.jsonl',
                        help='Output file path')
    parser.add_argument('--processed_log_file', type=str, default='log/data/openspiel/hex_processed.log',
                        help='Log file to track all processed task_ids (wins and losses)')
    parser.add_argument('--katahex_path', type=str,
                        default='external/katahex/build/katahex',
                        help='Path to katahex binary')
    parser.add_argument('--model_path', type=str,
                        default='external/katahex/katahex_model_20220618.bin.gz',
                        help='Path to neural network model')
    parser.add_argument('--config_path', type=str,
                        default='external/katahex/config_deterministic.cfg',
                        help='Path to config file (use config_deterministic.cfg for reproducible output)')
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
        task_ids = [random.randint(600_000_000, 699_999_999) for _ in range(args.num_random)]
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

        # Get hex tasks from sampling list
        hex_tasks = sampling_list.get('GAME', {}).get('hex', [])

        # Filter out already processed tasks
        task_ids = []
        seeds = []
        skipped_existing = 0

        for task in hex_tasks:
            task_id = task['task_id']
            seed = task['seed']
            if task_id in processed_task_ids:
                skipped_existing += 1
                continue
            task_ids.append(task_id)
            seeds.append(seed)

        print(f"Found {len(hex_tasks)} hex tasks in merged.json")
        print(f"Skipped {skipped_existing} already processed tasks")
        print(f"Remaining {len(task_ids)} new hex tasks to process")

    if args.limit:
        task_ids = task_ids[:args.limit]
        seeds = seeds[:args.limit]

    if len(task_ids) == 0:
        print("No tasks to process. Exiting.")
        return

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.processed_log_file) or '.', exist_ok=True)

    print(f"\nKataHex Dataset Generation")
    print(f"=" * 50)
    print(f"Merged file: {args.merged_file}")
    print(f"Number of tasks: {len(task_ids)}")
    print(f"Max visits: {args.max_visits}")
    print(f"Workers: {args.workers}")
    print(f"Output file: {args.output_file}")
    print(f"Processed log file: {args.processed_log_file}")
    print()

    trajectories = generate_trajectories_batch(
        task_ids=task_ids,
        seeds=seeds,
        katahex_path=args.katahex_path,
        model_path=args.model_path,
        config_path=args.config_path,
        max_visits=args.max_visits,
        output_file=args.output_file,
        processed_log_file=args.processed_log_file,
        workers=args.workers,
    )

    print(f"\nGeneration complete!")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Output file: {args.output_file}")
    print(f"Processed log file: {args.processed_log_file}")


if __name__ == "__main__":
    main()

