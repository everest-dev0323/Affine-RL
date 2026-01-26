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
"""

import os
import sys
import subprocess
import json
import argparse
from typing import Optional, List, Dict, Any

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


# Create Actor instance for using _create_opponent_bot
_actor = Actor()


def generate_katahex_trajectory(
    task_id: int,
    seed: int,
    katahex: KataHexGTP,
    max_visits: int = 500,
) -> List[Dict[str, Any]]:
    """
    Generate winning trajectory using KataHex vs MCTS opponent.

    - KataHex plays as the "learner" (same player_id as LLM in evaluation)
    - MCTS bot plays as opponent (exact same seed as evaluation)

    This ensures the trajectory teaches how to beat the exact MCTS opponent
    that will be used during evaluation.

    Args:
        task_id: Task ID (determines board size for hex)
        seed: Random seed (same as evaluation seed)
        katahex: KataHexGTP instance
        max_visits: MCTS visits per move for KataHex

    Returns:
        List with single trajectory dict for KataHex player
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

    # Setup KataHex
    katahex.set_board_size(board_size)
    katahex.clear_board()

    # Create OpenSpiel game for state tracking
    game = pyspiel.load_game("hex", {"board_size": board_size})
    state = game.new_initial_state()

    # Create MCTS opponent using Actor._create_opponent_bot (exact same as evaluation)
    # In evaluation: _create_opponent_bot(opponent, player_id, seed + 2 + player_id, game, agent)
    mcts_opponent = _actor._create_opponent_bot(
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
                break

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

    # Build trajectory for KataHex player only
    trajectory = {
        'conversation': conversation,
        'score': score,
        'success': score > 0.5,
        'task_id': task_id,
        'seed': seed,
        'game_name': 'hex',
        'player_id': katahex_player_id,
        'returns': list(returns),
        'board_size': board_size,
        'total_moves': len(moves),
        'generator': 'katahex',
        'max_visits': max_visits,
        'katahex_won': score > 0.5,
    }

    return [trajectory]


def generate_trajectories_batch(
    task_ids: List[int],
    seeds: List[int],
    katahex_path: str = "external/katahex/build/katahex",
    model_path: str = "external/katahex/katahex_model_20220618.bin.gz",
    config_path: str = "external/katahex/config.cfg",
    max_visits: int = 500,
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate multiple trajectories using KataHex.

    Args:
        task_ids: List of task IDs
        seeds: List of seeds (same length as task_ids)
        katahex_path: Path to katahex binary
        model_path: Path to neural network model
        config_path: Path to config file
        max_visits: MCTS visits per move
        output_file: Optional file to write results

    Returns:
        List of all generated trajectories
    """
    if len(task_ids) != len(seeds):
        raise ValueError("task_ids and seeds must have same length")

    katahex = KataHexGTP(
        katahex_path=katahex_path,
        model_path=model_path,
        config_path=config_path,
        max_visits=max_visits,
    )

    all_trajectories = []

    print(f"Starting KataHex...")
    if not katahex.start():
        print("Failed to start KataHex")
        return []

    try:
        from tqdm import tqdm
        iterator = tqdm(zip(task_ids, seeds), total=len(task_ids), desc="Generating")
    except ImportError:
        iterator = zip(task_ids, seeds)

    try:
        for task_id, seed in iterator:
            try:
                trajectories = generate_katahex_trajectory(
                    task_id=task_id,
                    seed=seed,
                    katahex=katahex,
                    max_visits=max_visits,
                )
                all_trajectories.extend(trajectories)

                # Write to file if specified
                if output_file:
                    with open(output_file, 'a') as f:
                        for traj in trajectories:
                            f.write(json.dumps(traj) + '\n')

            except Exception as e:
                print(f"Error generating trajectory for task_id={task_id}, seed={seed}: {e}")

    finally:
        katahex.stop()

    return all_trajectories



def main():
    parser = argparse.ArgumentParser(description='Generate Hex dataset using KataHex')
    parser.add_argument('--tasks_file', type=str, default='data/openspiel/hex_tasks.jsonl',
                        help='JSONL file with task_id and seed pairs')
    parser.add_argument('--max_visits', type=int, default=500,
                        help='KataHex MCTS visits per move')
    parser.add_argument('--output_file', type=str, default='data/openspiel/hex_katahex_dataset.jsonl',
                        help='Output file path')
    parser.add_argument('--katahex_path', type=str,
                        default='external/katahex/build/katahex',
                        help='Path to katahex binary')
    parser.add_argument('--model_path', type=str,
                        default='external/katahex/katahex_model_20220618.bin.gz',
                        help='Path to neural network model')
    parser.add_argument('--config_path', type=str,
                        default='external/katahex/config.cfg',
                        help='Path to config file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tasks to process')
    args = parser.parse_args()

    # Read task_id and seed pairs from file
    task_ids = []
    seeds = []

    with open(args.tasks_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                task_ids.append(data['task_id'])
                seeds.append(data['seed'])

    if args.limit:
        task_ids = task_ids[:args.limit]
        seeds = seeds[:args.limit]

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)

    # Clear output file if exists
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    print(f"KataHex Dataset Generation")
    print(f"=" * 50)
    print(f"Tasks file: {args.tasks_file}")
    print(f"Number of tasks: {len(task_ids)}")
    print(f"Max visits: {args.max_visits}")
    print(f"Output file: {args.output_file}")
    print()

    trajectories = generate_trajectories_batch(
        task_ids=task_ids,
        seeds=seeds,
        katahex_path=args.katahex_path,
        model_path=args.model_path,
        config_path=args.config_path,
        max_visits=args.max_visits,
        output_file=args.output_file,
    )

    print(f"\nGeneration complete!")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Output file: {args.output_file}")


if __name__ == "__main__":
    main()

