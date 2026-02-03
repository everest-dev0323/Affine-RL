#!/usr/bin/env python3
"""
Generate Othello training dataset from sample JSONL file.

Input format (each line is a JSON array):
[{"move": "e6", "player": "black", "probabilities": {"c4": 0.25, ...}}, ...]

Output format (JSONL, one per turn):
{
    "prompt": [{"role": "system", ...}, {"role": "user", ...}, ...],
    "label": "26",
    "probabilities": {"c4": 0.25, ...},
    "env": "openspiel",
    "task": "othello"
}

Usage:
    python scripts/data/generate_othello_dataset_from_sample.py
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AffineFoundation/affinetes/environments/openspiel'))

import pyspiel
from agents import GAME_AGENTS


def othello_coord_to_action(coord: str) -> int:
    """Convert Othello coordinate (e.g., 'c4', 'pass') to OpenSpiel action index."""
    if coord.lower() == 'pass':
        return 64

    col_char = coord[0].lower()
    row_str = coord[1:]

    col = ord(col_char) - ord('a')
    row = int(row_str) - 1

    return row * 8 + col


def generate_dataset_for_player(input_file: str, output_file: str, target_player: int):
    """Generate dataset from sample file for a specific player.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        target_player: 0 for black, 1 for white
    """
    game = pyspiel.load_game("othello")
    agent = GAME_AGENTS['othello']()
    system_prompt = agent.generate_system_prompt()

    total_samples = 0
    total_games = 0

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in, desc=f"Processing games ({'black' if target_player == 0 else 'white'})"):
            line = line.strip()
            if not line:
                continue

            try:
                moves = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(moves, list) or len(moves) == 0:
                continue

            total_games += 1
            state = game.new_initial_state()

            # Conversation history for target player only
            history = [{"role": "system", "content": system_prompt}]

            for move_data in moves:
                if state.is_terminal():
                    break

                current_player = state.current_player()
                move_str = move_data['move']
                player_str = move_data['player']
                probabilities = move_data.get('probabilities', {})

                # Verify player matches
                expected_player = 0 if player_str == 'black' else 1
                if current_player != expected_player:
                    continue

                try:
                    action = othello_coord_to_action(move_str)
                except:
                    break

                legal_actions = state.legal_actions(current_player)
                if action not in legal_actions:
                    continue

                # Only generate samples for target player
                if current_player == target_player:
                    # Generate user prompt
                    user_prompt = agent.generate_user_prompt(
                        state=state,
                        player_id=current_player,
                        legal_actions=legal_actions
                    )

                    # Build prompt: history + current user message
                    prompt = history + [{"role": "user", "content": user_prompt}]

                    # Write sample
                    sample = {
                        "prompt": prompt,
                        "label": str(action),
                        "probabilities": probabilities,
                        "env": "openspiel",
                        "task": "othello",
                    }
                    f_out.write(json.dumps(sample) + '\n')
                    total_samples += 1

                    # Update history with this turn
                    history.append({"role": "user", "content": user_prompt})
                    history.append({"role": "assistant", "content": str(action)})

                # Apply action (always, to keep game state in sync)
                state.apply_action(action)

    print(f"\nGeneration complete for {'black' if target_player == 0 else 'white'}!")
    print(f"Total games: {total_games}")
    print(f"Total samples: {total_samples}")

    return total_samples


def generate_dataset(input_file: str, output_file: str):
    """Generate dataset from sample file for both players into one file."""
    game = pyspiel.load_game("othello")
    agent = GAME_AGENTS['othello']()
    system_prompt = agent.generate_system_prompt()

    total_samples = 0
    total_games = 0

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in, desc="Processing games"):
            line = line.strip()
            if not line:
                continue

            try:
                moves = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(moves, list) or len(moves) == 0:
                continue

            total_games += 1
            state = game.new_initial_state()

            # Separate conversation history for each player
            histories = {
                0: [{"role": "system", "content": system_prompt}],  # black
                1: [{"role": "system", "content": system_prompt}],  # white
            }

            for move_data in moves:
                if state.is_terminal():
                    break

                current_player = state.current_player()
                move_str = move_data['move']
                player_str = move_data['player']
                probabilities = move_data.get('probabilities', {})

                # Verify player matches
                expected_player = 0 if player_str == 'black' else 1
                if current_player != expected_player:
                    continue

                try:
                    action = othello_coord_to_action(move_str)
                except:
                    break

                legal_actions = state.legal_actions(current_player)
                if action not in legal_actions:
                    continue

                # Generate user prompt
                user_prompt = agent.generate_user_prompt(
                    state=state,
                    player_id=current_player,
                    legal_actions=legal_actions
                )

                # Build prompt: history + current user message
                prompt = histories[current_player] + [{"role": "user", "content": user_prompt}]

                # Convert probabilities keys from coordinates to action strings
                converted_probabilities = {}
                for coord, prob in probabilities.items():
                    action_id = othello_coord_to_action(coord)
                    converted_probabilities[str(action_id)] = prob

                # Write sample
                sample = {
                    "prompt": prompt,
                    "label": str(action),
                    "probabilities": converted_probabilities,
                    "env": "openspiel",
                    "task": "othello",
                }
                f_out.write(json.dumps(sample) + '\n')
                total_samples += 1

                # Update history for this player
                histories[current_player].append({"role": "user", "content": user_prompt})
                histories[current_player].append({"role": "assistant", "content": str(action)})

                # Apply action (always, to keep game state in sync)
                state.apply_action(action)

    print(f"\nGeneration complete!")
    print(f"Total games: {total_games}")
    print(f"Total samples: {total_samples}")

    return total_samples


def main():
    parser = argparse.ArgumentParser(description='Generate Othello dataset from sample file')
    parser.add_argument('--input', type=str, default='dataset/sample/openspiel/othello_sample.jsonl',
                        help='Input sample file')
    parser.add_argument('--output', type=str, default='data/openspiel/othello_dataset.jsonl',
                        help='Output dataset file')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print(f"Othello Dataset Generation")
    print(f"=" * 50)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()

    count = generate_dataset(args.input, args.output)
    print(f"\nDataset saved to: {args.output}")
    print(f"Total samples: {count}")


if __name__ == "__main__":
    main()

