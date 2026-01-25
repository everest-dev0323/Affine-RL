#!/usr/bin/env python3
"""
Generate reasoning content for SFT datasets using DeepSeek API.

Takes existing expert conversation datasets (without reasoning) and generates
reasoning content for each assistant response using DeepSeek-chat model.

Supports:
- Multiple data directories (lgc-v2, openspiel, print)
- Parallel processing with asyncio + semaphore
- Resume from interrupted runs
- Streaming writes to output files

python scripts/data/generate_reasoning_dataset.py --file openspiel/goofspiel.jsonl --workers 10 --max_items 455

python scripts/data/generate_reasoning_dataset.py --file openspiel/goofspiel.jsonl --workers 10 --winner_only
"""

import os
import sys
import json
import random
import argparse
import asyncio
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import httpx
from dotenv import load_dotenv


# DeepSeek API configuration
# DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_URL = "https://api.openai.com/v1/chat/completions"
# DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MODEL = "gpt-5.2"


REASONING_SYSTEM_PROMPT = """You are solving a problem. Given a conversation history and the correct answer, generate strategic reasoning that leads to that answer.

Write thinking in FIRST PERSON. Keep it MODERATE length (4-8 sentences). Do NOT:
- Say "the expert chose" or "the answer is"
- Reference that you already know the answer
- Repeat the full game state or problem description

DO:
- Consider what the opponent might do and why
- Analyze the strategic tradeoffs between options
- Explain the key insight that makes this move strong
- Arrive at the answer through logical deduction

Output ONLY the thinking wrapped in <think> tags. No final answer."""


async def call_deepseek_api(
    client: httpx.AsyncClient,
    messages: List[Dict[str, str]],
    api_key: str,
    model: str = DEFAULT_MODEL,
    # temperature: float = 0.7,
    # max_tokens: int = 4096,
) -> str:
    """Call DeepSeek API and return the reasoning content."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        # "temperature": temperature,
        # "max_tokens": max_tokens,
    }
    
    response = await client.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"] # , result["choices"][0]["message"].get("reasoning_content", "")


def build_reasoning_prompt(conversation: List[Dict[str, str]], assistant_idx: int, game_name: str, player_id: int) -> List[Dict[str, str]]:
    """Build prompt for generating reasoning for a specific assistant turn."""
    # Get context up to and including the assistant response
    context = conversation[:assistant_idx + 1]
    
    # Format the conversation for the prompt
    conv_text = ""
    for i, msg in enumerate(context):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "system":
            conv_text += f"[System Message]\n{content}\n\n"
        elif role == "user":
            if i == len(context) -2:
                conv_text += f"[Current State]\n{content}\n\n"
            else:
                conv_text += f"[Previous State]\n{content}\n\n"
        elif role == "assistant":
            if i == len(context) - 1:
                conv_text += f"[Your Answer]\n{content}\n\n"
            else:
                conv_text += f"[Your Previuos Choice]\n{content}\n\n"
    
    if game_name == 'clobber':
        if player_id:
            system_prompt = REASONING_SYSTEM_PROMPT + "\nNOTE: You are player 1. (black pieces, 'x')"
        else:
            system_prompt = REASONING_SYSTEM_PROMPT + "\nNOTE: You are player 0. (white pieces, 'o')"
    elif game_name == 'hex':
        if player_id:
            system_prompt = REASONING_SYSTEM_PROMPT + "\nNOTE: You are player 1. (Blue 'o')"
        else:
            system_prompt = REASONING_SYSTEM_PROMPT + "\nNOTE: You are player 0. (Red 'x')"

        system_prompt = system_prompt + """
NOTE: The Hex board indentation is purely visual. Column positions (a, b, c, d, e) are always counted from the LEFT of each row, ignoring the leading spaces.
Example: In row "  . x . . .", the 'x' is at column b (2nd position), not column c.

BOARD ORIENTATION: The board is displayed as a diamond rotated 45 degrees.
- TOP edge of display = Top-Left side (Red's connection side)
- BOTTOM edge of display = Bottom-Right side (Red's connection side)
- LEFT edge of display = Bottom-Left side (Blue's connection side)
- RIGHT edge of display = Top-Right side (Blue's connection side)

Red (x) wins by connecting TOP edge to BOTTOM edge.
Blue (o) wins by connecting LEFT edge to RIGHT edge."""
    elif game_name == 'othello':
        if player_id:
            system_prompt = REASONING_SYSTEM_PROMPT + "\nNOTE: You are player 1. (White, 'o')"
        else:
            system_prompt = REASONING_SYSTEM_PROMPT + "\nNOTE: You are player 0. (Black, 'x')"
    else:
        system_prompt = REASONING_SYSTEM_PROMPT
        
    return [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": f"Here is the conversation with the correct answer. Generate realistic first-person thinking that leads to this answer:\n\n{conv_text}"},
    ]


async def process_single_item(
    client: httpx.AsyncClient,
    item: Dict[str, Any],
    api_key: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single conversation item and add reasoning."""
    async with semaphore:
        try:
            conversation = item.get("conversation", [])
            new_conversation = []
            game_name = item.get('game_name')
            player_id = item.get('player_id')
            
            for i, msg in enumerate(conversation):
                if msg.get("role") == "assistant":
                    # Generate reasoning for this assistant turn
                    prompt = build_reasoning_prompt(conversation, i, game_name, player_id)
                    reasoning = await call_deepseek_api(
                        client, prompt, api_key, model
                    )
                    
                    # Remove <think> tags from reasoning
                    reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()
                    new_conversation.append({"role": "assistant", "content": msg.get("content"), "reasoning_content": reasoning})
                else:
                    new_conversation.append(msg.copy())
            
            result = item.copy()
            result["conversation"] = new_conversation
            result["has_reasoning"] = True
            return result
            
        except Exception as e:
            return {"error": str(e), "original": item}


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def count_existing(filepath: str) -> int:
    """Count existing items in output file."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


async def process_file(
    input_file: str,
    output_file: str,
    api_key: str,
    model: str,
    max_workers: int,
    max_items: int = None,
    winner_only: bool = False,
) -> int:
    """Process a single JSONL file and generate reasoning."""
    # Load data
    data = load_jsonl(input_file)

    # Filter to winner rollouts only if requested
    if winner_only:
        original_count = len(data)
        data = [item for item in data if item.get("success", False)]
        print(f"  Filtered to winners: {len(data)}/{original_count} items")

    if max_items:
        data = data[:max_items]

    # Shuffle for diversity
    random.shuffle(data)

    # Check existing progress
    existing_count = count_existing(output_file)
    if existing_count > 0:
        print(f"  Resuming from {existing_count} existing items")
        data = data[existing_count:]

    if not data:
        print(f"  Already complete!")
        return existing_count

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_workers)

    # Process with httpx client
    async with httpx.AsyncClient() as client:
        # Process in batches for progress tracking
        total_processed = existing_count
        errors = 0

        with open(output_file, "a", encoding="utf-8") as f:
            # Create tasks for all items
            tasks = [
                process_single_item(client, item, api_key, model, semaphore)
                for item in data
            ]

            # Process with progress bar
            pbar = tqdm(total=len(tasks), desc=f"  Processing")

            for coro in asyncio.as_completed(tasks):
                result = await coro

                if "error" in result:
                    errors += 1
                    print(f"\n  Error: {result['error']}")
                else:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    total_processed += 1

                pbar.update(1)

            pbar.close()

    print(f"  Processed: {total_processed}, Errors: {errors}")
    return total_processed


async def main_async(args):
    """Async main function."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all input files
    input_path = Path(args.input_dir)

    if args.file:
        # Process specific file
        input_files = [input_path / args.file]
    else:
        # Process all jsonl files
        input_files = list(input_path.rglob("*.jsonl"))

    print(f"Reasoning Dataset Generation")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Max workers: {args.workers}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Files to process: {len(input_files)}")
    print(f"=" * 60)

    total_items = 0

    for input_file in input_files:
        if not input_file.exists():
            print(f"Skipping {input_file} (not found)")
            continue

        # Preserve directory structure
        rel_path = input_file.relative_to(input_path)
        output_file = Path(args.output_dir) / rel_path.parent / f"{rel_path.stem}_reasoning.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {input_file}")
        print(f"Output: {output_file}")

        count = await process_file(
            str(input_file),
            str(output_file),
            api_key,
            args.model,
            args.workers,
            args.max_items,
            args.winner_only,
        )
        total_items += count

    print(f"\n{'=' * 60}")
    print(f"Complete! Total items processed: {total_items}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description='Generate reasoning for SFT datasets')
    parser.add_argument('--input_dir', type=str, default='data',
                        help='Input directory containing JSONL files')
    parser.add_argument('--output_dir', type=str, default='data_reasoning',
                        help='Output directory for reasoning-augmented data')
    parser.add_argument('--file', type=str, default=None,
                        help='Specific file to process (relative to input_dir)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='OpenAI model name')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel API calls')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Maximum items to process per file')
    parser.add_argument('--winner_only', action='store_true',
                        help='Only process rollouts where success is true')
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()

