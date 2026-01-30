#!/usr/bin/env python3
"""
Tokenize datasets for SFT training and save to disk.

This script tokenizes all datasets once and saves them, so the multi-GPU
training script can load pre-tokenized data without redundant processing.

Usage:
    python scripts/sft/tokenize_datasets.py --config config/sft_reasoning_trl.yaml
    python scripts/sft/tokenize_datasets.py --config config/sft_reasoning_trl.yaml --output_dir data/tokenized
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

# Import from the training script
from sft_multigpu import (
    load_config,
    load_conversations_for_sft,
    create_tokenized_dataset,
)
from datasets import concatenate_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data/tokenized",
                        help="Directory to save tokenized datasets")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=128,
                        help="Number of processes for tokenization")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.model_name:
        config.model_name = args.model_name

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {config}")
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set the same chat template as training script
    tokenizer.chat_template = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1, thinking=false) %}
{%- for message in messages[::-1] %}
    {%- if message.role == "system" and message.content and message.content | trim != "" %}
        {%- set ns.thinking = true %}
    {%- endif %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if (loop.last and message.reasoning_content is defined) or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {%- if ns.thinking %}
        {{- '<|im_start|>assistant\n' }}
    {%- else %}
        {{- '<|im_start|>assistant\n<think>' }}
    {%- endif %}
{%- endif %}"""

    # Save tokenizer for reference
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    print(f"Tokenizer saved to {output_dir / 'tokenizer'}")

    # Load and tokenize each dataset
    print("\n" + "="*60)
    print("Loading and tokenizing datasets...")
    print("="*60)

    # Anchor dataset
    print("\n[1/4] Processing anchor dataset...")
    anchor_dataset = load_conversations_for_sft(data_dir=config.data_dir + "/anchor", reasoning=True)
    tokenized_anchor = create_tokenized_dataset(
        conversations=list(anchor_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        num_proc=args.num_proc,
        reasoning=True,
    )
    tokenized_anchor.save_to_disk(str(output_dir / "anchor"))
    print(f"  Saved {len(tokenized_anchor)} samples to {output_dir / 'anchor'}")

    # Dyck dataset
    print("\n[2/4] Processing dyck dataset...")
    dyck_dataset = load_conversations_for_sft(data_dir=config.data_dir + "/dyck", reasoning=True)
    tokenized_dyck = create_tokenized_dataset(
        conversations=list(dyck_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        num_proc=args.num_proc,
        reasoning=True,
    )
    tokenized_dyck.save_to_disk(str(output_dir / "dyck"))
    print(f"  Saved {len(tokenized_dyck)} samples to {output_dir / 'dyck'}")

    # Game reasoning dataset
    print("\n[3/4] Processing game reasoning dataset...")
    game_reasoning_dataset = load_conversations_for_sft(
        data_dir=config.data_dir + "/openspiel/reasoning", reasoning=True
    )
    tokenized_game_reasoning = create_tokenized_dataset(
        conversations=list(game_reasoning_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        num_proc=args.num_proc,
        reasoning=True,
    )
    tokenized_game_reasoning.save_to_disk(str(output_dir / "game_reasoning"))
    print(f"  Saved {len(tokenized_game_reasoning)} samples to {output_dir / 'game_reasoning'}")

    # Game non-reasoning dataset
    print("\n[4/4] Processing game non-reasoning dataset...")
    game_none_reasoning_dataset = load_conversations_for_sft(
        data_dir=config.data_dir + "/openspiel/non-reasoning", reasoning=False
    )
    tokenized_game_none_reasoning = create_tokenized_dataset(
        conversations=list(game_none_reasoning_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        num_proc=args.num_proc,
        reasoning=False,
    )
    tokenized_game_none_reasoning.save_to_disk(str(output_dir / "game_none_reasoning"))
    print(f"  Saved {len(tokenized_game_none_reasoning)} samples to {output_dir / 'game_none_reasoning'}")

    # Create combined train/eval splits
    print("\n" + "="*60)
    print("Creating combined train/eval splits...")
    print("="*60)

    # Split game_none_reasoning for eval
    game_split = tokenized_game_none_reasoning.train_test_split(test_size=300, seed=42)
    tokenized_game_none_reasoning_train = game_split["train"]
    tokenized_game_none_reasoning_test = game_split["test"]

    # Upsample and combine
    upsampled_datasets = []
    for dataset, multiplier, name in [
        (tokenized_anchor, 5, "anchor"),
        (tokenized_dyck, 1, "dyck"),
        (tokenized_game_reasoning, 2, "game_reasoning"),
        (tokenized_game_none_reasoning_train, 1, "game_none_reasoning_train"),
    ]:
        print(f"  {name}: {len(dataset)} samples x {multiplier} = {len(dataset) * multiplier}")
        for _ in range(multiplier):
            upsampled_datasets.append(dataset)

    combined_dataset = concatenate_datasets(upsampled_datasets).shuffle(seed=42)
    print(f"\nTotal combined samples: {len(combined_dataset)}")

    # Split into train/eval
    split = combined_dataset.train_test_split(test_size=1000, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Save final datasets
    train_dataset.save_to_disk(str(output_dir / "train"))
    eval_dataset.save_to_disk(str(output_dir / "eval"))
    tokenized_game_none_reasoning_test.save_to_disk(str(output_dir / "non_reasoning_eval"))

    print(f"\n" + "="*60)
    print("DONE! Saved datasets:")
    print("="*60)
    print(f"  Train:              {output_dir / 'train'} ({len(train_dataset)} samples)")
    print(f"  Eval (validation):  {output_dir / 'eval'} ({len(eval_dataset)} samples)")
    print(f"  Eval (non-reason):  {output_dir / 'non_reasoning_eval'} ({len(tokenized_game_none_reasoning_test)} samples)")
    print(f"\nYou can now run training with:")
    print(f"  accelerate launch scripts/sft/sft_multigpu.py --config <config> --tokenized_data_dir {output_dir}")


if __name__ == "__main__":
    main()

