#!/usr/bin/env python3
"""
SFT Reasoning Trainer using TRL SFTTrainer with Packing.

Loads JSONL conversation files and trains with SFTTrainer.
Uses packing for efficient training (up to 2-3x speedup).

Usage:
    python scripts/sft/sft_reasoning_trl_custom.py --config config/sft_reasoning_trl.yaml
    python scripts/sft/sft_reasoning_trl_custom.py \
    --config config/sft_reasoning_trl.yaml \
    --resume_from_checkpoint outputs/sft_reasoning_trl/checkpoint-3000
    
    accelerate config
    accelerate launch scripts/sft/sft_reasoning_trl_custom.py --config config/sft_reasoning_trl.yaml
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


@dataclass
class TrainingConfig:
    """Configuration for SFT training."""
    # Model
    model_name: str = "Qwen/Qwen3-14B"
    max_seq_length: int = 32768
    load_in_4bit: bool = False

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Training
    output_dir: str = "outputs/sft_reasoning_trl"
    data_dir: str = "data_reasoning/openspiel"  # Directory with JSONL files
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Efficiency
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "game-reasoning-sft"
    wandb_run_name: Optional[str] = None


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Load config from YAML file."""
    config = TrainingConfig()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return config


def filter_nan_text_rows(model, dataset, tokenizer, text_column="text", max_length=8192, device="cuda"):
    """
    Filter out rows that cause NaN in model outputs.

    Args:
        model: The model to test with
        dataset: Dataset to filter
        tokenizer: Tokenizer for the model
        text_column: Column name containing text
        max_length: Maximum sequence length for tokenization
        device: Device to run inference on

    Returns:
        List of valid dataset rows (can be converted back to Dataset)
    """
    model.eval()
    valid_dataset = []

    print(f"Scanning for NaN-causing rows in column: '{text_column}'...")

    for i in tqdm(range(len(dataset)), desc="Filtering NaN rows"):
        raw_text = dataset[i][text_column]

        # Tokenize the row
        inputs = tokenizer(
            raw_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)

        # Test for NaN
        with torch.no_grad():
            try:
                outputs = model(**inputs)
                logits = outputs.logits

                if not torch.isnan(logits).any():
                    valid_dataset.append(dataset[i])
                else:
                    print(f"\nRow {i} triggered NaN. Preview: {raw_text[:100]}...")
            except Exception as e:
                print(f"\nRow {i} caused an error: {e}")

    print(f"Filtered: {len(dataset)} -> {len(valid_dataset)} rows ({len(dataset) - len(valid_dataset)} removed)")
    return valid_dataset


def tokenize_conversation_with_labels(
    conversation: list,
    tokenizer,
    max_length: int = 32768,
    instruction_marker: str = "<|im_start|>",
    response_marker: str = "<|im_start|>assistant\n",
) -> dict:
    """
    Tokenize a conversation and create labels with prompt tokens masked.

    Args:
        conversation: List of message dicts with 'role' and 'content'
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        instruction_marker: String that marks the start of a new turn (used to find end of response)
        response_marker: String that marks the start of assistant response

    Returns:
        Dict with 'input_ids', 'labels', and 'attention_mask'
    """
    # Apply chat template to get full text
    full_text = tokenizer.apply_chat_template(conversation, tokenize=False)

    # Tokenize the full conversation
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,  # Return lists, not tensors
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Create labels: start with all -100 (masked)
    labels = [-100] * len(input_ids)

    # Find all assistant responses using response_marker and instruction_marker
    # Unmask tokens between response_marker and the next instruction_marker (or end of text)
    search_start = 0
    while True:
        response_start = full_text.find(response_marker, search_start)
        if response_start == -1:
            break

        # Find the end of this response (next instruction_marker or end of text)
        response_content_start = response_start + len(response_marker)
        next_instruction = full_text.find(instruction_marker, response_content_start)
        if next_instruction == -1:
            response_end = len(full_text)
        else:
            response_end = next_instruction

        # Tokenize text up to response content start to find token index
        prefix_text = full_text[:response_content_start]
        prefix_tokenized = tokenizer(
            prefix_text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        response_start_idx = len(prefix_tokenized["input_ids"]) if response_marker == "<|im_start|>assistant\n" else len(prefix_tokenized["input_ids"]) - 1

        # Tokenize text up to response end to find token index
        prefix_end_text = full_text[:response_end]
        prefix_end_tokenized = tokenizer(
            prefix_end_text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        response_end_idx = len(prefix_end_tokenized["input_ids"])

        # Unmask the response tokens (copy input_ids to labels for response part)
        for i in range(response_start_idx, min(response_end_idx, len(labels))):
            labels[i] = input_ids[i]

        search_start = response_end

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def create_tokenized_dataset(
    conversations: list,
    tokenizer,
    max_length: int = 32768,
    num_proc: int = 128,
    reasoning: bool = True
) -> Dataset:
    """
    Create a tokenized dataset with input_ids and labels from conversations.

    Args:
        conversations: List of conversations (each is a list of message dicts)
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        num_proc: Number of processes for parallel processing

    Returns:
        Dataset with 'input_ids', 'labels', and 'attention_mask'
    """
    # Create initial dataset with conversations
    dataset = Dataset.from_list([{"conversation": conv} for conv in conversations])

    response_marker = "<|im_start|>assistant\n<think>" if reasoning else "<|im_start|>assistant\n"

    def tokenize_fn(examples):
        """Tokenize a batch of conversations."""
        results = {"input_ids": [], "labels": [], "attention_mask": []}

        for conv in examples["conversation"]:
            sample = tokenize_conversation_with_labels(
                conversation=conv,
                tokenizer=tokenizer,
                max_length=max_length,
                response_marker=response_marker,
            )
            results["input_ids"].append(sample["input_ids"])
            results["labels"].append(sample["labels"])
            results["attention_mask"].append(sample["attention_mask"])

        return results

    print(f"Tokenizing {len(conversations)} conversations with {num_proc} processes...")
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        remove_columns=["conversation"],
        desc="Tokenizing",
    )

    return tokenized_dataset


def extract_sub_conversations(conversation: list) -> list:
    """
    Extract sub-conversations from a multi-turn conversation.

    For a conversation with N assistant messages, creates N sub-conversations,
    each ending at one assistant response. This ensures:
    1. Each assistant's reasoning content is properly preserved
    2. The model learns from ALL assistant turns, not just the last one

    Example:
        Input: [user1, assistant1, user2, assistant2, user3, assistant3]
        Output: [
            [user1, assistant1],                                    # Train on assistant1
            [user1, assistant1, user2, assistant2],                 # Train on assistant2
            [user1, assistant1, user2, assistant2, user3, assistant3]  # Train on assistant3
        ]

    Args:
        conversation: Full multi-turn conversation

    Returns:
        List of sub-conversations, each ending at an assistant message
    """
    sub_conversations = []

    for idx, msg in enumerate(conversation):
        if msg.get('role') == 'assistant':
            # Ensure assistant message has reasoning_content (default to empty string)
            if 'reasoning_content' not in msg:
                msg['reasoning_content'] = ""
            # Create sub-conversation up to and including this assistant message
            sub_conv = conversation[:idx + 1]
            sub_conversations.append(sub_conv)

    return sub_conversations


def load_conversations_for_sft(data_dir: str, max_samples_per_file: int = 5000, reasoning: bool = True) -> Dataset:
    """
    Load JSONL conversation files for SFTTrainer.

    For multi-turn conversations with multiple assistant messages (each with reasoning),
    we extract sub-conversations to ensure each assistant turn is trained on properly.

    SFTTrainer expects a dataset with 'messages' field containing conversations
    in the standard chat format: [{"role": "user/assistant/system", "content": "..."}]

    Args:
        data_dir: Directory containing JSONL files
        max_samples_per_file: Maximum number of messages to load from each file (default: 5000)

    Returns:
        Dataset with 'messages' field for each sub-conversation
    """
    data_path = Path(data_dir)
    samples = []

    # Load all conversations
    print("Loading JSONL files...")
    jsonl_files = list(data_path.rglob("*.jsonl"))
    for jsonl_file in tqdm(jsonl_files, desc="Loading files"):
        file_samples = []
        with open(jsonl_file) as f:
            for line in f:
                if len(file_samples) >= max_samples_per_file:
                    break
                if line.strip():
                    traj = json.loads(line)
                    conv = traj.get("conversation", []) or (traj.get("extra") or {}).get("conversation", [])
                    if conv:
                        if reasoning:
                            # Extract sub-conversations for each assistant turn
                            sub_convs = extract_sub_conversations(conv)
                            for sub_conv in sub_convs:
                                # SFTTrainer expects 'messages' field
                                file_samples.append({"messages": sub_conv})
                        else:
                            # Remove reasoning_content from last message when reasoning is disabled
                            cleaned_conv = []
                            for msg in conv:
                                cleaned_msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
                                cleaned_conv.append(cleaned_msg)
                            file_samples.append({"messages": cleaned_conv})

        print(f"  Loaded {len(file_samples)} samples from {jsonl_file.name}")
        samples.extend(file_samples)

    print(f"Loaded {len(samples)} training samples from conversations")
    return Dataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint directory to resume training from")
    parser.add_argument("--ignore_data_skip", action="store_true",
                        help="When resuming, don't skip data already seen (restart data from beginning)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model_name:
        config.model_name = args.model_name

    print(f"Config: {config}")

    # Setup wandb (only on main process to avoid duplicate logging)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if config.use_wandb and local_rank == 0:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Check if resuming from checkpoint
    resume_from_checkpoint = args.resume_from_checkpoint

    # Load model with Unsloth for efficiency
    if resume_from_checkpoint:
        # Resume from LoRA checkpoint - load the adapter
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=resume_from_checkpoint,  # Load from checkpoint (has adapter_config.json)
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )
    else:
        # Fresh training - load base model
        print(f"Loading model: {config.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )

    # Use left padding for LLM training (consistent with inference)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
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

    # Add LoRA adapters only if not resuming (checkpoint already has adapters)
    if not resume_from_checkpoint:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth" if config.gradient_checkpointing else False,
        )

    # Load conversations for SFTTrainer
    anchor_dataset = load_conversations_for_sft(data_dir=config.data_dir + "/anchor", reasoning=True)
    dyck_dataset = load_conversations_for_sft(data_dir=config.data_dir + "/dyck", reasoning=True)
    print_dataset = load_conversations_for_sft(data_dir=config.data_dir + "/print", reasoning=True)
    game_reasoning_dataset = load_conversations_for_sft(data_dir=config.data_dir + "/game_reasoning", reasoning=True)
    game_none_reasoning_dataset = load_conversations_for_sft(data_dir=config.data_dir + "/game_none_reasoning", reasoning=False)
    gold_dataset = load_conversations_for_sft(data_dir=config.data_dir + "/gold", reasoning=False)

    tokenized_anchor_dataset = create_tokenized_dataset(
        conversations=list(anchor_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        reasoning=True,
    )
    
    tokenized_dyck_dataset = create_tokenized_dataset(
        conversations=list(dyck_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        reasoning=True,
    )
    
    tokenized_print_dataset = create_tokenized_dataset(
        conversations=list(print_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        reasoning=True,
    )
    
    tokenized_game_reasoning_dataset = create_tokenized_dataset(
        conversations=list(game_reasoning_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        reasoning=True,
    )
    
    tokenized_game_none_reasoning_dataset = create_tokenized_dataset(
        conversations=list(game_none_reasoning_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        reasoning=False,
    )
    
    tokenized_gold_dataset = create_tokenized_dataset(
        conversations=list(gold_dataset["messages"]),
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        reasoning=False,
    )
    
    # Upsample datasets with multipliers
    # anchor * 2, dyck * 1, print * 3, game_reasoning * 1, game_none_reasoning * 1, gold * 30
    upsampled_datasets = []
    for dataset, multiplier in [
        (tokenized_anchor_dataset, 2),
        (tokenized_dyck_dataset, 1),
        (tokenized_print_dataset, 3),
        (tokenized_game_reasoning_dataset, 1),
        (tokenized_game_none_reasoning_dataset, 1),
        (tokenized_gold_dataset, 30),
    ]:
        for _ in range(multiplier):
            upsampled_datasets.append(dataset)

    tokenized_dataset = concatenate_datasets(upsampled_datasets).shuffle(seed=42)

    # Split into train/eval
    split = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Use multiple eval datasets: main validation + gold dataset for monitoring
    eval_datasets = {
        "validation": eval_dataset,
        "gold": tokenized_gold_dataset,  # Monitor performance on gold dataset separately
    }

    print(f"Train: {len(train_dataset)}, Eval (validation): {len(eval_dataset)}, Eval (gold): {len(tokenized_gold_dataset)}")

    # SFTConfig - no dataset_text_field since we provide pre-tokenized input_ids/labels
    training_args = SFTConfig(
        output_dir=config.output_dir,
        packing=False,  # Packing not compatible with pre-tokenized data
        optim="adamw_8bit",
        # Training settings
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="linear",
        weight_decay=0.001,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        # Logging
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_loss",  # Use validation split (not gold) for best model selection
        greater_is_better=False,
        report_to="wandb" if config.use_wandb else "none",
        dataloader_pin_memory=True,
        # No dataset_text_field - we provide pre-tokenized input_ids and labels
    )

    # Create SFTTrainer - no train_on_responses_only needed since labels are pre-masked
    # Use eval_dataset dict to monitor both validation and gold dataset separately
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,  # Dict of multiple eval datasets
        processing_class=tokenizer,
    )
    
    # Train!
    print("Starting training...")
    if resume_from_checkpoint:
        # Delete optimizer/scheduler state files so trainer uses fresh LR from config
        opt_path = Path(resume_from_checkpoint) / "optimizer.pt"
        sched_path = Path(resume_from_checkpoint) / "scheduler.pt"
        if opt_path.exists():
            print(f"Removing {opt_path} to use fresh learning rate from config")
            opt_path.unlink()
        if sched_path.exists():
            print(f"Removing {sched_path} to use fresh scheduler from config")
            sched_path.unlink()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    # Save merged model (optional)
    merged_dir = Path(config.output_dir) / "merged"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    print("Training complete!")


if __name__ == "__main__":
    main()
