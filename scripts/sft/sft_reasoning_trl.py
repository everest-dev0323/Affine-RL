#!/usr/bin/env python3
"""
SFT Reasoning Trainer using TRL SFTTrainer with Packing.

Loads JSONL conversation files and trains with SFTTrainer.
Uses packing for efficient training (up to 2-3x speedup).

Usage:
    python scripts/sft/sft_reasoning_trl.py --config config/sft_reasoning_trl.yaml
    python scripts/sft/sft_reasoning_trl.py --data_dir data_reasoning/openspiel --output_dir outputs/sft_trl
    python scripts/sft/sft_reasoning_trl.py \
    --config config/sft_reasoning_trl.yaml \
    --resume_from_checkpoint outputs/sft_reasoning_trl/checkpoint-3000
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig


@dataclass
class TrainingConfig:
    """Configuration for SFT training."""
    # Model
    model_name: str = "Qwen/Qwen3-14B"
    max_seq_length: int = 8192
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
            # Create sub-conversation up to and including this assistant message
            sub_conv = conversation[:idx + 1]
            sub_conversations.append(sub_conv)

    return sub_conversations


def load_conversations_for_sft(data_dir: str) -> Dataset:
    """
    Load JSONL conversation files for SFTTrainer.

    For multi-turn conversations with multiple assistant messages (each with reasoning),
    we extract sub-conversations to ensure each assistant turn is trained on properly.

    SFTTrainer expects a dataset with 'messages' field containing conversations
    in the standard chat format: [{"role": "user/assistant/system", "content": "..."}]

    Args:
        data_dir: Directory containing JSONL files

    Returns:
        Dataset with 'messages' field for each sub-conversation
    """
    data_path = Path(data_dir)
    samples = []

    # Load all conversations
    print("Loading JSONL files...")
    jsonl_files = list(data_path.rglob("*.jsonl"))
    for jsonl_file in tqdm(jsonl_files, desc="Loading files"):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    traj = json.loads(line)
                    conv = traj.get("conversation", []) or (traj.get("extra") or {}).get("conversation", [])
                    if conv:
                        # Extract sub-conversations for each assistant turn
                        sub_convs = extract_sub_conversations(conv)
                        for sub_conv in sub_convs:
                            # SFTTrainer expects 'messages' field
                            samples.append({"messages": sub_conv})

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

    # Setup wandb
    if config.use_wandb:
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

    # Load conversations for SFTTrainer (expects 'messages' field or 'text' field)
    dataset = load_conversations_for_sft(
        data_dir=config.data_dir,
    )
    
    reasoning_conversations = tokenizer.apply_chat_template(
        list(dataset["messages"]),
        tokenize = False,
    )

    # Split into train/eval
    split = Dataset.from_dict({"text": reasoning_conversations}).train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Filter out NaN-causing rows from eval dataset
    print("Filtering eval dataset for NaN-causing rows...")
    valid_eval_rows = filter_nan_text_rows(
        model=model,
        dataset=eval_dataset,
        tokenizer=tokenizer,
        text_column="text",
        max_length=config.max_seq_length,
        device="cuda",
    )
    eval_dataset = Dataset.from_list(valid_eval_rows)
    print(f"Eval dataset after filtering: {len(eval_dataset)}")

    # SFTConfig with packing enabled for faster training
    # Packing combines multiple sequences into one to reduce padding waste
    training_args = SFTConfig(
        output_dir=config.output_dir,
        # Packing settings (key for speedup!)
        packing=False,  # Enable packing for 2-3x speedup
        optim="adamw_8bit",
        # Use the tokenizer's actual eos_token (don't let TRL guess)
        eos_token=tokenizer.eos_token,
        # Training settings
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type = "linear",
        weight_decay = 0.001,
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if config.use_wandb else "none",
        dataloader_pin_memory=True,
        # Dataset settings
        dataset_text_field="text", 
        dataset_num_proc=8,  # Parallel tokenization
    )

    # Create SFTTrainer with packing
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    qwen_kwargs = dict(instruction_part = "<|im_start|>", response_part = "<|im_start|>assistant\n<think>\n",)
    trainer = train_on_responses_only(
        trainer,
        **qwen_kwargs,
    )
    
    # Train!
    print("Starting training...")
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
