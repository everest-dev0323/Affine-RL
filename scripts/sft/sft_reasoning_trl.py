#!/usr/bin/env python3
"""
SFT Reasoning Trainer using HuggingFace Trainer.

Loads JSONL conversation files, tokenizes them with proper label masking,
and trains with standard Trainer.

Usage:
    python scripts/sft/sft_reasoning_trl.py --config config/sft_reasoning_trl.yaml
    python scripts/sft/sft_reasoning_trl.py --data_dir data_reasoning/openspiel --output_dir outputs/sft_trl
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from unsloth import FastLanguageModel


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


def load_and_tokenize_data(data_dir: str, tokenizer, max_seq_length: int) -> Dataset:
    """
    Load JSONL conversation files and tokenize them.

    Creates input_ids and labels with proper masking:
    - labels = -100 for system/user messages (not trained on)
    - labels = token_id for assistant messages (trained on)
    """
    data_path = Path(data_dir)
    samples = []

    for jsonl_file in data_path.rglob("*.jsonl"):
        print(f"Loading {jsonl_file}")
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    traj = json.loads(line)
                    conv = traj.get("conversation", []) or (traj.get("extra") or {}).get("conversation", [])
                    if conv:
                        samples.append(conv)

    print(f"Loaded {len(samples)} conversations")

    # Tokenize with label masking
    def tokenize_conversation(conversation: List[Dict]) -> List[Dict]:
        """Tokenize a conversation and create labels with masking."""
        
        samples = []

        for idx, msg in enumerate(conversation):
            if msg['role'] == 'assistant':
                input_ids = tokenizer.apply_chat_template(
                    conversation[:idx + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                
                labels = list(input_ids)
                
                prompt_len = len(tokenizer.apply_chat_template(
                    conversation[:idx],
                    tokenize=True,
                    add_generation_prompt=True,
                ))
                
                # Update labels
                labels = [-100] * prompt_len + labels[prompt_len:]
                attention_mask = [1] * len(input_ids)

                # Truncate if needed
                if len(input_ids) > max_seq_length:
                    input_ids = input_ids[:max_seq_length]
                    labels = labels[:max_seq_length]
                    attention_mask = attention_mask[:max_seq_length]
                    
                samples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask
                })

        return samples

    # Process all conversations
    tokenized_samples = []
    for conv in samples:
        tokenized_subsamples = tokenize_conversation(conv)
        if len(tokenized_subsamples) > 0:
            tokenized_samples.extend(tokenized_subsamples)

    print(f"Tokenized {len(tokenized_samples)} samples")

    return Dataset.from_list(tokenized_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
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

    # Load model with Unsloth for efficiency
    print(f"Loading model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )

    # Use left padding for LLM training (consistent with inference)
    tokenizer.padding_side = "left"

    # Add LoRA adapters
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

    # Load JSONL files and tokenize
    dataset = load_and_tokenize_data(config.data_dir, tokenizer, config.max_seq_length)

    # Split into train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Data collator for pre-tokenized data
    # Pads input_ids and labels, handles -100 masking correctly
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,  # Efficient for tensor cores
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if config.use_wandb else "none",
        remove_unused_columns=False,  # Keep input_ids and labels
        dataloader_pin_memory=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Train!
    print("Starting training...")
    trainer.train()

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
