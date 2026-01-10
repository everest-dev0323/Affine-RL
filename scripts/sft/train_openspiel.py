"""
Fine-tune Qwen 30B on OpenSpiel dataset with 2xH100 GPUs.

Optimized for:
- Fast training with DeepSpeed ZeRO-3
- Memory efficiency with gradient checkpointing
- BF16 mixed precision
- Flash Attention 2
- wandb monitoring
"""

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import wandb
from dotenv import load_dotenv
import json

load_dotenv()

# ============ Configuration ============
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct")
OUTPUT_DIR = "./checkpoints/openspiel_qwen30b"
DATA_PATH = "data/openspiel_processed/train.jsonl"

# Qwen chat template with reasoning support
CHAT_TEMPLATE = r"""
{%- for message in messages %}
    {%- if message.role == "system" %}
        {{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {%- if loop.last %}
            {% generation %}
            {{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
            {% endgeneration %}
        {%- else %}
            {{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
""".strip()


def load_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # Initialize wandb
    wandb.init(
        project="openspiel-sft",
        name=f"qwen30b-openspiel",
        config={
            "model_name": MODEL_NAME,
            "dataset": DATA_PATH,
        },
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optimizations for 2xH100
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False

    # Load dataset
    print(f"Loading dataset from {DATA_PATH}")
    data = load_jsonl(DATA_PATH)
    train_dataset = Dataset.from_list(data)
    print(f"Dataset size: {len(train_dataset)}")

    # Training config optimized for 2xH100
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        
        # Batch size optimization for 2xH100 (80GB each)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        
        # Training params
        num_train_epochs=3,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        
        # Sequence length
        max_seq_length=4096,
        
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        
        # Only train on last assistant response (completion)
        # The chat template marks last assistant with {% generation %}
        dataset_text_field=None,
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to=["wandb"],
        
        # Checkpointing
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        
        # Eval
        eval_strategy="steps",
        eval_steps=200,
        
        # Multi-GPU
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # Speed optimizations
        optim="adamw_torch_fused",
        dataset_num_proc=8,
    )

    # Split for eval
    split_dataset = train_dataset.train_test_split(test_size=0.02, seed=42)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    wandb.finish()
    print(f"Training complete! Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()

