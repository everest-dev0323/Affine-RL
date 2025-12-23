import os
from functools import partial
import json
import random
import torch
from datasets import Dataset, load_dataset, concatenate_datasets, interleave_datasets
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
import wandb
import numpy as np

import sys
from pathlib import Path

from src.data import get_affine_dataset, get_prime_dataset, load_jsonl
from src.config import (
    AFFINE_DATASET_REPOSITORY,
    PRIME_DATASET_REPOSITORY,
    AGENTGYM_DATASET_REPOSITORY,
    TRAIN_DATA_PATH,
)

# Verify
print(f"Current working directory: {os.getcwd()}")

from dotenv import load_dotenv

load_dotenv()

# RESUME CONFIGURATION
RESUME_FROM_CHECKPOINT = "./sft_output/checkpoint-500"  # Set to None to train from scratch
MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_DIR = "./sft_output"

CHAT_TEMPLATE = r"""
{%- if tools %}
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
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {% generation %}
        {{- '<|im_start|>' + message.role + '\n' + content }}
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
        {% endgeneration %}
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
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
""".strip()

# Load model and tokenizer
# If resuming, load from checkpoint; otherwise load base model
if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
    print(f"Resuming from checkpoint: {RESUME_FROM_CHECKPOINT}")
    model_path = RESUME_FROM_CHECKPOINT
else:
    print(f"Starting fresh training with model: {MODEL_NAME}")
    model_path = MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.chat_template = CHAT_TEMPLATE

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# If we added a token, resize embeddings
if model.config.vocab_size != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))

# Load datasets
agentgym_dataset = Dataset.from_list(load_jsonl(TRAIN_DATA_PATH + "/agentgym.jsonl")).remove_columns("task_id")
swe_dataset = Dataset.from_list(load_jsonl(TRAIN_DATA_PATH + "/swe.jsonl")).remove_columns("task_id")
cde_dataset = Dataset.from_list(load_jsonl(TRAIN_DATA_PATH + "/cde.jsonl")).remove_columns("task_id")

prime_dataset = get_prime_dataset(PRIME_DATASET_REPOSITORY)
affine_dataset = get_affine_dataset(AFFINE_DATASET_REPOSITORY)

prime_dataset = prime_dataset.remove_columns("task_id")
affine_dataset = affine_dataset.remove_columns("task_id")

train_dataset = interleave_datasets(
    [agentgym_dataset, prime_dataset, affine_dataset, swe_dataset, cde_dataset], 
    probabilities=[0.1, 0.3, 0.3, 0.1, 0.2],  # AgentGym, Prime, Affine, SWE, CDE
    seed=42,
    stopping_strategy="all_exhausted",
)

train_dataset = train_dataset.rename_column(
    "conversations", "messages"
)

print("="*60)
print(f"Train dataset size: {len(train_dataset)}")
print(f"AgentGym: {len(agentgym_dataset)}")
print(f"Prime: {len(prime_dataset)}")
print(f"Affine: {len(affine_dataset)}")
print(f"SWE: {len(swe_dataset)}")
print(f"CDE: {len(cde_dataset)}")
print("="*60)

training_args = SFTConfig(
    model_init_kwargs={"torch_dtype": torch.bfloat16},
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    eval_accumulation_steps=1,
    
    num_train_epochs=3,
    learning_rate=1e-05,
    lr_scheduler_type="cosine",
    
    warmup_steps=128,
    max_length=12880,
    max_grad_norm=1.0,
    dataset_num_proc=16,
    
    # Memory optimization
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    
    # Conversation control
    assistant_only_loss=True,
            
    # Logging
    logging_strategy="steps",
    logging_steps=4,
    
    # Evaluation
    do_eval=True,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=3,  # Reduced from 8 to save storage
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    
    # Checkpointing
    output_dir=OUTPUT_DIR,
    save_strategy="steps",
    save_steps=100,
    bf16=True,
    remove_unused_columns=False,
    report_to=["wandb"],
    
    # Resume settings
    resume_from_checkpoint=RESUME_FROM_CHECKPOINT if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT) else None,
    save_only_model=True
)

# Initialize wandb
wandb.init(
    project="SFT",
    name="multi-dataset-training-resume" if RESUME_FROM_CHECKPOINT else "multi-dataset-training",
    config={
        "model_name": MODEL_NAME,
        "train_dataset": "AgentGym + PrimeIntellect + Affine + CDE + SWE",
        "resumed_from": RESUME_FROM_CHECKPOINT if RESUME_FROM_CHECKPOINT else "scratch",
        **training_args.to_dict(),
    },
    resume="allow",  # Allow resuming wandb run
)

def random_subset(dataset, subset_size):
    indices = random.sample(range(len(dataset)), subset_size)
    return dataset.select(indices=indices)


class MyTrainer(SFTTrainer):
    def __init__(self, *args, eval_subset_size=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_subset_size = eval_subset_size
    
    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = self.train_dataset

        subset = random_subset(
            eval_dataset,
            min(self.eval_subset_size, len(eval_dataset))
        )

        return DataLoader(
            subset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
        )


# Create dummy initial eval dataset
initial_eval = train_dataset.shuffle(seed=42).select(list(range(min(100, len(train_dataset)))))

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=initial_eval,
    processing_class=tokenizer,
)

# Train (will automatically resume from checkpoint if specified)
trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT) else None)

wandb.finish()