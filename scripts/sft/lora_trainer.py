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

# Import PEFT for LoRA
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_DIR = "./sft_output_lora"

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


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.chat_template = CHAT_TEMPLATE

# Load model with LoRA-friendly settings
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=False,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                           # LoRA rank (you can adjust: 8, 16, 32, 64)
    lora_alpha=32,                  # LoRA alpha (typically 2*r)
    lora_dropout=0.1,               # Dropout probability
    target_modules=[                # Target modules to apply LoRA
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    inference_mode=False,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# If we added a token, resize embeddings
if model.config.vocab_size != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))

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

NUM_PROC = 8

training_args = SFTConfig(
    model_init_kwargs={"torch_dtype": torch.bfloat16},
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=15,
    learning_rate=1e-05,
    warmup_steps=128,
    max_length=12880,
    max_grad_norm=1.0,
    dataset_num_proc=8,
    
    # Conversation control
    assistant_only_loss=True,
            
    # Logging
    logging_strategy="steps",
    logging_steps=16,
    
    # Evaluation
    do_eval=True,
    eval_strategy="steps",
    eval_steps=16,
    save_total_limit=3,
    metric_for_best_model="loss",
    greater_is_better=False,
    
    # Checkpointing
    output_dir=OUTPUT_DIR,
    save_strategy="steps",
    save_steps=128,
    bf16=True,
    remove_unused_columns=False,
    report_to=["wandb"],
    
    # Memory optimization
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
)


# Initialize wandb
wandb.init(
    project="SFT",
    name="multi-dataset-training-lora",
    config={
        "model_name": MODEL_NAME,
        "train_dataset": "AgentGym + PrimeIntellect + Affine + CDE + SWE",
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "training_method": "LoRA",
        "eval_every_n_steps": 500,
        "num_epochs": 15,
        **training_args.to_dict(),
    }
)

def random_subset(dataset, subset_size):
    indices = random.sample(range(len(dataset)), subset_size)
    return dataset.select(indices=indices)


class MyTrainer(SFTTrainer):
    def __init__(self, *args, eval_subset_size=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_subset_size = eval_subset_size
    
    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset or self.train_dataset

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

trainer.train()

# Save the LoRA adapter
model.save_pretrained(OUTPUT_DIR + "/final_lora_adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/final_lora_adapter")

wandb.finish()

print("Training complete! LoRA adapter saved to:", OUTPUT_DIR + "/final_lora_adapter")