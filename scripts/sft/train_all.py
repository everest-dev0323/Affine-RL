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

# from peft import LoraConfig, get_peft_model, TaskType
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


class StratifiedEvalCallback(TrainerCallback):
    def __init__(self, sub_datasets, sample_size_per_dataset=300):
        """
        Args:
            sub_datasets: dict of {"dataset_name": dataset}
            sample_size_per_dataset: number of samples to take from each dataset
        """
        self.sub_datasets = sub_datasets
        self.sample_size_per_dataset = sample_size_per_dataset
    
    def on_evaluate(self, args, state, control, **kwargs):
        # Sample from each sub-dataset
        eval_subsets = []
        for name, dataset in self.sub_datasets.items():
            n_samples = min(self.sample_size_per_dataset, len(dataset))
            indices = random.sample(range(len(dataset)), n_samples)
            subset = dataset.select(indices)
            eval_subsets.append(subset)
        
        # Concatenate all samples
        combined_eval = concatenate_datasets(eval_subsets)
        kwargs['trainer'].eval_dataset = combined_eval
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # This will be called after evaluation
        if logs and 'eval_loss' in logs:
            trainer = kwargs.get('trainer')
            if trainer and hasattr(trainer, 'eval_dataset'):
                # Evaluate each dataset separately
                self.evaluate_per_dataset(trainer, logs, state)
        return control
    
    def evaluate_per_dataset(self, trainer, logs, state):
        """Evaluate each sub-dataset separately and log metrics"""
        import torch.nn.functional as F
        
        metrics = {}
        all_losses = []
        
        for name, dataset in self.sub_datasets.items():
            # Sample from this specific dataset
            n_samples = min(self.sample_size_per_dataset, len(dataset))
            indices = random.sample(range(len(dataset)), n_samples)
            subset = dataset.select(indices)
            
            # Create dataloader
            eval_dataloader = trainer.get_eval_dataloader(subset)
            
            # Evaluate
            model = trainer.model
            model.eval()
            
            total_loss = 0
            total_tokens = 0
            correct_tokens = 0
            total_perplexity = 0
            n_batches = 0
            
            with torch.no_grad():
                for batch in eval_dataloader:
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    
                    outputs = model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    # Loss
                    total_loss += loss.item()
                    
                    # Token accuracy
                    if 'labels' in batch:
                        labels = batch['labels']
                        # Shift for causal LM
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        
                        # Only count non-padding tokens
                        mask = shift_labels != -100
                        predictions = shift_logits.argmax(dim=-1)
                        
                        correct_tokens += ((predictions == shift_labels) & mask).sum().item()
                        total_tokens += mask.sum().item()
                    
                    # Perplexity
                    total_perplexity += torch.exp(loss).item()
                    n_batches += 1
            
            # Calculate metrics for this dataset
            avg_loss = total_loss / n_batches
            avg_perplexity = total_perplexity / n_batches
            token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
            
            all_losses.append(avg_loss)
            
            # Log per-dataset metrics
            metrics[f"eval_{name}_loss"] = avg_loss
            metrics[f"eval_{name}_perplexity"] = avg_perplexity
            metrics[f"eval_{name}_token_accuracy"] = token_accuracy
        
        # Calculate average loss across all datasets
        metrics["eval_loss"] = np.mean(all_losses)
        
        # Log all metrics
        trainer.log(metrics)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Evaluation at step {state.global_step}")
        print(f"{'='*60}")
        for name in self.sub_datasets.keys():
            print(f"{name}:")
            print(f"  Loss: {metrics[f'eval_{name}_loss']:.4f}")
            print(f"  Perplexity: {metrics[f'eval_{name}_perplexity']:.4f}")
            print(f"  Token Accuracy: {metrics[f'eval_{name}_token_accuracy']:.4f}")
        print(f"\nAverage Loss: {metrics['eval_loss']:.4f}")
        print(f"{'='*60}\n")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.chat_template = CHAT_TEMPLATE
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

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
    probabilities=[0.1, 0.3, 0.3, 0.1, 0.2],
    seed=42,
    stopping_strategy="all_exhausted", 
)

training_args = SFTConfig(
    model_init_kwargs={"torch_dtype": torch.bfloat16},
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    # dataloader_num_workers=2,
    num_train_epochs=15,
    learning_rate=5e-06,
    warmup_ratio=0.02,
    max_length=2048,
    
    # Conversation control
    # dataset_text_field="conversations",
    assistant_only_loss=True,
            
    # Logging
    logging_strategy="steps",
    logging_steps=10,
    
    # Evaluation
    do_eval=True,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
    metric_for_best_model="loss",
    greater_is_better=False,
    
    # Checkpointing
    output_dir=OUTPUT_DIR,
    save_strategy="steps",
    save_steps=100,
    bf16=True,                # set True if using GPU with mixed precision
    remove_unused_columns=False,  # important because we pass labels
    report_to=["wandb"],
)

train_dataset = train_dataset.rename_column(
    "conversations", "messages"
)

agentgym_eval = agentgym_dataset.shuffle(seed=42).select(range(min(200, len(agentgym_dataset))))
prime_eval = prime_dataset.shuffle(seed=42).select(range(min(200, len(prime_dataset))))
affine_eval = affine_dataset.shuffle(seed=42).select(range(min(200, len(affine_dataset))))
cde_eval = cde_dataset.shuffle(seed=42).select(range(min(200, len(cde_dataset))))
swe_eval = swe_dataset.shuffle(seed=42).select(range(min(200, len(swe_dataset))))


eval_datasets = {
    "AgentGym": agentgym_eval,
    "PrimeIntellect": prime_eval,
    "Affine": affine_eval,
    "CDE": cde_eval,
    "SWE": swe_eval,
}

# Initialize wandb
wandb.init(
    project="SFT",
    name="multi-dataset-training",
    config={
        "model_name": MODEL_NAME,
        "train_dataset": "AgentGym + PrimeIntellect + Affine + CDE + SWE",
        "eval_datasets": list(eval_datasets.keys()),
        "eval_every_n_steps": 500,
        "num_epochs": 15,
        **training_args.to_dict(),
    }
)

# Create dummy initial eval dataset
initial_eval = train_dataset.shuffle(seed=42).select(list(range(min(100, len(train_dataset)))))

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=initial_eval,
    processing_class=tokenizer,
)

sub_datasets = {
    "AgentGym": agentgym_dataset,
    "PrimeIntellect": prime_dataset,
    "Affine": affine_dataset,
    "CDE": cde_dataset,
    "SWE": swe_dataset,
}

eval_callback = StratifiedEvalCallback(
    sub_datasets=sub_datasets,
    sample_size_per_dataset=300
)
trainer.add_callback(eval_callback)
trainer.train()

wandb.finish()