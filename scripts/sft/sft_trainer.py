import os
from functools import partial
import json
import torch
from datasets import Dataset, load_dataset, concatenate_datasets, interleave_datasets
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
import wandb

from src.data import get_affine_dataset, get_prime_dataset, load_jsonl
from src.config import (
    AFFINE_DATASET_REPOSITORY,
    PRIME_DATASET_REPOSITORY,
    AGENTGYM_DATASET_REPOSITORY,
    TRAIN_DATA_PATH,
)

from dotenv import load_dotenv

load_dotenv()

# from peft import LoraConfig, get_peft_model, TaskType
MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_DIR = "./sft_output"
NUM_EPOCHS = 6
TEST_SIZE = 0.1

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

class MultiDatasetEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_datasets, eval_every_n_steps=500):
        self.trainer = trainer
        self.eval_datasets = eval_datasets
        self.eval_every_n_steps = eval_every_n_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0 and state.global_step > 0:
            print(f"\n--- Evaluation at step {state.global_step} ---")
            
            # Save model state
            self.trainer.model.eval()
            
            for name, dataset in self.eval_datasets.items():
                # Create dataloader for this eval dataset
                eval_dataloader = self.trainer.get_eval_dataloader(dataset)
                
                total_loss = 0
                total_samples = 0
                
                with torch.no_grad():
                    for batch in tqdm(eval_dataloader, desc=f"Evaluating {name}"):
                        # Move batch to device
                        batch = {k: v.to(self.trainer.model.device) for k, v in batch.items()}
                        
                        outputs = self.trainer.model(**batch)
                        loss = outputs.loss
                        
                        total_loss += loss.item() * batch['input_ids'].size(0)
                        total_samples += batch['input_ids'].size(0)
                
                avg_loss = total_loss / total_samples
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                
                # Log to wandb
                wandb_metrics = {
                    f"{name}/loss": avg_loss,
                    f"{name}/perplexity": perplexity,
                }
                wandb.log(wandb_metrics, step=state.global_step)
                print(f"{name}: loss={avg_loss:.4f}, perplexity={perplexity:.4f}")
            
            # Restore training mode
            self.trainer.model.train()
            

def main():
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

    agentgym_dataset = Dataset.from_list(load_jsonl(TRAIN_DATA_PATH + "/agentgym.jsonl"))
    swe_dataset = Dataset.from_list(load_jsonl(TRAIN_DATA_PATH + "/swe.jsonl")).remove_columns("task_id")
    cde_dataset = Dataset.from_list(load_jsonl(TRAIN_DATA_PATH + "/cde.jsonl")).remove_columns("task_id")

    prime_dataset = get_prime_dataset(PRIME_DATASET_REPOSITORY)
    affine_dataset = get_affine_dataset(AFFINE_DATASET_REPOSITORY)

    trained_dataset = concatenate_datasets(
        [agentgym_dataset, prime_dataset, affine_dataset]
    ).remove_columns("task_id")
    
    training_args = SFTConfig(
        model_init_kwargs={"torch_dtype": torch.bfloat16},
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
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
        do_eval=False,
        eval_strategy="no",
        # eval_steps=500,
        
        # Checkpointing
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,                # set True if using GPU with mixed precision
        remove_unused_columns=False,  # important because we pass labels
        report_to=["wandb"],
    )

    train_dataset = interleave_datasets(
        [trained_dataset, cde_dataset, swe_dataset],
        probabilities=[0.1, 0.3, 0.6],
        seed=42,
        stopping_strategy="first_exhausted",  # Stop when smallest dataset runs out
    )
    
    train_dataset = train_dataset.rename_column(
        "conversations", "messages"
    )
    
    A_eval = trained_dataset.shuffle(seed=42).select(range(min(500, len(trained_dataset))))
    B_eval = cde_dataset.shuffle(seed=42).select(range(min(200, len(cde_dataset))))
    C_eval = swe_dataset.shuffle(seed=42).select(range(min(200, len(swe_dataset))))
    
    eval_datasets = {
        "dataset_A": A_eval,
        "dataset_B": B_eval,
        "dataset_C": C_eval,
    }
    
    # Initialize wandb
    wandb.init(
        project="SFT",
        name="multi-dataset-training",
        config={
            "dataset_A_size": len(trained_dataset),
            "dataset_B_size": len(cde_dataset),
            "dataset_C_size": len(swe_dataset),
            "sampling_probs": [0.1, 0.3, 0.6],
            "num_epochs": 15,
        }
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    eval_callback = MultiDatasetEvalCallback(trainer, eval_datasets)
    trainer.add_callback(eval_callback)
    trainer.train()
    
    wandb.finish()


if __name__ == "__main__":
    main()
