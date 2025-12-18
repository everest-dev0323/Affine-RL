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



def tokenize_item(item, tokenizer, max_input_length=12880):
    """Processes a single item (conversation) using the logic from tokenize_fn."""

    # It returns a list of dictionaries, as one conversation can generate multiple training examples (input_ids/labels pairs).
    task_id, conversations = (item["task_id"], item["conversations"])

    input_ids = []
    labels = []

    # 2. Iterate through messages to build token sequences
    for idx, message in enumerate(conversations):
        text = (
            "<|im_start|>"
            + message["role"]
            + "\n"
            + message["content"]
            + "<|im_end|>"
            + "\n"
        )
        input_encode = tokenizer(text, add_special_tokens=False)

        if message["role"] == "assistant" and (len(conversations) == 2 or idx > 1):
            input_ids.extend(input_encode["input_ids"])
            labels.extend(
                [-100] * 3 + input_encode["input_ids"][3:]
            )  # Standard behavior: train on the assistant response

        elif idx != len(conversations) - 1:
            # For all other roles (user, system) or the first two turns (i<=1)
            input_ids.extend(input_encode["input_ids"])
            labels.extend(
                [-100] * len(input_encode["input_ids"])
            )  # Do not train on user/system turns

        # 3. Handle conversation-ending example (if last message is assistant AND no reasoning content)
    attention_mask = [1] * len(input_ids)
    
    input_ids = input_ids[: max_input_length]
    labels = labels[: max_input_length]
    attention_mask = attention_mask[: max_input_length]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "task_id": str(task_id),
        "input_ids_max_length": len(
            input_ids
        ),  # Using current length as a placeholder; typically used to find the split point.
    }


class MyTrainer(SFTTrainer):
    
    def collate_fn(self, batch):
        """Collate function for dataloaders"""
        # The tokenizer can pad all fields at once if they're the same length
        max_len = max(len(x["input_ids"]) for x in batch)
        
        # Manually pad labels since tokenizer doesn't handle them
        padded_labels = []
        for x in batch:
            pad_len = max_len - len(x["labels"])
            padded_labels.append(x["labels"] + [-100] * pad_len)
        
        # Use tokenizer for input_ids and attention_mask
        to_pad = {
            "input_ids": [x["input_ids"] for x in batch],
            "attention_mask": [x["attention_mask"] for x in batch]
        }
        padded = self.processing_class.pad(to_pad, padding=True, return_tensors="pt")
        
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": torch.LongTensor(padded_labels)
        }
    def get_train_dataloader(self):
        """
        create train_dataset and train_dataloader using a for loop
        """
        
        # def collate_fn(batch):
        #     to_pad = {
        #         "input_ids": [x["input_ids"] for x in batch],
        #         "attention_mask": [x["attention_mask"] for x in batch]
        #     }
        #     padded = self.processing_class.pad(
        #         to_pad,
        #         padding=True
        #     )
        #     input_ids = torch.LongTensor(padded["input_ids"])

        #     max_len = input_ids.shape[1]

        #     attention_mask = torch.zeros_like(input_ids)
        #     labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        #     for i, x in enumerate(batch):
        #         mask = torch.tensor(x["attention_mask"], dtype=torch.long)
        #         attention_mask[i, : mask.shape[0]] = mask

        #         lab = torch.tensor(x["labels"], dtype=torch.long)
        #         labels[i, : lab.shape[0]] = lab

        #     padded = {
        #         "input_ids": input_ids,
        #         "attention_mask": attention_mask,
        #         "labels": labels
        #     }

        #     return padded

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=False,
            collate_fn=partial(self.collate_fn),
        )
        return train_dataloader
    
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
    swe_dataset = Dataset.from_list(load_jsonl(TRAIN_DATA_PATH + "/swe.jsonl"))
    cde_dataset = Dataset.from_list(load_jsonl(TRAIN_DATA_PATH + "/cde.jsonl"))

    prime_dataset = get_prime_dataset(PRIME_DATASET_REPOSITORY)
    affine_dataset = get_affine_dataset(AFFINE_DATASET_REPOSITORY)

    trained_dataset = concatenate_datasets(
        [agentgym_dataset, prime_dataset, affine_dataset]
    )
    
    # Tokenization with tqdm
    trained_dataset = trained_dataset.map(
        lambda x: tokenize_item(x, tokenizer),
        batched=False,
        num_proc=2,
    ).remove_columns(['task_id'])
    
    cde_dataset = cde_dataset.map(
        lambda x: tokenize_item(x, tokenizer),
        batched=False,
        num_proc=2,
    ).remove_columns(['task_id'])
    
    swe_dataset = swe_dataset.map(
        lambda x: tokenize_item(x, tokenizer),
        batched=False,
        num_proc=2,
    ).remove_columns(['task_id'])

    training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
        num_train_epochs=15,
        learning_rate=5e-06,
        warmup_ratio=0.02,
        max_length=2048,
                
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
        # deepspeed="./deepspeed_zero3.json",
        # bf16=True,                # set True if using GPU with mixed precision
        remove_unused_columns=False,  # important because we pass labels
        report_to=["wandb"],
    )

    train_dataset = interleave_datasets(
        [trained_dataset, cde_dataset, swe_dataset],
        probabilities=[0.1, 0.3, 0.6],
        seed=42,
        stopping_strategy="first_exhausted",  # Stop when smallest dataset runs out
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
        project="llm-finetuning",
        name="multi-dataset-training",
        config={
            "dataset_A_size": len(trained_dataset),
            "dataset_B_size": len(cde_dataset),
            "dataset_C_size": len(swe_dataset),
            "sampling_probs": [0.1, 0.3, 0.6],
            "num_epochs": 15,
        }
    )
    
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    eval_callback = MultiDatasetEvalCallback(trainer, eval_datasets)
    # trainer.add_callback(eval_callback)
    trainer.train()
    
    wandb.finish()


if __name__ == "__main__":
    main()
