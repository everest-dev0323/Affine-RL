"""
S-GRPO (Sequential Group Relative Policy Optimization) Training Script.

This script implements a custom GRPO trainer for fine-tuning language models
on multi-turn conversation trajectories using reinforcement learning from
human feedback (RLHF) principles.

Key components:
- SGRPOTrainer: Custom trainer class implementing the GRPO algorithm
- SGRPOTrainingArguments: Configuration dataclass for training hyperparameters
- PPO-style clipped objective with KL divergence penalty

Usage:
    python sqrpo_train.py --model_name "demonwizard0/affine-new-3" --output_dir "./output"
"""

from glob import glob
import json
import os
import random
import time
from tqdm import tqdm
import transformers
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


def get_gpu_memory_stats():
    """
    Get current GPU memory statistics.

    Returns:
        dict: Dictionary containing memory stats in GB:
            - allocated: Currently allocated memory
            - reserved: Reserved memory by PyTorch
            - max_allocated: Peak allocated memory
            - free: Available memory
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - reserved

    return {
        "gpu_memory_allocated_gb": round(allocated, 2),
        "gpu_memory_reserved_gb": round(reserved, 2),
        "gpu_memory_max_allocated_gb": round(max_allocated, 2),
        "gpu_memory_free_gb": round(free, 2),
        "gpu_memory_total_gb": round(total, 2),
        "gpu_memory_utilization_pct": round((reserved / total) * 100, 1),
    }


def log_gpu_memory(prefix: str = ""):
    """
    Log GPU memory statistics to console.

    Args:
        prefix: Optional prefix for the log message
    """
    stats = get_gpu_memory_stats()
    if stats:
        prefix_str = f"[{prefix}] " if prefix else ""
        print(f"{prefix_str}GPU Memory: {stats['gpu_memory_allocated_gb']:.2f}GB allocated, "
              f"{stats['gpu_memory_reserved_gb']:.2f}GB reserved, "
              f"{stats['gpu_memory_free_gb']:.2f}GB free, "
              f"{stats['gpu_memory_utilization_pct']:.1f}% utilization")


class SGRPOTrainer:
    """
    Sequential Group Relative Policy Optimization Trainer.

    Implements GRPO algorithm for training language models on trajectory data
    with advantage estimation and PPO-style clipped objectives.

    Attributes:
        args: Training configuration arguments
        model: The PEFT model to be trained (LoRA-adapted)
        tokenizer: Tokenizer for processing text
        beta: KL divergence penalty coefficient
        accelerator: Hugging Face Accelerator for distributed training
    """

    def __init__(self, args, model: PeftModel, tokenizer):
        """
        Initialize the SGRPO trainer.

        Args:
            args: Training arguments containing hyperparameters
            model: PEFT model with LoRA adapters
            tokenizer: Tokenizer compatible with the model
        """
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.beta = 0.01  # KL divergence penalty weight
        self.train_dataset = None
        self.eval_dataset = None
        self.best_eval_loss = float('inf')  # Track best evaluation loss for checkpointing

        self.accelerator = None

        # Initialize accelerator for distributed training
        self.create_accelerator()

        # Initialize wandb for logging and monitoring
        self.init_wandb()

        # Setup optimizer with AdamW
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.args.learning_rate
        )

        # Prepare model and optimizer for distributed training
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # Log initial GPU memory state
        log_gpu_memory("Initialization complete")

    def create_accelerator(self):
        """Initialize the Hugging Face Accelerator for distributed training."""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

    def init_wandb(self):
        """
        Initialize Weights & Biases for experiment tracking.

        Sets up wandb with training configuration and hyperparameters.
        Only initializes on the main process to avoid duplicate logging.
        """
        if self.args.wandb_log:
            if self.accelerator.is_main_process:
                wandb.init(
                    project=self.args.wandb_project,
                    name=self.args.run_name,
                    config={
                        "model_name": self.args.model_name,
                        "learning_rate": self.args.learning_rate,
                        "num_epochs": self.args.num_train_epochs,
                        "batch_size": self.args.per_device_train_batch_size,
                        "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                        "lora_r": self.args.lora_r,
                        "lora_alpha": self.args.lora_alpha,
                        "lora_dropout": self.args.lora_dropout,
                        "clip_range": self.args.clip_range,
                        "clip_grad_norm": self.args.clip_grad_norm,
                        "num_generations": self.args.num_generations,
                        "beta_kl": self.beta,
                        "temperature": self.args.temperature,
                        "top_p": self.args.top_p,
                        "max_prompt_length": self.args.max_prompt_length,
                        "max_completion_length": self.args.max_completion_length,
                    },
                    tags=["sgrpo", "rlhf", "lora"],
                )
                print(f"Wandb initialized: {wandb.run.name}")

    def get_raw_dataset(self):
        """
        Load training data from JSONL files in the dataset directory.

        Each JSONL file represents a task, containing trajectory histories
        with conversation data and rewards. Files are expected to be in format:
        {"conversations": [...], "reward": float, ...}
        """
        train_list = []
        for path in glob(self.args.train_dataset_path + "/*.jsonl"):
            fname = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                history = [json.loads(line) for line in f]
                train_list.append({
                    "history": history,
                    "task_id": fname
                })
        self.train_dataset = Dataset.from_list(train_list)

    def compute_advantages(self, trajectories):
        """
        Compute normalized advantages using Group Relative Policy Optimization.

        Advantages are computed by normalizing rewards across the trajectory group
        using z-score normalization (mean subtraction and std division).

        Args:
            trajectories: List of trajectory dicts containing "reward" field

        Returns:
            Tensor of normalized advantages for each trajectory
        """
        rewards = torch.tensor([traj["reward"] for traj in trajectories])
        mean = torch.mean(rewards)
        std = torch.std(rewards) + 1e-8  # Add epsilon for numerical stability
        advantages = (rewards - mean) / std
        return advantages

    def compute_log_probs(self, model, trajectories, requires_grad=True):
        """
        Compute log probabilities for assistant responses in trajectories.

        For each trajectory, computes the sum of log probabilities for all
        assistant turns, which is used for policy gradient estimation.

        Args:
            model: The language model to compute log probs with
            trajectories: List of trajectory dicts with "conversations" field
            requires_grad: Whether to track gradients (False for reference model)

        Returns:
            List of total log probability for each trajectory
        """
        all_log_probs = []  # FIX: Renamed to avoid collision with logits variable
        device = next(model.parameters()).device

        for traj in trajectories:
            log_prob = torch.tensor(0.0, device=device, requires_grad=requires_grad)

            for index in range(len(traj["conversations"])):
                # Compute log probs for assistant responses only (skip system/first user)
                if index > 1 and traj["conversations"][index]["role"] == "assistant":
                    # Tokenize conversation up to current turn (including current assistant response)
                    input_ids = self.tokenizer.apply_chat_template(
                        traj["conversations"][:index + 1],  # FIX: Include current response
                        tokenize=True,
                        add_generation_prompt=False,
                        return_tensors="pt",  # FIX: Return as tensor
                        enable_thinking=True
                    ).to(device)

                    attention_mask = torch.ones_like(input_ids)

                    # Get prompt length (conversation before assistant response)
                    prompt_len = len(self.tokenizer.apply_chat_template(
                        traj["conversations"][:index],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False
                    ))

                    # Forward pass to get logits
                    outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
                    logits = outputs.logits  # FIX: Use different variable name

                    # Shift logits and labels for next-token prediction
                    # We only want the response portion, not the prompt
                    shift_logits = logits[:, prompt_len:-1].contiguous()
                    shift_labels = input_ids[:, prompt_len + 1:].contiguous()
                    shift_attention_mask = attention_mask[:, prompt_len + 1:].contiguous()

                    # Compute log softmax and gather log probs for actual tokens
                    token_log_probs = F.log_softmax(shift_logits, dim=-1)
                    selected_log_probs = token_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

                    # Sum log probs for this response (masked by attention)
                    response_log_prob = torch.sum(selected_log_probs * shift_attention_mask)
                    log_prob = log_prob + response_log_prob

                    # Free intermediate tensors to reduce GPU memory
                    del outputs, logits, shift_logits, token_log_probs, selected_log_probs
                    del input_ids, attention_mask, shift_labels, shift_attention_mask

            all_log_probs.append(log_prob)

        return all_log_probs
    
    def train_one_epoch(self, epoch, global_step):
        """
        Train the model for one epoch over all tasks.

        For each task:
        1. Sample trajectories from the task history
        2. Compute advantages using GRPO normalization
        3. Compute policy loss with PPO-style clipping
        4. Add KL divergence penalty to prevent policy drift
        5. Update model parameters

        Args:
            epoch: Current epoch number for logging
            global_step: Global training step counter

        Returns:
            Updated global_step count
        """
        clip_grad_norm = self.args.clip_grad_norm
        self.model.train()
        self.optimizer.zero_grad()

        epoch_losses = []
        epoch_policy_losses = []
        epoch_kl_divs = []
        epoch_rewards = []

        for task in self.train_dataset.shuffle():
            step_start_time = time.time()

            # FIX: sorted() returns a new list; list.sort() returns None
            history = sorted(task["history"], key=lambda x: x["reward"])

            # Sample trajectories for this training step
            # Ensure we don't sample more than available
            num_samples = min(self.args.num_generations, len(history))
            trajectories = random.sample(history, num_samples)

            # Track rewards for logging
            step_rewards = [traj["reward"] for traj in trajectories]
            epoch_rewards.extend(step_rewards)

            # Compute normalized advantages across sampled trajectories
            # Move to device and detach since advantages are not part of the computation graph
            device = next(self.model.parameters()).device
            advantages = self.compute_advantages(trajectories).to(device).detach()

            # Compute log probabilities under current policy
            current_log_probs = self.compute_log_probs(self.model, trajectories, requires_grad=True)

            # Compute log probabilities under reference policy (frozen base model)
            # This is used for KL divergence computation
            with torch.no_grad():
                ref_log_probs = self.compute_log_probs(self.model.base_model, trajectories, requires_grad=False)

            # Stack log probs into tensors for vectorized computation
            current_log_probs = torch.stack(current_log_probs)
            ref_log_probs = torch.stack(ref_log_probs).detach()  # Ensure ref is detached

            # Compute KL divergence (approximated as log ratio)
            kl_div = current_log_probs - ref_log_probs

            # Compute probability ratio for importance sampling
            ratio = torch.exp(kl_div)

            # PPO-style clipped surrogate objective
            # surr1: unclipped objective
            # surr2: clipped objective to prevent large policy updates
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) * advantages

            # Take minimum of clipped and unclipped objective (pessimistic bound)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Add KL penalty to encourage staying close to reference policy
            kl_penalty = self.beta * kl_div.mean()

            # Total loss combines policy gradient and KL regularization
            loss = policy_loss + kl_penalty

            # Store metrics for logging (detach to avoid memory issues)
            loss_value = loss.detach().item()
            policy_loss_value = policy_loss.detach().item()
            kl_div_value = kl_div.mean().detach().item()
            epoch_losses.append(loss_value)
            epoch_policy_losses.append(policy_loss_value)
            epoch_kl_divs.append(kl_div_value)

            # Backward pass and optimization step
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Free memory after each task to prevent GPU memory accumulation
            del current_log_probs, ref_log_probs, kl_div, ratio
            del surr1, surr2, policy_loss, kl_penalty, loss, advantages
            torch.cuda.empty_cache()

            global_step += 1
            step_time = time.time() - step_start_time

            # Log metrics at logging_steps interval
            if global_step % self.args.logging_steps == 0 and self.accelerator.is_main_process:
                gpu_stats = get_gpu_memory_stats()
                log_dict = {
                    "train/loss": loss_value,
                    "train/policy_loss": policy_loss_value,
                    "train/kl_divergence": kl_div_value,
                    "train/mean_reward": sum(step_rewards) / len(step_rewards),
                    "train/step_time_seconds": step_time,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                if gpu_stats:
                    log_dict.update(gpu_stats)
                wandb.log(log_dict, step=global_step)

                print(f"Step {global_step}: loss={loss_value:.4f}, "
                      f"policy_loss={policy_loss_value:.4f}, kl={kl_div_value:.4f}")

        # Log epoch summary
        if self.accelerator.is_main_process and epoch_losses:
            epoch_summary = {
                "epoch/mean_loss": sum(epoch_losses) / len(epoch_losses),
                "epoch/mean_policy_loss": sum(epoch_policy_losses) / len(epoch_policy_losses),
                "epoch/mean_kl_divergence": sum(epoch_kl_divs) / len(epoch_kl_divs),
                "epoch/mean_reward": sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0,
                "epoch/number": epoch,
            }
            wandb.log(epoch_summary, step=global_step)
            print(f"Epoch {epoch} complete: mean_loss={epoch_summary['epoch/mean_loss']:.4f}")

        return global_step

    def save_checkpoint(self, global_step: int, checkpoint_name: Optional[str] = None):
        """
        Save model checkpoint to disk.

        Args:
            global_step: Current training step for checkpoint naming
            checkpoint_name: Optional custom name for the checkpoint
        """
        if not self.accelerator.is_main_process:
            return

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint-{global_step}"

        checkpoint_path = os.path.join(self.args.output_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Unwrap model from accelerator and save
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        training_state = {
            "global_step": global_step,
            "best_eval_loss": self.best_eval_loss,
        }
        torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))

        print(f"Checkpoint saved to {checkpoint_path}")

        # Log checkpoint to wandb
        wandb.log({"checkpoint/global_step": global_step}, step=global_step)

    def train(self):
        """
        Main training loop that runs for the specified number of epochs.

        Handles:
        - Dataset loading
        - Epoch iteration with progress bar
        - Model checkpointing based on save_steps
        - Logging and monitoring via wandb
        """
        global_step = 0
        n_epochs = self.args.num_train_epochs
        training_start_time = time.time()

        # Create output directory for model saves
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Load the training dataset before starting
        self.get_raw_dataset()

        # Log dataset info
        if self.accelerator.is_main_process:
            print(f"Training dataset loaded: {len(self.train_dataset)} tasks")
            wandb.log({"dataset/num_tasks": len(self.train_dataset)})
            log_gpu_memory("Before training")

        # Training loop with progress tracking
        with tqdm(range(1, n_epochs + 1), total=n_epochs, desc="Epochs") as t:
            for epoch in t:
                epoch_start_time = time.time()

                global_step = self.train_one_epoch(epoch, global_step)

                epoch_time = time.time() - epoch_start_time

                # Log epoch timing
                if self.accelerator.is_main_process:
                    wandb.log({
                        "epoch/time_seconds": epoch_time,
                        "epoch/time_minutes": epoch_time / 60,
                    }, step=global_step)
                    log_gpu_memory(f"End of epoch {epoch}")

                # Save checkpoint at end of each epoch
                if self.accelerator.is_main_process:
                    self.save_checkpoint(global_step, checkpoint_name=f"checkpoint-epoch-{epoch}")

                # Update progress bar
                t.set_postfix({"step": global_step, "epoch_time": f"{epoch_time:.1f}s"})

        # Save final model
        if self.accelerator.is_main_process:
            self.save_checkpoint(global_step, checkpoint_name="final-model")

            total_time = time.time() - training_start_time
            final_stats = {
                "training/total_time_seconds": total_time,
                "training/total_time_minutes": total_time / 60,
                "training/total_steps": global_step,
            }
            wandb.log(final_stats, step=global_step)
            print(f"Training complete! Total time: {total_time/60:.2f} minutes, Total steps: {global_step}")

            # Finish wandb run
            wandb.finish()
        


@dataclass
class SGRPOTrainingArguments:
    """
    Configuration dataclass for S-GRPO training.

    This class defines all hyperparameters and settings for the SGRPO trainer,
    including model configuration, LoRA settings, training parameters, and
    logging options. Compatible with HuggingFace's HfArgumentParser for CLI usage.
    """

    # Model configuration
    model_name: str = "demonwizard0/affine-new-3"  # HuggingFace model identifier
    use_flash_attention: bool = True  # Use Flash Attention 2 for efficiency
    load_in_4bit: bool = False  # Enable 4-bit quantization
    load_in_8bit: bool = False  # Enable 8-bit quantization

    # Dataset
    train_dataset_path: str = "datasets/train_exp"  # Path to training JSONL files

    # LoRA configuration for parameter-efficient fine-tuning
    lora_r: int = 16  # LoRA rank (lower = fewer params, higher = more expressive)
    lora_alpha: int = 32  # LoRA scaling factor (typically 2x rank)
    lora_dropout: float = 0.05  # Dropout for LoRA layers
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
        "gate_proj", "up_proj", "down_proj"  # MLP projections
    ])

    # Training configuration
    output_dir: str = "./models/qwen3-sgrpo-output"  # Directory for checkpoints and logs
    num_train_epochs: int = 3  # Total training epochs
    per_device_train_batch_size: int = 1  # Batch size per GPU
    gradient_accumulation_steps: int = 16  # Steps before gradient update
    learning_rate: float = 5e-6  # Peak learning rate
    warmup_ratio: float = 0.1  # Fraction of steps for LR warmup
    clip_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    clip_range: float = 0.2  # FIX: Added missing PPO clip range parameter

    # GRPO specific parameters
    num_generations: int = 4  # Number of trajectories to sample per task
    max_prompt_length: int = 512  # Maximum prompt token length
    max_completion_length: int = 2048  # Maximum completion token length
    temperature: float = 0.7  # Sampling temperature for generation
    top_p: float = 0.9  # Nucleus sampling probability threshold

    # S-GRPO specific (multi-round conversation penalties)
    max_rounds: int = 10  # Maximum conversation rounds before penalty
    round_penalty_weight: float = 0.1  # Penalty weight for exceeding max rounds
    length_penalty_weight: float = 0.05  # Penalty weight for overly long responses

    # Optimization settings
    bf16: bool = True  # Use bfloat16 mixed precision
    gradient_checkpointing: bool = True  # Trade compute for memory
    optim: str = "adamw_8bit"  # Optimizer type

    # Logging and checkpointing
    logging_steps: int = 10  # Log metrics every N steps
    save_steps: int = 100  # Save checkpoint every N steps
    eval_steps: int = 100  # Run evaluation every N steps
    report_to: str = "wandb"  # Logging backend (wandb, tensorboard, none)
    run_name: str = "qwen3-sgrpo"  # Name for the training run
    
    # wandb stuff
    wandb_log: bool = True  # Whether to log to wandb
    wandb_project: str = "sgrpo-training"  # Wandb project name

    # vLLM configuration (for inference acceleration)
    use_vllm: bool = True  # Use vLLM for faster inference
    vllm_gpu_memory_utilization: float = 0.7  # GPU memory fraction for vLLM

    # Reproducibility
    seed: int = 42  # Random seed for reproducibility


def main():
    """
    Main entry point for S-GRPO training.

    This function:
    1. Parses command-line arguments into SGRPOTrainingArguments
    2. Loads the base model and tokenizer from HuggingFace
    3. Applies LoRA adapters for parameter-efficient training
    4. Initializes the SGRPOTrainer and starts training

    Usage:
        python sqrpo_train.py --model_name "model/path" --output_dir "./output"
    """
    # Parse arguments from command line using HuggingFace's argument parser
    parser = transformers.HfArgumentParser(SGRPOTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    # Load tokenizer with trust_remote_code for custom tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load base model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,  # Reduce CPU memory during loading
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        device_map="auto"  # Automatically distribute across available GPUs
    )

    # Configure LoRA for parameter-efficient fine-tuning
    # LoRA only trains low-rank adapter matrices, keeping base model frozen
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",  # Don't train bias terms
        task_type="CAUSAL_LM"  # Causal language modeling task
    )

    # Apply LoRA adapters to the model
    model = get_peft_model(model, lora_config)

    # Print trainable parameter count for verification
    model.print_trainable_parameters()

    # Initialize the SGRPO trainer with model and configuration
    trainer = SGRPOTrainer(
        args=args,
        model=model,
        tokenizer=tokenizer
    )

    # Start the training loop
    trainer.train()


if __name__ == "__main__":
    main()