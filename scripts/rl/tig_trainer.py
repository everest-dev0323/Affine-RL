"""
Think in Games (TiG) Trainer

Implementation of the TiG algorithm from "Think in Games: Learning to Reason
in Games via Reinforcement Learning with Large Language Models".

TiG trains LLMs to develop procedural understanding by:
1. Loading expert dataset with states and ground-truth actions
2. Generating multiple responses (with reasoning) per state
3. Computing binary rewards based on action correctness
4. Training with GRPO using group-relative advantages

Key differences from standard GRPO:
- Uses offline expert data instead of environment interaction
- Rule-based binary reward (1 if action matches, 0 otherwise)
- No format rewards needed

Usage:
    python tig_trainer.py --config config/tig.yaml
    python tig_trainer.py --model_name "Qwen/Qwen2.5-7B-Instruct" --dataset_path "data/openspiel-mcts"
"""

import argparse
from datetime import datetime
import gc
import json
import os
import random
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from tqdm import tqdm
from unsloth import FastLanguageModel


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}


class TiGConfig:
    """Configuration for TiG training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-0.6B"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    max_seq_length: int = 8192

    # Dataset
    dataset_path: str = "data/openspiel-mcts"  # Path to expert JSONL files
    max_data_per_game: int = 10000

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules = None  # Auto-detect if None

    # Unsloth optimizations
    use_gradient_checkpointing = "unsloth"
    use_rslora: bool = False
    use_dora: bool = False

    # Training configuration
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 16

    # TiG/GRPO specific parameters
    num_generations: int = 4  # G completions per state (paper uses 4-8)
    max_new_tokens: int = 2048  # Max completion length
    temperature: float = 1.0  # Sampling temperature
    clip_epsilon: float = 0.2  # PPO clip range
    kl_coef: float = 0.01  # KL divergence penalty weight

    # Checkpoint settings
    checkpoint_dir: str = "./checkpoints/tig"
    save_steps: int = 100
    keep_last_n_checkpoints: int = 3

    # WandB settings
    use_wandb: bool = True
    wandb_project: str = "tig-training"
    wandb_run_name: str = "tig-experiment"

    # Reproducibility
    seed: int = 42


class TiGTrainer:
    """
    Think in Games Trainer with Unsloth for maximum efficiency.

    Implements TiG algorithm for training LLMs on expert datasets using GRPO.

    Key features:
    - Unsloth for 2-5x faster training and 60% less memory
    - Offline training on expert demonstrations
    - Rule-based binary reward (action matching)
    - Group-relative advantage estimation
    - PPO-style clipped objective with KL penalty
    """

    def __init__(self, config: TiGConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # Memory management settings
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.max_grad_norm = config.max_grad_norm
        self.max_new_tokens = config.max_new_tokens
        self.max_seq_length = config.max_seq_length

        # Checkpoint settings
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = config.save_steps
        self.keep_last_n_checkpoints = config.keep_last_n_checkpoints
        self.global_step = 0

        # WandB settings
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config) if hasattr(config, '__dict__') else {},
            )
            print(f"WandB initialized: {wandb.run.name}")

        # Detect optimal dtype
        dtype = None  # Auto-detect (fp16 or bf16)

        # Load model with Unsloth
        print(f"Loading model with Unsloth: {config.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=dtype,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

        self.tokenizer.padding_side = "left"

        # Apply Unsloth-optimized LoRA
        print("Applying Unsloth-optimized LoRA...")
        if config.lora_target_modules is None:
            config.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=config.lora_r,
            target_modules=config.lora_target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            random_state=config.seed,
            use_rslora=config.use_rslora,
            loftq_config=None,
            use_dora=config.use_dora,
        )

        # Print trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"Total params: {total_params:,}")

        # Log model info to wandb
        if self.use_wandb:
            wandb.config.update({
                'total_params': total_params,
                'trainable_params': trainable_params,
                'trainable_params_pct': 100 * trainable_params / total_params
            })

        # Load reference model (frozen)
        print("Loading reference model...")
        self.ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=dtype,
            load_in_4bit=config.load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer (only trainable parameters)
        trainable_params_list = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = bnb.optim.AdamW8bit(trainable_params_list, lr=config.learning_rate)

        # Hyperparameters
        self.clip_epsilon = config.clip_epsilon
        self.kl_coef = config.kl_coef
        self.num_generations = config.num_generations
        self.temperature = config.temperature

        # Load expert dataset
        self._load_dataset()

        print(f"TiG Trainer initialized with Unsloth optimizations")
        self._print_memory_stats()

    def _print_memory_stats(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def _clear_memory(self):
        """Clear GPU cache and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_dataset(self):
        """
        Load expert dataset from JSONL files.
        """
        data_list = []
        dataset_path = self.config.dataset_path

        # Handle both single file and directory
        if os.path.isfile(dataset_path):
            files = [dataset_path]
        else:
            files = glob(os.path.join(dataset_path, "*.jsonl"))

        for filepath in files:
            item_list = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        trajectory = json.loads(line)
                        conversation = trajectory.get('conversation', [])
                        for i in range(len(conversation)):
                            if conversation[i].get('role') == 'assistant':
                                item_list.append({
                                    'prompt': conversation[:i],
                                    'ground_truth_action': conversation[i].get('content')
                                })
            data_list.extend(item_list[:self.config.max_data_per_game])
        self.dataset = data_list
        print(f"Loaded dataset: {len(data_list)} samples from {len(files)} file(s)")

    def extract_action_from_response(self, response: str) -> Optional[str]:
        """
        Extract predicted action from model response.

        Looks for action after "</think>" tag first, if this tag doesn't exist, whole response

        Args:
            response: Model's generated response

        Returns:
            Extracted action string or None if not found
        """
        think_tag_index = response.rfind("</think>")
        if think_tag_index != -1:
            action_str = response[think_tag_index + len("</think>"):].strip()
        else:
            action_str = response.strip()

        return action_str if action_str else None

    @torch.no_grad()
    def generate_completions(
        self,
        prompt: list,
        num_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate multiple completions for a given state using Unsloth.

        Args:
            prompt: list of messages
            num_samples: Number of completions to generate

        Returns:
            List of dicts with 'response', 'full_tokens', 'log_probs', 'prompt_length' keys
        """
        if num_samples is None:
            num_samples = self.num_generations

        # Enable fast inference mode
        FastLanguageModel.for_inference(self.model)

        # Tokenize with chat template
        prompt_text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        input_ids = self.tokenizer.encode(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)

        prompt_length = input_ids.shape[1]
        completions = []

        # Generate num_samples completions
        for _ in range(num_samples):
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

            generated_ids = outputs[0]
            completion_ids = generated_ids[prompt_length:]

            # Decode response
            response = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Compute log probs for the completion
            log_probs = self._compute_log_probs_for_completion(
                input_ids, generated_ids.unsqueeze(0), prompt_length
            )

            completions.append({
                'response': response,
                'full_tokens': generated_ids,
                'log_probs': log_probs,
                'prompt_length': prompt_length,
            })

        return completions

    @torch.no_grad()
    def _compute_log_probs_for_completion(
        self,
        prompt_ids: torch.Tensor,
        full_ids: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """
        Compute log probabilities for completion tokens.

        Args:
            prompt_ids: Tokenized prompt (1, prompt_len)
            full_ids: Full sequence including completion (1, full_len)
            prompt_length: Length of prompt tokens

        Returns:
            Tensor of log probabilities for completion tokens
        """
        # Forward pass
        outputs = self.model(full_ids)
        logits = outputs.logits

        # Compute log probs for completion tokens only
        log_probs = []
        for j in range(prompt_length, full_ids.shape[1]):
            token_id = full_ids[0, j]
            token_logits = logits[0, j - 1]
            token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
            log_probs.append(token_log_prob)

        return torch.stack(log_probs) if log_probs else torch.tensor([])

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute group-relative advantages (GRPO).

        From paper: A_i = (r_i - mean(r)) / std(r)

        Args:
            rewards: List of rewards for each completion in the group

        Returns:
            Normalized advantages tensor
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        if len(rewards) < 2:
            return torch.zeros_like(rewards_tensor)

        mean = rewards_tensor.mean()
        std = rewards_tensor.std() + 1e-8

        advantages = (rewards_tensor - mean) / std
        return advantages

    def train_step(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Execute one training step on a batch of expert examples.

        For each example:
        1. Generate G completions with reasoning
        2. Extract predicted actions and compute rewards
        3. Compute group-relative advantages
        4. Compute GRPO loss with KL penalty

        Args:
            samples: List of expert data items with 'state' and 'ground_truth_action'

        Returns:
            Dict of training metrics
        """

        self.optimizer.zero_grad()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl_div = 0.0
        total_reward = 0.0
        num_completions = 0
        policy_losses = []
        kl_divs = []


        for item in samples:
            prompt = item.get("prompt", [])
            ground_truth_action = item["ground_truth_action"]
            
            # Generate completions
            completions = self.generate_completions(prompt)

            # Compute rewards for each completion
            rewards = []
            for comp in completions:
                predicted_action = self.extract_action_from_response(comp['response'])
                reward = 1.0 if predicted_action == ground_truth_action else 0.0
                rewards.append(reward)
                comp['reward'] = reward
                total_reward += reward

            # Ensure training mode
            FastLanguageModel.for_training(self.model)

            # Skip if all rewards are same (no gradient signal) - DAPO dynamic sampling
            if len(set(rewards)) <= 1:
                continue

            # Compute group-relative advantages
            advantages = self.compute_advantages(rewards).to(self.device)

            # Compute loss for each completion
            for i, comp in enumerate(completions):
                full_tokens = comp['full_tokens'].unsqueeze(0).to(self.device)
                prompt_length = comp['prompt_length']
                old_log_probs = comp['log_probs'].detach().to(self.device)
                advantage = advantages[i]

                # Forward pass for current policy
                outputs = self.model(full_tokens)
                logits = outputs.logits

                # Compute new log probs for completion tokens
                new_log_probs = []
                for j in range(prompt_length, full_tokens.shape[1]):
                    token_id = full_tokens[0, j]
                    token_logits = logits[0, j - 1]
                    token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
                    new_log_probs.append(token_log_prob)

                if not new_log_probs:
                    continue

                new_log_probs = torch.stack(new_log_probs)

                # Ensure shapes match
                min_len = min(len(new_log_probs), len(old_log_probs))
                new_log_probs = new_log_probs[:min_len]
                old_log_probs_slice = old_log_probs[:min_len]

                # Compute ratio
                ratio = torch.exp(new_log_probs - old_log_probs_slice)

                # PPO-style clipped objective
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                )
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                ).mean()

                # KL divergence with reference model
                with torch.no_grad():
                    ref_outputs = self.ref_model(full_tokens)
                    ref_logits = ref_outputs.logits

                    ref_log_probs = []
                    for j in range(prompt_length, full_tokens.shape[1]):
                        token_id = full_tokens[0, j]
                        token_logits = ref_logits[0, j - 1]
                        ref_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
                        ref_log_probs.append(ref_log_prob)

                    if ref_log_probs:
                        ref_log_probs = torch.stack(ref_log_probs)[:min_len]

                kl_div = (new_log_probs - ref_log_probs).mean()

                # Total loss
                loss = policy_loss + self.kl_coef * kl_div
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Track metrics
                total_loss += loss.item() * self.gradient_accumulation_steps
                total_policy_loss += policy_loss.item()
                total_kl_div += kl_div.item()
                # total_reward += comp['reward']
                num_completions += 1
                policy_losses.append(policy_loss.item())
                kl_divs.append(kl_div.item())

                # Gradient accumulation step
                if num_completions % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Save checkpoint if needed
                    if self.global_step % self.save_steps == 0:
                        self._save_checkpoint(self.global_step)

                    # Clear cache periodically
                    if num_completions % (self.gradient_accumulation_steps * 4) == 0:
                        self._clear_memory()

        # Final optimizer step
        if num_completions % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            if self.global_step % self.save_steps == 0:
                self._save_checkpoint(self.global_step)

        self._clear_memory()

        n = max(num_completions, 1)
        metrics = {
            'loss': total_loss / n,
            'policy_loss': np.mean(policy_losses) if policy_losses else 0,
            'kl_div': np.mean(kl_divs) if kl_divs else 0,
            'mean_reward': total_reward / (len(samples) * self.num_generations),
            'num_completions': num_completions,
            'global_step': self.global_step,
        }

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'train/loss': metrics['loss'],
                'train/policy_loss': metrics['policy_loss'],
                'train/kl_div': metrics['kl_div'],
                'train/mean_reward': metrics['mean_reward'],
                'train/num_completions': metrics['num_completions'],
            }, step=self.global_step)

        return metrics

    def _save_checkpoint(self, step: int, prefix: str = "step"):
        """Save checkpoint (LoRA adapter only)."""
        checkpoint_path = self.checkpoint_dir / f"{prefix}_{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))

        print(f"Checkpoint saved to {checkpoint_path}")

        if self.use_wandb:
            wandb.log({'checkpoint_saved': step}, step=step)

        # Clean up old checkpoints
        if self.keep_last_n_checkpoints is not None:
            self._cleanup_old_checkpoints(prefix)

    def _cleanup_old_checkpoints(self, prefix: str = "step"):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir()
             if d.is_dir() and d.name.startswith(prefix)],
            key=lambda x: int(x.name.split('_')[-1])
        )

        if len(checkpoints) > self.keep_last_n_checkpoints:
            import shutil
            for old_checkpoint in checkpoints[:-self.keep_last_n_checkpoints]:
                shutil.rmtree(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")

    def save_model(self, output_dir: str, save_method: str = "lora"):
        """
        Save the model with Unsloth optimizations.

        Args:
            output_dir: Directory to save model
            save_method:
                - "lora" - Save only LoRA adapters (smallest)
                - "merged_16bit" - Merge and save in 16bit
                - "merged_4bit" - Merge and save in 4bit
        """
        if save_method == "lora":
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"LoRA adapters saved to {output_dir}")
        elif save_method == "merged_16bit":
            self.model.save_pretrained_merged(
                output_dir,
                self.tokenizer,
                save_method="merged_16bit"
            )
            print(f"Merged model (16bit) saved to {output_dir}")
        elif save_method == "merged_4bit":
            self.model.save_pretrained_merged(
                output_dir,
                self.tokenizer,
                save_method="merged_4bit"
            )
            print(f"Merged model (4bit) saved to {output_dir}")
        else:
            raise ValueError(f"Unknown save_method: {save_method}")

    def finish(self):
        """Clean up and finish wandb run."""
        if self.use_wandb:
            wandb.finish()
            print("WandB run finished")


def main():
    """Main entry point for TiG training."""
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Think in Games (TiG) Trainer")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of samples per training step")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--kl_coef", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    cli_args = parser.parse_args()

    # Create config
    config = TiGConfig()

    # Override with config file if provided
    if cli_args.config:
        yaml_config = load_config_from_yaml(cli_args.config)
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Override with CLI args
    if cli_args.model_name:
        config.model_name = cli_args.model_name
    if cli_args.dataset_path:
        config.dataset_path = cli_args.dataset_path
    if cli_args.checkpoint_dir:
        config.checkpoint_dir = cli_args.checkpoint_dir
    if cli_args.learning_rate:
        config.learning_rate = cli_args.learning_rate
    if cli_args.num_generations:
        config.num_generations = cli_args.num_generations
    if cli_args.kl_coef:
        config.kl_coef = cli_args.kl_coef
    if cli_args.temperature:
        config.temperature = cli_args.temperature
    if cli_args.wandb_project:
        config.wandb_project = cli_args.wandb_project
    if cli_args.wandb_run_name:
        config.wandb_run_name = cli_args.wandb_run_name
    if cli_args.no_wandb:
        config.use_wandb = False

    # Set seed
    config.seed = datetime.now().timestamp()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Initialize trainer
    trainer = TiGTrainer(config)

    # Training parameters
    num_epochs = cli_args.num_epochs
    batch_size = cli_args.batch_size

    # Training loop
    print("\n" + "="*60)
    print("Starting TiG Training")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Generations per sample: {config.num_generations}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"KL penalty: {config.kl_coef}")
    print("="*60 + "\n")

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Log epoch to wandb
        if trainer.use_wandb:
            wandb.log({'epoch': epoch + 1}, step=trainer.global_step)

        # Shuffle dataset
        random.shuffle(trainer.dataset)

        # Process in batches
        epoch_loss = 0.0
        epoch_reward = 0.0
        num_batches = 0

        for i in range(0, len(trainer.dataset), batch_size):
            batch = trainer.dataset[i:i + batch_size]

            print(f"\nBatch {num_batches + 1}: Processing {len(batch)} samples...")

            metrics = trainer.train_step(batch)

            epoch_loss += metrics['loss']
            epoch_reward += metrics['mean_reward']
            num_batches += 1

            print(f"  Loss: {metrics['loss']:.4f}, Reward: {metrics['mean_reward']:.4f}, "
                  f"KL: {metrics['kl_div']:.4f}, Completions: {metrics['num_completions']}")

        # Epoch summary
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_reward = epoch_reward / num_batches
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Reward: {avg_reward:.4f}")

            if trainer.use_wandb:
                wandb.log({
                    'epoch/loss': avg_loss,
                    'epoch/reward': avg_reward,
                }, step=trainer.global_step)

        # Save epoch checkpoint
        epoch_checkpoint_path = trainer.checkpoint_dir / f"epoch_{epoch + 1}"
        epoch_checkpoint_path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(epoch_checkpoint_path))
        trainer.tokenizer.save_pretrained(str(epoch_checkpoint_path))
        print(f"Epoch checkpoint saved to {epoch_checkpoint_path}")

        trainer._print_memory_stats()

    # Save final model
    print("\n" + "="*60)
    print("Training complete! Saving final model...")
    print("="*60)

    trainer.save_model(str(trainer.checkpoint_dir / "final_lora"), save_method="lora")

    # Finish wandb
    trainer.finish()

    print("\nAll done!")


if __name__ == "__main__":
    main()
