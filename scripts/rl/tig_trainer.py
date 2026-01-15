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
import asyncio
from datetime import datetime
import gc
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from unsloth import FastLanguageModel

from src.evaluation import reward_response


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}


class TiGConfig:
    """Configuration for TiG training."""

    # Model configuration
    model_name: str
    load_in_4bit: bool
    load_in_8bit: bool
    max_seq_length: int

    # Dataset
    dataset_path: str
    max_data_per_task: int

    # LoRA configuration
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules = None

    # Unsloth optimizations
    use_gradient_checkpointing = "unsloth"
    use_rslora: bool
    use_dora: bool

    # Training configuration
    num_epochs: int
    learning_rate: float
    max_grad_norm: float
    gradient_accumulation_steps: int

    # TiG/GRPO specific parameters
    prompts_per_step: int
    num_generations: int # G completions per state (paper uses 4-8)
    max_new_tokens: int # Max completion length
    temperature: float  # Sampling temperature
    clip_epsilon: float  # PPO clip range
    clip_epsilon_high: float
    kl_coef: float
    train_micro_batch_size: int = 4  # Micro-batch size for training forward pass (to avoid OOM)

    # Checkpoint settings
    checkpoint_dir: str
    save_steps: int
    keep_last_n_checkpoints: int

    # WandB settings
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str

    # Reproducibility
    seed: int


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
        self.clip_epsilon_high = config.clip_epsilon_high
        self.kl_coef = config.kl_coef
        self.num_generations = config.num_generations
        self.temperature = config.temperature
        self.train_micro_batch_size = getattr(config, 'train_micro_batch_size', 4)

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
        path_obj = Path(dataset_path)

        envs = [x for x in path_obj.iterdir() if x.is_dir()]

        print("\n" + "=" * 60)
        print("📂 LOADING DATASET")
        print("=" * 60)
        print(f"Dataset path: {dataset_path}")
        print(f"Environments found: {len(envs)}")
        print("-" * 60)

        for env in envs:
            task_files = list(env.glob("*.jsonl"))
            task_counts = {}

            for task_file in task_files:
                item_list = []
                task_sample_count = 0
                with open(task_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            trajectory = json.loads(line)
                            conversation = trajectory.get('conversation', [])
                            for i in range(len(conversation)):
                                if conversation[i].get('role') == 'assistant':
                                    item_list.append({
                                        'prompt': conversation[:i],
                                        'ground_truth_action': conversation[i].get('content'),
                                        'env': env.stem,
                                        'task': task_file.stem,
                                        'task_id': trajectory.get('task_id', None),
                                        'seed': trajectory.get('seed', None),
                                    })
                                    task_sample_count += 1
                data_list.extend(item_list[:self.config.max_data_per_task])
                task_counts[task_file.stem] = task_sample_count

            print(f"\n🌍 Environment: {env.stem}")
            print(f"   Tasks: {len(task_files)}")
            for task_name, count in task_counts.items():
                print(f"     • {task_name}: {count} samples")

        self.dataset = data_list
        print("\n" + "-" * 60)
        print(f"✅ DATASET LOADED: {len(data_list)} total samples")
        print("=" * 60 + "\n")

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
    def generate_completions_batched(
        self,
        prompts: List[list],
        num_samples: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Generate multiple completions for multiple prompts in a batched manner.

        This is more efficient than processing prompts one by one as it:
        1. Batches all prompts together with padding
        2. Generates G completions for all prompts in one batched generation

        Args:
            prompts: List of prompts, where each prompt is a list of messages
            num_samples: Number of completions to generate per prompt

        Returns:
            List of lists: outer list corresponds to prompts, inner list contains
            dicts with 'response', 'full_tokens', 'log_probs', 'prompt_length' keys
        """
        if num_samples is None:
            num_samples = self.num_generations

        if not prompts:
            return []

        # Enable fast inference mode
        FastLanguageModel.for_inference(self.model)

        # Tokenize all prompts with chat template
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            for prompt in prompts
        ]

        # Batch tokenize with padding (left padding for generation)
        tokenized = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)

        input_ids = tokenized.input_ids  # (batch_size, max_prompt_len)
        attention_mask = tokenized.attention_mask  # (batch_size, max_prompt_len)

        batch_size = input_ids.shape[0]

        # Track original prompt lengths (before padding)
        # With left padding, the actual content starts where mask becomes 1
        prompt_lengths = attention_mask.sum(dim=1).tolist()  # List of actual prompt lengths

        # Repeat each prompt num_samples times for batched generation
        # Shape: (batch_size * num_samples, max_prompt_len)
        expanded_input_ids = input_ids.repeat_interleave(num_samples, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(num_samples, dim=0)

        # Generate all completions in one batch
        outputs = self.model.generate(
            expanded_input_ids,
            attention_mask=expanded_attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )  # (batch_size * num_samples, max_prompt_len + max_new_tokens)

        # Compute log probs for all completions in a batch
        all_log_probs = self._compute_log_probs_batched(
            expanded_input_ids, outputs, expanded_attention_mask
        )

        # Organize results by prompt
        all_completions = []
        max_prompt_len = input_ids.shape[1]

        for prompt_idx in range(batch_size):
            prompt_completions = []
            prompt_length = prompt_lengths[prompt_idx]

            for sample_idx in range(num_samples):
                flat_idx = prompt_idx * num_samples + sample_idx
                generated_ids = outputs[flat_idx]

                # Extract completion tokens (after the padded prompt)
                completion_ids = generated_ids[max_prompt_len:]

                # Decode response
                response = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

                prompt_completions.append({
                    'response': response,
                    'full_tokens': generated_ids,
                    'log_probs': all_log_probs[flat_idx],
                    'prompt_length': max_prompt_len,  # Use padded length for consistency
                    'actual_prompt_length': prompt_length,
                })

            all_completions.append(prompt_completions)

        return all_completions

    @torch.no_grad()
    def _compute_log_probs_batched(
        self,
        prompt_ids: torch.Tensor,
        full_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Compute log probabilities for completion tokens in a batched manner.

        Args:
            prompt_ids: Tokenized prompts (batch_size, prompt_len)
            full_ids: Full sequences including completions (batch_size, full_len)
            attention_mask: Attention mask for prompts (batch_size, prompt_len)

        Returns:
            List of tensors, each containing log probabilities for completion tokens
        """
        batch_size = full_ids.shape[0]
        prompt_length = prompt_ids.shape[1]

        # Forward pass on full sequences
        outputs = self.model(full_ids)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Compute log probs for each sequence
        all_log_probs = []
        for i in range(batch_size):
            log_probs = []
            for j in range(prompt_length, full_ids.shape[1]):
                token_id = full_ids[i, j]
                # Skip padding tokens
                if token_id == self.tokenizer.pad_token_id:
                    break
                token_logits = logits[i, j - 1]
                token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
                log_probs.append(token_log_prob)

            if log_probs:
                all_log_probs.append(torch.stack(log_probs))
            else:
                all_log_probs.append(torch.tensor([], device=self.device))

        return all_log_probs

    @torch.no_grad()
    def generate_completions(
        self,
        prompt: list,
        num_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate multiple completions for a given state using Unsloth.

        Note: For better efficiency with multiple prompts, use generate_completions_batched().

        Args:
            prompt: list of messages
            num_samples: Number of completions to generate

        Returns:
            List of dicts with 'response', 'full_tokens', 'log_probs', 'prompt_length' keys
        """
        # Delegate to batched version with single prompt
        results = self.generate_completions_batched([prompt], num_samples)
        return results[0] if results else []

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

    async def _compute_rewards_for_completions(
        self,
        completions: List[Dict],
        env: str,
        task_id: int,
        ground_truth_action: str
    ) -> List[float]:
        """
        Compute rewards for a list of completions using reward_response.

        Args:
            completions: List of completion dicts with 'response' key
            env: Environment name
            task_id: Task ID for the evaluation
            ground_truth_action: Expected answer

        Returns:
            List of rewards for each completion
        """
        # Create reward computation tasks for all completions
        reward_tasks = [
            reward_response(
                env=env,
                task_id=task_id,
                response=comp['response'],
                ground_truth=ground_truth_action
            )
            for comp in completions
        ]

        # Compute rewards in parallel
        rewards = await asyncio.gather(*reward_tasks)
        return list(rewards)

    async def train_step(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Execute one training step on a batch of expert examples.

        For each example:
        1. Generate G completions with reasoning (batched for efficiency)
        2. Extract predicted actions and compute rewards using reward_response
        3. Compute group-relative advantages
        4. Compute GRPO loss with KL penalty

        Args:
            samples: List of expert data items with 'prompt', 'ground_truth_action', 'env', 'task_id'

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

        # Collect all prompts and metadata for batched generation
        prompts = [item.get("prompt", []) for item in samples]
        ground_truth_actions = [item["ground_truth_action"] for item in samples]
        envs = [item.get("env", "game") for item in samples]
        task_ids = [item.get("task_id", 0) for item in samples]

        # Generate completions for all prompts in one batch
        all_completions = self.generate_completions_batched(prompts)

        # Compute rewards for all completions in parallel
        reward_tasks = []
        for completions, env, task_id, ground_truth_action in zip(
            all_completions, envs, task_ids, ground_truth_actions
        ):
            reward_tasks.append(
                self._compute_rewards_for_completions(
                    completions, env, task_id, ground_truth_action
                )
            )

        all_rewards = await asyncio.gather(*reward_tasks)

        # Flatten all completions with their metadata for batched processing
        flat_completions = []  # List of (comp, advantage, sample_idx)

        for sample_idx, (completions, ground_truth_action) in enumerate(
            zip(all_completions, ground_truth_actions)
        ):
            rewards = all_rewards[sample_idx]

            # Assign rewards to completions
            for comp, reward in zip(completions, rewards):
                comp['reward'] = reward
                total_reward += reward

            # Skip if all rewards are same (no gradient signal) - DAPO dynamic sampling
            if len(set(rewards)) <= 1:
                continue

            # Compute group-relative advantages
            advantages = self.compute_advantages(rewards).to(self.device)

            # Add completions with their advantages to flat list
            for i, comp in enumerate(completions):
                flat_completions.append({
                    'comp': comp,
                    'advantage': advantages[i],
                })

        if not flat_completions:
            # No valid completions to train on
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'kl_div': 0.0,
                'mean_reward': total_reward / max(len(samples) * self.num_generations, 1),
                'num_completions': 0,
                'global_step': self.global_step,
            }

        # Ensure training mode
        FastLanguageModel.for_training(self.model)

        # Process completions in micro-batches
        micro_batch_size = self.train_micro_batch_size

        for batch_start in range(0, len(flat_completions), micro_batch_size):
            batch_end = min(batch_start + micro_batch_size, len(flat_completions))
            batch_items = flat_completions[batch_start:batch_end]

            # Prepare batch tensors with padding
            batch_tokens = []
            batch_prompt_lengths = []
            batch_old_log_probs = []
            batch_advantages = []

            max_seq_len = max(item['comp']['full_tokens'].shape[0] for item in batch_items)

            for item in batch_items:
                comp = item['comp']
                tokens = comp['full_tokens'].to(self.device)
                prompt_length = comp['prompt_length']

                # Pad to max_seq_len
                if tokens.shape[0] < max_seq_len:
                    pad_length = max_seq_len - tokens.shape[0]
                    pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    padding = torch.full((pad_length,), pad_token_id, device=self.device, dtype=tokens.dtype)
                    tokens = torch.cat([tokens, padding])

                batch_tokens.append(tokens)
                batch_prompt_lengths.append(prompt_length)
                batch_old_log_probs.append(comp['log_probs'].detach().to(self.device))
                batch_advantages.append(item['advantage'])

            # Stack into batch tensor
            batch_input_ids = torch.stack(batch_tokens)  # (batch_size, max_seq_len)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (batch_input_ids != (self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)).long()

            # Batched forward pass for policy model
            outputs = self.model(batch_input_ids, attention_mask=attention_mask)
            policy_logits = outputs.logits  # (batch_size, max_seq_len, vocab_size)

            # Batched forward pass for reference model
            with torch.no_grad():
                ref_outputs = self.ref_model(batch_input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits  # (batch_size, max_seq_len, vocab_size)

            # Compute loss for each item in the batch
            batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for idx, item in enumerate(batch_items):
                prompt_length = batch_prompt_lengths[idx]
                old_log_probs = batch_old_log_probs[idx]
                advantage = batch_advantages[idx]
                tokens = batch_tokens[idx]

                # Compute new log probs for completion tokens
                new_log_probs = []
                ref_log_probs_list = []

                for j in range(prompt_length, len(tokens)):
                    token_id = tokens[j]
                    # Skip padding tokens
                    if token_id == (self.tokenizer.pad_token_id or self.tokenizer.eos_token_id):
                        break

                    # Policy log prob
                    token_logits = policy_logits[idx, j - 1]
                    token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
                    new_log_probs.append(token_log_prob)

                    # Reference log prob
                    ref_token_logits = ref_logits[idx, j - 1]
                    ref_log_prob = F.log_softmax(ref_token_logits, dim=-1)[token_id]
                    ref_log_probs_list.append(ref_log_prob)

                if not new_log_probs:
                    continue

                new_log_probs = torch.stack(new_log_probs)
                ref_log_probs = torch.stack(ref_log_probs_list)

                # Ensure shapes match with old log probs
                min_len = min(len(new_log_probs), len(old_log_probs))
                new_log_probs = new_log_probs[:min_len]
                old_log_probs_slice = old_log_probs[:min_len]
                ref_log_probs = ref_log_probs[:min_len]

                # Compute ratio
                ratio = torch.exp(new_log_probs - old_log_probs_slice)

                # DAPO: Asymmetric clipping
                if advantage >= 0:
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - self.clip_epsilon,
                        1 + self.clip_epsilon_high
                    )
                else:
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - self.clip_epsilon,
                        1 + self.clip_epsilon
                    )
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                ).mean()

                # KL divergence
                kl_div = (new_log_probs - ref_log_probs).mean()

                # Accumulate loss
                item_loss = policy_loss + self.kl_coef * kl_div
                batch_loss = batch_loss + item_loss

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_kl_div += kl_div.item()
                policy_losses.append(policy_loss.item())
                kl_divs.append(kl_div.item())
                num_completions += 1

            # Average loss over batch and apply gradient accumulation scaling
            if num_completions > 0:
                batch_loss = batch_loss / len(batch_items) / self.gradient_accumulation_steps
                batch_loss.backward()
                total_loss += batch_loss.item() * self.gradient_accumulation_steps * len(batch_items)

            # Gradient accumulation step
            if (batch_end) % (self.gradient_accumulation_steps * micro_batch_size) == 0 or batch_end == len(flat_completions):
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
                self._clear_memory()

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


async def main():
    """Main entry point for TiG training."""
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Think in Games (TiG) Trainer")
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    cli_args = parser.parse_args()

    # Create config
    config = TiGConfig()

    # Override with config file if provided
    if cli_args.config:
        yaml_config = load_config_from_yaml(cli_args.config)
        for key, value in yaml_config.items():
            setattr(config, key, value)

    # Set seed
    config.seed = int(datetime.now().timestamp()) 
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Initialize trainer
    trainer = TiGTrainer(config)

    # Training parameters
    num_epochs = config.num_epochs
    batch_size = config.batch_size

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

            metrics = await trainer.train_step(batch)

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
    asyncio.run(main())
