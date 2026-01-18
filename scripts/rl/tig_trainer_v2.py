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
import copy
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
    num_epochs: int  # Number of epochs; each epoch has min_dataset_length batches
    learning_rate: float
    max_grad_norm: float
    gradient_accumulation_steps: int

    # TiG/GRPO specific parameters
    # Note: batch_size removed - each step uses 1 sample from 1 env/task
    num_generations: int # G completions per state (paper uses 4-8)
    temperature: float  # Sampling temperature
    clip_epsilon: float  # PPO clip range
    clip_epsilon_high: float
    kl_coef: float
    train_micro_batch_size: int = 2  # Micro-batch size for training forward pass (to avoid OOM)

    # Checkpoint settings
    checkpoint_dir: str
    save_steps: int
    ref_replace_steps: int = 200  # Replace reference model every N steps
    keep_last_n_checkpoints: int

    # WandB settings
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str

    # Trajectory logging settings
    save_trajectories: bool = True
    trajectory_dir: str = "trajectories"

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

        # Trajectory saving settings
        self.save_trajectories = getattr(config, 'save_trajectories', True)
        self.trajectory_dir = Path(getattr(config, 'trajectory_dir', 'trajectories'))
        if self.save_trajectories:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            print(f"Trajectory directory: {self.trajectory_dir}")

        # Per-task reward tracking: {(env, task): [rewards]}
        self.task_reward_tracker: Dict[tuple, List[float]] = {}

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

        # Create reference model as deep copy of policy model (with LoRA weights initialized to zero)
        # This ensures ref_model has the same architecture including LoRA adapters
        print("Creating reference model (deep copy of policy model with LoRA)...")
        self.ref_model = copy.deepcopy(self.model)
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
        self.train_micro_batch_size = getattr(config, 'train_micro_batch_size', 2)
        self.ref_replace_steps = getattr(config, 'ref_replace_steps', 200)

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

    def _replace_reference_model(self):
        """
        Replace reference model with current policy model.

        This updates the reference model to match the current policy,
        which can help stabilize training by preventing the policy from
        drifting too far from the updated reference.
        """
        print(f"\n{'='*60}")
        print(f"🔄 Replacing reference model at global_step={self.global_step}")
        print(f"{'='*60}")

        # Delete old reference model to free memory
        del self.ref_model
        self._clear_memory()

        # Create new reference model as deep copy of current policy
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        print(f"✅ Reference model replaced successfully")
        self._print_memory_stats()

        if self.use_wandb:
            wandb.log({'ref_model_replaced': self.global_step}, step=self.global_step)

    def _load_dataset(self):
        """
        Load expert dataset from JSONL files.

        Data is loaded per env/task and stored in a dict keyed by (env, task).
        All datasets are cut to the minimum length across all env/tasks to ensure
        balanced training.
        """
        # Store data per env/task: {(env, task): [samples]}
        self.task_datasets: Dict[tuple, List[Dict]] = {}
        dataset_path = self.config.dataset_path
        path_obj = Path(dataset_path)

        envs = [x for x in path_obj.iterdir() if x.is_dir() and not x.name.startswith(".")]

        print("\n" + "=" * 60)
        print("📂 LOADING DATASET (per env/task)")
        print("=" * 60)
        print(f"Dataset path: {dataset_path}")
        print(f"Environments found: {len(envs)}")
        print("-" * 60)

        task_counts = {}  # For logging: {(env, task): count}

        for env in envs:
            task_files = list(env.glob("*.jsonl"))

            for task_file in task_files:
                item_list = []
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

                # Apply max_data_per_task limit
                item_list = item_list[:self.config.max_data_per_task]

                if item_list:  # Only add non-empty datasets
                    key = (env.stem, task_file.stem)
                    self.task_datasets[key] = item_list
                    task_counts[key] = len(item_list)
                    self.task_reward_tracker[key] = []

            print(f"\n🌍 Environment: {env.stem}")
            print(f"   Tasks: {len(task_files)}")
            for (e, t), count in task_counts.items():
                if e == env.stem:
                    print(f"     • {t}: {count} samples")

        # Find minimum length across all datasets
        if self.task_datasets:
            min_length = min(len(samples) for samples in self.task_datasets.values())
            print(f"\n📏 Minimum dataset length: {min_length}")

            # Cut all datasets to minimum length
            total_samples = 0
            for key in self.task_datasets:
                self.task_datasets[key] = self.task_datasets[key][:min_length]
                total_samples += len(self.task_datasets[key])

            self.min_dataset_length = min_length
        else:
            self.min_dataset_length = 0
            total_samples = 0

        # Store list of task keys for iteration
        self.task_keys = list(self.task_datasets.keys())

        # For backward compatibility, also create flat dataset (used for logging total count)
        self.dataset = []
        for samples in self.task_datasets.values():
            self.dataset.extend(samples)

        print("\n" + "-" * 60)
        print(f"✅ DATASET LOADED:")
        print(f"   • {len(self.task_keys)} env/task combinations")
        print(f"   • {min_length} samples per env/task (balanced)")
        print(f"   • {total_samples} total samples")
        print("=" * 60 + "\n")

    def shuffle_task_datasets(self):
        """Shuffle samples within each task dataset."""
        for key in self.task_keys:
            random.shuffle(self.task_datasets[key])

    def get_sample_for_epoch(self, epoch_idx: int, task_key: tuple) -> Dict:
        """
        Get a single sample from a specific task dataset at the given epoch index.

        Args:
            epoch_idx: The epoch index (0 to min_dataset_length-1)
            task_key: Tuple of (env, task)

        Returns:
            Single sample dict
        """
        return self.task_datasets[task_key][epoch_idx]

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
        Generate multiple completions for multiple prompts.

        Processes one prompt at a time, generating all num_samples completions
        for that prompt in a single batch. This avoids OOM by limiting batch size
        to num_samples (e.g., 8) instead of batch_size * num_samples (e.g., 512).

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

        all_completions = []
        num_prompts = len(prompts)

        for prompt_idx, prompt in enumerate(prompts):
            print(f"    Generating completions for prompt {prompt_idx + 1}/{num_prompts}...", end="\r")

            # Tokenize single prompt with chat template
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )

            tokenized = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)

            input_ids = tokenized.input_ids  # (1, prompt_len)
            attention_mask = tokenized.attention_mask  # (1, prompt_len)
            prompt_length = input_ids.shape[1]

            # Repeat prompt num_samples times for batched generation
            # Shape: (num_samples, prompt_len)
            expanded_input_ids = input_ids.repeat(num_samples, 1)
            expanded_attention_mask = attention_mask.repeat(num_samples, 1)

            # Generate all completions for this prompt in one batch
            outputs = self.model.generate(
                expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_length=self.max_seq_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )  # (num_samples, prompt_len + generated_len)

            # Compute log probs for all completions (both policy and reference)
            # Precomputing ref_log_probs here avoids redundant forward passes during training
            policy_log_probs_list, ref_log_probs_list = self._compute_log_probs_batched(
                prompt_length, outputs
            )

            # Build completions for this prompt
            prompt_completions = []
            for sample_idx in range(num_samples):
                generated_ids = outputs[sample_idx]

                # Extract completion tokens (after the prompt)
                completion_ids = generated_ids[prompt_length:]

                # Decode response
                response = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

                prompt_completions.append({
                    'response': response,
                    'full_tokens': generated_ids,
                    'log_probs': policy_log_probs_list[sample_idx],
                    'ref_log_probs': ref_log_probs_list[sample_idx],  # Precomputed ref log probs
                    'prompt_length': prompt_length,
                    'actual_prompt_length': prompt_length,
                })

            all_completions.append(prompt_completions)

            # Clear cache between prompts
            self._clear_memory()

        print()  # Clear the progress line
        return all_completions

    @torch.no_grad()
    def _compute_log_probs_batched(
        self,
        prompt_length: int,
        full_ids: torch.Tensor,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute log probabilities for completion tokens in a batched manner.

        Computes log probs for both policy model and reference model to avoid
        redundant forward passes during training (ref_model is frozen).

        Memory optimization: Process each model separately and clear cache between
        forward passes to avoid holding both sets of logits in memory simultaneously.

        Args:
            prompt_length: Length of the prompt (same for all sequences in batch)
            full_ids: Full sequences including completions (batch_size, full_len)

        Returns:
            Tuple of (policy_log_probs, ref_log_probs), each a list of tensors
            containing log probabilities for completion tokens
        """
        batch_size = full_ids.shape[0]
        completion_tokens = full_ids[:, prompt_length:]  # (batch_size, completion_len)
        pad_token_id = self.tokenizer.pad_token_id

        # --- Policy model forward pass ---
        policy_logits = self.model(full_ids).logits  # (batch_size, seq_len, vocab_size)
        # Compute log softmax for completion tokens immediately
        policy_log_probs_all = F.log_softmax(
            policy_logits[:, prompt_length - 1:-1, :], dim=-1
        )  # (batch_size, completion_len, vocab_size)
        # Gather log probs for actual tokens
        policy_log_probs_gathered = policy_log_probs_all.gather(
            -1, completion_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, completion_len)
        # Free policy logits memory before ref model forward pass
        del policy_logits, policy_log_probs_all
        self._clear_memory()

        # --- Reference model forward pass ---
        ref_logits = self.ref_model(full_ids).logits  # (batch_size, seq_len, vocab_size)
        # Compute log softmax for completion tokens immediately
        ref_log_probs_all = F.log_softmax(
            ref_logits[:, prompt_length - 1:-1, :], dim=-1
        )  # (batch_size, completion_len, vocab_size)
        # Gather log probs for actual tokens
        ref_log_probs_gathered = ref_log_probs_all.gather(
            -1, completion_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, completion_len)
        # Free ref logits memory
        del ref_logits, ref_log_probs_all

        # Split into list and handle padding
        all_policy_log_probs = []
        all_ref_log_probs = []

        for i in range(batch_size):
            # Find actual completion length (exclude padding)
            tokens = completion_tokens[i]
            if pad_token_id is not None:
                non_pad_mask = tokens != pad_token_id
                actual_len = non_pad_mask.sum().item()
            else:
                actual_len = len(tokens)

            if actual_len > 0:
                all_policy_log_probs.append(policy_log_probs_gathered[i, :actual_len])
                all_ref_log_probs.append(ref_log_probs_gathered[i, :actual_len])
            else:
                all_policy_log_probs.append(torch.tensor([], device=self.device))
                all_ref_log_probs.append(torch.tensor([], device=self.device))

        return all_policy_log_probs, all_ref_log_probs

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

    # def _save_trajectory(
    #     self,
    #     env: str,
    #     task: str,
    #     prompt: list,
    #     completions: List[Dict],
    #     ground_truth_action: str,
    #     batch_idx: int,
    #     sample_idx: int
    # ):
    #     """
    #     Save a single trajectory to local JSONL file (organized by env/task).

    #     Args:
    #         env: Environment name
    #         task: Task name
    #         prompt: The prompt conversation
    #         completions: List of completions with responses and rewards
    #         ground_truth_action: The expected correct action
    #         batch_idx: Current batch index
    #         sample_idx: Sample index within the batch
    #     """
    #     if not self.save_trajectories:
    #         return

    #     # Create directory structure: trajectory_dir/env/task.jsonl
    #     env_dir = self.trajectory_dir / env
    #     env_dir.mkdir(parents=True, exist_ok=True)
    #     trajectory_file = env_dir / f"{task}.jsonl"

    #     # Build trajectory record
    #     trajectory_record = {
    #         'batch_idx': batch_idx,
    #         'sample_idx': sample_idx,
    #         'global_step': self.global_step,
    #         'env': env,
    #         'task': task,
    #         'prompt': prompt,
    #         'ground_truth_action': ground_truth_action,
    #         'completions': [
    #             {
    #                 'response': comp['response'],
    #                 'reward': comp.get('reward', 0.0),
    #             }
    #             for comp in completions
    #         ],
    #         'mean_reward': sum(comp.get('reward', 0.0) for comp in completions) / max(len(completions), 1),
    #     }

    #     # Append to JSONL file
    #     with open(trajectory_file, 'a', encoding='utf-8') as f:
    #         f.write(json.dumps(trajectory_record, ensure_ascii=False) + '\n')

    # def _log_per_task_rewards(self):
    #     """
    #     Log average rewards per env/task to wandb.
    #     """
    #     if not self.use_wandb or not self.task_reward_tracker:
    #         return

    #     task_reward_log = {}
    #     for (env, task), rewards in self.task_reward_tracker.items():
    #         if rewards:
    #             avg_reward = sum(rewards) / len(rewards)
    #             # Use hierarchical naming: reward/{env}/{task}
    #             task_reward_log[f'reward/{env}/{task}'] = avg_reward
    #             task_reward_log[f'reward/{env}/{task}_count'] = len(rewards)

    #     if task_reward_log:
    #         wandb.log(task_reward_log, step=self.global_step)

    def _log_trajectories_to_wandb(self, trajectories: List[Dict]):
        """
        Log trajectories to wandb as a table.

        Args:
            trajectories: List of trajectory records
        """
        if not self.use_wandb or not trajectories:
            return

        # Create a wandb Table for trajectories
        columns = ['batch_idx', 'env', 'task', 'mean_reward', 'prompt_preview', 'completion_preview']
        table = wandb.Table(columns=columns)

        for traj in trajectories:
            # Create preview of prompt (last message)
            prompt_preview = ""
            if traj['prompt']:
                last_msg = traj['prompt'][-1]
                content = last_msg.get('content', '')
                prompt_preview = content[:200] + "..." if len(content) > 200 else content

            # Create preview of best completion
            completions = traj.get('completions', [])
            if completions:
                best_comp = max(completions, key=lambda x: x.get('reward', 0))
                comp_preview = best_comp['response'][:200] + "..." if len(best_comp['response']) > 200 else best_comp['response']
            else:
                comp_preview = ""

            table.add_data(
                traj['batch_idx'],
                traj['env'],
                traj['task'],
                traj['mean_reward'],
                prompt_preview,
                comp_preview
            )

        wandb.log({'trajectories': table}, step=self.global_step)

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
            batch_idx: Current batch index for logging

        Returns:
            Dict of training metrics
        """

        self.optimizer.zero_grad()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl_div = 0.0
        total_reward = 0.0
        num_completions = 0
        num_skipped_samples = 0  # Track samples skipped due to DAPO (same rewards)
        policy_losses = []
        kl_divs = []

        # Collect all prompts and metadata for batched generation
        prompts = [item.get("prompt", []) for item in samples]
        ground_truth_actions = [item["ground_truth_action"] for item in samples]
        envs = [item.get("env", "game") for item in samples]
        tasks = [item.get("task", "default") for item in samples]
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
            env = envs[sample_idx]
            task = tasks[sample_idx]
            prompt = prompts[sample_idx]

            # Assign rewards to completions
            for comp, reward in zip(completions, rewards):
                comp['reward'] = reward
                total_reward += reward

            # Update per-task reward tracker (add mean reward)
            self.task_reward_tracker[(env, task)].append(sum(rewards) / max(len(rewards), 1))

            # Skip if all rewards are same (no gradient signal) - DAPO dynamic sampling
            if len(set(rewards)) <= 1:
                num_skipped_samples += 1
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
            metrics = {
                'loss': 0.0,
                'policy_loss': 0.0,
                'kl_div': 0.0,
                'mean_reward': total_reward / max(len(samples) * self.num_generations, 1),
                'num_completions': 0,
                'num_skipped_samples': num_skipped_samples,
                'num_samples': len(samples),
                'global_step': self.global_step,
                'backprop_happened': False,
            }
            # Log to wandb even when no training happens (for monitoring)
            if self.use_wandb:
                wandb.log({
                    'global_step': self.global_step,
                    'loss': 0.0,
                    'mean_reward': 0,
                    'num_completions': 0,
                    'num_skipped_samples': num_skipped_samples,
                    'backprop_happened': 0,
                })
            return metrics

        # Ensure training mode
        FastLanguageModel.for_training(self.model)

        num_completions = len(flat_completions)
        # Process completions one by one and accumulate gradients
        for item in flat_completions:
            comp = item['comp']
            advantage = item['advantage']

            tokens = comp['full_tokens'].to(self.device)
            prompt_length = comp['prompt_length']
            old_log_probs = comp['log_probs'].detach().to(self.device)
            ref_log_probs = comp['ref_log_probs'].detach().to(self.device)

            # Forward pass
            input_ids = tokens.unsqueeze(0)  # (1, seq_len)
            outputs = self.model(input_ids)
            policy_logits = outputs.logits[0]  # (seq_len, vocab_size)

            # Vectorized log prob computation for completion tokens
            # logits at position j-1 predict token at position j
            completion_logits = policy_logits[prompt_length - 1:-1]  # (completion_len, vocab_size)
            completion_tokens = tokens[prompt_length:]  # (completion_len,)

            if len(completion_tokens) == 0:
                continue

            # Compute log softmax and gather log probs for actual tokens
            log_probs_all = F.log_softmax(completion_logits, dim=-1)  # (completion_len, vocab_size)
            new_log_probs = log_probs_all.gather(-1, completion_tokens.unsqueeze(-1)).squeeze(-1)  # (completion_len,)

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

            # KL divergence (approximate, always >= 0)
            log_ratio = new_log_probs - ref_log_probs
            kl_ratio = torch.exp(log_ratio)
            kl_div = ((kl_ratio - 1) - log_ratio).mean()

            # Compute loss and accumulate gradient
            item_loss = (policy_loss + self.kl_coef * kl_div) / num_completions
            item_loss.backward()

            # Track metrics
            total_loss += item_loss.item()
            total_policy_loss += policy_loss.item()
            total_kl_div += kl_div.item()
            policy_losses.append(policy_loss.item())
            kl_divs.append(kl_div.item())

        # Single optimizer step per train_step
        if num_completions > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()
            self.global_step += 1

            # Save checkpoint if needed
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint(self.global_step)

            # Replace reference model if needed
            if self.ref_replace_steps > 0 and self.global_step % self.ref_replace_steps == 0:
                self._replace_reference_model()

        self._clear_memory()

        backprop_happened = num_completions > 0
        metrics = {
            'loss': total_loss,
            'policy_loss': np.mean(policy_losses) if policy_losses else 0,
            'kl_div': np.mean(kl_divs) if kl_divs else 0,
            'mean_reward': total_reward / (len(samples) * self.num_generations),
            'num_completions': num_completions,
            'num_skipped_samples': num_skipped_samples,
            'num_samples': len(samples),
            'global_step': self.global_step,
            'backprop_happened': backprop_happened,
        }

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'global_step': self.global_step,
                'loss': metrics['loss'],
                'policy_loss': metrics['policy_loss'],
                'kl_div': metrics['kl_div'],
                'mean_reward': metrics['mean_reward'],
                'num_completions': num_completions,
                'num_skipped_samples': num_skipped_samples,
                'backprop_happened': 1 if backprop_happened else 0,
            }, step=self.global_step)

            # Log per-task rewards
            # self._log_per_task_rewards()

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
    num_tasks = len(trainer.task_keys)
    batches_per_epoch = trainer.min_dataset_length  # Number of batches per epoch

    # Training loop
    print("\n" + "="*60)
    print("Starting TiG Training")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Epochs: {num_epochs}")
    print(f"Env/Task combinations: {num_tasks}")
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Total training batches: {num_epochs * batches_per_epoch}")
    print(f"Generations per sample: {config.num_generations}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"KL penalty: {config.kl_coef}")
    print("="*60 + "\n")

    # Shuffle all task datasets at the beginning
    trainer.shuffle_task_datasets()

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Log epoch to wandb
        if trainer.use_wandb:
            wandb.log({'epoch': epoch + 1}, step=trainer.global_step)

        epoch_loss = 0.0
        epoch_reward = 0.0
        num_steps = 0
        epoch_global_step_start = trainer.global_step

        # Each epoch has batches_per_epoch batches, each batch contains 1 sample from each task
        for batch_idx in range(batches_per_epoch):
            # Collect one sample from each task at the current batch index
            batch = []
            for task_key in trainer.task_keys:
                sample = trainer.task_datasets[task_key][batch_idx]
                batch.append(sample)
            num_steps += 1

            print(f"\n[Epoch {epoch + 1}/{num_epochs}] Batch {num_steps}/{batches_per_epoch} "
                  f"({num_tasks} samples) (global_step={trainer.global_step})")

            metrics = await trainer.train_step(batch)

            epoch_loss += metrics['loss']
            epoch_reward += metrics['mean_reward']

            # Detailed logging
            backprop_status = "✓ BACKPROP" if metrics['backprop_happened'] else "✗ NO BACKPROP"
            print(f"  [{backprop_status}] global_step: {metrics['global_step']} (+{metrics['global_step_delta']})")
            print(f"  Loss: {metrics['loss']:.4f}, Reward: {metrics['mean_reward']:.4f}, "
                  f"KL: {metrics['kl_div']:.4f}")
            print(f"  Completions: {metrics['num_completions']}, "
                  f"Skipped (DAPO): {metrics['num_skipped_samples']}/{metrics['num_samples']} samples")

        # Epoch summary
        if num_steps > 0:
            avg_loss = epoch_loss / num_steps
            avg_reward = epoch_reward / num_steps
            epoch_global_step_delta = trainer.global_step - epoch_global_step_start
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Reward: {avg_reward:.4f}")
            print(f"  Steps processed: {num_steps}")
            print(f"  Global step progress: {epoch_global_step_start} -> {trainer.global_step} "
                  f"(+{epoch_global_step_delta} optimizer steps)")
            print(f"{'='*60}")

            if trainer.use_wandb:
                wandb.log({
                    'epoch/loss': avg_loss,
                    'epoch/reward': avg_reward,
                    'epoch/num_steps': num_steps,
                    'epoch/global_step_delta': epoch_global_step_delta,
                }, step=trainer.global_step)

        # Save epoch checkpoint periodically (not every epoch to save disk space)
        if (epoch + 1) % config.save_steps == 0 or epoch == num_epochs - 1:
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
