import random
import asyncio
import torch
import torch.nn.functional as F
from torch.optim import Adam
from unsloth import FastLanguageModel, is_bfloat16_supported
from typing import List, Dict, Tuple, Optional
import numpy as np
import gc
import wandb
import os
from pathlib import Path

from src.evaluation import ACTOR_MAPPING, ENVS_WITH_TASK_TYPE
from src import get_base_task_name


class TaskDataset:
    """
    Dataset class for task_env and range format.
    
    Format: Each item has task_env (str) and range [start, end]
    """
    def __init__(self, data: List[Dict]):
        """
        Args:
            data: List of dicts with format:
                  [{'task_env': 'math', 'range': [0, 100]},
                   {'task_env': 'coding', 'range': [0, 50]}, ...]
        """
        self.data = data
    
    def sample(self, n: int) -> List[Dict]:
        """Sample n random tasks."""
        samples = []
        
        for _ in range(n):
            # Randomly choose a task environment
            config = random.choice(self.data)
            
            # Randomly choose an ID within that environment's range
            task_id = random.randint(config['range'][0], config['range'][1] - 1)
            
            samples.append({
                'task_env': config['task_env'],
                'task_id': task_id
            })
        
        return samples
    

class GRPO_DAPO_Trainer:
    """
    GRPO/DAPO implementation with Unsloth for maximum efficiency.
    
    Key features:
    - Unsloth for 2-5x faster training and 60% less memory
    - LoRA with Unsloth optimizations
    - Memory-efficient training with gradient accumulation
    - Native support for 4-bit/8-bit quantization
    - Optimized for 30B+ models
    - WandB integration for experiment tracking
    - Periodic checkpoint saving
    """
    
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        clip_epsilon_high: float = 5.0,
        kl_coef: float = 0.01,
        num_samples_per_prompt: int = 4,
        use_token_level_loss: bool = False,
        # LoRA parameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0,
        lora_target_modules: Optional[List[str]] = None,
        # Memory management
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        load_in_4bit: bool = False,
        # Generation settings
        max_seq_length: int = 2048,
        max_new_tokens: int = 512,
        # Unsloth specific
        use_gradient_checkpointing: str = "unsloth",  # "unsloth" is faster
        use_rslora: bool = False,  # Rank stabilized LoRA
        use_dora: bool = False,  # Weight decomposed LoRA (slower but better)
        # WandB settings
        use_wandb: bool = True,
        wandb_project: str = "dapo-training",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        # Checkpoint settings
        checkpoint_dir: str = "./checkpoints",
        save_steps: int = 100,  # Save checkpoint every N training steps
        keep_last_n_checkpoints: Optional[int] = 3,  # Keep only last N checkpoints (None = keep all)
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Memory management settings
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        
        # Checkpoint settings
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = save_steps
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.global_step = 0
        
        # WandB settings
        self.use_wandb = use_wandb
        if self.use_wandb:
            # Prepare config for wandb
            config = {
                'model_name': model_name,
                'learning_rate': learning_rate,
                'clip_epsilon': clip_epsilon,
                'clip_epsilon_high': clip_epsilon_high,
                'kl_coef': kl_coef,
                'num_samples_per_prompt': num_samples_per_prompt,
                'use_token_level_loss': use_token_level_loss,
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'max_grad_norm': max_grad_norm,
                'load_in_4bit': load_in_4bit,
                'max_seq_length': max_seq_length,
                'max_new_tokens': max_new_tokens,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'use_rslora': use_rslora,
                'use_dora': use_dora,
                'save_steps': save_steps,
            }
            
            # Merge with user-provided config
            if wandb_config:
                config.update(wandb_config)
            
            # Initialize wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config
            )
            print(f"WandB initialized: {wandb.run.name}")
        
        # Detect optimal dtype
        dtype = None  # Auto-detect (fp16 or bf16)
        if is_bfloat16_supported():
            print("Using bfloat16 for better numerical stability")
        
        # Load model with Unsloth (MUCH faster than HF)
        print(f"Loading model with Unsloth: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # Unsloth optimizations
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Apply Unsloth-optimized LoRA
        print("Applying Unsloth-optimized LoRA...")
        if lora_target_modules is None:
            # Unsloth will auto-detect the best modules
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=42,
            use_rslora=use_rslora,
            loftq_config=None,
            # DoRA (weight decomposed LoRA) - slower but sometimes better
            use_dora=use_dora,
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
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Optimizer (only trainable parameters)
        trainable_params_list = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(trainable_params_list, lr=learning_rate)
        
        # Hyperparameters
        self.clip_epsilon = clip_epsilon
        self.clip_epsilon_high = clip_epsilon_high
        self.kl_coef = kl_coef
        self.num_samples = num_samples_per_prompt
        self.use_token_level_loss = use_token_level_loss
        
        print(f"Trainer initialized with Unsloth optimizations")
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
    
    def _save_checkpoint(self, step: int, prefix: str = "step"):
        """
        Save checkpoint (LoRA adapter only).
        
        Args:
            step: Current training step
            prefix: Prefix for checkpoint folder name
        """
        checkpoint_path = self.checkpoint_dir / f"{prefix}_{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        self.model.save_pretrained(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))
        
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({'checkpoint_saved': step}, step=step)
        
        # Clean up old checkpoints if needed
        if self.keep_last_n_checkpoints is not None:
            self._cleanup_old_checkpoints(prefix)
    
    def _cleanup_old_checkpoints(self, prefix: str = "step"):
        """Remove old checkpoints, keeping only the last N."""
        # Get all checkpoint directories
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() 
             if d.is_dir() and d.name.startswith(prefix)],
            key=lambda x: int(x.name.split('_')[-1])
        )
        
        # Remove old checkpoints
        if len(checkpoints) > self.keep_last_n_checkpoints:
            for old_checkpoint in checkpoints[:-self.keep_last_n_checkpoints]:
                import shutil
                shutil.rmtree(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
    
    @torch.no_grad()
    def generate_trajectories(
        self,
        tasks: List[Dict],
    ) -> List[Dict]:
        """
        Generate trajectories by running evaluation on tasks.

        Args:
            tasks: List of dicts with 'task_env' and 'task_id' keys
                   e.g., [{'task_env': 'game', 'task_id': 123}, ...]

        Returns:
            List of trajectory dicts containing:
                - task_env: str
                - task_id: int
                - rollout: list of conversation messages
                - reward: float (reward from environment)
                - success: bool
                - time_taken: float
        """
        # Enable fast inference mode (2x faster generation)
        FastLanguageModel.for_inference(self.model)

        trajectories = []

        total_trajectories = len(tasks) * self.num_samples
        print(f"Generating {total_trajectories} trajectories ({len(tasks)} tasks x {self.num_samples} samples)...")

        for task_idx, task in enumerate(tasks):
            task_env = task['task_env']
            task_id = task['task_id']

            # Get the actor class for this environment
            actor_cls = ACTOR_MAPPING.get(task_env.lower())
            if actor_cls is None:
                print(f"Warning: Unknown environment '{task_env}', skipping task {task_id}")
                continue

            # Generate num_samples trajectories for the same task
            for sample_idx in range(self.num_samples):
                # Initialize actor (fresh instance per sample)
                actor = actor_cls()

                # Build evaluation kwargs
                eval_kwargs = {
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "task_id": task_id,
                }

                # Add task_type for specific environments
                if task_env.lower() in ENVS_WITH_TASK_TYPE:
                    eval_kwargs["task_type"] = get_base_task_name(task_env)

                try:
                    # Run evaluation (generates the trajectory)
                    result = asyncio.run(actor.local_evaluate(**eval_kwargs))

                    # Extract rollout from result
                    rollout = []
                    if "extra" in result and "conversation" in result["extra"]:
                        rollout = result["extra"]["conversation"]
                        
                        log_probs, full_tokens = self._compute_log_probs_efficient(rollout)
                        log_probs = log_probs.to(self.device)

                    trajectories.append({
                        'task_env': task_env,
                        'task_id': task_id,
                        'sample_idx': sample_idx,
                        'rollout': rollout,
                        'log_probs': log_probs,
                        'full_tokens': full_tokens,
                        'reward': result.get('score', 0.0),
                        'success': result.get('success', False),
                        'time_taken': result.get('time_taken', 0.0),
                        'task_metadata': task,
                    })

                except Exception as e:
                    print(f"Error evaluating task {task_env}:{task_id} (sample {sample_idx}): {e}")
                    continue

                # Clear cache periodically
                if (len(trajectories)) % 4 == 0:
                    self._clear_memory()

            if (task_idx + 1) % 10 == 0:
                print(f"  Completed {task_idx + 1}/{len(tasks)} tasks ({len(trajectories)} trajectories)")

        # Switch back to training mode
        FastLanguageModel.for_training(self.model)

        self._clear_memory()
        
        # Log trajectory statistics to wandb
        if self.use_wandb and trajectories:
            wandb.log({
                'trajectories/total_generated': len(trajectories),
                'trajectories/mean_reward': np.mean([t['reward'] for t in trajectories]),
                'trajectories/success_rate': np.mean([t['success'] for t in trajectories]),
            }, step=self.global_step)
        
        return trajectories
    
    @torch.no_grad()
    def _compute_log_probs_efficient(self, conversations):
        """Compute log probabilities efficiently."""
        full_text = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
        )
        
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
            
        # Forward pass with Unsloth optimizations
        outputs = self.model(full_tokens)
        logits = outputs.logits
        
        # Compute log probs
        log_probs = []
        
        response_mask = self._get_response_mask(conversations)
        
        for j in range(full_tokens.shape[1]):
            token_id = full_tokens[0, j]
            if response_mask[j] == 1:
                token_logits = logits[0, j - 1]
                token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
                log_probs.append(token_log_prob)
        
        return torch.tensor(log_probs), full_tokens
    
    def compute_advantages(self, trajectories: List[Dict]) -> torch.Tensor:
        """Compute group-based advantages (GRPO approach)."""
        prompt_groups = {}
        for i, traj in enumerate(trajectories):
            prompt = f"{traj['task_env']}_{traj['task_id']}"
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append((i, traj['reward']))
        
        advantages = torch.zeros(len(trajectories))
        
        for prompt, group in prompt_groups.items():
            indices = [idx for idx, _ in group]
            rewards = torch.tensor([reward for _, reward in group])
            
            group_mean = rewards.mean()
            group_std = rewards.std() + 1e-8
            group_advantages = (rewards - group_mean) / group_std
            
            for idx, adv in zip(indices, group_advantages):
                advantages[idx] = adv
        
        return advantages
    
    def dynamic_sampling_filter(self, trajectories: List[Dict]) -> List[Dict]:
        """DAPO: Filter out zero-gradient prompts."""
        taskid_groups = {}
        for traj in trajectories:
            taskid = f"{traj['task_env']}_{traj['task_id']}"
            if taskid not in taskid_groups:
                taskid_groups[taskid] = []
            taskid_groups[taskid].append(traj)
        
        filtered = []
        for taskid, group in taskid_groups.items():
            rewards = [t['reward'] for t in group]
            if len(set(rewards)) > 1:
                filtered.extend(group)
        
        return filtered if filtered else trajectories
    
    def train_step(
        self, 
        trajectories: List[Dict],
        use_dynamic_sampling: bool = True,
    ) -> Dict[str, float]:
        """
        Memory-efficient training step with Unsloth optimizations.
        """
        # Ensure training mode
        FastLanguageModel.for_training(self.model)
        
        # DAPO: Dynamic sampling
        if use_dynamic_sampling:
            trajectories = self.dynamic_sampling_filter(trajectories)
        
        if len(trajectories) == 0:
            return {'error': 'No valid trajectories after filtering'}
        
        # Compute advantages
        advantages = self.compute_advantages(trajectories)
        
        # Gradient accumulation
        self.optimizer.zero_grad()
        
        total_loss = 0
        policy_losses = []
        kl_divs = []
        
        for i, traj in enumerate(trajectories):
            # Move tensors to device
            old_log_probs = traj['log_probs'].to(self.device)
            
            full_tokens = traj['full_tokens'].to(self.device)
            
            # Forward pass with Unsloth optimizations
            outputs = self.model(full_tokens)
            logits = outputs.logits
            
            # Compute new log probs
            new_log_probs = []
            
            response_mask = self._get_response_mask(traj.get('rollout'))
            
            for j in range(full_tokens.shape[1]):
                token_id = full_tokens[0, j]
                if response_mask[j] == 1:
                    token_logits = logits[0, j - 1]
                    token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
                    new_log_probs.append(token_log_prob)
            
            new_log_probs = torch.stack(new_log_probs)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            advantage = advantages[i]
            
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
            
            # Policy loss
            if self.use_token_level_loss:
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                ).sum()
            else:
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                ).mean()
            
            # KL divergence with reference model
            with torch.no_grad():
                ref_outputs = self.ref_model(full_tokens)
                ref_logits = ref_outputs.logits
                
                ref_log_probs = []
                for j in range(full_tokens.shape[1]):
                    token_id = full_tokens[0, j]
                    if response_mask[j] == 1:
                        token_logits = ref_logits[0, j - 1]
                        ref_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
                        ref_log_probs.append(ref_log_prob)
                
                ref_log_probs = torch.stack(ref_log_probs)
            
            kl_div = (new_log_probs - ref_log_probs).mean()
            
            # Total loss
            loss = policy_loss + self.kl_coef * kl_div
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass (Unsloth optimized)
            loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            policy_losses.append(policy_loss.item())
            kl_divs.append(kl_div.item())
            
            # Gradient accumulation step
            if (i + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Increment global step
                self.global_step += 1
                
                # Save checkpoint if needed
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(self.global_step)
                
                # Clear cache periodically
                if (i + 1) % (self.gradient_accumulation_steps * 4) == 0:
                    self._clear_memory()
        
        # Final optimizer step
        if len(trajectories) % self.gradient_accumulation_steps != 0:
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
        
        self._clear_memory()
        
        metrics = {
            'total_loss': total_loss,
            'policy_loss': np.mean(policy_losses),
            'kl_div': np.mean(kl_divs),
            'num_trajectories': len(trajectories),
            'global_step': self.global_step
        }
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'train/total_loss': metrics['total_loss'],
                'train/policy_loss': metrics['policy_loss'],
                'train/kl_div': metrics['kl_div'],
                'train/num_trajectories': metrics['num_trajectories'],
            }, step=self.global_step)
        
        return metrics
    
    def save_model(self, output_dir: str, save_method: str = "merged_16bit"):
        """
        Save the model with Unsloth optimizations.
        
        Args:
            output_dir: Directory to save model
            save_method: 
                - "lora" - Save only LoRA adapters (smallest)
                - "merged_16bit" - Merge and save in 16bit (recommended)
                - "merged_4bit" - Merge and save in 4bit (smallest full model)
        """
        if save_method == "lora":
            # Save only LoRA adapters
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"LoRA adapters saved to {output_dir}")
            
        elif save_method == "merged_16bit":
            # Merge LoRA and save full model in 16bit
            self.model.save_pretrained_merged(
                output_dir,
                self.tokenizer,
                save_method="merged_16bit"
            )
            print(f"Merged model (16bit) saved to {output_dir}")
            
        elif save_method == "merged_4bit":
            # Merge LoRA and save in 4bit
            self.model.save_pretrained_merged(
                output_dir,
                self.tokenizer,
                save_method="merged_4bit"
            )
            print(f"Merged model (4bit) saved to {output_dir}")
        
        else:
            raise ValueError(f"Unknown save_method: {save_method}")
    
    def push_to_hub(
        self, 
        repo_id: str, 
        save_method: str = "lora",
        token: Optional[str] = None
    ):
        """
        Push model to Hugging Face Hub.
        
        Args:
            repo_id: HF repo name (e.g., "username/model-name")
            save_method: "lora", "merged_16bit", or "merged_4bit"
            token: HF token (optional)
        """
        if save_method == "lora":
            self.model.push_to_hub(repo_id, token=token)
            self.tokenizer.push_to_hub(repo_id, token=token)
        else:
            self.model.push_to_hub_merged(
                repo_id,
                self.tokenizer,
                save_method=save_method,
                token=token
            )
        print(f"Model pushed to {repo_id}")
    
    def finish(self):
        """Clean up and finish wandb run."""
        if self.use_wandb:
            wandb.finish()
            print("WandB run finished")

    # these functions will move to BasicTrainer
    def _get_response_mask(self, conversations):
        """
        Get a mask for the assistant responses in a conversation.
        
        Args:
            conversations: List of conversation turns with "role" field
        
        Returns:
            Mask tensor of shape (seq_len,) with 1s for assistant responses
        """
        mask = []
        for msg in conversations:
            input_ids = self.tokenizer.apply_chat_template(
                [msg],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False
            )
            if msg['role'] == 'assistant':
                mask.extend([1] * len(input_ids))
            else:
                mask.extend([0] * len(input_ids))
        return torch.tensor(mask)


if __name__ == "__main__":
    # =====================================================================
    # 1. Setup your task dataset
    # =====================================================================
    
    # Define your dataset with task_env and ranges
    dataset_config = [
        {'task_env': 'math', 'range': [0, 100]},      # Math problems 0-99
        {'task_env': 'coding', 'range': [0, 50]},     # Coding tasks 0-49
        {'task_env': 'dialogue', 'range': [0, 200]},  # Dialogue tasks 0-199
    ]
    
    # Create dataset
    dataset = TaskDataset(dataset_config)
    
    # =====================================================================
    # 2. Initialize trainer with Unsloth and WandB
    # =====================================================================
    
    trainer = GRPO_DAPO_Trainer(
        model_name="Qwen/Qwen3-0.6B",  # or your 30B model
        learning_rate=2e-5,
        num_samples_per_prompt=4,
        # LoRA settings
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        # Memory management
        gradient_accumulation_steps=8,
        load_in_4bit=True,
        max_seq_length=2048,
        max_new_tokens=256,
        # Unsloth optimizations
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        use_dora=False,
        # WandB settings
        use_wandb=True,
        wandb_project="grpo-dapo-training",
        wandb_run_name="qwen-grpo-experiment-1",
        wandb_config={
            'dataset': 'mixed_tasks',
            'note': 'GRPO with DAPO on mixed task dataset'
        },
        # Checkpoint settings
        checkpoint_dir="./checkpoints",
        save_steps=100,  # Save every 100 training steps
        keep_last_n_checkpoints=3,  # Keep only last 3 checkpoints
    )
    
    # =====================================================================
    # 3. Training loop with task-based sampling
    # =====================================================================
    
    num_epochs = 3
    tasks_per_epoch = 20  # Sample 20 tasks per epoch
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Log epoch to wandb
        if trainer.use_wandb:
            wandb.log({'epoch': epoch + 1}, step=trainer.global_step)
        
        # Sample tasks from dataset
        sampled_tasks = dataset.sample(tasks_per_epoch)
        print(f"Sampled {len(sampled_tasks)} tasks")
        
        # Generate trajectories
        print("Generating trajectories...")
        trajectories = trainer.generate_trajectories(sampled_tasks)
        
        # Optional: Print some statistics by task_env
        env_stats = {}
        for traj in trajectories:
            env = traj['task_metadata']['task_env']
            if env not in env_stats:
                env_stats[env] = []
            env_stats[env].append(traj['reward'])
        
        print("\nReward statistics by environment:")
        for env, rewards in env_stats.items():
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            print(f"  {env}: mean={mean_reward:.3f}, std={std_reward:.3f}")
            
            # Log to wandb
            if trainer.use_wandb:
                wandb.log({
                    f'rewards/{env}_mean': mean_reward,
                    f'rewards/{env}_std': std_reward,
                }, step=trainer.global_step)
        
        # Train
        print("\nTraining...")
        metrics = trainer.train_step(trajectories, use_dynamic_sampling=True)
        print(f"Training metrics: {metrics}")
        
        trainer._print_memory_stats()
        
        # Save checkpoint at end of epoch (in addition to step-based saves)
        epoch_checkpoint_path = trainer.checkpoint_dir / f"epoch_{epoch + 1}"
        epoch_checkpoint_path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(epoch_checkpoint_path))
        trainer.tokenizer.save_pretrained(str(epoch_checkpoint_path))
        print(f"Epoch checkpoint saved to {epoch_checkpoint_path}")
    
    # =====================================================================
    # 4. Save final model
    # =====================================================================
    
    print("\n" + "="*60)
    print("Training complete! Saving final model...")
    print("="*60)
    
    # Save only LoRA adapters (smallest, ~few MB)
    trainer.save_model("./final_model/lora_adapter", save_method="lora")
    
    # Optional: Save merged models
    # trainer.save_model("./final_model/merged_16bit", save_method="merged_16bit")
    # trainer.save_model("./final_model/merged_4bit", save_method="merged_4bit")
    
    # Optional: Push to Hub
    # trainer.push_to_hub("your-username/grpo-model", save_method="lora")
    
    # Finish wandb run
    trainer.finish()
    
    print("\nAll done!")