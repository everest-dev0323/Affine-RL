"""
SFT Reasoning Trainer

Supervised Fine-Tuning trainer for multi-turn conversations with reasoning content.
Uses Unsloth for high performance, LoRA for memory efficiency, and KL-divergence
penalty to prevent overfitting.

Features:
- Loads JSONL data from data_reasoning folder
- Splits conversations by assistant messages
- Combines reasoning_content + content for training
- Uses Unsloth + LoRA for efficient 14B model training
- KL-divergence penalty against frozen reference model
- Validation loss monitoring

Usage:
    python sft_reasoning_trainer.py --config config/sft_reasoning.yaml
"""

import argparse
from dataclasses import dataclass
import gc
import json
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from unsloth import FastLanguageModel

from src.utils import load_config_from_yaml


@dataclass
class SFTReasoningConfig:
    """Configuration for SFT Reasoning training.

    All fields have sensible defaults. Override any value by loading
    from a YAML config file via ``--config``.
    """

    # Model configuration
    model_name: str = "Qwen/Qwen3-14B"
    load_in_4bit: bool = False  # Use full precision for best quality
    load_in_8bit: bool = False
    max_seq_length: int = 8192

    # Dataset
    dataset_path: str = "data_reasoning"
    train_split: float = 0.95  # Fraction for training, rest for validation

    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Unsloth optimizations
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = True
    use_dora: bool = False

    # Training configuration
    num_epochs: int = 3
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # KL-divergence penalty
    kl_coef: float = 0.1

    # Checkpoint settings
    checkpoint_dir: str = "checkpoints/sft_reasoning"
    save_steps: int = 200
    keep_last_n_checkpoints: int = 3

    # WandB settings
    use_wandb: bool = True
    wandb_project: str = "SFT-Reasoning"
    wandb_run_name: str = "sft-reasoning-14b"

    # Reproducibility
    seed: int = 42


class SFTReasoningTrainer:
    """
    SFT Trainer with reasoning content support.

    Key features:
    - Unsloth for 2-5x faster training and 60% less memory
    - LoRA for memory-efficient fine-tuning
    - KL-divergence penalty to prevent overfitting
    - Supports reasoning_content in assistant messages
    """

    def __init__(self, config: SFTReasoningConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.global_step = 0

        # Checkpoint settings
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # WandB settings
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config) if hasattr(config, '__dict__') else {},
            )
            print(f"WandB initialized: {wandb.run.name}")

        # Load model with Unsloth
        print(f"Loading model with Unsloth: {config.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

        self.tokenizer.padding_side = "right"  # For SFT, use right padding

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

        # Note: For KL penalty, we use the base model (LoRA disabled) as reference
        # No need for a separate reference model - just disable LoRA adapters temporarily
        # This saves ~28GB VRAM for a 14B model!
        print("Using base model (LoRA disabled) as reference for KL penalty")

        # Optimizer
        import bitsandbytes as bnb
        trainable_params_list = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = bnb.optim.AdamW8bit(trainable_params_list, lr=config.learning_rate)

        # Hyperparameters
        self.kl_coef = config.kl_coef
        self.max_grad_norm = config.max_grad_norm
        self.batch_size = config.batch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

        # Load dataset
        self._load_dataset()

        print(f"SFT Reasoning Trainer initialized")
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
        Load dataset from JSONL files in data_reasoning folder.

        Splits each conversation by assistant messages to create training samples.
        Each sample contains the conversation up to and including one assistant turn.
        """
        dataset_path = Path(self.config.dataset_path)
        all_samples = []

        print("\n" + "=" * 60)
        print("📂 LOADING DATASET")
        print("=" * 60)
        print(f"Dataset path: {dataset_path}")

        # Find all JSONL files
        jsonl_files = list(dataset_path.rglob("*.jsonl"))
        print(f"Found {len(jsonl_files)} JSONL files")

        for jsonl_file in jsonl_files:
            file_samples = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        trajectory = json.loads(line)
                        conversation = trajectory.get('conversation', [])

                        # Split by assistant messages
                        for i in range(len(conversation)):
                            if conversation[i].get('role') == 'assistant':
                                # Create sample with conversation up to this assistant turn
                                sample = {
                                    'messages': conversation[:i+1],
                                    'source_file': str(jsonl_file.relative_to(dataset_path)),
                                }
                                file_samples.append(sample)

            print(f"  {jsonl_file.name}: {len(file_samples)} samples")
            all_samples.extend(file_samples)

        # Shuffle and split into train/val
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * self.config.train_split)
        self.train_dataset = all_samples[:split_idx]
        self.val_dataset = all_samples[split_idx:]

        print(f"\n✅ DATASET LOADED:")
        print(f"   • Total samples: {len(all_samples)}")
        print(f"   • Train samples: {len(self.train_dataset)}")
        print(f"   • Val samples: {len(self.val_dataset)}")
        print("=" * 60 + "\n")

    def _format_messages_for_training(self, messages: List[Dict]) -> List[Dict]:
        """
        Format messages for training, combining reasoning_content + content for assistant.

        For assistant messages with reasoning_content, format as:
        <think>{reasoning_content}</think>{content}
        """
        formatted = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                reasoning = msg.get('reasoning_content', '')
                content = msg.get('content', '')

                if reasoning:
                    # Combine reasoning and content with think tags
                    combined_content = f"<think>{reasoning}</think>{content}"
                else:
                    combined_content = content

                formatted.append({
                    'role': 'assistant',
                    'content': combined_content
                })
            else:
                formatted.append({
                    'role': msg.get('role'),
                    'content': msg.get('content', '')
                })
        return formatted

    def _tokenize_sample(self, sample: Dict) -> Optional[Dict]:
        """
        Tokenize a sample for training.

        Returns dict with:
        - input_ids: Full tokenized sequence
        - labels: Labels with -100 for non-assistant tokens
        - attention_mask: Attention mask
        """
        messages = sample['messages']
        formatted_messages = self._format_messages_for_training(messages)

        # Apply chat template
        full_text = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize full sequence
        tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
            add_special_tokens=False,
        )

        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]

        # Create labels: -100 for non-assistant tokens
        # We need to find where assistant content starts
        labels = input_ids.clone()

        # Tokenize without the last assistant message to find where it starts
        if len(formatted_messages) > 1:
            prefix_messages = formatted_messages[:-1]
            prefix_text = self.tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,  # Add the assistant prompt
            )
            prefix_tokenized = self.tokenizer(
                prefix_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
                add_special_tokens=False,
            )
            prefix_len = prefix_tokenized.input_ids.shape[1]
        else:
            # Only one message (system or first assistant)
            prefix_len = 0

        # Mask everything before assistant content
        labels[:prefix_len] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'prefix_len': prefix_len,
        }

    def compute_loss_with_kl(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SFT loss with KL-divergence penalty.

        Loss = CE_loss + kl_coef * KL(policy || ref)

        Reference model is the base model with LoRA adapters disabled.
        This reuses the same weights, saving ~28GB VRAM for 14B models.

        Args:
            input_ids: Input token ids (batch_size, seq_len)
            labels: Labels with -100 for masked positions (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Dict with 'loss', 'ce_loss', 'kl_loss'
        """
        # Policy model forward pass (with LoRA enabled)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        policy_logits = outputs.logits  # (batch, seq_len, vocab)

        # Reference model forward pass (LoRA disabled = base model)
        # This reuses the same base weights, no extra VRAM needed!
        with torch.no_grad():
            self.model.disable_adapter_layers()
            ref_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ref_logits = ref_outputs.logits
            self.model.enable_adapter_layers()

        # Compute CE loss (only on non-masked positions)
        # Shift for next token prediction
        shift_logits = policy_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten for cross entropy
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        ce_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Compute KL divergence on all positions (or just assistant positions)
        # KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log P - log Q))
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Create mask for valid positions (where labels != -100)
        valid_mask = (labels != -100).float()  # (batch, seq_len)

        # Compute KL divergence per token
        # KL = exp(log_p) * (log_p - log_q) = p * (log_p - log_q)
        policy_probs = F.softmax(policy_logits, dim=-1)
        kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)  # (batch, seq_len)

        # Apply mask and average
        kl_loss = (kl_per_token * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # Total loss
        total_loss = ce_loss + self.kl_coef * kl_loss

        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        self.model.eval()

        total_ce_loss = 0.0
        total_kl_loss = 0.0
        num_samples = 0

        for sample in self.val_dataset:
            tokenized = self._tokenize_sample(sample)
            if tokenized is None:
                continue

            input_ids = tokenized['input_ids'].unsqueeze(0).to(self.device)
            labels = tokenized['labels'].unsqueeze(0).to(self.device)
            attention_mask = tokenized['attention_mask'].unsqueeze(0).to(self.device)

            losses = self.compute_loss_with_kl(input_ids, labels, attention_mask)

            total_ce_loss += losses['ce_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            num_samples += 1

            # Limit validation samples for speed
            if num_samples >= 100:
                break

        self.model.train()

        if num_samples == 0:
            return {'val_ce_loss': 0.0, 'val_kl_loss': 0.0, 'val_loss': 0.0}

        avg_ce = total_ce_loss / num_samples
        avg_kl = total_kl_loss / num_samples

        return {
            'val_ce_loss': avg_ce,
            'val_kl_loss': avg_kl,
            'val_loss': avg_ce + self.kl_coef * avg_kl,
        }

    def train_step(self, batch_samples: List[Dict]) -> Dict[str, float]:
        """
        Execute one training step on a batch of samples.

        Args:
            batch_samples: List of samples to train on

        Returns:
            Dict of training metrics
        """
        FastLanguageModel.for_training(self.model)

        total_loss = 0.0
        total_ce_loss = 0.0
        total_kl_loss = 0.0
        num_valid = 0

        for sample in batch_samples:
            tokenized = self._tokenize_sample(sample)
            if tokenized is None:
                continue

            input_ids = tokenized['input_ids'].unsqueeze(0).to(self.device)
            labels = tokenized['labels'].unsqueeze(0).to(self.device)
            attention_mask = tokenized['attention_mask'].unsqueeze(0).to(self.device)

            losses = self.compute_loss_with_kl(input_ids, labels, attention_mask)

            # Scale loss for gradient accumulation
            scaled_loss = losses['loss'] / len(batch_samples)
            scaled_loss.backward()

            total_loss += losses['loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            num_valid += 1

        if num_valid == 0:
            return {'loss': 0.0, 'ce_loss': 0.0, 'kl_loss': 0.0}

        return {
            'loss': total_loss / num_valid,
            'ce_loss': total_ce_loss / num_valid,
            'kl_loss': total_kl_loss / num_valid,
        }

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
        if self.config.keep_last_n_checkpoints is not None:
            self._cleanup_old_checkpoints(prefix)

    def _cleanup_old_checkpoints(self, prefix: str = "step"):
        """Remove old checkpoints, keeping only the last N."""
        import shutil
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir()
             if d.is_dir() and d.name.startswith(prefix)],
            key=lambda x: int(x.name.split('_')[-1])
        )

        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            for old_checkpoint in checkpoints[:-self.config.keep_last_n_checkpoints]:
                shutil.rmtree(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")

    def save_model(self, output_dir: str, save_method: str = "lora"):
        """Save the model."""
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

    def finish(self):
        """Clean up and finish wandb run."""
        if self.use_wandb:
            wandb.finish()
            print("WandB run finished")

    def train(self):
        """Main training loop."""
        num_epochs = self.config.num_epochs
        num_samples = len(self.train_dataset)
        steps_per_epoch = num_samples // self.batch_size
        eval_steps = self.config.save_steps

        print("\n" + "=" * 60)
        print("Starting SFT Reasoning Training")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Dataset: {self.config.dataset_path}")
        print(f"Epochs: {num_epochs}")
        print(f"Train samples: {num_samples}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"KL coefficient: {self.kl_coef}")
        print("=" * 60 + "\n")

        self.optimizer.zero_grad()
        accum_step = 0

        for epoch in range(num_epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 60}")

            # Shuffle training data
            random.shuffle(self.train_dataset)

            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_kl_loss = 0.0
            num_batches = 0

            for batch_start in range(0, num_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_samples)
                batch_samples = self.train_dataset[batch_start:batch_end]

                metrics = self.train_step(batch_samples)
                accum_step += 1

                epoch_loss += metrics['loss']
                epoch_ce_loss += metrics['ce_loss']
                epoch_kl_loss += metrics['kl_loss']
                num_batches += 1

                # Gradient accumulation step
                if accum_step % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % 10 == 0:
                        print(f"  Step {self.global_step}: loss={metrics['loss']:.4f}, "
                              f"ce={metrics['ce_loss']:.4f}, kl={metrics['kl_loss']:.4f}")

                    if self.use_wandb:
                        wandb.log({
                            'train/loss': metrics['loss'],
                            'train/ce_loss': metrics['ce_loss'],
                            'train/kl_loss': metrics['kl_loss'],
                            'epoch': epoch + 1,
                        }, step=self.global_step)

                    # Validation and checkpoint
                    if self.global_step % eval_steps == 0:
                        val_metrics = self.validate()
                        print(f"\n  📊 Validation at step {self.global_step}:")
                        print(f"     val_loss={val_metrics['val_loss']:.4f}, "
                              f"val_ce={val_metrics['val_ce_loss']:.4f}, "
                              f"val_kl={val_metrics['val_kl_loss']:.4f}\n")

                        if self.use_wandb:
                            wandb.log(val_metrics, step=self.global_step)

                        self._save_checkpoint(self.global_step)

                self._clear_memory()

            # Epoch summary
            if num_batches > 0:
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Avg Loss: {epoch_loss / num_batches:.4f}")
                print(f"  Avg CE Loss: {epoch_ce_loss / num_batches:.4f}")
                print(f"  Avg KL Loss: {epoch_kl_loss / num_batches:.4f}")

            self._print_memory_stats()

        # Save final model
        print("\n" + "=" * 60)
        print("Training complete! Saving final model...")
        print("=" * 60)

        self.save_model(str(self.checkpoint_dir / "final_lora"), save_method="lora")
        self.finish()

        print("\nAll done!")


def main():
    """Main entry point for SFT Reasoning training."""
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="SFT Reasoning Trainer")
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    cli_args = parser.parse_args()

    # Create config
    config = SFTReasoningConfig()

    # Override with config file if provided
    if cli_args.config:
        yaml_config = load_config_from_yaml(cli_args.config)
        for key, value in yaml_config.items():
            setattr(config, key, value)

    # Set seed
    if config.seed is None:
        config.seed = int(datetime.now().timestamp())
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Initialize and run trainer
    trainer = SFTReasoningTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

