#!/usr/bin/env python3
"""
SFT Trainer for Kimi K2.5 Model (1T MoE with Native INT4 Quantization)

This script is designed for fine-tuning the Kimi K2.5 model which is:
- 1T total parameters (32B activated) MoE architecture
- 595GB model size with native INT4 quantization
- Uses MLA attention and MoonViT vision encoder
- 384 experts with 8 selected per token, plus 1 shared expert
- 61 layers (1 dense layer)
- 256K context length

========================================================================
RECOMMENDED APPROACH: KTransformers + LLaMA-Factory (for LIMITED VRAM)
========================================================================

For users with LIMITED GPU VRAM, we STRONGLY recommend using KTransformers kt-sft.
This approach allows fine-tuning with just 4× RTX 4090 GPUs (~96GB total VRAM).

Hardware Requirements for KTransformers:
- GPU: 4× RTX 4090 24GB (or equivalent with ~48GB+ total available VRAM)
- CPU: x86 CPU with AMX support (Intel Sapphire Rapids or newer)
- RAM: At least 2TB system memory
- Storage: ~600GB for model weights

KTransformers Setup:
1. Create conda environment:
   conda create -n kt-sft python=3.11
   conda activate kt-sft
   conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64
   conda install -y -c nvidia/label/cuda-11.8.0 cuda-runtime

2. Install LLaMA-Factory:
   git clone https://github.com/hiyouga/LlamaFactory.git
   cd LlamaFactory && pip install -e .

3. Install KTransformers (from releases):
   pip install ktransformers-<version>.whl
   pip install flash_attn-<version>.whl

4. Create training YAML (see scripts/sft/kimik25_kt_lora.yaml):
   model_name_or_path: moonshotai/Kimi-K2.5
   trust_remote_code: true
   stage: sft
   finetuning_type: lora
   lora_rank: 8
   lora_target: all
   bf16: true
   use_kt: true
   kt_optimize_rule: KimiK2.5-sft-amx-multi-gpu.yaml
   cpu_infer: 32
   chunk_size: 8192

5. Run training:
   USE_KT=1 llamafactory-cli train scripts/sft/kimik25_kt_lora.yaml

KTransformers Documentation:
- Tutorial: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Kimi-K2.5.md
- SFT Guide: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/SFT_Installation_Guide_KimiK2.5.md
- kt-sft README: https://github.com/kvcache-ai/ktransformers/tree/main/kt-sft

========================================================================
ALTERNATIVE: DeepSpeed ZeRO-3 (requires SUBSTANTIAL GPU resources)
========================================================================

This Python script below uses DeepSpeed ZeRO-3 approach for users with:
- 8× H100 80GB GPUs (or equivalent with ~500GB+ total GPU VRAM)
- Large system RAM (>500GB) for CPU offloading

Key considerations for this large model:
1. Use DeepSpeed ZeRO-3 with CPU offloading for memory efficiency
2. Gradient checkpointing is essential
3. Small batch sizes with gradient accumulation
4. LoRA/QLoRA for parameter-efficient fine-tuning
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to model configuration."""
    model_name_or_path: str = field(
        default="moonshotai/Kimi-K2.5",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Enable trust_remote_code for custom model architectures"}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2 for faster training"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model weights (bfloat16 recommended for A100/H100)"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to dataset configuration."""
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the dataset to use from HuggingFace Hub"}
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local dataset directory"}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length for training (reduce for memory)"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of processes for data preprocessing"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    use_lora: bool = field(
        default=True,
        metadata={"help": "Enable LoRA for parameter-efficient fine-tuning (highly recommended for 1T model)"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension (rank)"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha parameter for scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Target modules for LoRA. If None, will use default for Kimi K2.5"}
    )


@dataclass
class SFTArguments(TrainingArguments):
    """Extended training arguments for SFT."""
    output_dir: str = field(
        default="./outputs/kimi_k25_sft",
        metadata={"help": "Output directory for checkpoints and logs"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device (keep at 1 for large MoE model)"}
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate for AdamW optimizer"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio for learning rate scheduler"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory (essential for large models)"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 mixed precision training"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every N steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    deepspeed: Optional[str] = field(
        default="config/ds_kimi_k25_zero3.json",
        metadata={"help": "Path to DeepSpeed config file"}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "DDP find unused parameters (set False for efficiency)"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of dataloader workers"}
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "Report to wandb for experiment tracking"}
    )


# Default LoRA target modules for Kimi K2.5 MoE architecture
# Based on MLA attention and MoE expert layers
KIMI_K25_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_kimi_k25_model(model_args: ModelArguments):
    """
    Load Kimi K2.5 model with proper configuration for training.

    The model uses native INT4 quantization (compressed-tensors format).
    For training, we load it and apply LoRA adapters.
    """
    from transformers import AutoModel

    logger.info(f"Loading Kimi K2.5 model from {model_args.model_name_or_path}")
    logger.info("This is a 595GB model - loading may take significant time...")

    torch_dtype = get_torch_dtype(model_args.torch_dtype)

    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": torch_dtype,
        "device_map": "auto",  # Let accelerate/deepspeed handle device placement
    }

    # Enable Flash Attention 2 if available
    if model_args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")

    # Load the model - Kimi K2.5 is a multimodal model (image-text-to-text)
    # Use AutoModel with trust_remote_code for custom architecture
    try:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )
    except Exception as e:
        logger.warning(f"Failed to load with AutoModel: {e}")
        logger.info("Attempting to load with custom class...")
        # Fallback: try importing the model class directly
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
        model = AutoModel.from_config(config, trust_remote_code=True)

    logger.info(f"Model loaded. Type: {type(model)}")
    logger.info(f"Total parameters: {model.num_parameters():,}")

    return model


def load_tokenizer_and_processor(model_args: ModelArguments):
    """Load tokenizer and processor for Kimi K2.5."""
    logger.info("Loading tokenizer and processor...")

    # Load processor (handles both text and image inputs)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Also load tokenizer separately for text-only processing
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, processor


def apply_lora(model, lora_args: LoraArguments):
    """Apply LoRA adapters to the model for parameter-efficient fine-tuning."""
    if not lora_args.use_lora:
        logger.info("LoRA disabled, training full model (not recommended for 1T model)")
        return model

    logger.info("Applying LoRA for parameter-efficient fine-tuning")

    # Prepare model for k-bit training (handles gradient checkpointing for quantized models)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Configure LoRA
    target_modules = lora_args.lora_target_modules or KIMI_K25_LORA_TARGET_MODULES

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied. Trainable params: {trainable_params:,} / {all_params:,} "
        f"({100 * trainable_params / all_params:.2f}%)"
    )

    return model



def create_chat_format_function(tokenizer, max_seq_length: int):
    """Create a function to format data into chat format for Kimi K2.5."""

    def format_chat(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a single example into chat format.

        Expected input format (flexible):
        - {"messages": [{"role": "user/assistant/system", "content": "..."}]}
        - {"instruction": "...", "response": "..."}
        - {"prompt": "...", "completion": "..."}
        """
        messages = []

        # Handle different input formats
        if "messages" in example:
            messages = example["messages"]
        elif "instruction" in example and "response" in example:
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["response"]}
            ]
            if "system" in example:
                messages.insert(0, {"role": "system", "content": example["system"]})
        elif "prompt" in example and "completion" in example:
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]}
            ]
        else:
            raise ValueError(f"Unknown data format. Keys: {example.keys()}")

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )

        # Create labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    return format_chat


def load_and_prepare_dataset(data_args: DataArguments, tokenizer):
    """Load and prepare dataset for training."""
    logger.info("Loading dataset...")

    if data_args.dataset_name:
        dataset = load_dataset(data_args.dataset_name)
    elif data_args.dataset_path:
        dataset = load_dataset("json", data_files=data_args.dataset_path)
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")

    # Get train split
    if "train" in dataset:
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset

    logger.info(f"Dataset loaded. Number of examples: {len(train_dataset)}")

    # Create formatting function
    format_fn = create_chat_format_function(tokenizer, data_args.max_seq_length)

    # Process dataset
    train_dataset = train_dataset.map(
        format_fn,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="Formatting dataset",
    )

    logger.info(f"Dataset processed. Number of examples: {len(train_dataset)}")

    return train_dataset


class KimiK25DataCollator:
    """Custom data collator for Kimi K2.5 training."""

    def __init__(self, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack all features
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }
        return batch


def main():
    """Main training function."""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, SFTArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed for reproducibility
    set_seed(training_args.seed)

    logger.info("=" * 60)
    logger.info("Kimi K2.5 SFT Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info(f"LoRA enabled: {lora_args.use_lora}")
    logger.info(f"LoRA rank: {lora_args.lora_r}")
    logger.info("=" * 60)

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # Load tokenizer and processor
    tokenizer, _processor = load_tokenizer_and_processor(model_args)

    # Load model
    model = load_kimi_k25_model(model_args)

    # Apply LoRA
    model = apply_lora(model, lora_args)

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Load and prepare dataset
    train_dataset = load_and_prepare_dataset(data_args, tokenizer)

    # Create data collator
    data_collator = KimiK25DataCollator(tokenizer, data_args.max_seq_length)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

