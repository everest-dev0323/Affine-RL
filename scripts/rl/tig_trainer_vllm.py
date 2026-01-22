"""
TiG Trainer using TRL's GRPOTrainer with vLLM colocate mode.

Single-GPU:
    python tig_trainer_vllm.py --config config/tig.yaml

Multi-GPU (with DeepSpeed ZeRO-3):
    accelerate launch --config_file config/accelerate_deepspeed_zero3.yaml \
        scripts/rl/tig_trainer_vllm.py --config config/tig.yaml
"""

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from src.evaluation import reward_response


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_dataset_from_path(dataset_path: str, max_per_task: int = 10000) -> Dataset:
    """Load dataset from JSONL files into HuggingFace Dataset format."""
    items = []
    path_obj = Path(dataset_path)

    for env_dir in path_obj.iterdir():
        if not env_dir.is_dir() or env_dir.name.startswith("."):
            continue

        for task_file in env_dir.glob("*.jsonl"):
            count = 0
            with open(task_file) as f:
                for line in f:
                    if not line.strip() or count >= max_per_task:
                        continue
                    traj = json.loads(line)
                    conv = traj.get("conversation", [])
                    for i, msg in enumerate(conv):
                        if msg.get("role") == "assistant":
                            items.append({
                                "prompt": conv[:i],
                                "ground_truth": msg.get("content", ""),
                                "env": env_dir.stem,
                                "task_id": traj.get("task_id", 0),
                            })
                            count += 1

    print(f"Loaded {len(items)} samples from {dataset_path}")
    return Dataset.from_list(items)


def make_reward_func(loop: asyncio.AbstractEventLoop):
    """Create reward function compatible with TRL's GRPOTrainer."""

    def reward_func(completions, env, task_id, ground_truth, **kwargs):
        async def compute():
            tasks = [
                reward_response(e, tid, c, gt)
                for c, e, tid, gt in zip(completions, env, task_id, ground_truth)
            ]
            return await asyncio.gather(*tasks)

        return list(loop.run_until_complete(compute()))

    return reward_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load model with Unsloth + vLLM fast inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg.get("max_seq_length", 8192),
        load_in_4bit=cfg.get("load_in_4bit", False),
        fast_inference=True,
        gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.5),
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    dataset = load_dataset_from_path(
        cfg["dataset_path"],
        cfg.get("max_data_per_task", 10000),
    )

    # Create async event loop for reward computation
    loop = asyncio.new_event_loop()

    # Training config with vLLM colocate mode (multi-GPU supported)
    training_args = GRPOConfig(
        output_dir=cfg.get("checkpoint_dir", "outputs/tig"),
        num_train_epochs=cfg.get("num_epochs", 1),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 1e-5),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        # GRPO params
        num_generations=cfg.get("num_generations", 4),
        max_completion_length=8192,
        temperature=cfg.get("temperature", 1.0),
        beta=cfg.get("kl_coef", 0.0),
        # vLLM colocate mode with multi-GPU support
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.5),
        vllm_tensor_parallel_size=cfg.get("vllm_tensor_parallel_size", 1),
        # Misc
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        report_to="wandb" if cfg.get("use_wandb", False) else "none",
        run_name=cfg.get("wandb_run_name", "tig-grpo"),
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=make_reward_func(loop),
    )

    trainer.train()
    trainer.save_model(cfg.get("checkpoint_dir", "outputs/tig") + "/final")
    loop.close()


if __name__ == "__main__":
    main()
