#!/usr/bin/env python3
"""
Universal evaluation script for affine environments using affinetes

Connects to locally deployed environment services via URL mode.

Usage:
    ./evaluate_local_model.py --env sat --model your-model --base-url http://host.docker.internal:30000/v1 --samples 10
    ./evaluate_local_model.py --env alfworld --task-id 2 --model deepseek-ai/DeepSeek-V3 --base-url https://llm.chutes.ai/v1 --samples 5
"""

import time
from dataclasses import asdict, dataclass, field
import asyncio
import argparse
import sys
import os
import json
from typing import Optional
from dotenv import load_dotenv
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import affinetes as af_env
from agentenv.envs import (
    AlfWorldTask,
    BabyAITask,
    SciworldTask,
    TextCraftTask,
    WebshopTask
)
from agentenv.trainer.distributed_evaluator import DistributedEvaluator
from agentenv.controller import Agent, ChatMLTemplate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

from src.config import ENVIRONMENT_NAMES
from src import get_base_task_name

# Environment name to port mapping (from .env)
ENVIRONMENT_PORTS = {
    'webshop': int(os.getenv('AGENTGYM_WEBSHOP_PORT', 36001)),
    'alfworld': int(os.getenv('AGENTGYM_ALFWORLD_PORT', 36002)),
    'sciworld': int(os.getenv('AGENTGYM_SCIWORLD_PORT', 36003)),
    'textcraft': int(os.getenv('AGENTGYM_TEXTCRAFT_PORT', 36004)),
    'babyai': int(os.getenv('AGENTGYM_BABYAI_PORT', 36005)),
}

@dataclass
class EvalArguments:
    model_path: str = field(default="Nagata99999/Affine-99999", metadata={"help": "Path of initial train model"})
    # required path
    output_file: str = field(default="result/agentgym_inference.jsonl", metadata={"help": "Output file path for JSON results (optional)"})
    inference_file: str = field(default="./datasets/evaluation/alfworld_test.json", metadata={"help": "Inference dataset."})
    task_name: str = field(default="alfworld", metadata={"help": "Task name for evaluation"})

    # eval config
    eval_batch_size: int = field(default=8, metadata={"help": "Batch size for evaluation."})
    num_workers: int = field(default=8, metadata={"help": "Number of subprocesses to use for data loading."})
    seed: int = field(default=42)
    do_sample: bool = field(default=False, metadata={"help": "Do sampling or not."})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})

    # conversation rounds
    max_round: int = field(
        default=30,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

    weight_decay: float = field(default=1e-6, metadata={"help": "Weight decay for regularization."})

    # environment parameters
    env_server_base: str = field(default="http://localhost:36002")
    data_len: int = field(default=200)
    timeout: int = field(default=2400)

async def main():
    """Main evaluation function"""
    parser = transformers.HfArgumentParser(EvalArguments)
    (args,) = parser.parse_args_into_dataclasses()
    
    model_name = os.getenv('MODEL_NAME')
    
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "babyai": BabyAITask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask
    }
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    

    task_class = task_classes.get(args.task_name.lower(), None)
    if task_class is None:
        raise ValueError(f"Unsupported task name: {args.task_name}")
    
    # set environment parameters
    env_args = {
        "env_server_base": args.env_server_base,
        "data_len": args.data_len,
        "timeout": args.timeout,
    }

    distributed_evaluator = DistributedEvaluator(
        Agent(
            model=model, tokenizer=tokenizer, chat_template=ChatMLTemplate()
        ),
        [task_class(client_args=env_args, n_clients=1)],
        args,
    )
    
    start_time = time.time()
    distributed_evaluator.generate()
    process_time = time.time() - start_time
    print(f"==== {process_time} seconds ====")
    

if __name__ == "__main__":
    asyncio.run(main())