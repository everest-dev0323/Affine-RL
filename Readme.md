<div align="center">

# Affine-RL

**Miner Training Toolkit for [Affine](https://affine.io) — Bittensor Subnet 120**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Bittensor SN120](https://img.shields.io/badge/Bittensor-SN120-green.svg)](https://taostats.io/subnets)
[![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen.svg)](tests/)

</div>

---

Affine-RL is the official miner training repository for **Affine (SN120)**, Bittensor's reinforcement-learning subnet. It provides end-to-end tooling for fine-tuning, evaluating, and benchmarking language models on the diverse reasoning tasks that Affine validators score — from program synthesis and mathematical reasoning to multi-turn agent tasks and strategic game-play.

> 💡 Bittensor founder Jacob Steeves has cited Affine as *"one of the largest subnets on Bittensor"* and praised it as an example of the most competitive incentive mechanisms on the entire network.

---

## Table of Contents

- [What is Bittensor?](#what-is-bittensor)
- [What is the Affine Subnet?](#what-is-the-affine-subnet)
- [Supported Environments & Tasks](#supported-environments--tasks)
- [RL Algorithms](#rl-algorithms)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [License](#license)

---

## What is Bittensor?

[Bittensor](https://bittensor.com) is an open, decentralized AI protocol that applies Bitcoin-style mining incentives to machine intelligence. The network:

- Issues the native **TAO** token to reward contributors based on their AI output quality.
- Runs ~128 independent **subnets**, each targeting a specific AI task (inference, training, RL, storage, etc.).
- Uses **validators** to score miners and set emission weights — no central authority controls rewards.

In short: *useful AI production → market rewards → network consensus*.

---

## What is the Affine Subnet?

**Affine (SN120)** is Bittensor's dedicated reinforcement-learning subnet. Its incentive mechanism works in three steps:

1. **Miners** fine-tune language models on Affine's multi-domain reasoning tasks and deploy them via [Chutes](https://chutes.ai).
2. **Validators** query miner endpoints, score responses with verifiable reward functions, and assign TAO emissions proportional to performance.
3. **Competition** is continuous — only models that improve against live benchmarks earn meaningful rewards.

The subnet's task portfolio spans logical reasoning, code generation, mathematical problem-solving, strategic game-play, and agentic web tasks, making it one of the most demanding and diverse RL training environments in the decentralized AI ecosystem.

---

## Supported Environments & Tasks

### Affine Core Tasks

| Task | ID | Description |
|------|----|-------------|
| Abductive Deduction | `affine:abd-v2` | Infer program inputs from observed outputs |
| Deductive Reasoning | `affine:ded-v2` | Write Python programs from problem specifications |
| Logic & Games | `LGC` | Sudoku, Game of 24, Boolean expression evaluation |
| Mathematics | `MTH` | Step-by-step math with `\boxed{}` answers |
| Coding | `CDE` | Function-level Python synthesis |
| Science | `SCI` | Multi-domain science question answering |
| Software Engineering | `SWE-PRO` | Repository-level bug fixing |

### Game Environments (via OpenSpiel)

Hex · Othello · Goofspiel · Liar's Dice · Clobber

### AgentGym Environments

WebShop · ALFWorld · BabyAI · SciWorld · TextCraft · WebArena · SQLGym · Maze · Wordle · Weather · Todo · Movie · Sheet · Academia

---

## RL Algorithms

| Algorithm | Script | Notes |
|-----------|--------|-------|
| **TiG** (Think in Games) | `scripts/rl/tig_trainer_vllm.py` | GRPO with vLLM colocate; length-penalty; KL regularization |
| **GRPO** | `scripts/rl/r1_grpo.py` | Multi-GPU DDP + Accelerate |
| **S-GRPO** (Sequential) | `scripts/rl/sqrpo_train.py` | PPO-style clip + KL on multi-turn trajectories |
| **DAPO** | `scripts/rl/dapo.py` | Dynamic advantage policy optimization |
| **OpenRLHF GRPO** | `scripts/rl/openrlhf/` | Ray + vLLM distributed; scales to 70 B+ models |

All trainers support:
- LoRA / QLoRA via [Unsloth](https://github.com/unslothai/unsloth) (4-bit and 8-bit quantization)
- WandB experiment tracking
- YAML-based configuration (`config/`)
- DeepSpeed ZeRO-3 and Accelerate DDP for multi-GPU training

---

## Quick Start

### Prerequisites

- Linux with CUDA-capable GPU(s)
- [Miniconda / Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)

### 1. Clone with Submodules

```bash
git clone https://github.com/Bambuuai/Affine-RL.git --recurse-submodules
cd Affine-RL
```

### 2. Create Conda Environment

```bash
conda create -n affine python=3.12
conda activate affine
```

### 3. Install Dependencies

```bash
pip install -e AffineFoundation/affinetes
pip install -r requirements.txt
pip install -r AffineFoundation/affinetes/environments/openspiel/requirements.txt
pip install -r AffineFoundation/affinetes/environments/arc_gen/requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.sample .env
# Edit .env — fill in AFFINE_ENDPOINT, CHUTES_API_KEY, and any other required keys
```

---

## Training

### Download Datasets

```bash
# Main RL dataset
hf download --repo-type dataset <YOUR HUGGINGFACE DATASET ID> --local-dir <LOCAL DATASET DIRECTORY>
```

To generate datasets locally:

```bash
python scripts/data/generate_mcts_dataset.py --game liars_dice --num_trajectories 1000 --workers 180
python scripts/data/generate_lgc_dataset.py --task_type all --num_trajectories 100000 --workers 180
```

### RL Training

#### Think in Games (TiG) — recommended for Affine tasks

```bash
# Single-node, multi-GPU (vLLM colocate mode)
accelerate launch --num_processes 8 \
    --config_file config/accelerate_ddp.yaml \
    scripts/rl/tig_trainer_vllm.py --config config/tig.yaml
```

#### GRPO (R1-style)

```bash
accelerate launch --num_processes 8 \
    --config_file config/accelerate_ddp.yaml \
    scripts/rl/r1_grpo.py --config config/r1_grpo.yaml
```

#### OpenRLHF — multi-node, large models (70 B+)

```bash
# Start Ray cluster
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "."}' \
    -- python scripts/rl/openrlhf/openrlhf_script.py --config config/tig.yaml
```

### Supervised Fine-Tuning (SFT)

```bash
# LoRA SFT (single GPU)
python scripts/sft/sft_trainer.py

# Multi-GPU SFT with Accelerate
accelerate launch --config_file config/accelerate_ddp.yaml scripts/sft/train_all.py

# Reasoning-aware SFT (chain-of-thought)
python scripts/sft/sft_reasoning_trainer.py --config config/sft_reasoning.yaml
```

---

## Evaluation

### Start vLLM Inference Server

```bash
source scripts/vllm_serve.sh
```

### Evaluate All Tasks

```bash
python scripts/evaluation/evaluate_all_vllm.py
```

### Programmatic API

```python
from src.evaluation import evaluate_one_vllm, evaluate_env_vllm

# Evaluate a single task instance
result = await evaluate_one_vllm(model, "LGC", task_id=10)

# Evaluate across an entire environment
results = await evaluate_env_vllm(model, "affine:ded-v2", task_ids=range(1, 50))
```

---

## Project Structure

```
Affine-RL/
├── config/                      # YAML training configs (tig, grpo, sft, deepspeed, …)
├── doc/                         # Extended documentation
│   ├── reinforcement-learning.md
│   ├── kimik25-sft.md
│   └── data.md
├── scripts/
│   ├── rl/                      # RL training scripts (TiG, GRPO, DAPO, S-GRPO, OpenRLHF)
│   ├── sft/                     # SFT / LoRA fine-tuning scripts
│   ├── evaluation/              # Evaluation runners (vLLM, local, AgentGym)
│   └── data/                    # Dataset generation utilities
├── src/
│   ├── evaluation/              # Evaluation package
│   │   ├── _constants.py        #   Environment registry & actor imports
│   │   ├── rewards.py           #   Response parsing & reward computation
│   │   ├── sudoku.py            #   Sudoku-specific reward logic
│   │   ├── vllm.py              #   vLLM-based evaluation
│   │   ├── api.py               #   OpenAI-compatible API evaluation
│   │   ├── hf.py                #   HuggingFace model evaluation
│   │   └── __init__.py          #   Re-exports all public symbols
│   ├── config.py                # Environment names, prompts, dataset paths
│   ├── utils.py                 # Shared utilities (load_config_from_yaml, …)
│   └── __init__.py              # get_base_task_name & lazy sub-package loader
├── tests/                       # Smoke test suite (43 tests, pytest)
├── AffineFoundation/            # Submodule: environment actors & task verifiers
├── AgentGym/                    # Submodule: AgentGym task environments
├── dataset/                     # Local dataset storage
└── requirements.txt             # Pinned Python dependencies
```

---

## Configuration

All training runs are driven by YAML files in `config/`. Key options for the default TiG config:

| Key | Default | Description |
|-----|---------|-------------|
| `model_name` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | Base model (HuggingFace hub ID) |
| `max_seq_length` | `16384` | Maximum token context window |
| `num_generations` | `8` | GRPO completions sampled per state |
| `clip_epsilon` | `0.2` | PPO surrogate clip range |
| `kl_coef` | `0.001` | KL divergence penalty weight |
| `length_penalty_weight` | `0.05` | Verbosity penalty (0 = disabled) |
| `vllm_gpu_memory_utilization` | `0.5` | vLLM VRAM share in colocate mode |
| `lora_r` | `16` | LoRA rank |

See [`config/tig.yaml`](config/tig.yaml) for the full annotated reference.

---

## Testing

Run the full smoke test suite with:

```bash
conda run -n affine python -m pytest tests/ -v
```

| Test File | Coverage |
|-----------|----------|
| `test_configs.py` | `TiGConfig`, `DapoConfig`, `SFTReasoningConfig` — field types, defaults, mutable isolation |
| `test_evaluation_rewards.py` | `parse_response` — think-block stripping, edge cases |
| `test_evaluation_sudoku.py` | `parse_sudoku_grid`, `extract_sudoku_answer`, `sudoku_reward` |
| `test_src_init.py` | `get_base_task_name` — prefix and version-suffix normalization |

**43 tests, 0 failures.**

---

## License

This project is licensed under the [MIT License](LICENSE).
