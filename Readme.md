## Environment Setup

```bash
git clone https://github.com/Bambuuai/Affine-H.git --recurse-submodules
```

### 1. Create Conda Environment
📖 **Reference Documentation**  
[Conda Installation Guide - Linux Terminal Installer](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)

```bash
conda create -n affine python=3.12
conda activate affine
```

### 2. Install Python Libraries
```bash
pip install -e AffineFoundation/affinetes
pip install -r requirements.txt
pip install -r AffineFoundation/affinetes/environments/openspiel/requirements.txt
pip install -r AffineFoundation/affinetes/environments/arc_gen/requirements.txt
```

<!-- ### 3. Install AgentGym
Navigate to the AgentGym directory and install it in editable mode:
```bash
cd AgentGym
pip install -e .
``` -->

### 3. Set up .env file
```bash
cp .env.sample .env
```

<!-- ### 6. Pull Docker Images and Run Docker Containers
```bash
source scripts/dockerenv_setup.sh
``` -->


## Evaluation

### 1. Evaluate with docker environment

```bash
source scripts/vllm_serve.sh
```

```bash
python scripts/evaluate_all_tasks.py
```

```python
from src.evaluation import evaluate_one, evaluate_env

# Evaluate single task on Affine environment
result = await evaluate_one(model, "sat", task_id=10)

# Evaluate single task on AgentGym environment
result = await evaluate_one(model, "alfworld", task_id=5)

# Evaluate multiple tasks
results = await evaluate_env(model, "alfworld", task_ids=[1, 2, 3, 4, 5])
```

## Training

### 1. Download dataset
```bash
hf download --repo-type dataset hongyu05/affine-dataset-1219 --local-dir datasets/train/all

hf download --repo-type dataset hongyu05/affine-rl --local-dir data

hf download --repo-type dataset hongyu05/affine-sft --local-dir data_sft
```

dataset generation
```bash
python scripts/data/generate_mcts_dataset.py --game liars_dice --num_trajectories 1000 --workers 180
python scripts/data/generate_lgc_dataset.py --task_type all --num_trajectories 100000 --workers 180
```
