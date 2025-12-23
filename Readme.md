## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n affine python=3.11
conda activate affine
```

### 2. Install Affinetes
Navigate to the Affinetes directory and install it in editable mode:
```bash
cd AffineFoundation/affinetes
pip install -e .
```

### 3. Install AgentGym
Navigate to the AgentGym directory and install it in editable mode:
```bash
cd AgentGym
pip install -e .
```

### 4. Install additional libraries
```bash
pip install trl wandb peft vllm mini-swe-agent
```

### 5. Set up .env file
```bash
cp .env.sample .env
```

### 6. Pull Docker Images and Run Docker Containers
```bash
source scripts/dockerenv_setup.sh
```


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
```