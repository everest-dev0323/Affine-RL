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
pip install vllm mini-swe-agent
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