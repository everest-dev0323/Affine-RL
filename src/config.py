import os
import requests

# ========== Environment names ==========
ENVIRONMENT_NAMES = ['affine:abd-v2', 'affine:ded-v2', 'ALFWORLD', 'WEBSHOP', 'SCIWORLD', 'TEXTCRAFT', 'MTH', 'CDE', 'SCI', 'LGC', 'SWE-PRO'] # SAT, BABYAI


# ========== Huggingface Dataset ==========
AFFINE_DATASET_REPOSITORY = "AffineFoundation/rl-python"
PRIME_DATASET_REPOSITORY = "PrimeIntellect/INTELLECT-3-RL"
LGC_DATASET_REPOSITORY = "AffineFoundation/affine-lgc"
AGENTGYM_DATASET_REPOSITORY = "JuniorJR124/Old_dataset"

TRAIN_DATA_PATH = "datasets/train/all"

# ========== Prompts ==========
ABD_PROMPT_TEMPLATE = """You are a programming expert. Given a Python program and its expected output, you need to determine the exact input that would produce this output.

Program:
```python
{program}
```

Expected Output:
```
{output}
```

Task: Analyze the program to understand what input format it expects from stdin, then provide the input data that would produce the expected output.

You can provide any explanations, analysis, or reasoning you want. However, you MUST include the input data within <INPUT> </INPUT> tags.

Format the input data like this:
<INPUT>
[input data here - each line on a separate line as the program expects]
</INPUT>

I will extract only the content between these tags.

Requirements for the input data within the tags:
1. Each line of input should be on a separate line
2. Use the exact format the program expects  
3. Provide the raw input values that should be fed to stdin
4. Do not include any prefixes or extra formatting within the INPUT tags

Please analyze the program and provide the required input:"""
DED_EXTRA_HINT = (
    "\n\n---\n"
    "⚠️ **Instructions** ⚠️\n"
    "Write a complete **Python 3** program that\n"
    "• reads *all* input from **STDIN** (using `input()` / `sys.stdin`),\n"
    "• writes *only* the required answer(s) to **STDOUT** using `print`,\n"
    "• contains no additional prompts or debug text, and\n"
    "• is returned as a single ```python … ``` fenced block.\n"
)
CDE_PROMPT = """Write a Python function to solve the following problem. 
    Put your complete solution in a markdown code block (```python ... ```).
    Make sure your function handles all test cases correctly."""
MTH_PROMPT = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."
SCI_PROMPT = "Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}."


# ========== Functions ==========
def get_sampling_range():
    """Get the sampling range from the environment"""
    response = requests.get(f"{os.getenv('AFFINE_ENDPOINT')}/config/environments")
    return response.json()['param_value']
