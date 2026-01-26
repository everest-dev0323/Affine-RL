import json
from datasets import Dataset, load_dataset, concatenate_datasets
from src.data.logic import process_logic
from src.config import (
    AFFINE_DATASET_REPOSITORY,
    PRIME_DATASET_REPOSITORY,
    LGC_DATASET_REPOSITORY,
    ABD_PROMPT_TEMPLATE,
    DED_EXTRA_HINT,
    MTH_PROMPT,
    SCI_PROMPT,
)


def handle_abd_data(example):
    prompt = ABD_PROMPT_TEMPLATE.format(
        program=example["program"], output=example["output"]
    )
    inputs = example["inputs"]
    response = f"<INPUT>{inputs}</INPUT>"
    return {
        "task_id": "abd" + str(example["sample_index"]),
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def handle_ded_data(example):
    prompt = example["prompt"].rstrip() + DED_EXTRA_HINT
    response = example["program"]

    return {
        "task_id": "ded" + str(example["sample_index"]),
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def convert_to_conversation(dataset, task_name):
    converted = []
    for i, data in enumerate(dataset):
        processed = {
            "task_id": task_name + "_" + str(i),
            "conversations": [
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["answer"]},
            ],
        }
        converted.append(processed)
    return Dataset.from_list(converted)


def get_abd_dataset(path: str = AFFINE_DATASET_REPOSITORY):
    train_data = load_dataset(path, split="train")
    abd_data = train_data.map(
        handle_abd_data, remove_columns=train_data.column_names
    )
    return abd_data

def get_ded_dataset(path: str = AFFINE_DATASET_REPOSITORY):
    train_data = load_dataset(path, split="train")
    ded_data = train_data.map(
        handle_ded_data, remove_columns=train_data.column_names
    )
    return ded_data

def get_mth_dataset(path: str = PRIME_DATASET_REPOSITORY):
    mth_dataset = (
        load_dataset(path, "math", split="train").map(
            lambda x: {
                "question": MTH_PROMPT + "\n\n" + x["question"],
                "answer": x["answer"],
            }
        )
    )
    mth_dataset = convert_to_conversation(mth_dataset, "MTH")
    return mth_dataset

def get_sci_dataset(path: str = PRIME_DATASET_REPOSITORY):
    sci_dataset = (
        load_dataset(path, "science", split="train").map(
            lambda x: {
                "question": SCI_PROMPT + "\n\n" + x["question"],
                "answer": x["answer"],
            }
        )
    )
    sci_dataset = convert_to_conversation(sci_dataset, "SCI")
    return sci_dataset

def get_lgc_dataset(path: str = LGC_DATASET_REPOSITORY):
    lgc_dataset = (
        load_dataset(path, split="train")
        .map(process_logic)
        .select_columns(["question", "answer"])
    )
    lgc_dataset = convert_to_conversation(lgc_dataset, "LGC")
    return lgc_dataset


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_sampling_list(data: list, latest_n: int = 200) -> dict:
    """
    Extract sampling list from merged.json data.

    Args:
        data: List of dictionaries from merged.json
        latest_n: Number of latest tasks to collect per env (default: 200)

    Returns:
        Dictionary structure:
        {
            'game': {
                'othello': [{'task_id': ..., 'seed': ...}, ...],
                'hex': [{'task_id': ..., 'seed': ...}, ...],
                ...
            },
            'lgc-v2': {
                'task_type': [{'task_id': ..., 'seed': ...}, ...],
                ...
            },
            'print': {
                'default': [{'task_id': ..., 'seed': ...}, ...],
            },
            ...
        }
    """
    from collections import defaultdict

    # First, group all items by env (preserving order for "latest" logic)
    env_items = defaultdict(list)
    for item in data:
        env = item.get('env')
        if env:
            env_items[env].append(item)

    result = {}

    for env, items in env_items.items():
        # Take only the latest N items per env
        latest_items = items[:latest_n]

        # Group by task_type within this env
        task_type_dict = defaultdict(list)

        for item in latest_items:
            extra = item.get('extra', {})
            request = extra.get('request', {})

            task_id = request.get('task_id')
            seed = request.get('seed')

            # Skip if no task_id or seed
            if task_id is None or seed is None:
                continue

            # Get task_type (print env doesn't have task_type)
            task_type = extra.get('task_type', 'default')

            task_type_dict[task_type].append({
                'task_id': task_id,
                'seed': seed
            })

        if task_type_dict:
            result[env] = dict(task_type_dict)

    return result

