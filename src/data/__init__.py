import json
from datasets import Dataset, load_dataset, concatenate_datasets
from src.data.logic import process_logic
from src.config import (
    AFFINE_DATASET_REPOSITORY,
    PRIME_DATASET_REPOSITORY,
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


def get_affine_dataset(path: str = AFFINE_DATASET_REPOSITORY):
    train_data = load_dataset(path, split="train")
    abd_data = train_data.map(
        handle_abd_data, remove_columns=train_data.column_names
    ).select(range(20000, 23301))

    ded_data = train_data.map(
        handle_ded_data, remove_columns=train_data.column_names
    ).select(range(20000, 23301))

    return concatenate_datasets([abd_data, ded_data])


def get_prime_dataset(path: str = PRIME_DATASET_REPOSITORY):
    mth_dataset = (
        load_dataset(path, "math", split="train").map(
            lambda x: {
                "question": MTH_PROMPT + "\n\n" + x["question"],
                "answer": x["answer"],
            }
        )
    ).select(range(1000))
    mth_dataset = convert_to_conversation(mth_dataset, "MTH")

    sci_dataset = (
        load_dataset(path, "science", split="train").map(
            lambda x: {
                "question": SCI_PROMPT + "\n\n" + x["question"],
                "answer": x["answer"],
            }
        )
    ).select(range(1000))
    sci_dataset = convert_to_conversation(sci_dataset, "SCI")

    lgc_dataset = (
        load_dataset(path, "logic", split="train")
        .map(process_logic)
        .select_columns(["question", "answer"])
    ).select(range(1000))
    lgc_dataset = convert_to_conversation(lgc_dataset, "LGC")

    return concatenate_datasets([mth_dataset, sci_dataset, lgc_dataset])


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

