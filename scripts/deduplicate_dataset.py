import json
import os

def deleteDuplicates(lst):
    unique = []
    for item in lst:
        if item not in unique:
            unique.append(item)
    return unique

dataset_dir = "./datasets/train"
files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
for file in files:
    data = []  # noqa: B909
    if file.split(".")[-1] != "jsonl":  # noqa: B909
        continue
    filepath = os.path.join(dataset_dir, file)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    data = deleteDuplicates(data)
    # Write as JSONL
    with open(filepath, "w", encoding="utf-8") as out:
        for obj in data:  # noqa: B909
            out.write(json.dumps(obj) + "\n")
