import dotenv
import json
import os
import random
import asyncio
from multiprocessing import Pool
from pathlib import Path

from src.evaluation import evaluate_one_api

dotenv.load_dotenv()
# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME  = "accounts/fireworks/models/kimi-k2p5"
BASE_URL    = "https://api.fireworks.ai/inference/v1"
API_KEY     = os.getenv("FIREWORKS_API_KEY")

OUTPUT_FILE = "dataset/rollouts_600M.jsonl"
NUM_WORKERS = 20   # parallel processes

TASK_ID_MIN = 600_000_000
TASK_ID_MAX = 699_999_999
# ───────────────────────────────────────────────────────────────────────────────


def make_seed(task_id: int) -> int:
    """Random seed; must be even when task_id % 4 == 0."""
    seed = random.randint(0, 2**31 - 1)
    if task_id % 4 == 0 and seed % 2 != 0:
        seed += 1
    return seed


def worker(task_id: int):
    """Evaluate one task in a child process."""
    seed = make_seed(task_id)
    try:
        result = asyncio.run(evaluate_one_api(
            model=MODEL_NAME,
            base_url=BASE_URL,
            api_key=API_KEY,
            env="game",
            task_id=task_id,
            seed=seed,
        ))
        return {
            "task_id":    task_id,
            "seed":       seed,
            "score":      result["score"],
            "success":    result["success"],
            "time_taken": result["time_taken"],
            "rollout":    result["rollout"],
        }
    except Exception as e:
        print(f"[ERROR] task_id={task_id}: {type(e).__name__}: {e}", flush=True)
        return None


def load_done_ids(output_file: str) -> set:
    """Return the set of task_ids already written to the output file."""
    done = set()
    path = Path(output_file)
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["task_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def random_task_ids(done: set):
    """Yield random task_ids where id % 4 in {1, 2}, skipping already done."""
    while True:
        t = random.randint(TASK_ID_MIN, TASK_ID_MAX)
        if t % 4 in (0, 1, 2) and t not in done:
            done.add(t)
            yield t


def main():
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(OUTPUT_FILE)
    print(f"Already done: {len(done_ids):,}  |  workers: {NUM_WORKERS}")

    saved = 0
    with open(OUTPUT_FILE, "a", buffering=1) as out_f:
        with Pool(processes=NUM_WORKERS) as pool:
            for record in pool.imap_unordered(worker, random_task_ids(done_ids)):
                if record is None:
                    continue
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1
                print(
                    f"[{saved:>8,}] task_id={record['task_id']}  "
                    f"seed={record['seed']}  score={record['score']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
