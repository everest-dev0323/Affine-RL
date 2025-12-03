"""
Dataset Update Script (Standalone)

This script keeps the training dataset up to date by:
1. Reading the last processed JSON file from a log file
2. Fetching new JSON files from the index URL
3. Processing new data (clearing, reward normalization, round penalty)
4. Appending to existing training dataset files
5. Updating the log file with the new last processed file

Note: Conversations with reasoning_content are skipped.

Usage:
    python scripts/dataset_update.py
    python scripts/dataset_update.py --log_file ./datasets/update_log.json
    python scripts/dataset_update.py --init 006980400
"""

import json
import os
import re
import requests
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any

# Default paths
DEFAULT_LOG_FILE = "./datasets/update_log.json"
DEFAULT_TRAIN_DIR = "./datasets/train"
INDEX_URL = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/affine/index.json"
BASE_URL = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/"

# 8 environments
ENVS = ["abd", "ded", "sat", "alfworld", "sciworld", "babyai", "textcraft", "webshop"]


# ============================================================================
# Data Clearing Functions (from dataset_clear.py)
# ============================================================================

def detect_env(raw_str: str) -> Optional[str]:
    """Detect environment from raw string"""
    raw_lower = raw_str.lower()
    for env in ENVS:
        if f'"{env}"' in raw_lower or f":{env}" in raw_lower or f'"{env}:' in raw_lower:
            return env
    return None


def extract_field_str(raw_str: str, field_name: str) -> Optional[str]:
    """Extract a string field value using regex"""
    pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
    match = re.search(pattern, raw_str)
    if match:
        return match.group(1)
    return None


def extract_field_number(raw_str: str, field_name: str) -> Optional[float]:
    """Extract a numeric field value using regex"""
    pattern = rf'"{field_name}"\s*:\s*([0-9.]+)'
    match = re.search(pattern, raw_str)
    if match:
        try:
            val = match.group(1)
            return float(val) if '.' in val else int(val)
        except:
            return None
    return None


def extract_conversations_from_json(data: Dict) -> Optional[List[Dict]]:
    """Try to extract conversations from parsed JSON data"""
    conversations = None

    # Path 1: evaluation.extra.details[0].experiences[0].conversation
    try:
        if "evaluation" in data and "extra" in data["evaluation"]:
            extra = data["evaluation"]["extra"]
            if "details" in extra:
                details = extra["details"]
                if isinstance(details, list) and len(details) > 0:
                    detail_item = details[0]
                    if "experiences" in detail_item:
                        exp = detail_item["experiences"]
                        if isinstance(exp, list) and len(exp) > 0:
                            if "conversation" in exp[0]:
                                conversations = exp[0]["conversation"]
                elif isinstance(details, dict) and "experiences" in details:
                    exp = details["experiences"]
                    if isinstance(exp, list) and len(exp) > 0:
                        if "conversation" in exp[0]:
                            conversations = exp[0]["conversation"]
                    elif isinstance(exp, dict):
                        if "challenge" in exp and "llm_response" in exp:
                            conversations = [
                                {"role": "user", "content": exp["challenge"]},
                                {"role": "assistant", "content": exp["llm_response"]}
                            ]
    except:
        pass

    # Path 2: extra.conversation
    if not conversations:
        try:
            if "extra" in data and "conversation" in data["extra"]:
                conversations = data["extra"]["conversation"]
        except:
            pass

    # Path 3: Affine env format - challenge.prompt + response.response
    if not conversations:
        try:
            if "challenge" in data and "prompt" in data["challenge"]:
                prompt = data["challenge"]["prompt"]
                response = data.get("response", {}).get("response", "")
                if prompt and len(prompt) > 10:
                    conversations = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
        except:
            pass

    return conversations


def extract_score(raw_str: str, data: Dict) -> Optional[float]:
    """Extract score from raw string or parsed data"""
    try:
        if "evaluation" in data and "score" in data["evaluation"]:
            return data["evaluation"]["score"]
    except:
        pass
    try:
        if "score" in data:
            return data["score"]
    except:
        pass
    return extract_field_number(raw_str, "score")


def extract_task_id(raw_str: str, data: Dict) -> Optional[str]:
    """Extract task_id from raw string or parsed data"""
    try:
        if "evaluation" in data and "extra" in data["evaluation"]:
            if "task_id" in data["evaluation"]["extra"]:
                return data["evaluation"]["extra"]["task_id"]
    except:
        pass
    try:
        if "challenge" in data and "extra" in data["challenge"]:
            if "task_id" in data["challenge"]["extra"]:
                return data["challenge"]["extra"]["task_id"]
    except:
        pass
    try:
        if "challenge" in data and "challenge_id" in data["challenge"]:
            return data["challenge"]["challenge_id"]
    except:
        pass
    try:
        if "task_id" in data:
            return data["task_id"]
    except:
        pass
    task_id = extract_field_str(raw_str, "task_id")
    if not task_id:
        task_id = extract_field_str(raw_str, "challenge_id")
    return task_id


def extract_max_round(raw_str: str, data: Dict) -> Optional[int]:
    """Extract max_round from raw string or parsed data"""
    try:
        if "evaluation" in data and "extra" in data["evaluation"]:
            if "max_round" in data["evaluation"]["extra"]:
                return data["evaluation"]["extra"]["max_round"]
    except:
        pass
    try:
        if "max_round" in data:
            return data["max_round"]
    except:
        pass
    return extract_field_number(raw_str, "max_round")


def extract_reward(raw_str: str, data: Dict) -> Optional[float]:
    """Extract reward from raw string or parsed data"""
    try:
        if "evaluation" in data and "extra" in data["evaluation"]:
            extra = data["evaluation"]["extra"]
            if "details" in extra:
                details = extra["details"]
                if isinstance(details, list) and len(details) > 0:
                    if "reward" in details[0]:
                        return details[0]["reward"]
                    if "experiences" in details[0]:
                        exp = details[0]["experiences"]
                        if isinstance(exp, list) and len(exp) > 0 and "reward" in exp[0]:
                            return exp[0]["reward"]
    except:
        pass
    try:
        if "evaluation" in data and "reward" in data["evaluation"]:
            return data["evaluation"]["reward"]
    except:
        pass
    try:
        if "reward" in data:
            return data["reward"]
    except:
        pass
    return extract_field_number(raw_str, "reward")


def clear_data(raw_str: str, data: Dict) -> Optional[Dict]:
    """Parse rollout data using string operations and JSON fallback"""
    env = detect_env(raw_str)
    if not env:
        return None

    score = extract_score(raw_str, data)
    task_id = extract_task_id(raw_str, data)
    conversations = extract_conversations_from_json(data)

    if score is None or not conversations:
        return None

    result = {
        "env": env,
        "task_id": task_id,
        "score": score,
        "conversations": conversations
    }

    reward = extract_reward(raw_str, data)
    if reward is not None:
        result["reward"] = reward

    max_round = extract_max_round(raw_str, data)
    if max_round is not None:
        result["max_round"] = max_round

    try:
        if "miner" in data and "model" in data["miner"]:
            result["model"] = data["miner"]["model"]
    except:
        pass

    try:
        if "challenge" in data and "timestamp" in data["challenge"]:
            result["timestamp"] = data["challenge"]["timestamp"]
        elif "timestamp" in data:
            result["timestamp"] = data["timestamp"]
    except:
        pass

    return result


# ============================================================================
# Reward Functions (from dataset_prepare.py)
# ============================================================================

def normalize_sciworld_reward(reward: float) -> float:
    """Normalize sciworld reward from [-100, 100] to [0, 1]"""
    reward = max(-100, min(100, reward))
    return (reward + 100) / 200


def get_base_reward(data: Dict) -> float:
    """Get base reward from data, normalizing if necessary."""
    env = data.get("env", "")
    reward = data.get("score")
    if reward is None:
        reward = data.get("reward", 0)
    if reward is None:
        reward = 0

    if env == "sciworld":
        reward = normalize_sciworld_reward(reward)
    else:
        reward = max(0, min(1, float(reward)))

    return reward


def calculate_round_penalty(
    num_rounds: int,
    max_rounds: int = 10,
    penalty_weight: float = 0.05
) -> float:
    """Calculate penalty based on number of conversation rounds."""
    if num_rounds <= 1:
        return 0.0
    if num_rounds > max_rounds + 1:
        penalty = 1
    elif num_rounds < 6:
        penalty = 0
    else:
        penalty = penalty_weight * (num_rounds - 5)
    return penalty


def calculate_final_reward(
    base_reward: float,
    num_rounds: int,
    max_rounds: int = 10,
    round_penalty_weight: float = 0.05
) -> float:
    """Calculate final reward with round penalty."""
    penalty = calculate_round_penalty(num_rounds, max_rounds, round_penalty_weight)
    final_reward = base_reward - penalty
    return max(0.0, min(1.0, final_reward))


# ============================================================================
# Task ID Functions (from dataset_prepare.py)
# ============================================================================

def extract_task_number(task_id: Any) -> str:
    """Extract a numeric identifier from task_id."""
    if task_id is None:
        return "unknown"
    if isinstance(task_id, (int, float)):
        return str(int(task_id))
    task_str = str(task_id)
    match = re.match(r'^(\d+)', task_str)
    if match:
        return match.group(1)
    return str(abs(hash(task_str)) % 100000)


def get_output_key(env: str, task_id: Any) -> str:
    """Generate output file key from env and task_id."""
    task_number = extract_task_number(task_id)
    return f"train_{env}_{task_number}"


# ============================================================================
# Log Functions
# ============================================================================

def load_log(log_file: str) -> Dict:
    """Load the update log file."""
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_processed_file": None,
        "last_update_time": None,
        "total_processed_files": 0,
        "total_processed_items": 0,
        "history": []
    }


def save_log(log_file: str, log_data: Dict):
    """Save the update log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)


def get_new_files(index_list: List[str], last_processed: Optional[str]) -> List[str]:
    """Get list of new files to process after the last processed file.

    Supports both exact match and prefix match (e.g., '006980400' will match
    'affine/results/006980400-5Fn7rj78bfSrNcFQCHShC7aoVSneGLbiPD7xFHu3zhwFrQhs.json')
    """
    if last_processed is None:
        return index_list

    # Try exact match first
    if last_processed in index_list:
        last_idx = index_list.index(last_processed)
        return index_list[last_idx + 1:]

    # Try prefix match (for file numbers like '006980400')
    for i, file_path in enumerate(index_list):
        # Extract the file number from path like 'affine/results/006980400-xxx.json'
        filename = file_path.split('/')[-1]  # Get filename
        file_number = filename.split('-')[0]  # Get number before first dash

        if file_number == last_processed or last_processed in file_path:
            return index_list[i + 1:]

    # Last processed file not found in index
    print(f"Warning: Last processed file '{last_processed}' not found in index")
    return index_list


def append_to_train_file(train_dir: str, output_key: str, items: List[Dict]):
    """Append processed items to the training file."""
    output_path = Path(train_dir) / f"{output_key}.jsonl"
    with open(output_path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def has_reasoning_content(conversations: List[Dict]) -> bool:
    """Check if any assistant message has reasoning_content."""
    for conv in conversations:
        if conv.get("role") == "assistant" and conv.get("reasoning_content"):
            return True
    return False


def process_single_item_no_reasoning(
    data: Dict,
    max_rounds: int = 10,
    round_penalty_weight: float = 0.05
) -> Optional[Dict]:
    """
    Process a single rollout item, skipping if it has reasoning_content.

    Returns:
        Processed item dict, or None if skipped
    """
    env = data.get("env")
    task_id = data.get("task_id")
    conversations = data.get("conversations", [])

    if not env or not conversations:
        return None

    # Skip if any conversation has reasoning_content
    if has_reasoning_content(conversations):
        return None

    # Get base reward and calculate final reward with round penalty
    base_reward = get_base_reward(data)
    num_rounds = len([c for c in conversations if c.get("role") == "assistant"])
    final_reward = calculate_final_reward(
        base_reward, num_rounds, max_rounds, round_penalty_weight
    )

    return {
        "env": env,
        "task_id": task_id,
        "conversations": conversations,
        "reward": final_reward,
        "base_reward": base_reward,
        "num_rounds": num_rounds,
        "original_score": data.get("score"),
        "original_reward": data.get("reward"),
    }


def process_json_file(file_url: str) -> Dict:
    """Download and process a single JSON file.

    Returns:
        Dict with keys: items, total, success, failed, skipped_reasoning, skipped_clear
    """
    result = {
        "items": [],
        "total": 0,
        "success": 0,
        "failed": 0,
        "skipped_reasoning": 0,
        "skipped_clear": 0,
    }

    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"  Error downloading: {e}")
        result["failed"] = 1
        return result

    raw_content = response.content.decode('utf-8')
    try:
        data_list = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"  Error parsing JSON: {e}")
        result["failed"] = 1
        return result

    result["total"] = len(data_list)

    for item in data_list:
        # Step 1: Clear data (from dataset_clear.py)
        item_str = json.dumps(item)
        cleared = clear_data(item_str, item)

        if cleared is None:
            result["skipped_clear"] += 1
            continue

        # Step 2: Check for reasoning_content
        conversations = cleared.get("conversations", [])
        if has_reasoning_content(conversations):
            result["skipped_reasoning"] += 1
            continue

        # Step 3: Process item
        prepared = process_single_item_no_reasoning(cleared)
        if prepared:
            result["items"].append(prepared)
            result["success"] += 1
        else:
            result["failed"] += 1

    return result


def update_dataset(
    log_file: str = DEFAULT_LOG_FILE,
    train_dir: str = DEFAULT_TRAIN_DIR,
) -> Dict:
    """Main update function."""
    # Load log
    log_data = load_log(log_file)
    last_processed = log_data.get("last_processed_file")
    
    print("=" * 60)
    print("Dataset Update")
    print("=" * 60)
    print(f"Log file: {log_file}")
    print(f"Train directory: {train_dir}")
    print(f"Last processed file: {last_processed}")
    print()
    
    # Fetch index
    print("Fetching index...")
    try:
        response = requests.get(INDEX_URL, timeout=30)
        response.raise_for_status()
        index_list = json.loads(response.content.decode('utf-8'))
    except Exception as e:
        print(f"Error fetching index: {e}")
        return {"error": str(e)}
    
    print(f"Total files in index: {len(index_list)}")
    
    # Get new files to process
    new_files = get_new_files(index_list, last_processed)
    print(f"New files to process: {len(new_files)}")
    
    if not new_files:
        print("No new files to process.")
        return {"new_files": 0, "new_items": 0}
    
    # Create train directory if needed
    os.makedirs(train_dir, exist_ok=True)
    
    # Process new files
    stats = {
        "new_files": 0,
        "new_items": 0,
        "total_records": 0,
        "total_success": 0,
        "total_failed": 0,
        "total_skipped_reasoning": 0,
        "total_skipped_clear": 0,
        "env_counts": defaultdict(int),
    }

    for i, index_file in enumerate(new_files[::-1]):
        print(f"[{i+1}/{len(new_files)}] Processing: {index_file}")
        file_url = BASE_URL + index_file

        file_result = process_json_file(file_url)

        # Update stats
        stats["total_records"] += file_result["total"]
        stats["total_success"] += file_result["success"]
        stats["total_failed"] += file_result["failed"]
        stats["total_skipped_reasoning"] += file_result["skipped_reasoning"]
        stats["total_skipped_clear"] += file_result["skipped_clear"]

        processed_items = file_result["items"]
        if processed_items:
            # Group by output key and write immediately
            file_output: Dict[str, List[Dict]] = defaultdict(list)
            for item in processed_items:
                env = item.get("env")
                task_id = item.get("task_id")
                output_key = get_output_key(env, task_id)
                file_output[output_key].append(item)
                stats["env_counts"][env] += 1

            # Write this file's items immediately to disk
            for output_key, items in file_output.items():
                append_to_train_file(train_dir, output_key, items)

            stats["new_items"] += len(processed_items)

        # Detailed per-file log
        print(f"  -> total: {file_result['total']}, success: {file_result['success']}, "
              f"failed: {file_result['failed']}, skipped_reasoning: {file_result['skipped_reasoning']}, "
              f"skipped_clear: {file_result['skipped_clear']}")

        stats["new_files"] += 1

    last_successful_file = new_files[-1]

    # Update log
    log_data["last_processed_file"] = last_successful_file
    log_data["last_update_time"] = datetime.now().isoformat()
    log_data["total_processed_files"] += stats["new_files"]
    log_data["total_processed_items"] += stats["new_items"]
    log_data["history"].append({
        "timestamp": datetime.now().isoformat(),
        "files_processed": stats["new_files"],
        "items_added": stats["new_items"],
        "last_file": last_successful_file
    })
    save_log(log_file, log_data)

    # Print summary
    print("\n" + "=" * 60)
    print("Update Summary")
    print("=" * 60)
    print(f"Files processed: {stats['new_files']}")
    print(f"Total records: {stats['total_records']}")
    print(f"Total success: {stats['total_success']}")
    print(f"Total failed: {stats['total_failed']}")
    print(f"Total skipped (reasoning): {stats['total_skipped_reasoning']}")
    print(f"Total skipped (clear): {stats['total_skipped_clear']}")
    print(f"Last processed file: {last_successful_file}")
    print("\nItems per environment:")
    for env, count in sorted(stats["env_counts"].items()):
        print(f"  {env}: {count}")

    return stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Update training dataset with new data")
    parser.add_argument(
        "--log_file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="Path to the update log file"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=DEFAULT_TRAIN_DIR,
        help="Directory containing training data"
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="Initialize log with last processed file (e.g., '006980400')"
    )

    args = parser.parse_args()

    # Handle initialization
    if args.init:
        log_data = load_log(args.log_file)
        log_data["last_processed_file"] = args.init
        log_data["last_update_time"] = datetime.now().isoformat()
        save_log(args.log_file, log_data)
        print(f"Initialized log with last_processed_file: {args.init}")
        print(f"Log saved to: {args.log_file}")
        return

    update_dataset(
        log_file=args.log_file,
        train_dir=args.train_dir,
    )


if __name__ == "__main__":
    main()

