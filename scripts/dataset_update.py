"""
Dataset Update Script

This script keeps the training dataset up to date by:
1. Reading the last processed JSON file from a log file
2. Fetching new JSON files from the index URL
3. Processing new data using dataset_clear and dataset_prepare logic
4. Appending to existing training dataset files
5. Updating the log file with the new last processed file

Usage:
    python scripts/dataset_update.py
    python scripts/dataset_update.py --log_file ./datasets/update_log.json
"""

import json
import os
import requests
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

# Import functions from existing scripts
from dataset_clear import clear_data, ENVS
from dataset_prepare import (
    process_single_item,
    get_output_key,
)

# Default paths
DEFAULT_LOG_FILE = "./datasets/update_log.json"
DEFAULT_TRAIN_DIR = "./datasets/train"
INDEX_URL = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/affine/index.json"
BASE_URL = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/"


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


def process_json_file(file_url: str) -> List[Dict]:
    """Download and process a single JSON file."""
    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"  Error downloading: {e}")
        return []
    
    raw_content = response.content.decode('utf-8')
    try:
        data_list = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"  Error parsing JSON: {e}")
        return []
    
    processed_items = []
    for item in data_list:
        # Step 1: Clear data (from dataset_clear.py)
        item_str = json.dumps(item)
        cleared = clear_data(item_str, item)
        
        if cleared is None:
            continue
        
        # Step 2: Process single item (from dataset_prepare.py)
        prepared_items = process_single_item(cleared)
        processed_items.extend(prepared_items)
    
    return processed_items


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
        "env_counts": defaultdict(int),
        "errors": []
    }
    
    output_data: Dict[str, List[Dict]] = defaultdict(list)
    last_successful_file = last_processed

    for i, index_file in enumerate(new_files):
        print(f"[{i+1}/{len(new_files)}] Processing: {index_file}")
        file_url = BASE_URL + index_file

        processed_items = process_json_file(file_url)

        if processed_items:
            # Group by output key
            for item in processed_items:
                env = item.get("env")
                task_id = item.get("task_id")
                output_key = get_output_key(env, task_id)
                output_data[output_key].append(item)
                stats["env_counts"][env] += 1

            stats["new_items"] += len(processed_items)
            last_successful_file = index_file
            print(f"  -> {len(processed_items)} items processed")
        else:
            print(f"  -> 0 items (skipped or error)")

        stats["new_files"] += 1

    # Append to training files
    print(f"\nAppending to training files in {train_dir}...")
    for output_key, items in output_data.items():
        append_to_train_file(train_dir, output_key, items)
        print(f"  {output_key}.jsonl: +{len(items)} items")

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
    print(f"Items added: {stats['new_items']}")
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

