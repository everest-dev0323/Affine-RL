"""
Preprocess OpenSpiel dataset for reasoning model fine-tuning.

Each record with N assistant messages is split into N sub-conversations.
In each sub-conversation, only the last assistant message becomes the completion
with <think></think> reasoning tags prepended.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def split_conversation(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a multi-turn conversation into sub-conversations.
    
    For a conversation with N assistant messages, generate N sub-conversations.
    Each sub-conversation ends at a different assistant message, and only
    the last assistant message gets <think></think> reasoning tags.
    """
    # Get the conversation - try common field names
    conversation = record.get("conversation") or record.get("conversations") or record.get("messages") or []
    
    if not conversation:
        return []
    
    sub_conversations = []
    assistant_indices = []
    
    # Find all assistant message indices
    for i, msg in enumerate(conversation):
        if msg.get("role") == "assistant":
            assistant_indices.append(i)
    
    if not assistant_indices:
        return []
    
    # Create sub-conversations ending at each assistant message
    for end_idx in assistant_indices:
        # Get messages up to and including this assistant message
        sub_conv = []
        for i in range(end_idx + 1):
            msg = conversation[i].copy()
            
            # Only modify the last message (which is an assistant message)
            if i == end_idx:
                original_content = msg.get("content", "")
                # Prepend empty thinking tags for reasoning model
                msg["content"] = f"<think>\n\n</think>\n\n{original_content}"
            
            sub_conv.append(msg)
        
        sub_conversations.append({
            "messages": sub_conv,
            "source": record.get("source", "openspiel"),
        })
    
    return sub_conversations


def preprocess_dataset(input_dir: str, output_path: str) -> None:
    """
    Preprocess all JSONL files in the input directory.
    """
    input_path = Path(input_dir)
    all_samples = []
    
    jsonl_files = list(input_path.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")
    
    for file_path in tqdm(jsonl_files, desc="Processing files"):
        records = load_jsonl(str(file_path))
        game_name = file_path.stem
        
        for record in tqdm(records, desc=f"Processing {game_name}", leave=False):
            record["source"] = game_name
            sub_convs = split_conversation(record)
            all_samples.extend(sub_convs)
    
    print(f"\nTotal samples after splitting: {len(all_samples)}")
    save_jsonl(all_samples, output_path)
    print(f"Saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess OpenSpiel dataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/openspiel",
        help="Input directory containing JSONL files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/openspiel_processed/train.jsonl",
        help="Output JSONL file path",
    )
    
    args = parser.parse_args()
    preprocess_dataset(args.input_dir, args.output_path)


if __name__ == "__main__":
    main()

