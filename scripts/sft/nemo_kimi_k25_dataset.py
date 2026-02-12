# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Custom dataset loader for Kimi K2.5 fine-tuning with NeMo Automodel
# Loads JSONL with 'conversations' field in OpenAI-like format
#
# Key features:
# - Converts multi-turn conversations to text using chat template
# - Masks all turns EXCEPT the last assistant response for loss calculation
# - Supports reasoning_content in <think> blocks

import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from torch.utils.data import Dataset


class KimiK25SFTDataset(Dataset):
    """
    PyTorch Dataset for Kimi K2.5 SFT with proper loss masking.

    Only the LAST assistant response contributes to the loss.
    All previous turns (user messages, earlier assistant responses) are masked.
    """

    IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss ignores this index

    def __init__(
        self,
        path_or_dataset: str,
        tokenizer,
        max_length: int = 16000,
        num_samples_limit: Optional[int] = None,
        include_reasoning: bool = True,
    ):
        """
        Args:
            path_or_dataset: Path to JSONL file with conversations
            tokenizer: HuggingFace tokenizer with chat_template
            max_length: Maximum sequence length
            num_samples_limit: Optional limit on number of samples
            include_reasoning: Whether to include reasoning_content in <think> blocks
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_reasoning = include_reasoning

        # Load raw conversations from JSONL
        self.conversations = self._load_jsonl(path_or_dataset, num_samples_limit)
        print(f"Loaded {len(self.conversations)} conversations from {path_or_dataset}")

    def _load_jsonl(
        self,
        path: str,
        limit: Optional[int] = None
    ) -> List[List[Dict[str, str]]]:
        """Load conversations from JSONL file."""
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        conversations = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if limit is not None and line_num >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num + 1}: {e}")
                    continue

                raw_conv = item.get("conversations", [])
                if not raw_conv:
                    continue

                conversations.append(raw_conv)

        return conversations

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Process a single conversation and return tokenized inputs with labels.

        Labels are masked (-100) for all tokens EXCEPT the last assistant response.
        """
        conversation = self.conversations[idx]

        # Convert conversation to text and get token positions
        input_ids, attention_mask, labels = self._process_conversation(conversation)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _process_conversation(
        self,
        conversation: List[Dict[str, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert conversation to tokens with proper loss masking.

        Strategy:
        1. Tokenize full conversation (all turns)
        2. Tokenize conversation WITHOUT last assistant turn
        3. Mask everything up to where the last assistant starts
        """
        # Full conversation text
        assert conversation[-1]['role'] == 'assistant'
        assert conversation[-1]['reasoning_content']
        
        prompt_text = self.tokenizer.apply_chat_template(
            conversation[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        last_assistant_text = f'{conversation[-1].get("reasoning_content")}</think>{conversation[-1]["content"]}<|im_end|>'
        full_text = prompt_text + last_assistant_text

        # Tokenize full conversation
        full_tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = full_tokenized["input_ids"].squeeze(0)
        attention_mask = full_tokenized["attention_mask"].squeeze(0)
        
        # Tokenize the prefix
        prompt_tokenized = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # The last assistant response starts after the prefix
        assistant_start = prompt_tokenized["input_ids"].shape[1]

        # Create labels: mask everything except last assistant response
        labels = input_ids.clone()
        labels[:assistant_start] = self.IGNORE_INDEX

        # Also mask padding tokens
        labels[attention_mask == 0] = self.IGNORE_INDEX

        return input_ids, attention_mask, labels


def make_kimi_k25_sft_dataset(
    path_or_dataset: str,
    tokenizer_name_or_path: str = "moonshotai/Kimi-K2.5",
    max_length: int = 16000,
    split: str = "train",
    num_samples_limit: Optional[int] = None,
    include_reasoning: bool = True,
) -> KimiK25SFTDataset:
    """
    Factory function to create Kimi K2.5 SFT dataset.
    This function can be referenced in YAML config via _target_.

    Args:
        path_or_dataset: Path to JSONL file with conversations
        tokenizer_name_or_path: Tokenizer to use for chat template
        max_length: Maximum sequence length
        split: Dataset split (for compatibility)
        num_samples_limit: Optional sample limit
        include_reasoning: Whether to include reasoning_content in output

    Returns:
        KimiK25SFTDataset instance
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return KimiK25SFTDataset(
        path_or_dataset=path_or_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples_limit=num_samples_limit,
        include_reasoning=include_reasoning,
    )


# ============================================================================
# Legacy function for backward compatibility
# ============================================================================
def load_conversations_dataset(
    path_or_dataset: str,
    split: str = "train",
    num_samples_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Legacy function - loads raw conversations without tokenization.
    Use KimiK25SFTDataset for proper training with loss masking.
    """
    data_path = Path(path_or_dataset)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path_or_dataset}")

    conversations = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if num_samples_limit is not None and line_num >= num_samples_limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_conv = item.get("conversations", [])
            if raw_conv:
                conversations.append({"conversation": raw_conv})

    return conversations


if __name__ == "__main__":
    # Test the dataset loader
    import sys

    print("=" * 60)
    print("Testing KimiK25SFTDataset")
    print("=" * 60)

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/total.jsonl"

    # Test with a mock tokenizer first (without loading full model)
    try:
        from transformers import AutoTokenizer

        print(f"\nLoading tokenizer from moonshotai/Kimi-K2.5...")
        tokenizer = AutoTokenizer.from_pretrained(
            "moonshotai/Kimi-K2.5",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading dataset from {dataset_path}...")
        dataset = KimiK25SFTDataset(
            path_or_dataset=dataset_path,
            tokenizer=tokenizer,
            max_length=4096,  # Shorter for testing
            num_samples_limit=3,
        )

        print(f"\nDataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nFirst sample:")
            print(f"  input_ids shape: {sample['input_ids'].shape}")
            print(f"  attention_mask shape: {sample['attention_mask'].shape}")
            print(f"  labels shape: {sample['labels'].shape}")

            # Count masked vs unmasked tokens
            num_masked = (sample['labels'] == -100).sum().item()
            num_unmasked = (sample['labels'] != -100).sum().item()
            print(f"\n  Masked tokens (not in loss): {num_masked}")
            print(f"  Unmasked tokens (in loss): {num_unmasked}")
            print(f"  Loss computed on {num_unmasked / (num_masked + num_unmasked) * 100:.1f}% of tokens")

            # Show a snippet of the decoded text
            print(f"\n  First 500 chars of decoded input:")
            decoded = tokenizer.decode(sample['input_ids'][:500], skip_special_tokens=False)
            print(f"  {decoded[:500]}...")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

