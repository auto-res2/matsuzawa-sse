"""Dataset preprocessing utilities for GSM8K."""

from typing import Dict, List, Optional
from datasets import load_dataset
from pathlib import Path


def load_gsm8k(
    split: str = "test",
    cache_dir: Optional[str] = None,
    subset_size: Optional[int] = None,
    subset_start: int = 0,
) -> List[Dict]:
    """Load GSM8K dataset.

    Args:
        split: Dataset split ('train' or 'test')
        cache_dir: Directory to cache dataset
        subset_size: Number of samples to load (None = all)
        subset_start: Starting index for subset

    Returns:
        List of dataset samples with 'question' and 'answer' fields
    """
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Convert to list of dicts
    samples = []
    for item in dataset:
        samples.append(
            {
                "question": item["question"],
                "answer": item["answer"],
            }
        )

    # Apply subsetting if specified
    if subset_size is not None:
        end_idx = min(subset_start + subset_size, len(samples))
        samples = samples[subset_start:end_idx]

    return samples


def extract_gold_answer(answer_text: str) -> str:
    """Extract the numerical answer from GSM8K answer field.

    GSM8K answers are in format: "reasoning steps\n#### numerical_answer"

    Args:
        answer_text: Raw answer text from dataset

    Returns:
        Extracted numerical answer as string
    """
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


def normalize_number(num_str: str) -> float:
    """Normalize a number string to float for comparison.

    Args:
        num_str: Number as string (may contain commas, spaces, etc.)

    Returns:
        Normalized float value
    """
    # Remove common formatting
    num_str = num_str.replace(",", "").replace(" ", "").strip()

    # Handle dollar signs and other currency symbols
    num_str = num_str.replace("$", "").replace("€", "").replace("£", "")

    try:
        return float(num_str)
    except ValueError:
        # If conversion fails, return as-is for string comparison
        return num_str
