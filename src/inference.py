"""Inference script for LLM-based reasoning on GSM8K."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import wandb
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from src.preprocess import load_gsm8k, extract_gold_answer, normalize_number


def get_llm_client(provider: str, model: str):
    """Initialize LLM client based on provider."""
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client, "openai"
    elif provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return client, "anthropic"
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def call_llm(
    client,
    provider: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
) -> str:
    """Call LLM API and return response text."""
    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response.choices[0].message.content
    elif provider == "anthropic":
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_final_answer(response: str, pattern: str) -> Optional[str]:
    """Extract final answer from model response using regex pattern."""
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def evaluate_answer(
    predicted: Optional[str], gold: str, normalize: bool = True
) -> bool:
    """Compare predicted answer with gold answer."""
    if predicted is None:
        return False

    if normalize:
        try:
            pred_norm = normalize_number(predicted)
            gold_norm = normalize_number(gold)
            return pred_norm == gold_norm
        except:
            pass

    # Fallback to string comparison
    return predicted.strip().lower() == gold.strip().lower()


def run_inference(cfg) -> Dict:
    """Run inference on dataset and compute metrics."""

    # Initialize WandB if enabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"WandB run: {wandb.run.url}")

    # Load dataset
    print(f"Loading dataset: {cfg.run.dataset.name}")
    samples = load_gsm8k(
        split=cfg.run.dataset.split,
        cache_dir=cfg.run.inference.cache_dir,
        subset_size=cfg.run.dataset.subset_size,
        subset_start=cfg.run.dataset.subset_start,
    )
    print(f"Loaded {len(samples)} samples")

    # Initialize LLM client
    print(f"Initializing {cfg.run.model.provider} client: {cfg.run.model.name}")
    client, provider = get_llm_client(cfg.run.model.provider, cfg.run.model.name)

    # Run inference
    results = []
    correct = 0
    total = 0
    total_response_length = 0

    for idx, sample in enumerate(tqdm(samples, desc="Running inference")):
        question = sample["question"]
        gold_answer = extract_gold_answer(sample["answer"])

        # Format prompt
        prompt = cfg.run.method.prompt_template.format(question=question)

        # Call LLM
        try:
            response = call_llm(
                client=client,
                provider=provider,
                model=cfg.run.model.name,
                prompt=prompt,
                temperature=cfg.run.model.temperature,
                max_tokens=cfg.run.model.max_tokens,
                top_p=cfg.run.model.top_p,
            )

            # Extract answer
            predicted_answer = extract_final_answer(
                response, cfg.run.evaluation.extract_answer_pattern
            )

            # Evaluate
            is_correct = evaluate_answer(
                predicted_answer,
                gold_answer,
                normalize=cfg.run.evaluation.normalize_numbers,
            )

            if is_correct:
                correct += 1
            total += 1

            total_response_length += len(response)

            # Store result
            result = {
                "idx": idx,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "response": response,
                "correct": is_correct,
                "response_length": len(response),
            }
            results.append(result)

            # Log to WandB
            if cfg.wandb.mode != "disabled":
                wandb.log(
                    {
                        "sample_idx": idx,
                        "correct": int(is_correct),
                        "response_length": len(response),
                    }
                )

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            results.append(
                {
                    "idx": idx,
                    "question": question,
                    "gold_answer": gold_answer,
                    "error": str(e),
                    "correct": False,
                }
            )

    # Compute final metrics
    accuracy = correct / total if total > 0 else 0.0
    avg_response_length = total_response_length / total if total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_response_length": avg_response_length,
    }

    print(f"\nFinal Metrics:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  Avg Response Length: {avg_response_length:.1f} chars")

    # Log summary to WandB
    if cfg.wandb.mode != "disabled":
        wandb.summary.update(metrics)
        wandb.finish()

    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_file}")

    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    # Perform validation based on mode
    if cfg.mode == "sanity":
        validate_sanity(metrics, results)
    elif cfg.mode == "pilot":
        validate_pilot(metrics, results)

    return metrics


def validate_sanity(metrics: Dict, results: List[Dict]) -> None:
    """Validate sanity mode execution."""
    passed = True
    reasons = []

    # Check: at least 5 samples processed
    if metrics["total"] < 5:
        passed = False
        reasons.append(f"insufficient_samples ({metrics['total']} < 5)")

    # Check: all outputs valid (not all errors)
    error_count = sum(1 for r in results if "error" in r)
    if error_count == len(results):
        passed = False
        reasons.append("all_outputs_failed")

    # Check: outputs are not all identical
    if len(results) > 1:
        responses = [r.get("response", "") for r in results if "response" in r]
        if len(set(responses)) == 1:
            passed = False
            reasons.append("identical_outputs")

    # Check: metrics are finite
    if not (0 <= metrics["accuracy"] <= 1):
        passed = False
        reasons.append("invalid_accuracy")

    # Print validation summary
    summary = {
        "samples": metrics["total"],
        "accuracy": metrics["accuracy"],
        "errors": error_count,
        "unique_responses": len(
            set(r.get("response", "") for r in results if "response" in r)
        ),
    }
    print(f"\nSANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")

    if passed:
        print("SANITY_VALIDATION: PASS")
    else:
        reason_str = ",".join(reasons)
        print(f"SANITY_VALIDATION: FAIL reason={reason_str}")


def validate_pilot(metrics: Dict, results: List[Dict]) -> None:
    """Validate pilot mode execution."""
    passed = True
    reasons = []

    # Check: at least 50 samples processed
    if metrics["total"] < 50:
        passed = False
        reasons.append(f"insufficient_samples ({metrics['total']} < 50)")

    # Check: primary metric computed and finite
    if not (0 <= metrics["accuracy"] <= 1):
        passed = False
        reasons.append("invalid_accuracy")

    # Check: outputs are non-trivial (not all identical)
    if len(results) > 1:
        responses = [r.get("response", "") for r in results if "response" in r]
        if len(set(responses)) == 1:
            passed = False
            reasons.append("identical_outputs")

    # Print validation summary
    summary = {
        "samples": metrics["total"],
        "primary_metric": "accuracy",
        "primary_metric_value": metrics["accuracy"],
        "outputs_unique": len(
            set(r.get("response", "") for r in results if "response" in r)
        ),
    }
    print(f"\nPILOT_VALIDATION_SUMMARY: {json.dumps(summary)}")

    if passed:
        print("PILOT_VALIDATION: PASS")
    else:
        reason_str = ",".join(reasons)
        print(f"PILOT_VALIDATION: FAIL reason={reason_str}")


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    args = parser.parse_args()

    # Load config
    with open(args.config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    # Run inference
    run_inference(cfg)


if __name__ == "__main__":
    main()
