"""Evaluation script to aggregate metrics and create comparison plots."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Dict:
    """Fetch WandB run data by display name.

    Args:
        entity: WandB entity
        project: WandB project name
        run_id: Run display name (run_id)

    Returns:
        Dictionary with run data including config, summary, and history
    """
    api = wandb.Api()

    # Find run by display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        raise ValueError(f"No run found with display_name={run_id}")

    # Get most recent run with this name
    run = runs[0]

    # Extract data
    return {
        "config": dict(run.config),
        "summary": dict(run.summary),
        "history": run.history(),
        "url": run.url,
    }


def export_per_run_metrics(
    results_dir: Path,
    run_id: str,
    run_data: Dict,
) -> None:
    """Export per-run metrics and create figures.

    Args:
        results_dir: Base results directory
        run_id: Run identifier
        run_data: Fetched WandB run data
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Export metrics
    metrics = run_data["summary"]
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Exported metrics: {metrics_file}")

    # Create per-run figure: accuracy bar
    if "accuracy" in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([run_id], [metrics["accuracy"]], color="steelblue")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy for {run_id}")
        ax.set_ylim([0, 1])

        fig_path = run_dir / f"{run_id}_accuracy.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Exported figure: {fig_path}")

    # Create per-run figure: response length
    if "avg_response_length" in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([run_id], [metrics["avg_response_length"]], color="coral")
        ax.set_ylabel("Avg Response Length (chars)")
        ax.set_title(f"Response Length for {run_id}")

        fig_path = run_dir / f"{run_id}_response_length.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Exported figure: {fig_path}")


def create_comparison_plots(
    results_dir: Path,
    run_ids: List[str],
    all_run_data: Dict[str, Dict],
) -> None:
    """Create comparison plots across all runs.

    Args:
        results_dir: Base results directory
        run_ids: List of run identifiers
        all_run_data: Dictionary mapping run_id to run data
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics for all runs
    metrics_by_run = {}
    for run_id in run_ids:
        metrics_by_run[run_id] = all_run_data[run_id]["summary"]

    # Comparison: Accuracy
    if all("accuracy" in metrics for metrics in metrics_by_run.values()):
        fig, ax = plt.subplots(figsize=(8, 5))
        accuracies = [metrics_by_run[rid]["accuracy"] for rid in run_ids]
        colors = ["steelblue" if "proposed" in rid else "coral" for rid in run_ids]

        ax.bar(run_ids, accuracies, color=colors)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Comparison Across Methods")
        ax.set_ylim([0, 1])
        ax.tick_params(axis="x", rotation=45)

        fig_path = comparison_dir / "comparison_accuracy.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Exported comparison figure: {fig_path}")

    # Comparison: Response Length
    if all("avg_response_length" in metrics for metrics in metrics_by_run.values()):
        fig, ax = plt.subplots(figsize=(8, 5))
        lengths = [metrics_by_run[rid]["avg_response_length"] for rid in run_ids]
        colors = ["steelblue" if "proposed" in rid else "coral" for rid in run_ids]

        ax.bar(run_ids, lengths, color=colors)
        ax.set_ylabel("Avg Response Length (chars)")
        ax.set_title("Response Length Comparison Across Methods")
        ax.tick_params(axis="x", rotation=45)

        fig_path = comparison_dir / "comparison_response_length.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Exported comparison figure: {fig_path}")

    # Comparison: Efficiency scatter (accuracy vs response length)
    if all(
        "accuracy" in metrics and "avg_response_length" in metrics
        for metrics in metrics_by_run.values()
    ):
        fig, ax = plt.subplots(figsize=(8, 6))

        for run_id in run_ids:
            acc = metrics_by_run[run_id]["accuracy"]
            length = metrics_by_run[run_id]["avg_response_length"]
            color = "steelblue" if "proposed" in run_id else "coral"
            ax.scatter([length], [acc], s=150, color=color, label=run_id, alpha=0.7)

        ax.set_xlabel("Avg Response Length (chars)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Efficiency: Accuracy vs Response Length")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig_path = comparison_dir / "comparison_efficiency.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Exported comparison figure: {fig_path}")


def export_aggregated_metrics(
    results_dir: Path,
    run_ids: List[str],
    all_run_data: Dict[str, Dict],
) -> None:
    """Export aggregated metrics across all runs.

    Args:
        results_dir: Base results directory
        run_ids: List of run identifiers
        all_run_data: Dictionary mapping run_id to run data
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics
    metrics_by_run = {}
    for run_id in run_ids:
        metrics_by_run[run_id] = all_run_data[run_id]["summary"]

    # Identify proposed and baseline runs
    proposed_runs = [rid for rid in run_ids if "proposed" in rid]
    baseline_runs = [rid for rid in run_ids if "comparative" in rid]

    # Compute best values
    best_proposed = None
    best_baseline = None

    if proposed_runs:
        best_proposed_id = max(
            proposed_runs, key=lambda r: metrics_by_run[r].get("accuracy", 0)
        )
        best_proposed = {
            "run_id": best_proposed_id,
            "accuracy": metrics_by_run[best_proposed_id].get("accuracy", 0),
        }

    if baseline_runs:
        best_baseline_id = max(
            baseline_runs, key=lambda r: metrics_by_run[r].get("accuracy", 0)
        )
        best_baseline = {
            "run_id": best_baseline_id,
            "accuracy": metrics_by_run[best_baseline_id].get("accuracy", 0),
        }

    # Compute gap
    gap = None
    if best_proposed and best_baseline:
        gap = best_proposed["accuracy"] - best_baseline["accuracy"]

    # Create aggregated metrics
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }

    # Export
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Exported aggregated metrics: {agg_file}")


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument(
        "--run-ids", type=str, required=True, help="JSON list of run IDs"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY", "airas")
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "2026-0312-matsuzawa-2"),
    )
    args = parser.parse_args()

    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    results_dir = Path(args.results_dir)

    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {results_dir}")
    print(f"WandB: {args.wandb_entity}/{args.wandb_project}")
    print("-" * 80)

    # Fetch data for all runs
    all_run_data = {}
    for run_id in run_ids:
        print(f"\nFetching data for: {run_id}")
        try:
            run_data = fetch_wandb_run(args.wandb_entity, args.wandb_project, run_id)
            all_run_data[run_id] = run_data
            print(f"  URL: {run_data['url']}")

            # Export per-run metrics and figures
            export_per_run_metrics(results_dir, run_id, run_data)

        except Exception as e:
            print(f"  Error fetching {run_id}: {e}")
            continue

    if not all_run_data:
        print("\nNo run data fetched. Exiting.")
        return

    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(results_dir, run_ids, all_run_data)

    # Export aggregated metrics
    print("\nExporting aggregated metrics...")
    export_aggregated_metrics(results_dir, run_ids, all_run_data)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
