"""Main orchestrator for CCR prompting experiment."""

import os
import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Orchestrate a single run_id inference experiment."""

    # Apply mode overrides
    if cfg.mode == "sanity":
        # Sanity mode: minimal execution
        cfg.dataset.subset_size = 10
        cfg.wandb.project = f"{cfg.wandb.project}-sanity"
    elif cfg.mode == "pilot":
        # Pilot mode: 20% of full dataset (at least 50 samples)
        original_size = cfg.dataset.subset_size
        cfg.dataset.subset_size = max(50, int(original_size * 0.2))
        cfg.wandb.project = f"{cfg.wandb.project}-pilot"

    print(f"Running experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Model: {cfg.run.model.name}")
    print(f"Dataset: {cfg.run.dataset.name} (size={cfg.run.dataset.subset_size})")
    print(f"Results dir: {cfg.results_dir}")
    print("-" * 80)

    # Create results directory
    results_path = Path(cfg.results_dir) / cfg.run.run_id
    results_path.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config_path = results_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Saved config to: {config_path}")

    # This is an inference-only task, so invoke inference.py
    print("\nInvoking inference.py...")

    # Convert config to command-line arguments for subprocess
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.inference",
        f"--config-path={config_path}",
    ]

    # Run inference as subprocess
    result = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        print(
            f"\nInference failed with return code {result.returncode}", file=sys.stderr
        )
        sys.exit(result.returncode)

    print(f"\nExperiment {cfg.run.run_id} completed successfully!")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
