"""Learning-curve plots from a run's training logs.

    python -m src.utils.plotting --config config/ppo_carracing.yaml

Writes ``results/<run>/learning_curve.png`` with two curves, both in raw
(unshaped) CarRacing reward:
    - training: rolling mean over finished training episodes (Monitor CSVs)
    - evaluation: mean +/- std of each periodic evaluation (evaluations.npz)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import load_results

from src.utils.config import RunPaths, load_config


def training_curve(monitor_dir: str | Path, window: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """(timesteps, rolling-mean raw return) across all training envs."""
    episodes = load_results(str(monitor_dir))  # sorted by wall-clock time
    timesteps = episodes["l"].cumsum().to_numpy()
    raw = episodes["raw_return"] if "raw_return" in episodes else episodes["r"]
    smoothed = pd.Series(raw).rolling(window, min_periods=1).mean().to_numpy()
    return timesteps, smoothed


def eval_curve(evaluations_npz: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(timesteps, mean return, std of return) for each periodic evaluation."""
    data = np.load(evaluations_npz)
    return data["timesteps"], data["results"].mean(axis=1), data["results"].std(axis=1)


def plot_learning_curve(logs_dir: str | Path, out_path: str | Path, title: str) -> Path:
    logs_dir, out_path = Path(logs_dir), Path(out_path)
    fig, ax = plt.subplots(figsize=(9, 5))

    steps, returns = training_curve(logs_dir / "monitor")
    ax.plot(steps, returns, label="Training (rolling mean)", color="tab:blue")

    evaluations = logs_dir / "evaluations.npz"
    if evaluations.exists():
        steps, mean, std = eval_curve(evaluations)
        ax.plot(steps, mean, label="Evaluation (mean ± std)", color="tab:orange", marker="o", ms=3)
        ax.fill_between(steps, mean - std, mean + std, color="tab:orange", alpha=0.2)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode return (raw reward)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a run's learning curve.")
    parser.add_argument("--config", required=True, help="Path to a config/*.yaml file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = RunPaths.for_run(cfg["run_name"])
    out = plot_learning_curve(
        paths.logs, paths.results / "learning_curve.png", f"{cfg['algo'].upper()} on {cfg['env']['id']}"
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
