"""Config loading and the per-run output folder layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

REPO_ROOT = Path(__file__).resolve().parents[2]

# Maps the config's ``algo`` key to its Stable-Baselines3 class. Stage 2 adds SAC.
ALGOS: dict[str, type[BaseAlgorithm]] = {"ppo": PPO}


def algo_class(name: str) -> type[BaseAlgorithm]:
    try:
        return ALGOS[name]
    except KeyError:
        raise ValueError(f"Unknown algo {name!r}; available: {sorted(ALGOS)}") from None


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config from ``config/``."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class RunPaths:
    """Where one run (e.g. ``ppo_carracing``) keeps its outputs."""

    models: Path
    logs: Path
    videos: Path
    results: Path

    @classmethod
    def for_run(cls, run_name: str, root: Path = REPO_ROOT) -> "RunPaths":
        return cls(
            models=root / "models" / run_name,
            logs=root / "logs" / run_name,
            videos=root / "videos" / run_name,
            results=root / "results" / run_name,
        )

    def create(self) -> "RunPaths":
        for path in (self.models, self.logs, self.videos, self.results):
            path.mkdir(parents=True, exist_ok=True)
        return self
