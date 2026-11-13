"""Seeding helpers so runs and before/after comparisons are repeatable."""

from __future__ import annotations

import random

import numpy as np
import torch

# Evaluation envs are seeded far away from the training envs (which use
# seed, seed + 1, ... one per parallel env) so no eval track is a training track.
EVAL_SEED_OFFSET = 10_000


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_seed(seed: int, episode: int = 0) -> int:
    """Seed for evaluation episode ``episode`` of a run seeded with ``seed``."""
    return seed + EVAL_SEED_OFFSET + episode
