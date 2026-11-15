"""Shared test fixtures."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest


class ScriptedEnv(gym.Env):
    """Tiny env that pays out a fixed list of rewards, one per step, then terminates.

    Records every action it receives so wrappers can be checked exactly.
    """

    observation_space = gym.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(5)

    def __init__(self, rewards: list[float]):
        self.rewards = list(rewards)
        self.actions: list[int] = []
        self.resets = 0
        self._t = 0

    def _obs(self) -> np.ndarray:
        return np.full(self.observation_space.shape, self._t % 256, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self.actions.append(int(action))
        reward = self.rewards[self._t]
        self._t += 1
        terminated = self._t >= len(self.rewards)
        return self._obs(), reward, terminated, False, {}


@pytest.fixture
def scripted_env():
    return ScriptedEnv


@pytest.fixture
def env_cfg():
    """The Stage 1 env settings, mirroring config/ppo_carracing.yaml."""
    return {
        "id": "CarRacing-v3",
        "n_envs": 2,
        "kwargs": {"continuous": False, "lap_complete_percent": 0.95, "domain_randomize": False},
        "skip_zoom_frames": 50,
        "frame_skip": 2,
        "frame_stack": 4,
        "reward_shaping": {"clip_max": 1.0, "patience": 100},
    }
