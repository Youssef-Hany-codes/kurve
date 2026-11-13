"""CarRacing-v3 environment construction: wrappers and vectorised env factory.

Wrapper stack for one env, innermost first:

    CarRacing-v3 -> [Telemetry] -> SkipZoom -> FrameSkip -> [RewardShaping] -> Monitor

and for the vectorised env:

    Subproc/DummyVecEnv -> VecFrameStack -> VecTransposeImage

``RewardShaping`` is applied to training envs only. Evaluation envs keep the
raw reward, so every reported metric is the true CarRacing score.
``Telemetry`` is opt-in and used by the before/after comparison; it only adds
entries to ``info`` and never changes observations or rewards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
)


# CarRacing-v3's discrete action ids, in order.
DISCRETE_ACTIONS = ("nothing", "steer right", "steer left", "gas", "brake")


class Telemetry(gym.Wrapper):
    """Report what the car is doing in ``info`` on every step.

    - ``speed``: the hull's speed, in the simulator's own units (the same
      number CarRacing draws as its speed bar).
    - ``wheels_on_track``: how many of the 4 wheels touch a road tile.
    - ``tiles_visited`` / ``track_tiles``: progress around this track.

    It sits directly on the base env so it reads the simulator state of the
    frame the step ended on.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        vx, vy = base.car.hull.linearVelocity
        info["speed"] = float(np.hypot(vx, vy))
        info["wheels_on_track"] = sum(1 for wheel in base.car.wheels if wheel.tiles)
        info["tiles_visited"] = int(base.tile_visited_count)
        info["track_tiles"] = len(base.track)
        return obs, reward, terminated, truncated, info


class SkipZoom(gym.Wrapper):
    """Play no-op actions through the opening camera zoom after every reset.

    CarRacing spends its first ~50 frames zooming in on the car. Those frames
    look nothing like the rest of the episode and no action matters during
    them, so the agent never sees them.
    """

    def __init__(self, env: gym.Env, frames: int = 50):
        super().__init__(env)
        self.frames = frames
        if isinstance(env.action_space, gym.spaces.Discrete):
            self._noop: Any = 0
        else:
            self._noop = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.frames):
            obs, _, _, _, info = self.env.step(self._noop)
        return obs, info


class FrameSkip(gym.Wrapper):
    """Repeat each action for ``skip`` simulator frames and sum the rewards."""

    def __init__(self, env: gym.Env, skip: int = 2):
        super().__init__(env)
        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class RewardShaping(gym.Wrapper):
    """Training-only reward shaping.

    - ``clip_max``: cap each step's reward. Tile bonuses arrive as rare spikes
      of ~3-7; capping them keeps the value targets well scaled.
    - ``patience``: truncate the episode after this many consecutive steps
      without a positive reward (the car is stuck or off the track), instead of
      collecting hundreds of useless frames until the time limit.

    The unshaped episode return is tracked and reported as
    ``info["raw_return"]`` on the final step, so training logs can show the
    true score next to the shaped one.
    """

    def __init__(
        self,
        env: gym.Env,
        clip_max: float | None = 1.0,
        patience: int | None = 100,
    ):
        super().__init__(env)
        self.clip_max = clip_max
        self.patience = patience
        self._raw_return = 0.0
        self._stalled_steps = 0

    def reset(self, **kwargs):
        self._raw_return = 0.0
        self._stalled_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        self._raw_return += reward
        self._stalled_steps = 0 if reward > 0 else self._stalled_steps + 1

        if (
            self.patience is not None
            and self._stalled_steps >= self.patience
            and not terminated
            and not truncated
        ):
            truncated = True
            info["early_stop"] = True

        if self.clip_max is not None:
            reward = min(reward, self.clip_max)
        if terminated or truncated:
            info["raw_return"] = self._raw_return
        return obs, reward, terminated, truncated, info


def make_env(
    env_cfg: dict[str, Any],
    *,
    shaped: bool,
    render_mode: str | None = None,
    monitor_file: str | Path | None = None,
    telemetry: bool = False,
) -> Callable[[], gym.Env]:
    """Return a factory for one wrapped env, built from the config's ``env`` section.

    ``shaped=True`` adds ``RewardShaping`` (training). ``monitor_file`` makes
    the ``Monitor`` write per-episode stats to ``<monitor_file>.monitor.csv``.
    ``telemetry=True`` adds ``Telemetry`` (speed, wheels on track, progress).
    """

    def _init() -> gym.Env:
        env = gym.make(env_cfg["id"], render_mode=render_mode, **env_cfg.get("kwargs", {}))
        if telemetry:
            env = Telemetry(env)
        if env_cfg.get("skip_zoom_frames", 0) > 0:
            env = SkipZoom(env, frames=env_cfg["skip_zoom_frames"])
        if env_cfg.get("frame_skip", 1) > 1:
            env = FrameSkip(env, skip=env_cfg["frame_skip"])
        info_keywords: tuple[str, ...] = ()
        if shaped:
            env = RewardShaping(env, **env_cfg.get("reward_shaping", {}))
            info_keywords = ("raw_return",)
        filename = str(monitor_file) if monitor_file is not None else None
        return Monitor(env, filename=filename, info_keywords=info_keywords)

    return _init


def make_vec_env(
    env_cfg: dict[str, Any],
    *,
    n_envs: int,
    seed: int,
    shaped: bool,
    render_mode: str | None = None,
    monitor_dir: str | Path | None = None,
    telemetry: bool = False,
) -> VecEnv:
    """Build the vectorised, frame-stacked, channel-first env the CNN policy consumes.

    Uses subprocesses when ``n_envs > 1`` and a single in-process env otherwise.
    Env ``i`` is seeded with ``seed + i``.
    """
    if monitor_dir is not None:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)
    env_fns = [
        make_env(
            env_cfg,
            shaped=shaped,
            render_mode=render_mode,
            monitor_file=Path(monitor_dir) / str(i) if monitor_dir is not None else None,
            telemetry=telemetry,
        )
        for i in range(n_envs)
    ]
    venv: VecEnv = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    venv.seed(seed)
    if env_cfg.get("frame_stack", 1) > 1:
        venv = VecFrameStack(venv, n_stack=env_cfg["frame_stack"])
    return VecTransposeImage(venv)


def agent_fps(env_cfg: dict[str, Any], simulator_fps: int = 50) -> float:
    """Frames per second of a video with one frame per agent step."""
    return simulator_fps / env_cfg.get("frame_skip", 1)
