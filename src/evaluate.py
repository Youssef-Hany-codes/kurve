"""Roll out a policy in the evaluation env.

Library half: ``run_episode`` / ``summarize`` / ``write_video`` are shared by
the training callbacks and ``src.utils.compare``.

CLI half: watch an agent drive in a live window.

    python -m src.evaluate --config config/ppo_carracing.yaml            # best saved model
    python -m src.evaluate --config config/ppo_carracing.yaml --random   # untrained baseline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from src.envs.carracing_wrapper import make_vec_env
from src.utils.config import RunPaths, algo_class, load_config
from src.utils.seeding import eval_seed


class Policy(Protocol):
    """Anything with a Stable-Baselines3 style ``predict``."""

    def predict(self, observation: np.ndarray, *, deterministic: bool = False) -> tuple[np.ndarray, Any]: ...


class RandomPolicy:
    """Uniformly random actions: the "before training" baseline."""

    def __init__(self, action_space, seed: int):
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, observation: np.ndarray, *, deterministic: bool = False):
        actions = [self.action_space.sample() for _ in range(len(observation))]
        return np.array(actions), None


@dataclass
class EpisodeResult:
    episode_return: float
    length: int  # agent steps
    frames: list[np.ndarray] = field(default_factory=list)
    # How the episode ended: "lap" (lap completed), "out_of_bounds" (drove off
    # the map), "time_limit", or "cut_off" (stopped by max_steps).
    outcome: str = "time_limit"
    actions: list[int] = field(default_factory=list)  # discrete action spaces only
    # Filled only when the env was built with telemetry=True:
    speeds: list[float] = field(default_factory=list)
    off_track_steps: int = 0  # steps that ended with no wheel on the road
    tiles_visited: int | None = None
    track_tiles: int | None = None

    @property
    def has_telemetry(self) -> bool:
        return self.track_tiles is not None

    @property
    def track_completion(self) -> float:
        """Share of the track's tiles the car reached, 0-1."""
        return self.tiles_visited / self.track_tiles


def run_episode(
    vec_env: VecEnv,
    policy: Policy,
    *,
    seed: int | None = None,
    deterministic: bool = True,
    capture: bool = False,
    max_steps: int | None = None,
) -> EpisodeResult:
    """Play one episode in a single-env ``vec_env`` and return its score.

    ``seed`` fixes the track. ``capture=True`` collects one rendered frame per
    agent step (the env must have been built with ``render_mode="rgb_array"``).
    """
    if vec_env.num_envs != 1:
        raise ValueError("run_episode needs a single-env VecEnv")
    if seed is not None:
        vec_env.seed(seed)
    obs = vec_env.reset()
    result = EpisodeResult(episode_return=0.0, length=0)
    discrete = isinstance(vec_env.action_space, gym.spaces.Discrete)
    while True:
        if capture:
            result.frames.append(vec_env.get_images()[0])
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)
        info = infos[0]  # on the last step this is still the finished episode's info
        result.episode_return += float(rewards[0])
        result.length += 1
        if discrete:
            result.actions.append(int(action[0]))
        if "track_tiles" in info:
            result.speeds.append(info["speed"])
            result.off_track_steps += info["wheels_on_track"] == 0
            result.tiles_visited = info["tiles_visited"]
            result.track_tiles = info["track_tiles"]
        if dones[0]:
            if "lap_finished" in info:
                result.outcome = "lap" if info["lap_finished"] else "out_of_bounds"
            return result
        if max_steps is not None and result.length >= max_steps:
            result.outcome = "cut_off"
            return result


SOLVED_RETURN = 900.0  # CarRacing's "solved" bar for the average score


def summarize(
    results: list[EpisodeResult],
    *,
    fps: float | None = None,
    action_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate episode results into the metrics stored in ``results/``.

    Score metrics are always present. ``fps`` (agent steps per second) adds
    durations in seconds, ``action_names`` adds how often each discrete action
    was used, and episodes run with telemetry add the driving metrics.
    """
    returns = np.array([r.episode_return for r in results], dtype=np.float64)
    lengths = np.array([r.length for r in results], dtype=np.float64)
    outcomes = [r.outcome for r in results]
    lap_lengths = [r.length for r in results if r.outcome == "lap"]
    metrics: dict[str, Any] = {
        "n_episodes": len(results),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "median_return": float(np.median(returns)),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "episodes_solved": int((returns >= SOLVED_RETURN).sum()),
        "mean_length": float(lengths.mean()),
        "laps_completed": outcomes.count("lap"),
        "out_of_bounds": outcomes.count("out_of_bounds"),
        "mean_lap_length": float(np.mean(lap_lengths)) if lap_lengths else None,
    }
    if fps is not None:
        metrics["mean_duration_s"] = metrics["mean_length"] / fps
        metrics["mean_lap_time_s"] = metrics["mean_lap_length"] / fps if lap_lengths else None
        metrics["best_lap_time_s"] = min(lap_lengths) / fps if lap_lengths else None

    telemetry = bool(results) and all(r.has_telemetry for r in results)
    if telemetry:
        completion = np.array([r.track_completion for r in results])
        speeds = np.concatenate([r.speeds for r in results])
        metrics.update(
            mean_track_completion=float(completion.mean()),
            min_track_completion=float(completion.min()),
            mean_speed=float(speeds.mean()),
            max_speed=float(speeds.max()),
            off_track_share=float(sum(r.off_track_steps for r in results) / lengths.sum()),
        )

    if action_names is not None and all(r.actions for r in results):
        counts = np.bincount(
            np.concatenate([r.actions for r in results]), minlength=len(action_names)
        )
        metrics["action_shares"] = {
            name: float(count / counts.sum()) for name, count in zip(action_names, counts)
        }

    metrics["returns"] = [float(r) for r in returns]
    metrics["lengths"] = [int(n) for n in lengths]
    metrics["outcomes"] = outcomes
    if telemetry:
        metrics["track_tiles"] = [r.track_tiles for r in results]
        metrics["tiles_visited"] = [r.tiles_visited for r in results]
        metrics["mean_speeds"] = [float(np.mean(r.speeds)) for r in results]
        metrics["off_track_shares"] = [r.off_track_steps / r.length for r in results]
    return metrics


def write_video(frames: list[np.ndarray], path: str | Path, fps: float) -> Path:
    """Encode RGB frames to an H.264 ``.mp4``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 keeps the native 600x400 frame instead of padding to a multiple of 16.
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path


def load_policy(cfg: dict[str, Any], vec_env: VecEnv, model_path: Path | None, random: bool) -> Policy:
    """The random baseline, or the trained model at ``model_path``."""
    if random:
        return RandomPolicy(vec_env.action_space, seed=cfg["seed"])
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Train one with "
            f"`python -m src.train --config <config>`, or pass --random to run the untrained baseline."
        )
    return algo_class(cfg["algo"]).load(model_path, device=cfg.get("device", "auto"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch an agent drive.")
    parser.add_argument("--config", required=True, help="Path to a config/*.yaml file")
    parser.add_argument("--model", type=Path, help="Model .zip (default: models/<run>/best_model.zip)")
    parser.add_argument("--random", action="store_true", help="Use random actions instead of a model")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-render", action="store_true", help="Run headless and only print scores")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of taking the most likely")
    parser.add_argument("--max-steps", type=int, help="Cut each episode off after this many agent steps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model or RunPaths.for_run(cfg["run_name"]).models / "best_model.zip"
    vec_env = make_vec_env(
        cfg["env"],
        n_envs=1,
        seed=eval_seed(cfg["seed"]),
        shaped=False,
        render_mode=None if args.no_render else "human",
    )
    try:
        policy = load_policy(cfg, vec_env, model_path, args.random)
        results = []
        for episode in range(args.episodes):
            result = run_episode(
                vec_env,
                policy,
                seed=eval_seed(cfg["seed"], episode),
                deterministic=not args.stochastic,
                max_steps=args.max_steps,
            )
            results.append(result)
            print(f"episode {episode + 1}: return {result.episode_return:8.1f}  steps {result.length}")
        stats = summarize(results)
        print(f"mean return {stats['mean_return']:.1f} +/- {stats['std_return']:.1f} over {len(results)} episodes")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
