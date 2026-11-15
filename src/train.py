"""Train an agent from a config file.

    python -m src.train --config config/ppo_carracing.yaml
    python -m src.train --config config/ppo_carracing.yaml --smoke   # ~1 minute pipeline check

Outputs, all under the config's ``run_name``:
    models/<run>/best_model.zip     best evaluation score so far
    models/<run>/final_model.zip    weights at the end of training
    models/<run>/checkpoints/       periodic snapshots
    logs/<run>/                     TensorBoard logs, Monitor CSVs, evaluations.npz
    videos/<run>/step_<n>.mp4       one recorded episode every eval.video_freq timesteps
    videos/<run>/step_final.mp4     the agent when training finished
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from typing import Any, Callable

from stable_baselines3.common.callbacks import CheckpointCallback

from src.envs.carracing_wrapper import agent_fps, make_vec_env
from src.utils.callbacks import RawReturnLogger, VideoEvalCallback
from src.utils.config import RunPaths, algo_class, load_config
from src.utils.seeding import eval_seed, set_global_seed

SMOKE_RUN_NAME = "_smoke"


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Learning rate that decays linearly from ``initial_value`` to 0 over training."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def algo_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """The config's algorithm section, with the learning-rate schedule resolved."""
    kwargs = dict(cfg[cfg["algo"]])
    learning_rate = float(kwargs["learning_rate"])
    schedule = kwargs.pop("lr_schedule", "constant")
    if schedule == "linear":
        kwargs["learning_rate"] = linear_schedule(learning_rate)
    elif schedule == "constant":
        kwargs["learning_rate"] = learning_rate
    else:
        raise ValueError(f"Unknown lr_schedule {schedule!r}; use 'linear' or 'constant'")
    return kwargs


def apply_smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Shrink a config to a tiny run that still exercises every code path."""
    cfg = {**cfg, "run_name": SMOKE_RUN_NAME, "total_timesteps": 1024}
    cfg["env"] = {**cfg["env"], "n_envs": 2}
    cfg[cfg["algo"]] = {**cfg[cfg["algo"]], "n_steps": 128, "batch_size": 64, "n_epochs": 2}
    cfg["eval"] = {**cfg["eval"], "freq": 512, "video_freq": 512, "n_episodes": 1}
    cfg["checkpoint"] = {**cfg["checkpoint"], "freq": 512}
    return cfg


def keep_system_awake() -> None:
    """Stop Windows from idle-sleeping while this process runs (a no-op elsewhere).

    A multi-hour run on a laptop would otherwise be suspended mid-training. The
    request ends with the process and changes no power settings.
    """
    if sys.platform == "win32":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def train(cfg: dict[str, Any]) -> None:
    keep_system_awake()
    paths = RunPaths.for_run(cfg["run_name"]).create()
    seed = cfg["seed"]
    n_envs = cfg["env"]["n_envs"]
    set_global_seed(seed)

    train_env = make_vec_env(
        cfg["env"], n_envs=n_envs, seed=seed, shaped=True, monitor_dir=paths.logs / "monitor"
    )
    eval_env = make_vec_env(
        cfg["env"], n_envs=1, seed=eval_seed(seed), shaped=False, render_mode="rgb_array"
    )

    model = algo_class(cfg["algo"])(
        cfg["policy"],
        train_env,
        seed=seed,
        device=cfg.get("device", "auto"),
        tensorboard_log=str(paths.logs / "tensorboard"),
        verbose=1,
        **algo_kwargs(cfg),
    )

    # Callback frequencies count calls, and each call advances n_envs timesteps.
    callbacks = [
        VideoEvalCallback(
            eval_env,
            video_dir=paths.videos,
            video_fps=agent_fps(cfg["env"]),
            video_freq=max(cfg["eval"]["video_freq"] // n_envs, 1),
            video_seed=eval_seed(seed),
            eval_freq=max(cfg["eval"]["freq"] // n_envs, 1),
            n_eval_episodes=cfg["eval"]["n_episodes"],
            deterministic=cfg["eval"]["deterministic"],
            best_model_save_path=str(paths.models),
            log_path=str(paths.logs),
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(cfg["checkpoint"]["freq"] // n_envs, 1),
            save_path=str(paths.models / "checkpoints"),
            name_prefix=cfg["run_name"],
        ),
        RawReturnLogger(),
    ]

    try:
        model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks, progress_bar=True)
        model.save(paths.models / "final_model")
        print(f"Saved final model to {paths.models / 'final_model.zip'}")
    finally:
        train_env.close()
        eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL racing agent.")
    parser.add_argument("--config", required=True, help="Path to a config/*.yaml file")
    parser.add_argument("--total-timesteps", type=int, help="Override the config's total_timesteps")
    parser.add_argument("--n-envs", type=int, help="Override the number of parallel training envs")
    parser.add_argument("--seed", type=int, help="Override the config's seed")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Tiny end-to-end run (1024 timesteps) written under the '{SMOKE_RUN_NAME}' run name",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        cfg["env"] = {**cfg["env"], "n_envs": args.n_envs}
    if args.seed is not None:
        cfg["seed"] = args.seed
    train(cfg)


# SubprocVecEnv spawns worker processes on Windows, which re-import this module.
if __name__ == "__main__":
    main()
