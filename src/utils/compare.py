"""Before/after comparison: random policy vs. trained agent, as video and stats.

    python -m src.utils.compare --config config/ppo_carracing.yaml --stage before
    python -m src.utils.compare --config config/ppo_carracing.yaml --stage after

``before`` (no model needed) writes:
    results/<run>/metrics_before.json
    videos/<run>/step_0_random.mp4

``after`` (needs a trained model) writes:
    results/<run>/metrics_after.json
    results/<run>/comparison_table.md
    videos/<run>/step_final.mp4
    videos/<run>/comparison_side_by_side.mp4
    docs/assets/<compare.gif_name>

Both policies drive the same tracks (same seeds), without reward shaping.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.envs.carracing_wrapper import agent_fps, make_vec_env
from src.evaluate import EpisodeResult, Policy, load_policy, run_episode, summarize, write_video
from src.utils.config import REPO_ROOT, RunPaths, load_config
from src.utils.seeding import eval_seed

GIF_MAX_SECONDS = 12
GIF_FRAME_STEP = 2  # keep every Nth frame
GIF_DOWNSCALE = 2


def evaluate(cfg: dict[str, Any], policy_spec: dict[str, Any], n_episodes: int) -> list[EpisodeResult]:
    """Run ``n_episodes`` on the fixed evaluation tracks; frames are kept for episode 0 only."""
    vec_env = make_vec_env(
        cfg["env"], n_envs=1, seed=eval_seed(cfg["seed"]), shaped=False, render_mode="rgb_array"
    )
    try:
        policy: Policy = load_policy(cfg, vec_env, policy_spec.get("model_path"), policy_spec["random"])
        return [
            run_episode(vec_env, policy, seed=eval_seed(cfg["seed"], ep), capture=(ep == 0))
            for ep in range(n_episodes)
        ]
    finally:
        vec_env.close()


def save_metrics(
    path: Path, cfg: dict[str, Any], policy_name: str, results: list[EpisodeResult], **extra: Any
) -> dict[str, Any]:
    metrics = {
        "policy": policy_name,
        **extra,
        "env_id": cfg["env"]["id"],
        "seeds": [eval_seed(cfg["seed"], ep) for ep in range(len(results))],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **summarize(results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a caption in the top-left corner of an RGB frame."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    left, top, right, bottom = draw.textbbox((10, 8), text, font=font)
    draw.rectangle((left - 6, top - 4, right + 6, bottom + 4), fill=(0, 0, 0))
    draw.text((10, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def side_by_side(
    before: list[np.ndarray], after: list[np.ndarray], after_label: str
) -> list[np.ndarray]:
    """Pair up two episodes frame by frame; the shorter one holds its last frame."""
    n = max(len(before), len(after))
    frames = []
    for i in range(n):
        left = label_frame(before[min(i, len(before) - 1)], "Before: random")
        right = label_frame(after[min(i, len(after) - 1)], after_label)
        frames.append(np.hstack([left, right]))
    return frames


def write_gif(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    """Write a small looping GIF for the README."""
    gif_fps = fps / GIF_FRAME_STEP
    kept = frames[::GIF_FRAME_STEP][: int(GIF_MAX_SECONDS * gif_fps)]
    small = [f[::GIF_DOWNSCALE, ::GIF_DOWNSCALE] for f in kept]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, small, duration=1000 / gif_fps, loop=0)
    return path


def comparison_table(before: dict[str, Any], after: dict[str, Any]) -> str:
    def row(label: str, key: str, fmt: str = "{:.1f}") -> str:
        b, a = before[key], after[key]
        return f"| {label} | {fmt.format(b)} | {fmt.format(a)} | {a - b:+.1f} |"

    return "\n".join(
        [
            f"# {after['env_id']}: before vs. after training",
            "",
            f"Raw environment reward over {after['n_episodes']} episodes on the same tracks "
            f"(seeds {after['seeds'][0]}-{after['seeds'][-1]}).",
            "",
            f"| Metric | Before ({before['policy']}) | After ({after['policy']}) | Change |",
            "|---|---|---|---|",
            row("Mean return", "mean_return"),
            row("Std of return", "std_return"),
            row("Median return", "median_return"),
            row("Min return", "min_return"),
            row("Max return", "max_return"),
            row("Mean episode length (agent steps)", "mean_length"),
            "",
        ]
    )


def run_before(cfg: dict[str, Any], paths: RunPaths) -> tuple[dict[str, Any], list[np.ndarray]]:
    results = evaluate(cfg, {"random": True}, cfg["compare"]["n_episodes"])
    metrics = save_metrics(paths.results / "metrics_before.json", cfg, "random", results)
    write_video(results[0].frames, paths.videos / "step_0_random.mp4", agent_fps(cfg["env"]))
    print(f"before: mean return {metrics['mean_return']:.1f} +/- {metrics['std_return']:.1f}")
    return metrics, results[0].frames


def run_after(cfg: dict[str, Any], paths: RunPaths, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}; train first or pass --model.")
    fps = agent_fps(cfg["env"])
    # Re-running the random baseline is cheap and keeps both sides on identical tracks.
    before_metrics, before_frames = run_before(cfg, paths)

    results = evaluate(cfg, {"random": False, "model_path": model_path}, cfg["compare"]["n_episodes"])
    policy_name = f"trained {cfg['algo'].upper()}"
    after_metrics = save_metrics(
        paths.results / "metrics_after.json", cfg, policy_name, results, model_path=str(model_path)
    )
    print(f"after:  mean return {after_metrics['mean_return']:.1f} +/- {after_metrics['std_return']:.1f}")

    write_video(results[0].frames, paths.videos / "step_final.mp4", fps)
    combined = side_by_side(before_frames, results[0].frames, f"After: {policy_name}")
    write_video(combined, paths.videos / "comparison_side_by_side.mp4", fps)
    write_gif(combined, REPO_ROOT / "docs" / "assets" / cfg["compare"]["gif_name"], fps)
    (paths.results / "comparison_table.md").write_text(
        comparison_table(before_metrics, after_metrics), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate before/after videos and metrics.")
    parser.add_argument("--config", required=True, help="Path to a config/*.yaml file")
    parser.add_argument("--stage", choices=["before", "after"], required=True)
    parser.add_argument("--model", type=Path, help="Model .zip (default: models/<run>/best_model.zip)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = RunPaths.for_run(cfg["run_name"]).create()
    if args.stage == "before":
        run_before(cfg, paths)
    else:
        run_after(cfg, paths, args.model or paths.models / "best_model.zip")


if __name__ == "__main__":
    main()
