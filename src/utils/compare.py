"""Before/after comparison: random policy vs. trained agent, as video and stats.

    python -m src.utils.compare --config config/ppo_carracing.yaml --stage before
    python -m src.utils.compare --config config/ppo_carracing.yaml --stage after

``before`` (no model needed) writes:
    results/<run>/metrics_before.json
    videos/<run>/step_0_random.mp4

``after`` (needs a trained model) writes:
    results/<run>/metrics_after.json
    results/<run>/comparison_table.md
    videos/<run>/step_final.mp4            (only when scoring the final model)
    videos/<run>/comparison_side_by_side.mp4
    docs/assets/<compare.gif_name>

Both policies drive the same tracks (same seeds), without reward shaping.

``after`` scores ``models/<run>/final_model.zip`` by default: the agent as
training left it. ``best_model.zip`` was picked by its score on the first few
of these same tracks, which would flatter it here; pass ``--model`` to score
it (or any checkpoint) anyway.
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

from src.envs.carracing_wrapper import DISCRETE_ACTIONS, agent_fps, make_vec_env
from src.evaluate import (
    SOLVED_RETURN,
    EpisodeResult,
    Policy,
    load_policy,
    run_episode,
    summarize,
    write_video,
)
from src.utils.config import REPO_ROOT, RunPaths, load_config
from src.utils.seeding import eval_seed

GIF_MAX_SECONDS = 12
GIF_FRAME_STEP = 2  # keep every Nth frame
GIF_DOWNSCALE = 2

OUTCOME_LABELS = {
    "lap": "lap completed",
    "out_of_bounds": "drove off the map",
    "time_limit": "ran out of time",
    "cut_off": "cut off",
}


def evaluate(cfg: dict[str, Any], policy_spec: dict[str, Any], n_episodes: int) -> list[EpisodeResult]:
    """Run ``n_episodes`` on the fixed evaluation tracks; frames are kept for episode 0 only."""
    vec_env = make_vec_env(
        cfg["env"],
        n_envs=1,
        seed=eval_seed(cfg["seed"]),
        shaped=False,
        render_mode="rgb_array",
        telemetry=True,
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
    discrete = not cfg["env"].get("kwargs", {}).get("continuous", True)
    metrics = {
        "policy": policy_name,
        **extra,
        "env_id": cfg["env"]["id"],
        "seeds": [eval_seed(cfg["seed"], ep) for ep in range(len(results))],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_steps_per_second": agent_fps(cfg["env"]),
        **summarize(
            results,
            fps=agent_fps(cfg["env"]),
            action_names=DISCRETE_ACTIONS if discrete else None,
        ),
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
    """The before/after report as Markdown: summary, action usage, and one row per track."""
    n = after["n_episodes"]
    fps = after["agent_steps_per_second"]

    def number(label: str, key: str, scale: float = 1.0) -> str:
        b, a = before.get(key), after.get(key)
        cells = ["n/a" if v is None else f"{v * scale:.1f}" for v in (b, a)]
        change = "n/a" if b is None or a is None else f"{(a - b) * scale:+.1f}"
        return f"| {label} | {cells[0]} | {cells[1]} | {change} |"

    def count(label: str, key: str) -> str:
        b, a = before[key], after[key]
        return f"| {label} | {b} / {n} | {a} / {n} | {a - b:+d} |"

    header = f"| Metric | Before ({before['policy']}) | After ({after['policy']}) | Change |"
    lines = [
        f"# {after['env_id']}: before vs. after training",
        "",
        f"{n} episodes per policy on the same tracks (seeds {after['seeds'][0]}-"
        f"{after['seeds'][-1]}), scored with the unmodified environment reward. "
        f"\"After\" is `{after['model_path']}`, acting deterministically.",
        "",
        "## Score",
        "",
        header,
        "|---|---|---|---|",
        number("Mean return", "mean_return"),
        number("Std of return", "std_return"),
        number("Median return", "median_return"),
        number("Min return", "min_return"),
        number("Max return", "max_return"),
        count(f"Episodes scoring {SOLVED_RETURN:.0f}+", "episodes_solved"),
        "",
        "## Driving",
        "",
        header,
        "|---|---|---|---|",
        count("Laps completed", "laps_completed"),
        count("Drove off the map", "out_of_bounds"),
        number("Mean track completion (%)", "mean_track_completion", 100),
        number("Lowest track completion (%)", "min_track_completion", 100),
        number("Mean lap time (s)", "mean_lap_time_s"),
        number("Fastest lap (s)", "best_lap_time_s"),
        number("Mean episode length (agent steps)", "mean_length"),
        number("Mean episode length (s)", "mean_duration_s"),
        number("Mean speed (simulator units)", "mean_speed"),
        number("Top speed (simulator units)", "max_speed"),
        number("Time with all wheels off the road (%)", "off_track_share", 100),
        "",
    ]

    if "action_shares" in before and "action_shares" in after:
        lines += [
            "## Action usage (% of steps)",
            "",
            f"| Action | Before ({before['policy']}) | After ({after['policy']}) |",
            "|---|---|---|",
            *(
                f"| {name} | {before['action_shares'][name] * 100:.1f} | {share * 100:.1f} |"
                for name, share in after["action_shares"].items()
            ),
            "",
        ]

    lines += [
        "## Per track",
        "",
        "| Seed | Track tiles | Before return | After return | Before completion (%) "
        "| After completion (%) | After result | After time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, seed in enumerate(after["seeds"]):
        lines.append(
            f"| {seed} | {after['track_tiles'][i]} "
            f"| {before['returns'][i]:.1f} | {after['returns'][i]:.1f} "
            f"| {before['tiles_visited'][i] / before['track_tiles'][i] * 100:.1f} "
            f"| {after['tiles_visited'][i] / after['track_tiles'][i] * 100:.1f} "
            f"| {OUTCOME_LABELS[after['outcomes'][i]]} "
            f"| {after['lengths'][i] / fps:.1f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- **Return** is the CarRacing score: +1000/N for each of the track's N tiles "
        "reached, -0.1 per simulator frame, -100 for driving off the map.",
        f"- **Episodes scoring {SOLVED_RETURN:.0f}+**: {SOLVED_RETURN:.0f} is the average "
        "score at which CarRacing counts as solved.",
        "- **Lap completed** means the car got back to the start having reached more than "
        "95% of the tiles, which ends the episode early. A random policy never gets that far, "
        "so its episodes run to the time limit.",
        "- **Track completion** is tiles reached / tiles on the track.",
        "- **Times** count from the moment the agent takes control, after the 1-second "
        f"opening zoom. One agent step is {1 / fps:.2f} s.",
        "- **Speed** is in the simulator's own units, the quantity CarRacing draws as its "
        "speed bar.",
        "- **All wheels off the road** is the share of steps on which no wheel touched a "
        "road tile.",
        "",
    ]
    return "\n".join(lines)


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
    try:
        shown_path = model_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        shown_path = str(model_path)
    after_metrics = save_metrics(
        paths.results / "metrics_after.json", cfg, policy_name, results, model_path=shown_path
    )
    print(f"after:  mean return {after_metrics['mean_return']:.1f} +/- {after_metrics['std_return']:.1f}")

    # step_final.mp4 means "the agent when training finished", so it is only
    # (re)written when that is the model being scored.
    if model_path.resolve() == (paths.models / "final_model.zip").resolve():
        write_video(results[0].frames, paths.videos / "step_final.mp4", fps)
    combined = side_by_side(before_frames, results[0].frames, f"After: {policy_name}")
    write_video(combined, paths.videos / "comparison_side_by_side.mp4", fps)
    write_gif(combined, REPO_ROOT / "docs" / "assets" / cfg["compare"]["gif_name"], fps)
    table_path = paths.results / "comparison_table.md"
    table_path.write_text(comparison_table(before_metrics, after_metrics), encoding="utf-8")
    print(f"Wrote {table_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate before/after videos and metrics.")
    parser.add_argument("--config", required=True, help="Path to a config/*.yaml file")
    parser.add_argument("--stage", choices=["before", "after"], required=True)
    parser.add_argument("--model", type=Path, help="Model .zip (default: models/<run>/final_model.zip)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = RunPaths.for_run(cfg["run_name"]).create()
    if args.stage == "before":
        run_before(cfg, paths)
    else:
        run_after(cfg, paths, args.model or paths.models / "final_model.zip")


if __name__ == "__main__":
    main()
