"""Training callbacks: periodic evaluation, video checkpoints, and raw-score logging."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import safe_mean

from src.evaluate import run_episode, write_video


class VideoEvalCallback(EvalCallback):
    """``EvalCallback`` that also records video checkpoints of the agent driving.

    Evaluation (scores, best-model saving) runs every ``eval_freq`` calls.
    Independently, every ``video_freq`` calls one episode is recorded to
    ``<video_dir>/step_<timesteps>.mp4``, and when training ends the final
    agent is recorded to ``<video_dir>/step_final.mp4``.

    Every video uses the same track (``video_seed``), so the checkpoints show
    progress on identical ground. The eval env is a single separate env built
    with ``render_mode="rgb_array"``; recording never touches the parallel
    training envs.
    """

    def __init__(
        self,
        eval_env,
        *,
        video_dir: str | Path,
        video_fps: float,
        video_freq: int,
        video_seed: int,
        **kwargs,
    ):
        super().__init__(eval_env, **kwargs)
        self.video_dir = Path(video_dir)
        self.video_fps = video_fps
        self.video_freq = video_freq
        self.video_seed = video_seed

    def _record(self, name: str) -> None:
        episode = run_episode(
            self.eval_env,
            self.model,
            seed=self.video_seed,
            deterministic=self.deterministic,
            capture=True,
        )
        path = write_video(episode.frames, self.video_dir / name, self.video_fps)
        if self.verbose >= 1:
            print(f"Saved video {path} (return {episode.episode_return:.1f})")

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        # The end of training gets step_final.mp4 instead of a numbered checkpoint.
        training_done = self.num_timesteps >= self.model._total_timesteps
        if self.video_freq > 0 and self.n_calls % self.video_freq == 0 and not training_done:
            self._record(f"step_{self.num_timesteps}.mp4")
        return continue_training

    def _on_training_end(self) -> None:
        self._record("step_final.mp4")


class RawReturnLogger(BaseCallback):
    """Log the unshaped training return as ``rollout/ep_raw_rew_mean``.

    ``rollout/ep_rew_mean`` is the shaped reward the agent optimises; this is
    the real CarRacing score over the same recent episodes.
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        raw_returns = [ep["raw_return"] for ep in self.model.ep_info_buffer if "raw_return" in ep]
        if raw_returns:
            self.logger.record("rollout/ep_raw_rew_mean", safe_mean(raw_returns))
