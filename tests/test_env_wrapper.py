"""Env wrappers and the vectorised env factory, against stub envs and the real CarRacing-v3."""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from src.envs.carracing_wrapper import FrameSkip, SkipZoom, agent_fps, make_env, make_vec_env
from src.evaluate import EpisodeResult, RandomPolicy, run_episode, summarize

# --- wrappers against a scripted stub env -----------------------------------


def test_frame_skip_repeats_action_and_sums_rewards(scripted_env):
    inner = scripted_env([1.0, 2.0, 4.0, 8.0, 16.0])
    env = FrameSkip(inner, skip=2)
    env.reset()
    _, reward, terminated, truncated, _ = env.step(3)
    assert reward == 3.0
    assert inner.actions == [3, 3]
    assert not terminated and not truncated


def test_frame_skip_stops_at_episode_end(scripted_env):
    inner = scripted_env([1.0, 2.0, 4.0])
    env = FrameSkip(inner, skip=2)
    env.reset()
    env.step(0)
    _, reward, terminated, _, _ = env.step(0)  # only one frame left
    assert reward == 4.0
    assert terminated
    assert len(inner.actions) == 3


def test_frame_skip_rejects_invalid_skip(scripted_env):
    with pytest.raises(ValueError):
        FrameSkip(scripted_env([1.0]), skip=0)


def test_skip_zoom_plays_noops_on_reset(scripted_env):
    inner = scripted_env([0.0] * 20)
    env = SkipZoom(inner, frames=5)
    obs, _ = env.reset()
    assert inner.actions == [0] * 5
    assert obs[0, 0, 0] == 5  # the observation after the 5th no-op frame


def test_agent_fps_accounts_for_frame_skip():
    assert agent_fps({"frame_skip": 2}) == 25
    assert agent_fps({}) == 50


# --- the real CarRacing-v3 ---------------------------------------------------


def test_single_env_spaces_match_the_locked_in_design(env_cfg):
    env = make_env(env_cfg, shaped=True)()
    try:
        assert env.observation_space.shape == (96, 96, 3)
        assert env.observation_space.dtype == np.uint8
        assert env.action_space.n == 5  # discrete actions for Stage 1
        obs, _ = env.reset(seed=0)
        assert obs.shape == (96, 96, 3)
    finally:
        env.close()


def test_single_env_passes_sb3_env_checker(env_cfg):
    env = make_env(env_cfg, shaped=True)()
    try:
        check_env(env, warn=True)
    finally:
        env.close()


def test_vec_env_is_frame_stacked_and_channel_first(env_cfg):
    vec_env = make_vec_env(env_cfg, n_envs=1, seed=0, shaped=True)
    try:
        assert vec_env.observation_space.shape == (12, 96, 96)  # 4 stacked RGB frames
        obs = vec_env.reset()
        assert obs.shape == (1, 12, 96, 96)
        assert obs.dtype == np.uint8
    finally:
        vec_env.close()


def test_same_seed_gives_the_same_track(env_cfg):
    def first_obs(seed):
        vec_env = make_vec_env(env_cfg, n_envs=1, seed=seed, shaped=False)
        try:
            return vec_env.reset()
        finally:
            vec_env.close()

    assert np.array_equal(first_obs(7), first_obs(7))
    assert not np.array_equal(first_obs(7), first_obs(8))


def test_shaped_env_stops_a_stalled_car_and_reports_raw_return(env_cfg):
    cfg = {**env_cfg, "reward_shaping": {"clip_max": 1.0, "patience": 5}}
    vec_env = make_vec_env(cfg, n_envs=1, seed=0, shaped=True)
    try:
        vec_env.reset()
        for step in range(1, 6):
            _, _, dones, infos = vec_env.step(np.array([0]))  # no-op: the car never moves
        assert step == 5 and dones[0]
        info = infos[0]
        assert info["early_stop"] is True
        assert info["TimeLimit.truncated"] is True
        # 5 agent steps x 2 frames x -0.1 per frame
        assert info["raw_return"] == pytest.approx(-1.0)
        assert info["episode"]["raw_return"] == pytest.approx(-1.0)  # what Monitor logs
    finally:
        vec_env.close()


def test_unshaped_env_does_not_stop_early(env_cfg):
    cfg = {**env_cfg, "reward_shaping": {"clip_max": 1.0, "patience": 5}}
    vec_env = make_vec_env(cfg, n_envs=1, seed=0, shaped=False)
    try:
        vec_env.reset()
        for _ in range(20):
            _, _, dones, _ = vec_env.step(np.array([0]))
            assert not dones[0]
    finally:
        vec_env.close()


def test_run_episode_scores_and_captures_frames(env_cfg):
    vec_env = make_vec_env(env_cfg, n_envs=1, seed=0, shaped=False, render_mode="rgb_array")
    try:
        policy = RandomPolicy(vec_env.action_space, seed=0)
        result = run_episode(vec_env, policy, seed=3, capture=True, max_steps=10)
        assert result.length == 10
        assert len(result.frames) == 10
        assert result.frames[0].shape == (400, 600, 3)
        stats = summarize([result])
        assert stats["n_episodes"] == 1
        assert stats["mean_return"] == pytest.approx(result.episode_return)
    finally:
        vec_env.close()


def test_telemetry_reports_driving_metrics_without_changing_the_score(env_cfg):
    def play(telemetry):
        vec_env = make_vec_env(env_cfg, n_envs=1, seed=0, shaped=False, telemetry=telemetry)
        try:
            policy = RandomPolicy(vec_env.action_space, seed=0)
            return run_episode(vec_env, policy, seed=3, max_steps=40)
        finally:
            vec_env.close()

    plain, measured = play(False), play(True)
    assert not plain.has_telemetry
    assert measured.has_telemetry
    assert measured.episode_return == pytest.approx(plain.episode_return)
    assert measured.actions == plain.actions

    assert measured.outcome == "cut_off"
    assert len(measured.speeds) == len(measured.actions) == 40
    assert all(speed >= 0 for speed in measured.speeds)
    assert 0 <= measured.off_track_steps <= 40
    assert 0 < measured.tiles_visited <= measured.track_tiles
    assert 0 < measured.track_completion <= 1


def test_summarize_adds_driving_metrics_when_telemetry_is_present():
    lap = EpisodeResult(
        episode_return=920.0, length=400, outcome="lap", actions=[3] * 300 + [1] * 100,
        speeds=[50.0] * 400, off_track_steps=0, tiles_visited=290, track_tiles=300,
    )
    crash = EpisodeResult(
        episode_return=-80.0, length=100, outcome="out_of_bounds", actions=[3] * 100,
        speeds=[100.0] * 100, off_track_steps=50, tiles_visited=30, track_tiles=300,
    )
    stats = summarize([lap, crash], fps=25.0, action_names=("nothing", "right", "left", "gas", "brake"))

    assert stats["episodes_solved"] == 1
    assert stats["laps_completed"] == 1
    assert stats["out_of_bounds"] == 1
    assert stats["mean_lap_time_s"] == pytest.approx(16.0)
    assert stats["best_lap_time_s"] == pytest.approx(16.0)
    assert stats["mean_duration_s"] == pytest.approx(10.0)
    assert stats["mean_track_completion"] == pytest.approx((290 / 300 + 30 / 300) / 2)
    assert stats["min_track_completion"] == pytest.approx(0.1)
    assert stats["mean_speed"] == pytest.approx(60.0)  # weighted by steps
    assert stats["max_speed"] == pytest.approx(100.0)
    assert stats["off_track_share"] == pytest.approx(0.1)
    assert stats["action_shares"] == pytest.approx(
        {"nothing": 0.0, "right": 0.2, "left": 0.0, "gas": 0.8, "brake": 0.0}
    )
    assert stats["outcomes"] == ["lap", "out_of_bounds"]


def test_summarize_without_telemetry_keeps_to_score_metrics():
    stats = summarize([EpisodeResult(episode_return=5.0, length=10)])
    assert stats["mean_return"] == pytest.approx(5.0)
    assert stats["mean_lap_length"] is None
    assert "mean_track_completion" not in stats
    assert "action_shares" not in stats
    assert "mean_duration_s" not in stats
