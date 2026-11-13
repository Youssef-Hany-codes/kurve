"""RewardShaping: reward clipping, early stopping, and raw-return bookkeeping."""

from __future__ import annotations

import pytest

from src.envs.carracing_wrapper import RewardShaping


def rollout(env, n_steps):
    """Step with action 0 up to n_steps times; return the list of step tuples."""
    env.reset()
    steps = []
    for _ in range(n_steps):
        steps.append(env.step(0))
        if steps[-1][2] or steps[-1][3]:
            break
    return steps


def test_clip_caps_positive_rewards_only(scripted_env):
    env = RewardShaping(scripted_env([5.0, 0.5, -0.1, -100.0]), clip_max=1.0, patience=None)
    rewards = [step[1] for step in rollout(env, 4)]
    assert rewards == [1.0, 0.5, -0.1, -100.0]


def test_no_clipping_when_disabled(scripted_env):
    env = RewardShaping(scripted_env([5.0, 7.0]), clip_max=None, patience=None)
    assert [step[1] for step in rollout(env, 2)] == [5.0, 7.0]


def test_truncates_after_patience_steps_without_progress(scripted_env):
    env = RewardShaping(scripted_env([-0.1] * 10), clip_max=None, patience=3)
    steps = rollout(env, 10)
    assert len(steps) == 3
    _, _, terminated, truncated, info = steps[-1]
    assert truncated and not terminated
    assert info["early_stop"] is True


def test_positive_reward_resets_the_patience_counter(scripted_env):
    # Two stalled steps, progress, then three stalled steps -> stops on step 6, not step 3.
    env = RewardShaping(scripted_env([-0.1, -0.1, 3.0, -0.1, -0.1, -0.1, -0.1]), clip_max=None, patience=3)
    steps = rollout(env, 7)
    assert len(steps) == 6
    assert steps[-1][3] is True


def test_zero_reward_counts_as_no_progress(scripted_env):
    env = RewardShaping(scripted_env([0.0] * 5), clip_max=None, patience=2)
    assert len(rollout(env, 5)) == 2


def test_patience_none_never_stops_early(scripted_env):
    env = RewardShaping(scripted_env([-0.1] * 6), clip_max=None, patience=None)
    steps = rollout(env, 6)
    assert len(steps) == 6
    assert steps[-1][2] is True  # the env's own termination
    assert "early_stop" not in steps[-1][4]


def test_raw_return_is_the_unclipped_sum(scripted_env):
    env = RewardShaping(scripted_env([5.0, -0.1, 4.0]), clip_max=1.0, patience=None)
    steps = rollout(env, 3)
    assert "raw_return" not in steps[0][4]  # only reported when the episode ends
    assert steps[-1][4]["raw_return"] == pytest.approx(8.9)
    assert sum(step[1] for step in steps) == pytest.approx(1.9)


def test_raw_return_reported_on_early_stop(scripted_env):
    env = RewardShaping(scripted_env([2.0, -0.1, -0.1, -0.1]), clip_max=1.0, patience=2)
    steps = rollout(env, 4)
    assert len(steps) == 3
    assert steps[-1][4]["raw_return"] == pytest.approx(1.8)


def test_reset_clears_episode_state(scripted_env):
    env = RewardShaping(scripted_env([-0.1] * 10), clip_max=None, patience=3)
    rollout(env, 2)  # two stalled steps, then a fresh episode
    steps = rollout(env, 10)
    assert len(steps) == 3
    assert steps[-1][4]["raw_return"] == pytest.approx(-0.3)


def test_natural_termination_is_not_marked_as_early_stop(scripted_env):
    # The env terminates on the same step the patience would run out.
    env = RewardShaping(scripted_env([-0.1, -0.1]), clip_max=None, patience=2)
    _, _, terminated, truncated, info = rollout(env, 2)[-1]
    assert terminated and not truncated
    assert "early_stop" not in info
