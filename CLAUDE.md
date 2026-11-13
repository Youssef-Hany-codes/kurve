# Kurve — An RL Racing Agent

Persistent project context. Read this before starting work in this repo.

## What this is

A reinforcement learning project that trains agents to drive: first in the
Gymnasium `CarRacing-v3` toy environment, then in Assetto Corsa, a real racing
simulator. The repo carries training code, configs, evaluation/comparison
tooling, and the recorded before/after artifacts for each stage.

## Status

- **Stage 1 (PPO on CarRacing-v3): trained, "after" comparison NOT built
  yet.** The 2M-timestep run finished on 2026-11-13 in 3 h 04 min. Training-time
  evals (5 fixed tracks, raw reward): 914.8 ± 5.1 at 2M, best 918.2 ± 6.0 at
  1.7M (`models/ppo_carracing/best_model.zip`); first above 900 at 400k. The
  random baseline is -43.0 ± 11.6 over 10 tracks
  (`results/ppo_carracing/metrics_before.json`). Still to do:
  `compare --stage after` (note it overwrites the training-end
  `step_final.mp4` with a best-model recording) and `plotting`.
- **Stage 2 and Stage 3: not started.** `config/sac_*.yaml`, `src/envs/ac_env.py`
  and `src/ac_interface/*` are empty placeholder files.

## Running things

Everything runs from the repo root as a module, using the venv at `.venv`
(Python 3.14). Full install steps are in `docs/SETUP.md`.

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m src.train --config config/ppo_carracing.yaml --smoke   # 1,024-step pipeline check
.venv/Scripts/python -m src.utils.compare --config config/ppo_carracing.yaml --stage before
.venv/Scripts/python -m src.train --config config/ppo_carracing.yaml           # real run, ~3 h
.venv/Scripts/python -m src.utils.compare --config config/ppo_carracing.yaml --stage after
.venv/Scripts/python -m src.utils.plotting --config config/ppo_carracing.yaml
.venv/Scripts/python -m src.evaluate --config config/ppo_carracing.yaml [--random]
```

## Stage 1 as built

Decisions made while building Stage 1, on top of the locked-in ones below:

- **Wrapper stack** (`src/envs/carracing_wrapper.py`): `SkipZoom` (50 no-op
  frames on reset) -> `FrameSkip` (2) -> `RewardShaping` (training only) ->
  `Monitor`, then `SubprocVecEnv` -> `VecFrameStack` (4) -> `VecTransposeImage`.
  Observations stay raw RGB; the policy sees a (12, 96, 96) uint8 stack.
- **Reward shaping is training-only**: per-step reward capped at 1.0, episode
  truncated after 100 agent steps without a positive reward. Evaluation envs
  are built with `shaped=False`, so every eval score, video and metrics file is
  the true CarRacing reward. The raw training return is logged as
  `rollout/ep_raw_rew_mean`.
- **A "timestep" is one agent decision** = `frame_skip` simulator frames. A
  full episode is 475 timesteps (1000 frames - 50 zoom, / 2).
- **Evaluation tracks are fixed**: eval seeds are `seed + 10000 + episode`
  (`src/utils/seeding.py`), shared by training-time evals and `compare.py`, so
  before and after are scored on identical tracks.
- **`src/evaluate.py` doubles as the rollout library** (`run_episode`,
  `summarize`, `write_video`) used by `callbacks.py` and `compare.py`.
- **Hyperparameters in `config/ppo_carracing.yaml` were never tuned**: the
  first full run with them reached ~915, so they were left as they are.
- **Dependencies**: `pygame-ce` instead of `pygame` (no 3.14 wheels), and
  `box2d-py` installed from a locally built wheel in git-ignored `wheels/`
  (no PyPI wheels exist; see `docs/SETUP.md` Route B for the rebuild).
- **Measured on this machine**: ~250 timesteps/s with 8 envs at the start of a
  run (16 envs only reaches ~300), ~180 averaged over the full 2M-step run,
  ~250 MiB VRAM for PPO.

Heads-up for Stage 2: an SAC replay buffer of stacked image observations is
large (1M transitions x 12x96x96 bytes is ~110 GB), so the buffer size and
frame stacking need sizing against the 32 GB of RAM.

## Sequencing (deliberate, de-risked)

1. **PPO on CarRacing-v3**
2. **SAC on CarRacing-v3**
3. **SAC on Assetto Corsa**

Each stage changes exactly one variable. Stage 2 isolates the algorithm change
(same env, new algorithm). Stage 3 isolates the environment change (same
algorithm, new env) — safe because Stage 2 already proved SAC works before it
meets a harder environment.

## Locked-in decisions

- **Env id is `CarRacing-v3`**, not v2. Farama renamed it; v3 also changed lap
  completion to properly *terminate* rather than truncate.
- **Stage 1 (PPO): discrete actions** first, for a simpler working baseline.
- **Stage 2 (SAC): `continuous=True`.** SAC needs continuous control to make
  sense. PPO gets retrained on continuous too, so the comparison is fair.
- **`CnnPolicy` throughout.** Observations are raw 96x96 RGB pixels. No
  hand-engineered feature vector — that's a possible v2 improvement, explicitly
  out of scope now.
- **Hardware: RTX 5070, 32GB DDR5.** Comfortably supports `SubprocVecEnv` with
  8 parallel envs and GPU-accelerated CNN training. No cloud/Colab needed.
  (It is the Laptop GPU variant: 8 GB VRAM, 20 logical CPU cores.)

## Visualization approach

- Train **headless and parallel** for speed.
- A periodic `EvalCallback` uses a **single separate rendered env**: it scores
  the agent every 50k steps and records a video checkpoint every **400k steps**
  plus one when training finishes (`step_final.mp4`) — no slowdown to main
  training. (Originally ~50k for videos; changed to 400k at the user's request.)
  All videos use the same track, the one in `step_0_random.mp4`.
- `evaluate.py` is a separate on-demand script for live `render_mode="human"`
  viewing.
- Every stage produces a **before/after (random vs. trained)** video and a
  metrics comparison, generated by `src/utils/compare.py`.

## Stage 3: Assetto Corsa specifics

- Telemetry is read from AC's **shared memory** (ctypes-based).
- Actions are sent via a **virtual gamepad (ViGEm)** — AC has no direct
  programmatic input API.
- **Windows-only** and by far the most fragile stage; `docs/SETUP.md` exists
  specifically to document this carefully.
- Reference prior art exists (open-source AC+RL repos) for both the
  shared-memory reading and the PPO/SAC training patterns.

## Target repository layout

```
kurve/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
├── setup.py / pyproject.toml
├── config/
│   ├── ppo_carracing.yaml
│   ├── sac_carracing.yaml
│   └── sac_assettocorsa.yaml
├── src/
│   ├── envs/
│   │   ├── carracing_wrapper.py
│   │   └── ac_env.py
│   ├── ac_interface/
│   │   ├── shared_memory_reader.py
│   │   └── controller.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       ├── config.py           # config loading, run output paths, algo registry
│       ├── callbacks.py
│       ├── plotting.py
│       ├── seeding.py
│       └── compare.py          # before/after video + stats comparisons
├── tests/
│   ├── conftest.py
│   ├── test_env_wrapper.py
│   └── test_reward_shaping.py
├── .github/
│   └── workflows/
│       └── tests.yml
├── models/
│   ├── ppo_carracing/
│   ├── sac_carracing/
│   └── sac_assettocorsa/
├── logs/
├── videos/
│   ├── ppo_carracing/
│   │   ├── step_0_random.mp4
│   │   ├── step_50000.mp4
│   │   ├── ...
│   │   ├── step_final.mp4
│   │   └── comparison_side_by_side.mp4
│   ├── sac_carracing/
│   └── sac_assettocorsa/
├── results/
│   ├── ppo_carracing/
│   │   ├── metrics_before.json
│   │   ├── metrics_after.json
│   │   └── comparison_table.md
│   ├── sac_carracing/
│   └── sac_assettocorsa/
├── notebooks/
│   └── results_analysis.ipynb
└── docs/
    ├── assets/
    │   ├── ppo_before_after.gif
    │   ├── sac_before_after.gif
    │   └── ac_before_after.gif
    └── SETUP.md
```
