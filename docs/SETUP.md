# Setup

Stages 1 and 2 (CarRacing-v3) run on Windows, Linux and macOS. Stage 3
(Assetto Corsa) is Windows-only and gets its own section when it is built.

Tested on: Windows 11, Python 3.14, RTX 5070 Laptop GPU (8 GB), PyTorch 2.14 +
CUDA 13.0, Stable-Baselines3 2.9, Gymnasium 1.4.

## 1. Virtual environment

From the repo root (commands are for Git Bash; in PowerShell use
`.venv\Scripts\python` instead of `.venv/Scripts/python`):

```bash
py -3.14 -m venv .venv
```

```bash
.venv/Scripts/python -m pip install --upgrade pip
```

Any Python from 3.10 to 3.14 works.

## 2. PyTorch with CUDA

Install PyTorch **before** the other requirements. On Windows a plain
`pip install torch` gives you the CPU-only build.

```bash
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
```

RTX 50-series cards need a CUDA 12.8 or newer build (`cu128`, `cu130`); older
builds do not support them.

## 3. Box2D (the fiddly part)

CarRacing runs on the Box2D physics engine. Its Python binding, `box2d-py`,
publishes **no prebuilt wheels**, so pip has to compile it, which needs
[SWIG](https://www.swig.org/) and a C++ compiler. Pick one route.

### Route A: reuse the wheel in `wheels/`

If `wheels/box2d_py-2.3.8-cp314-cp314-win_amd64.whl` exists (it is built once
per machine and is git-ignored), there is nothing to do: step 4 picks it up.
It only fits 64-bit Windows with Python 3.14.

### Route B: build the wheel with MSYS2 GCC (how the wheel above was made)

Needs [MSYS2](https://www.msys2.org/) with the UCRT64 toolchain
(`pacman -S mingw-w64-ucrt-x86_64-gcc`). Run in Git Bash from the repo root:

```bash
PY="$PWD/.venv/Scripts/python"
"$PY" -m pip install swig

# pip's swig launcher breaks inside pip's isolated build, so call the real binary.
SWIG_PKG="$("$PY" -c "import swig, os; print(os.path.dirname(swig.__file__))")"
export SWIG_LIB="$(cygpath -w "$(ls -d "$(cygpath -u "$SWIG_PKG")"/data/share/swig/*)")"
export PATH="$(cygpath -u "$SWIG_PKG")/data/bin:/c/msys64/ucrt64/bin:$PATH"

mkdir -p build_box2d wheels
"$PY" -m pip download box2d-py==2.3.8 --no-deps --no-binary=:all: -d build_box2d
tar xzf build_box2d/box2d-py-2.3.8.tar.gz -C build_box2d
cd build_box2d/box2d-py-2.3.8

# 1) compile with GCC instead of MSVC
printf '\n[build]\ncompiler = mingw32\n\n[build_ext]\ncompiler = mingw32\n' >> setup.cfg
# 2) GCC rejects the pointer-to-long casts MSVC only warns about
sed -i "s/    extra_args=\['-I.'\]$/    extra_args=['-I.', '-fpermissive', '-w']/" setup.py
# 3) link the GCC runtime statically so the wheel needs no MSYS2 DLLs at run time
sed -i "s/extra_compile_args=extra_args, language='c++')/extra_compile_args=extra_args, extra_link_args=['-static-libgcc', '-static-libstdc++', '-static'], language='c++')/" setup.py

"$PY" -m pip wheel . --no-deps -w ../../wheels
cd ../.. && rm -rf build_box2d
```

### Route C: Microsoft C++ Build Tools

Install the "Desktop development with C++" workload from the
[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/),
install SWIG so that `swig.exe` is on `PATH`, and step 4 builds `box2d-py` by
itself. Not tested on this project's machine.

### Route D: conda

`conda env create -f environment.yml` installs a prebuilt `box2d-py` from
conda-forge (then reinstall PyTorch with CUDA as described in that file). Not
tested on this project's machine.

### Linux / macOS

Install SWIG from your package manager (`sudo apt-get install swig`,
`brew install swig`); step 4 then builds `box2d-py` with the system compiler.
This is what CI does.

## 4. Everything else

```bash
.venv/Scripts/python -m pip install --find-links wheels -r requirements.txt
```

`requirements.txt` uses `pygame-ce` rather than `pygame`: the original has no
wheels for Python 3.14, and `pygame-ce` is a drop-in replacement that provides
the same `pygame` module.

## 5. Check the install

```bash
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
.venv/Scripts/python -m pytest -q
```

```bash
.venv/Scripts/python -m src.train --config config/ppo_carracing.yaml --smoke
```

The smoke run trains for 1,024 timesteps (about half a minute) and exercises
everything a real run does: parallel envs, the GPU, evaluation, video
recording, checkpoints. Its output goes to git-ignored `_smoke` folders under
`models/`, `logs/` and `videos/`, which are safe to delete.

## Stage 3: Assetto Corsa

Not written yet. This section will cover the Assetto Corsa install, shared
memory telemetry, and the ViGEm virtual gamepad driver.
