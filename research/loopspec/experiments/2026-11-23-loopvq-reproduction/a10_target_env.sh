#!/usr/bin/env bash
# Source in this task's A10 tmux windows. Existing CUDA/CPython tools are read-only;
# Python packages and compiler caches belong to this task. No credentials here.
export LOOPVQ_TASK_ROOT=/homes/ruoxi07/experiments/loopvq/2026-11-23-reproduction
export LOOPVQ_TARGET_REPO=/homes/ruoxi07/repo/sglang-loopspec-tp2
export LOOPVQ_MODEL=/homes/ruoxi07/experiments/loopspec/tp2-20261106/models/Ouro-2.6B-Thinking
export LOOPVQ_TOOLCHAIN=/homes/ruoxi07/experiments/loopspec/tp2-20261106/environment
export VIRTUAL_ENV="$LOOPVQ_TASK_ROOT/envs/sglang-loopvq-test"
export LOOPVQ_PYTHON="$VIRTUAL_ENV/bin/python"
export CUDA_HOME="$LOOPVQ_TOOLCHAIN/cuda13/nvidia/cu13"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$LOOPVQ_TOOLCHAIN/cargo/bin:$PATH"
export LD_LIBRARY_PATH="$LOOPVQ_TOOLCHAIN/compat:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$LOOPVQ_TARGET_REPO/python"
export RUSTUP_HOME="$LOOPVQ_TOOLCHAIN/rustup"
export CARGO_HOME="$LOOPVQ_TOOLCHAIN/cargo"
export CARGO_TARGET_DIR="$LOOPVQ_TASK_ROOT/cache/rust-target"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TMPDIR="$LOOPVQ_TASK_ROOT/tmp"
export HF_HOME="$LOOPVQ_TASK_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$LOOPVQ_TASK_ROOT/cache/xdg"
export SGLANG_CACHE_DIR="$LOOPVQ_TASK_ROOT/cache/sglang"
export SGLANG_JIT_CACHE_DIR="$LOOPVQ_TASK_ROOT/cache/sglang-jit"
export TORCH_EXTENSIONS_DIR="$LOOPVQ_TASK_ROOT/cache/torch-extensions"
export TRITON_CACHE_DIR="$LOOPVQ_TASK_ROOT/cache/triton"
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY=529513043-tsinghua-university
export WANDB_PROJECT=loopvq-ouro-2p6b-thinking
export WANDB_DIR="$LOOPVQ_TASK_ROOT/wandb"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$CARGO_TARGET_DIR" "$SGLANG_CACHE_DIR" \
  "$SGLANG_JIT_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
