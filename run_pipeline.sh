#!/usr/bin/env bash
# End-to-end pipeline: profiler (T3 input) -> build_strategy (T3) -> training (T4).
# Ties the four stages together so a schedule is profiled, solved, mapped, and
# consumed by the training driver in one command.
#
# Usage:
#   DGNN_GRAPH_ROOT=/path DGNN_DATA_ROOT=/path ./run_pipeline.sh
# Override any of the vars below via the environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${DATASET:-arxiv}"
MODEL="${MODEL:-wdgcn}"
WINDOW="${WINDOW:-4}"
NUM_SNAPSHOTS="${NUM_SNAPSHOTS:-30}"
STRIDE="${STRIDE:-1}"
SOLVER="${SOLVER:-greedy}"          # greedy (REAL-G) or ilp (REAL-L, needs gurobipy)
GPUS="${GPUS:-8}"                    # D: GPUs per iteration (scheduler)
WORLD_SIZE="${WORLD_SIZE:-8}"        # training processes
HIDDEN="${HIDDEN:-64}"
EPOCHS="${EPOCHS:-50}"
DEVICE="${DEVICE:-0}"
OUTDIR="${OUTDIR:-/tmp/real_pipeline}"

mkdir -p "$OUTDIR"
REUSE_LOG="$OUTDIR/reuse_${DATASET}_${MODEL}.txt"
STRATEGY="$OUTDIR/strategy_${DATASET}_${MODEL}_${SOLVER}.txt"

echo "==> [1/3] Profiling ($DATASET/$MODEL, window=$WINDOW, ${NUM_SNAPSHOTS} snapshots)"
python "$HERE/training_process/profiler.py" \
    --dataset "$DATASET" --model "$MODEL" --window "$WINDOW" \
    --num-snapshots "$NUM_SNAPSHOTS" --stride "$STRIDE" \
    --repeats 2 --warmup 1 --device "$DEVICE" --out "$REUSE_LOG"

echo "==> [2/3] Building strategy ($SOLVER, $GPUS GPUs) with group->start mapping"
python "$HERE/build_strategy.py" \
    --reuse-log "$REUSE_LOG" --solver "$SOLVER" --gpus "$GPUS" \
    --world-size "$WORLD_SIZE" --model "$MODEL" --out "$STRATEGY"

echo "==> [3/3] Training ($MODEL, world_size=$WORLD_SIZE)"
python "$HERE/training_process/training.py" \
    --dataset "$DATASET" --model "$MODEL" --world_size "$WORLD_SIZE" \
    --time_window_size "$WINDOW" --num_epochs "$EPOCHS" --hidden_size "$HIDDEN" \
    --strategy_path "$STRATEGY"

echo "==> pipeline complete. reuse_log=$REUSE_LOG strategy=$STRATEGY"
