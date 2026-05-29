#!/bin/bash
# IS4.5.0 training loop — identical logic to train_and_viz.sh but uses the
# Python 3.10 venv with IsaacSim 4.5.0 + Isaac Lab 2.1.0.
#
# Usage:
#   bash tools/train_and_viz_is45.sh           # auto-resume if checkpoint exists
#   bash tools/train_and_viz_is45.sh --clean   # force fresh start

set -euo pipefail

cd /workspace/TTRL-ICRA2026

PYTHON="$HOME/.venv/isaac45/bin/python"
TASK="k1_tt"
TASK_EVAL="k1_tt_eval"
NUM_ENVS=4096
MAX_ITERS=15000
VIZ_INTERVAL=500
VIZ_DURATION=90
LOG_ROOT="logs/k1_table_tennis"
CLEAN=false

for arg in "$@"; do
    [ "$arg" = "--clean" ] && CLEAN=true
done

echo "============================================================"
echo "  K1 Table Tennis — IS4.5.0 train + periodic VNC visualization"
echo "  Python:       $PYTHON"
echo "  Task:         $TASK  ($NUM_ENVS envs, headless)"
echo "  Viz task:     $TASK_EVAL  (1 env, DISPLAY=:1)"
echo "  Viz every:    $VIZ_INTERVAL iterations ($VIZ_DURATION s each)"
echo "  Max iters:    $MAX_ITERS"
echo "  Clean start:  $CLEAN"
echo "============================================================"
echo ""

latest_run() {
    ls -t "$LOG_ROOT" 2>/dev/null | head -1
}

latest_checkpoint() {
    local run="$1"
    ls -t "$LOG_ROOT/$run"/model_*.pt 2>/dev/null | head -1 | xargs -I{} basename {} 2>/dev/null || true
}

checkpoint_iter() {
    echo "$1" | grep -oP '\d+' | tail -1
}

CURRENT_ITER=0
RESUME_RUN=""
RESUME_CKPT=""

if ! $CLEAN; then
    _RUN=$(latest_run)
    if [ -n "$_RUN" ]; then
        _CKPT=$(latest_checkpoint "$_RUN")
        if [ -n "$_CKPT" ]; then
            CURRENT_ITER=$(checkpoint_iter "$_CKPT")
            RESUME_RUN="$_RUN"
            RESUME_CKPT="$_CKPT"
            echo "  Found existing checkpoint: $_RUN / $_CKPT (iter $CURRENT_ITER)"
            echo "  Resuming from iter $CURRENT_ITER — pass --clean to start fresh."
            echo ""
        fi
    fi
fi

if $CLEAN || [ -z "$RESUME_RUN" ]; then
    echo "  Starting fresh from iter 0."
    echo ""
fi

while [ "$CURRENT_ITER" -lt "$MAX_ITERS" ]; do
    STOP_AT=$(( CURRENT_ITER + VIZ_INTERVAL ))
    [ "$STOP_AT" -gt "$MAX_ITERS" ] && STOP_AT=$MAX_ITERS

    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  TRAINING  iter $CURRENT_ITER → $STOP_AT"
    echo "──────────────────────────────────────────────────────────"

    if [ -z "$RESUME_RUN" ]; then
        $PYTHON legged_lab/scripts/train.py \
            --task "$TASK" \
            --num_envs "$NUM_ENVS" \
            --headless \
            --logger tensorboard \
            --predictor \
            --max_iterations "$STOP_AT"
    else
        echo "  Resuming $RESUME_RUN / $RESUME_CKPT"
        $PYTHON legged_lab/scripts/train.py \
            --task "$TASK" \
            --num_envs "$NUM_ENVS" \
            --headless \
            --logger tensorboard \
            --predictor \
            --resume True \
            --load_run "$RESUME_RUN" \
            --checkpoint "$RESUME_CKPT" \
            --max_iterations "$VIZ_INTERVAL"
    fi

    CURRENT_ITER=$STOP_AT
    RESUME_RUN=$(latest_run)
    RESUME_CKPT=$(latest_checkpoint "$RESUME_RUN")

    echo ""
    echo "  ── Training stats at iter $CURRENT_ITER ──"
    $PYTHON tools/check_training.py || true

    if [ "$CURRENT_ITER" -ge "$MAX_ITERS" ]; then
        echo ""
        echo "Training complete at $CURRENT_ITER iterations."
        break
    fi

    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  VISUALIZATION  ($VIZ_DURATION s) — $RESUME_RUN / $RESUME_CKPT"
    echo "  Open http://localhost:5999/vnc.html to watch"
    echo "──────────────────────────────────────────────────────────"

    # WARNING: Isaac Sim 4.5 hangs on exit even with timeout --kill-after=10.
    # This stalls the training loop for hours (confirmed across multiple runs).
    # Prefer finetune_is45.sh + continue_is45.sh (no VIZ) with watchdog_is45.sh
    # running in a spare tmux window to auto-kill hung processes.
    DISPLAY=:1 timeout --kill-after=10 "$VIZ_DURATION" \
        $PYTHON legged_lab/scripts/play.py \
            --task "$TASK_EVAL" \
            --num_envs 1 \
            --predictor \
            --load_run "$RESUME_RUN" \
            --checkpoint "$RESUME_CKPT" \
            --/exts/omni.kit.renderer.core/present/enabled=true \
        || true

    echo ""
    echo "  Visualization done. Resuming training..."
done

echo ""
echo "============================================================"
echo "  All done. Final model:"
echo "  $LOG_ROOT/$RESUME_RUN/$RESUME_CKPT"
echo "============================================================"
