#!/bin/bash
# Alternates between headless training chunks and short VNC visualizations.
# Each Isaac Sim instance takes 3-5 min to start, so we use 1000-iter chunks
# to keep the startup overhead to ~17% of total wall time.
#
# Usage:
#   bash tools/train_and_viz.sh           # auto-resume if checkpoint exists
#   bash tools/train_and_viz.sh --clean   # force fresh start (ignore checkpoints)

set -euo pipefail

cd /workspace/TTRL-ICRA2026

TASK="k1_tt"
TASK_EVAL="k1_tt_eval"
NUM_ENVS=4096
MAX_ITERS=10000
VIZ_INTERVAL=500    # train this many iters between each visualization
VIZ_DURATION=90     # seconds to run play.py for visualization
LOG_ROOT="logs/k1_table_tennis"
PYTHON="python3"
CLEAN=false

# Parse flags
for arg in "$@"; do
    [ "$arg" = "--clean" ] && CLEAN=true
done

echo "============================================================"
echo "  K1 Table Tennis — train + periodic VNC visualization"
echo "  Task:         $TASK  ($NUM_ENVS envs, headless)"
echo "  Viz task:     $TASK_EVAL  (1 env, DISPLAY=:1)"
echo "  Viz every:    $VIZ_INTERVAL iterations ($VIZ_DURATION s each)"
echo "  Max iters:    $MAX_ITERS"
echo "  Clean start:  $CLEAN"
echo "============================================================"
echo ""

# ── helpers ──────────────────────────────────────────────────────────────────

latest_run() {
    ls -t "$LOG_ROOT" 2>/dev/null | head -1
}

latest_checkpoint() {
    local run="$1"
    ls -t "$LOG_ROOT/$run"/model_*.pt 2>/dev/null | head -1 | xargs -I{} basename {} 2>/dev/null || true
}

checkpoint_iter() {
    # extract iteration number from filename: model_4000.pt → 4000
    echo "$1" | grep -oP '\d+' | tail -1
}

# ── determine start iteration from existing checkpoint ───────────────────────

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

# ── main loop ─────────────────────────────────────────────────────────────────

while [ "$CURRENT_ITER" -lt "$MAX_ITERS" ]; do
    STOP_AT=$(( CURRENT_ITER + VIZ_INTERVAL ))
    [ "$STOP_AT" -gt "$MAX_ITERS" ] && STOP_AT=$MAX_ITERS

    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  TRAINING  iter $CURRENT_ITER → $STOP_AT"
    echo "──────────────────────────────────────────────────────────"

    if [ -z "$RESUME_RUN" ]; then
        # Fresh start — no resume flags, new run directory will be created.
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
    # After first chunk always resume from latest checkpoint
    RESUME_RUN=$(latest_run)
    RESUME_CKPT=$(latest_checkpoint "$RESUME_RUN")

    # Show progress stats before visualization
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

    # Isaac Sim ignores SIGTERM; --kill-after=10 escalates to SIGKILL after 10s
    DISPLAY=:1 timeout --kill-after=10 "$VIZ_DURATION" \
        $PYTHON legged_lab/scripts/play.py \
            --task "$TASK_EVAL" \
            --num_envs 1 \
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
echo "  logs/k1_table_tennis/$RESUME_RUN/$RESUME_CKPT"
echo "============================================================"
