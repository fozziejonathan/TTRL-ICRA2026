#!/bin/bash
# Continuation: resumes from the latest IS4.5.0 checkpoint and trains
# for EXTRA_ITERS more iterations. No VIZ — run play.py manually when needed.
#
# Usage:
#   bash tools/continue_is45.sh [EXTRA_ITERS]

set -euo pipefail

cd /workspace/TTRL-ICRA2026

PYTHON="$HOME/.venv/isaac45/bin/python"
TASK="k1_tt"
NUM_ENVS=4096
EXTRA_ITERS="${1:-5000}"
CHUNK_SIZE=500
LOG_ROOT="logs/k1_table_tennis"

latest_run() {
    for run in $(ls -t "$LOG_ROOT" 2>/dev/null); do
        if ls "$LOG_ROOT/$run"/model_*.pt &>/dev/null; then
            echo "$run"
            return
        fi
    done
}

latest_checkpoint() {
    local run="$1"
    ls -t "$LOG_ROOT/$run"/model_*.pt 2>/dev/null | head -1 | xargs -I{} basename {} 2>/dev/null || true
}

checkpoint_iter() {
    echo "$1" | grep -oP '\d+' | tail -1
}

RESUME_RUN=$(latest_run)
RESUME_CKPT=$(latest_checkpoint "$RESUME_RUN")
START_ITER=$(checkpoint_iter "$RESUME_CKPT")
END_ITER=$(( START_ITER + EXTRA_ITERS ))

echo "============================================================"
echo "  K1 Table Tennis — IS4.5.0 continuation"
echo "  Resume: $LOG_ROOT/$RESUME_RUN/$RESUME_CKPT (iter $START_ITER)"
echo "  Extra iters: $EXTRA_ITERS  (target iter ~$END_ITER)"
echo "============================================================"
echo ""

CURRENT_ITER="$START_ITER"

while [ "$CURRENT_ITER" -lt "$END_ITER" ]; do
    STOP_AT=$(( CURRENT_ITER + CHUNK_SIZE ))
    [ "$STOP_AT" -gt "$END_ITER" ] && STOP_AT="$END_ITER"

    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  TRAINING  iter $CURRENT_ITER → $STOP_AT"
    echo "──────────────────────────────────────────────────────────"

    $PYTHON legged_lab/scripts/train.py \
        --task "$TASK" \
        --num_envs "$NUM_ENVS" \
        --headless \
        --logger tensorboard \
        --predictor \
        --resume True \
        --load_run "$RESUME_RUN" \
        --checkpoint "$RESUME_CKPT" \
        --max_iterations "$CHUNK_SIZE"

    CURRENT_ITER=$STOP_AT
    RESUME_RUN=$(latest_run)
    RESUME_CKPT=$(latest_checkpoint "$RESUME_RUN")

    echo ""
    echo "  ── Stats at iter $CURRENT_ITER ──"
    $PYTHON tools/check_training.py || true
done

echo ""
echo "============================================================"
echo "  Continuation complete. Final model:"
echo "  $LOG_ROOT/$RESUME_RUN/$RESUME_CKPT"
echo "============================================================"
