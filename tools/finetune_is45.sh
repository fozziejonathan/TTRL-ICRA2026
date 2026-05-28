#!/bin/bash
# Fine-tuning run: actor-only warm start from model_14500.pt.
# reward_future_landing_dis uses clamped rectangular distance:
#   positive inside table, ZERO outside (never negative — avoids hitting-avoidance collapse).
# Critic re-initialised; actor + predictor loaded from seed.
# No VIZ between chunks — run play.py manually in window 1 when needed.
#
# Usage:
#   bash tools/finetune_is45.sh

set -euo pipefail

cd /workspace/TTRL-ICRA2026

PYTHON="$HOME/.venv/isaac45/bin/python"
TASK="k1_tt"
NUM_ENVS=4096
MAX_ITERS=15000
CHUNK_SIZE=500
LOG_ROOT="logs/k1_table_tennis"

SEED_RUN="2026-05-27_20-46-44"
SEED_CKPT="model_14500.pt"

echo "============================================================"
echo "  K1 Table Tennis — IS4.5.0 fine-tuning (actor-only warm start)"
echo "  Seed:         $LOG_ROOT/$SEED_RUN/$SEED_CKPT"
echo "  Reward:       landing_dis clamped to [0, inf) — positive inside, 0 outside"
echo "  Critic:       re-initialised from scratch"
echo "  Max iters:    $MAX_ITERS"
echo "============================================================"
echo ""

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

# First chunk: actor-only warm start from seed checkpoint
echo "──────────────────────────────────────────────────────────"
echo "  TRAINING  iter 0 → $CHUNK_SIZE  (actor-only warm start)"
echo "──────────────────────────────────────────────────────────"

$PYTHON legged_lab/scripts/train.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --logger tensorboard \
    --predictor \
    --actor_only \
    --resume True \
    --load_run "$SEED_RUN" \
    --checkpoint "$SEED_CKPT" \
    --max_iterations "$CHUNK_SIZE"

CURRENT_ITER=$CHUNK_SIZE
RESUME_RUN=$(latest_run)
RESUME_CKPT=$(latest_checkpoint "$RESUME_RUN")

echo ""
echo "  ── Stats at iter $CURRENT_ITER ──"
$PYTHON tools/check_training.py || true

# Subsequent chunks: normal resume
while [ "$CURRENT_ITER" -lt "$MAX_ITERS" ]; do
    STOP_AT=$(( CURRENT_ITER + CHUNK_SIZE ))
    [ "$STOP_AT" -gt "$MAX_ITERS" ] && STOP_AT=$MAX_ITERS

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
echo "  Fine-tuning complete at $CURRENT_ITER iterations."
echo "  Final model: $LOG_ROOT/$RESUME_RUN/$RESUME_CKPT"
echo "============================================================"
