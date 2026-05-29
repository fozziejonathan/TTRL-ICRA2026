#!/bin/bash
# Fine-tuning step 1: actor-only warm start from model_14500.pt (500 iters).
#
# reward_future_landing_dis uses an elliptical Gaussian centred on the
# opponent's half-centre (0.675, 0): sigma_x=0.58, sigma_y=0.65.
# All four table edges receive 50% of centre reward — gradient everywhere,
# no zero-gradient zone outside the table.
#
# Critic is re-initialised from scratch; actor + predictor are loaded from
# the IS4.5.0 base run seed.
#
# After this script completes (~500 iters), run train.py directly to the
# target iteration count — no chunking loop:
#
#   ~/.venv/isaac45/bin/python legged_lab/scripts/train.py \
#       --task k1_tt --num_envs 4096 --headless --logger tensorboard \
#       --predictor --resume True \
#       --load_run <run_from_this_script> --checkpoint <latest_model.pt> \
#       --max_iterations 15000
#
# Usage:
#   bash tools/finetune_is45.sh

set -euo pipefail

cd /workspace/TTRL-ICRA2026

PYTHON="$HOME/.venv/isaac45/bin/python"
TASK="k1_tt"
NUM_ENVS=4096
CHUNK_SIZE=500
LOG_ROOT="logs/k1_table_tennis"

SEED_RUN="2026-05-27_20-46-44"
SEED_CKPT="model_14500.pt"

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

echo "============================================================"
echo "  K1 Table Tennis — IS4.5.0 fine-tune: actor-only warm start"
echo "  Seed:    $LOG_ROOT/$SEED_RUN/$SEED_CKPT"
echo "  Reward:  landing_dis elliptical Gaussian (sigma_x=0.58, sigma_y=0.65)"
echo "  Critic:  re-initialised from scratch"
echo "  Iters:   0 → $CHUNK_SIZE  (actor-only)"
echo "============================================================"
echo ""

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
    --max_iterations "$CHUNK_SIZE" || { ec=$?; [ $ec -eq 137 ] || exit $ec; }

RESUME_RUN=$(latest_run)
RESUME_CKPT=$(latest_checkpoint "$RESUME_RUN")

echo ""
echo "  ── Stats at iter $CHUNK_SIZE ──"
$PYTHON tools/check_training.py || true

echo ""
echo "============================================================"
echo "  Actor-only warm start complete."
echo "  Run the following to continue straight to target iterations:"
echo ""
echo "  $PYTHON legged_lab/scripts/train.py \\"
echo "      --task $TASK --num_envs $NUM_ENVS --headless \\"
echo "      --logger tensorboard --predictor --resume True \\"
echo "      --load_run $RESUME_RUN --checkpoint $RESUME_CKPT \\"
echo "      --max_iterations 15000"
echo "============================================================"
