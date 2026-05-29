#!/bin/bash
# Watchdog for hung Isaac Sim training processes.
# Isaac Sim ignores SIGTERM and hangs after completing a training chunk,
# blocking continue_is45.sh / finetune_is45.sh for hours.
#
# This script detects the hang by watching the TensorBoard events file:
# if train.py is running but events haven't been written in STALE_SECS,
# it kill -9s the process so the training loop can advance to the next chunk.
#
# Usage (run in a spare tmux window, leave running alongside training):
#   bash tools/watchdog_is45.sh

STALE_SECS=600      # 10 min — ~3× the per-iter budget; definitively stale
CHECK_INTERVAL=60   # check every minute
LOG_ROOT="/workspace/TTRL-ICRA2026/logs/k1_table_tennis"

log() { echo "[watchdog $(date '+%H:%M:%S')] $*"; }

log "Started. Stale threshold: ${STALE_SECS}s, check interval: ${CHECK_INTERVAL}s"

while true; do
    sleep "$CHECK_INTERVAL"

    # Find the running train.py PID
    TRAIN_PID=$(pgrep -f 'python.*train\.py.*k1_tt' | head -1)
    if [ -z "$TRAIN_PID" ]; then
        log "No train.py running — idle"
        continue
    fi

    # Find the most recently modified TensorBoard events file
    EVENTS_FILE=$(find "$LOG_ROOT" -name "events.out.tfevents.*" -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$EVENTS_FILE" ]; then
        log "PID $TRAIN_PID running, no events file yet (Isaac Sim still initializing)"
        continue
    fi

    LAST_MOD=$(stat -c %Y "$EVENTS_FILE")
    NOW=$(date +%s)
    AGE=$(( NOW - LAST_MOD ))

    if [ "$AGE" -gt "$STALE_SECS" ]; then
        log "HUNG: PID $TRAIN_PID — events file stale for ${AGE}s — sending kill -9"
        kill -9 "$TRAIN_PID" 2>/dev/null \
            && log "Killed PID $TRAIN_PID — continue_is45.sh should advance to next chunk" \
            || log "Kill failed (process already gone?)"
    else
        log "PID $TRAIN_PID healthy — events file updated ${AGE}s ago"
    fi
done
