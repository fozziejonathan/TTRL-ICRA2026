# CLAUDE.md — K1 Table Tennis RL (Agent Quick-Start)

## What this is

Training a Booster K1 humanoid robot to play table tennis in Isaac Sim using PPO (RSL-RL).
Branch: `feature/k1-tt`. Goal: ≥96% hit rate, ≥92% success rate.

## Start here

**Read `TUNING_LOG.md`** — authoritative log of every code change, reward weight decision,
bug fix, and training run. The top section has the reconnect checklist and morning health
metrics. The most recent session is always at the bottom.

## Current training state (last updated 2026-05-26 evening)

| | |
|---|---|
| Run | `2026-05-26_04-04-16` |
| Iteration | 10482 — **training stopped at MAX_ITERS=10000, needs resume** |
| Hit rate | **94.5%** (target 96% — nearly there) |
| Success rate | **77.4%** (target 92% — still climbing) |
| Latest checkpoint | `logs/k1_table_tennis/2026-05-26_04-04-16/model_10482.pt` |

**To resume:** bump `MAX_ITERS` in `tools/train_and_viz.sh` (currently 10000) to 20000, then relaunch `bash tools/train_and_viz.sh` in tmux window 0. Script auto-detects the latest checkpoint.

## Tmux session: `k1_train`

| Window | Name | What's running |
|--------|------|----------------|
| 0 | bash | `train_and_viz.sh` — main loop (500-iter chunks + 90s viz) |
| 1 | play | manual viz — run play.py here when needed |
| 2 | services | noVNC (websockify port 5999 → VNC :1) |
| 3 | auto_tune | `while true; sleep 900; python3 tools/auto_tune.py` loop |

## Key commands

```bash
tmux attach -t k1_train
python3 tools/check_training.py     # quick health snapshot
tail -80 TUNING_LOG.md              # recent session notes
nvidia-smi                          # GPU status (training uses ~6.4 GB / 24.5 GB)
```

## VNC / TensorBoard (requires SSH tunnel)

```bash
ssh -L 5999:localhost:5999 -L 6006:localhost:6006 <user@host>
# VNC:         http://localhost:5999/vnc.html
# TensorBoard: http://localhost:6006
```

To manually launch a visualization (use window 1, kill training first or verify GPU headroom):
```bash
DISPLAY=:1 python3 legged_lab/scripts/play.py \
    --task k1_tt_eval --num_envs 1 \
    --load_run <run_dir> --checkpoint <model_N.pt> \
    --/exts/omni.kit.renderer.core/present/enabled=true
```

## Critical gotchas

- **Isaac Sim ignores SIGTERM** — always `kill -9` or `timeout --kill-after=10 <dur>`
- **No cron daemon in this container** — auto_tune runs as a tmux sleep loop (window 3), not cron; any log entries referencing cron job IDs are stale
- **Never resume with changed reward weights** — corrupts the PPO critic (see MorFiC, arXiv:2603.14554); always `--clean` restart when changing weights
- **VNC rendering disabled by default** — Isaac Sim needs `--/exts/omni.kit.renderer.core/present/enabled=true` to render on display `:1`
- **train_and_viz.sh chunk size** — pass `--max_iterations $VIZ_INTERVAL`, not `$STOP_AT`; RSL-RL's `learn()` adds to checkpoint iter so passing the absolute target causes exponential chunk growth

## auto_tune phases (fires every 15 min, tmux window 3)

| Phase | Trigger | Action |
|-------|---------|--------|
| 1 | hit_rate < 3% after iter ~3700 | bump contact_weight +50 (cap 300), clean restart |
| 2 | hit_rate > 5% and pass_net/landing_dis < 0.20 | widen std_h ×1.5 (cap 0.8), resume |
| 3 | hit_rate > 10% and success_rate < 5% | bump success_weight +50 (cap 500), clean restart |
| done | hit ≥ 96% and success ≥ 92% | marks `tools/auto_tune_state.json` done |

Current auto_tune state: `tools/auto_tune_state.json`

## Key source files

| File | Purpose |
|------|---------|
| `legged_lab/envs/base/tt_env.py` | Core env — rewards, mask_invalid, reset logic |
| `legged_lab/envs/k1_tt/k1_tt_config.py` | K1-specific reward weights and env config |
| `legged_lab/assets/booster/booster.py` | K1 actuator gains (`BOOSTER_K1_TT_CFG`) |
| `tools/train_and_viz.sh` | Training loop script |
| `tools/auto_tune.py` | Autonomous tuning logic |
| `tools/check_training.py` | Quick health snapshot from TensorBoard events |
| `tools/auto_tune_state.json` | auto_tune persistent state (current weights, interventions) |
