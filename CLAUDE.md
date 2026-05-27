# CLAUDE.md — K1 Table Tennis RL (Agent Quick-Start)

## What this is

Training a Booster K1 humanoid robot to play table tennis in Isaac Sim using PPO (RSL-RL).
Branch: `feature/k1-tt`. Goal: ≥96% hit rate, ≥92% success rate.

## Start here

**Read `TUNING_LOG.md`** — authoritative log of every code change, reward weight decision,
bug fix, and training run. The top section has the reconnect checklist and morning health
metrics. The most recent session is always at the bottom.

## Current training state (last updated 2026-05-27)

**IS5.1.0 run (complete, saved):**

| | |
|---|---|
| Run | `2026-05-26_04-04-16` |
| Iteration | 10482 (stopped, saved as `checkpoints/` — see MODELS.md) |
| Hit rate | ~94% (confirmed in play.py) |
| Success rate | ~17–20% in play.py — limited by IS5.x physics |

**IS4.5.0 run (complete — checkpoint saved as `model_14500.pt`):**
| Script | `tools/train_and_viz_is45.sh` |
| Python | `~/.venv/isaac45/bin/python` (IS4.5.0 + Isaac Lab 2.1.0) |
| Iteration | ~14500 |
| Hit rate | ~94% (plateaued) |
| Success rate | **>50%** (confirmed in play.py — well above 28% Kyle baseline) |
| Status | Stopped; checkpoint saved. Next run is a fine-tuning run, not continuation. |

**Next planned run — fine-tuning from model_14500.pt (actor-only warm start):**
| Change | `reward_future_landing_dis` threshold: 3.0 → 0.6 |
| Rationale | See TUNING_LOG 2026-05-27 afternoon session for full research notes |
| Method | Actor-only warm start: load actor + predictor weights from model_14500.pt, fresh critic |

## Tmux session: `k1_train`

| Window | Name | What's running |
|--------|------|----------------|
| 0 | bash | idle (IS5.1 run complete) |
| 1 | play | manual viz — run play.py here when needed |
| 2 | services | noVNC (websockify port 5999 → VNC :1) |
| 3 | auto_tune | `while true; sleep 900; python3 tools/auto_tune.py` loop |
| 4 | is45_setup | IS4.5.0 venv setup (complete) |
| 5 | is45_train | **`train_and_viz_is45.sh --clean`** — IS4.5.0 training, 15k iters |

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
    --task k1_tt_eval --num_envs 1 --predictor \
    --load_run <run_dir> --checkpoint <model_N.pt> \
    --/exts/omni.kit.renderer.core/present/enabled=true
```

## Critical gotchas

- **Always pass `--predictor` to play.py** — the policy was trained with `OnPolicyPredictorRegressionRunner`; omitting `--predictor` loads the wrong runner and produces garbage actions (0% hit rate)
- **Isaac Sim ignores SIGTERM** — always `kill -9` or `timeout --kill-after=10 <dur>`
- **No cron daemon in this container** — auto_tune runs as a tmux sleep loop (window 3), not cron; any log entries referencing cron job IDs are stale
- **Never resume with changed reward weights** — corrupts the PPO critic (see MorFiC, arXiv:2603.14554); always `--clean` restart when changing weights
- **VNC rendering disabled by default** — Isaac Sim needs `--/exts/omni.kit.renderer.core/present/enabled=true` to render on display `:1`
- **train_and_viz.sh chunk size** — pass `--max_iterations $VIZ_INTERVAL`, not `$STOP_AT`; RSL-RL's `learn()` adds to checkpoint iter so passing the absolute target causes exponential chunk growth
- **IsaacSim 5.x degrades success rate** — paper achieved 96%/92% on IsaacSim 4.5.0; we're on 5.1.0 and the authors explicitly warn 5.0+ hurts success rate; hit rate is fine but success plateau may be a simulator issue
- **TensorBoard success rate is inflated** — `Train/TT_success_rate` uses a position-based zone check, not a physics bounce; actual success rate in play.py is ~17–20% vs 77% shown in TB; hit rate (TB vs play.py) is accurate
- **`reward_future_landing_dis` threshold=3.0 is too generous for precision fine-tuning** — with threshold=3.0 and weight=60, a ball landing 0.1m outside the table still gets +156 reward vs +180 for a perfect shot; the boundary gradient is near-zero; tighten to 0.6 for precision runs (see TUNING_LOG 2026-05-27)
- **Do not add an out-of-bounds penalty** — the upstream purdue-tracelab repo has `penalty_ball_to_floor` and `penalty_table_fail` commented out; they were tried on this exact system and removed; the PACE paper achieves ≥92% success with positive shaping only

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
