# CLAUDE.md — K1 Table Tennis RL (Agent Quick-Start)

## What this is

Training a Booster K1 humanoid robot to play table tennis in Isaac Sim using PPO (RSL-RL).
Branch: `feature/k1-tt`. Goal: ≥96% hit rate, ≥92% success rate.

## Start here

**Read `TUNING_LOG.md`** — authoritative log of every code change, reward weight decision,
bug fix, and training run. The top section has the reconnect checklist and morning health
metrics. The most recent session is always at the bottom.

## Current training state (last updated 2026-05-29)

**IS5.1.0 run (complete, saved):**

| | |
|---|---|
| Run | `2026-05-26_04-04-16` |
| Iteration | 10482 (stopped, saved as `checkpoints/` — see MODELS.md) |
| Hit rate | ~94% (confirmed in play.py) |
| Success rate | ~17–20% in play.py — limited by IS5.x physics |

**IS4.5.0 base run (complete — best checkpoint `model_14500.pt`, final `model_15469.pt`):**
| Script | `tools/train_and_viz_is45.sh` |
| Python | `~/.venv/isaac45/bin/python` (IS4.5.0 + Isaac Lab 2.1.0) |
| Iteration | 15469 |
| Hit rate | ~95.8% (plateaued) |
| Success rate | ~20–22% — plateaued; root cause: weak gradient at table boundary |
| Status | Complete. Seed for fine-tune. |

**IS4.5.0 fine-tune run v2 (ABANDONED — plateaued at 37% success):**
| Script | `tools/finetune_is45.sh` + `continue_is45.sh` |
| Seed | `logs/k1_table_tennis/2026-05-27_20-46-44/model_14500.pt` |
| Reward change | `reward_future_landing_dis`: clamped rectangular — `max(signed_dist_to_table, 0)` |
| Peak | iter ~5988: 37% success / 94% hit |
| Plateau | iter 1400 → 7500+: stuck 31–37%, zero sustained improvement |
| Root cause | Zero gradient outside table → 64% of shots got no signal |
| Best ckpt | `k1_tt_IS4.5_finetune_v2_rect_reward_iter5988_succ37pct.pt` (workspace root) |
| Status | **Killed 2026-05-29. Seed for fine-tune v3.** |

**IS4.5.0 fine-tune run v3 (PENDING — Gaussian landing reward):**
| Script | `tools/finetune_is45.sh` (reuse — same actor-only warm start logic) |
| Seed | `logs/k1_table_tennis/2026-05-27_20-46-44/model_14500.pt` |
| Reward change | `reward_future_landing_dis`: elliptical Gaussian, σx=0.58, σy=0.65 |
| Status | **Not yet started — ready to launch** |

**Previous fine-tune attempt (ABANDONED — policy collapsed):**
| Reward used | Signed rectangular with negatives — caused hitting-avoidance collapse |
| Peak before collapse | iter ~4750: 93% hit / 36% success |
| Collapse | iter 6000–8000: hit rate 93% → 0.3% as policy learned not-hitting = safer than hitting-and-missing |
| Fix | Clamp reward to 0 from below; restart from model_14500.pt |

## Tmux session: `k1_train`

| Window | Name | What's running |
|--------|------|----------------|
| 0 | bash | idle (IS5.1 run complete) |
| 1 | play | manual viz — run play.py here when needed |
| 2 | services | noVNC (websockify port 5999 → VNC :1) |
| 3 | auto_tune | `while true; sleep 900; python3 tools/auto_tune.py` loop |
| 4 | is45_setup | IS4.5.0 venv setup (complete) |
| 6 | is45_train | idle — fine-tune v2 killed; ready to launch fine-tune v3 |

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
- **`reward_future_landing_dis` is now an elliptical Gaussian** — centred at (0.675, 0) (opponent's half centre), σx=0.58, σy=0.65 (50% reward at all table edges). Smooth everywhere, gradient always points toward centre, no zero-gradient zone. See `legged_lab/mdp/rewards.py`.
- **Never use negative landing_dis reward** — a previous attempt used the raw signed distance (negative outside table). The policy collapsed over ~3000 iters: it learned that not-hitting = 0 reward is safer than hitting-and-missing = negative reward. Hit rate went 93% → 0.3%. Always clamp to zero from below.
- **Do not use a soft exponential decay outside (Option A)** — has a discontinuity at the table boundary: reward jumps from ~0.001 (1mm inside) to ~0.997 (1mm outside). Creates a perverse incentive to barely miss the table. The Gaussian (Option B) avoids this entirely.
- **Rectangular clamped reward plateaued at 37%** — fine-tune v2 used `max(signed_rect_dist, 0)` and stuck at 31–37% success from iter 1400 onward (5000+ iters of zero progress). Root cause: 64% of shots land outside the table and get zero gradient. Gaussian provides gradient everywhere.
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
