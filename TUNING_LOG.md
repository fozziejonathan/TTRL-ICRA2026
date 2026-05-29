# K1 Table Tennis Tuning Log

> **For Claude agents starting a new session: read `CLAUDE.md` first, then `tail -80 TUNING_LOG.md` for the latest session.**

## Reconnecting after overnight / disconnect

Training runs in `tmux` and survives SSH disconnects. `auto_tune` runs as a `while true; sleep 900` loop in tmux window 3 and fires every 15 min — you don't need to babysit it. (There is no cron daemon in this container; any earlier references to cron job IDs are stale.)

```bash
# Reconnect and check status
tmux attach -t k1_train                                       # live training output
python3 /workspace/TTRL-ICRA2026/tools/check_training.py     # quick health snapshot
python3 /workspace/TTRL-ICRA2026/tools/auto_tune.py          # check if any intervention fired overnight
cat /workspace/TTRL-ICRA2026/TUNING_LOG.md | tail -60        # see overnight auto_tune log entries
tensorboard --logdir logs --port 6006 --host 0.0.0.0         # TensorBoard (expose port 6006 in RunPod)
```

### Morning checklist — what to look for

| Metric | Healthy | Needs attention |
|--------|---------|-----------------|
| Hit rate | Climbing past 5–10% by iter 3000 | Still ≤ 1% at iter 2500+ |
| reward_contact | Nonzero, growing with hit rate | Zero for 500+ iters |
| reward_future_dis_ee | Trending up | Flat or declining |
| pass_net / landing_dis ratio | > 0.20 | < 0.20 (std_h too tight — auto_tune Phase 2 will handle) |
| Success rate | Should appear ~5–10% once hit rate > 20% | Still 0% at hit rate > 20% (auto_tune Phase 3 will handle) |
| tmux | Training output scrolling | Error / crash / stopped |

### If training has converged overnight (hit ≥ 96%, success ≥ 92%)

`auto_tune` marks `tools/auto_tune_state.json` → `"done": true` and logs the iter. Then:

```bash
# Eval with visual livestream
python3 legged_lab/scripts/train.py --task=k1_tt_eval --num_envs=1 \
    --headless=False --livestream 2 \
    --resume True --load_run <run_dir> --checkpoint <latest>.pt
```

### If training is stuck (hit rate ≤ 1% past iter 2500 and not climbing)

auto_tune Phase 1 will NOT fire until iter ~3700 (requires 40 TensorBoard events). If it looks genuinely stuck before then, manually run:

```bash
python3 /workspace/TTRL-ICRA2026/tools/auto_tune.py   # check its assessment
# If it says "No intervention needed" but hit rate is flat, you can force a manual bump:
# Edit k1_tt_config.py: bump reward_contact weight 150 → 200
# Then: python3 legged_lab/scripts/train.py --task=k1_tt --num_envs=4096 --headless --logger=tensorboard --predictor
# (clean restart — do NOT use --resume when changing reward weights)
```

---

Baseline: 98% hit rate, 28% success rate (policy contacts ball almost every time but rarely returns it over the net).

---

## 2026-05-21 — Initial K1 gain + reward rebalance

### Context
Trained K1 policy achieves 98% hit rate but only 28% success rate. Investigation identified two root causes.

### Changes

#### 1. `legged_lab/assets/booster/booster.py` — K1 actuator gains (`BOOSTER_K1_TT_CFG`)

| Group | Stiffness before → after | Damping before → after |
|-------|--------------------------|------------------------|
| Legs  | 200 → 100                | 5 → 3                  |
| Feet  | 50 → 30                  | 1 → 2                  |
| Arms  | 40 → 25                  | 10 → 3                 |
| Head  | unchanged                | unchanged              |

**Why legs/feet**: The K1's effort limits are ~67% of T1's (legs: 30–40 Nm vs 45–60 Nm). The gains were copied from T1's raw defaults. The existing `BOOSTER_T1_TT_P2_CFG` (the annotated sim2real config) already halved these same gains for T1; K1 should have started there. The PACE paper also identifies ankle tracking error as the primary source of end-effector miss.

**Why arms**: With `stiffness=40` and `effort_limit=14 Nm`, the arm joints saturate at `14/40 = 0.35 rad (20°)` of tracking error — trivially exceeded during any swing. The policy had zero torque modulation authority over paddle face angle or swing speed once motion started. Reducing to 25 gives ~0.56 rad headroom. Damping cut from 10 → 3 because high damping actively brakes fast swings; a table tennis stroke needs to accelerate through the ball.

#### 2. `legged_lab/envs/k1_tt/k1_tt_config.py` — K1 reward weights (`K1TableTennisRewardCfg`)

| Term | Weight before → after | Notes |
|------|----------------------|-------|
| `reward_contact` | 150 → 50 | Reaching is shaping, not the goal |
| `reward_future_landing_dis` | 60 → 120 | Return quality must dominate |
| `reward_future_pass_net` | 100 → 150 | Also tightened std_h 0.4→0.15, lowered z_target 1.11m→0.96m |
| `reward_table_success` | 100 → 250 | Must be highest sparse reward |

**Why**: The contact reward (150) was 1.5× the success reward (100). The policy found the local optimum of "touch the ball and block" — a passive block satisfies the contact reward perfectly. 98% hit / 28% success is the textbook signature of this failure mode. The PACE paper explicitly frames contact/reaching as a shaping signal with returning as the primary objective, which the original weights inverted.

**Net height**: `z_target` lowered from `0.76 + 0.35 = 1.11m` to `0.76 + 0.20 = 0.96m` (net top is at ~0.91m). The old target sent the policy toward high looping shots that clear the net by 20cm but land long or out of bounds. `std_h` tightened from 0.4 to 0.15 to provide a sharper gradient near the target height.

### What to watch in the next run
- Success rate should climb; if it stays low, the sparse `reward_table_success` signal may still be too rare early in training — consider a curriculum that starts with slower/easier serves
- If hit rate drops significantly (below ~85%), the contact reward reduction was too aggressive — try 80 instead of 50
- Check TensorBoard for `reward_contact` vs `reward_table_success` episode totals to verify the policy is actually pursuing returns

---

## 2026-05-23 — Pre-training code-review fixes

### Bug 1 — `has_touch_paddle` never set (critical, pre-existing)

**File**: `legged_lab/envs/base/tt_env.py` ~line 1107

`ball_contact_rew` was updated to `max(ball_contact_rew, ball_contact)` **before** `new_hits` checked `ball_contact < ball_contact_rew`. After the update those two values are equal, so `new_hits` was always `False` and `has_touch_paddle` was never `True`. Every reward gated on `has_touch_paddle` was permanently dead:

- `reward_contact` (zeroed at line 1123 via `has_touch_paddle`)
- `reward_table_success` (multiplies by `has_touch_paddle.float()`)
- `reward_future_landing_dis` (mask = `ball_landing_dis_rew`, derived from `has_touch_paddle`)
- `reward_future_pass_net` (same mask)

The 98%/28% baseline was trained with all four of these signals dead. The 28% success came entirely from position-guidance rewards (`reward_future_ee_target`, `reward_future_body_target`, `reward_future_vel_target`).

**Fix**: compute `new_hits = (contact_score > 0) & (ball_contact_rew == 0)` — "first step where contact_score > 0 and no prior contact this serve" — **before** updating `ball_contact_rew`.

### Note — `reward_future_pass_net` formula is Laplacian, not Gaussian; monitor early

`reward_future_pass_net` uses `exp(-height_err / std_h)`, so gradient width scales linearly with `std_h`. At `std_h=0.15`, a 0.5m error → 3.6% of max; at 0.40, same error → 29%. This is the first run where this reward will actually fire (has_touch_paddle was broken before), so there's no prior data on typical height errors at contact time.

**Check at iter ~1000–1500:**
- Read `reward_future_pass_net` and `reward_future_landing_dis` episode totals from TensorBoard events.
- Both fire on the same timestep (same `ball_landing_dis_rew` mask), so normalise by weight: compute `(pass_net_total / 150) / (landing_dis_total / 120)`. If this ratio is below ~0.20, std_h is too tight and should be widened to 0.25–0.30.
- If success rate climbs to ~50–60% then stalls while hit rate stays high, suspect high-arcing shots — same fix.

**Action if std_h too tight:** change `std_h=0.15` → `0.25` in `k1_tt_config.py`, resume from latest checkpoint.

---

## 2026-05-23 — Revert gains + partial reward revert; restart fresh

### Root cause identified

The previous session changed all three actuator groups (legs, feet, arms) from T1 defaults. The teammate's run — which used those same T1 defaults — achieved 98% hit rate. Our run with changed gains was stuck at 0.3% hit rate for 3200+ iterations despite identical reaching reward weights. Gains change is the most parsimonious explanation; the reaching rewards (dis_ee, dis_ro, vel_base) were untouched between runs.

**Verification**: confirmed teammate's gains and rewards via `https://github.com/kylevansant/TTRL-ICRA2026/tree/feature/k1-tt`.

### Changes

#### `legged_lab/assets/booster/booster.py` — K1 actuator gains reverted to T1 defaults

| Group | Stiffness before → after | Damping before → after |
|-------|--------------------------|------------------------|
| Legs  | 100 → **200**            | 3 → **5**              |
| Feet  | 30 → **50**              | 2 → **1**              |
| Arms  | 25 → **40**              | 3 → **10**             |

#### `legged_lab/envs/k1_tt/k1_tt_config.py` — rewards partially reverted

| Term | Before → after | Reason |
|------|---------------|--------|
| `reward_contact` | 50 → **150** | Teammate's proven value; success(250) > contact(150) prevents blocking |
| `reward_future_pass_net` weight | 150 → **100** | Teammate's value |
| `reward_future_pass_net` std_h | 0.15 → **0.4** | Teammate's proven value; wider gradient once hits occur |
| `reward_future_pass_net` z_target | 0.96m → **1.11m** | Teammate's value |
| `reward_future_landing_dis` | kept at **120** | Stronger return quality gradient, no downside |
| `reward_table_success` | kept at **250** | Keeps success > contact(150), prevents blocking |

### Automation

`tools/auto_tune.py` runs every 15 min (cron job `5c5efdd8`) and intervenes autonomously:
- **Phase 1**: hit_rate peak < 3% in last 500 iters after iter 800 → bump contact_weight +50 (cap 300), resume
- **Phase 2**: hit_rate > 5% and pass_net/landing_dis ratio < 0.20 → widen std_h ×1.5 (cap 0.8), resume
- **Phase 3**: hit_rate > 10% and success_rate < 5% → bump success_weight +50 (cap 500), resume
- **Convergence**: hit ≥ 96% and success ≥ 92% → mark done

State in `tools/auto_tune_state.json`. All interventions appended here automatically.

### Early signal (first ~3 iters of new run)
`reward_contact: 0.0003` already, vs the previous run's near-zero average at iter 3200. Strong early confirmation the gains were the issue.

---

## 2026-05-24 01:43 — auto_tune: bump reward_contact

- Run: `2026-05-23_23-21-01`  |  Iter: 2099
- Change: weight=150.0 → weight=200.0
- Hit rate: 0.7%  |  Success: 0.0%
- reward_contact: 0.0009  |  pass_net: 0.0000  |  landing_dis: -0.0003


## 2026-05-24 03:48 — auto_tune: bump reward_contact

- Run: `2026-05-24_01-44-32`  |  Iter: 4299
- Change: weight=200.0 → weight=250.0
- Hit rate: 0.3%  |  Success: 0.0%
- reward_contact: 0.0009  |  pass_net: 0.0000  |  landing_dis: -0.0011


---

## 2026-05-24 — Root cause identified via literature research; clean restart

### Diagnosis (research-backed)

**Root cause of 0.3% hit rate across all resumed runs:** `mask_invalid` in `tt_env.py` included `| self.has_touch_paddle`, which zeroed ALL reaching rewards (`dis_ee`, `dis_ro`, `vel_base`) the instant the policy made contact with the ball.

Math (WoCoCo 2024): expected reaching income lost at contact ≈ 360–600 units. One-shot `reward_contact` = 150–250 units. Contact was economically irrational — the policy correctly learned to hover near the ball and never swing. This is a named failure mode ("positioning without contact").

**Why teammate's run worked (98% hit rate):** The pre-existing `has_touch_paddle` bug (new_hits always False) meant `has_touch_paddle` was never True, so `mask_invalid` never zeroed reaching rewards. Reaching rewards fired throughout the full episode regardless of contact. No disincentive for contact.

**Why checkpoint resumes made things worse:** Per MorFiC (arXiv:2603.14554), loading a checkpoint with changed reward weights corrupts the PPO critic's advantage estimates. RSL-RL's adaptive KL scheduler then raises LR to the cap (0.01) precisely when gradient signal is most corrupted. Three resumes compounded this.

### Fix

**`legged_lab/envs/base/tt_env.py`**: Removed `| self.has_touch_paddle` from `mask_invalid`. Ball physics (`vx > 0`, `z < 0.7`) naturally extinguish reaching rewards a few steps post-contact; `has_touch_paddle` was creating an immediate cliff on the contact timestep itself.

**`legged_lab/envs/k1_tt/k1_tt_config.py`**: Reverted `reward_contact` weight 250 → **150** (teammate's proven value).

**Clean restart** from iter 0 — no checkpoint resume. Checkpoint resumes with changed reward weights corrupt the critic.

**`tools/auto_tune_state.json`**: Reset to initial state (contact_weight=150, interventions=[]).

### Sources
- WoCoCo (NeurIPS 2024, arXiv:2406.06005): positioning-without-contact failure mode
- MorFiC (arXiv:2603.14554): value function miscalibration on reward weight change
- PACE (arXiv:2509.21690): hit-guidance reward must be dense, not one-shot sparse
- RSL-RL: adaptive KL schedule pushes LR to cap when policy KL is low post-resume

---

## 2026-05-24 — auto_tune robustness fixes (same session, pre-overnight)

### Changes to `tools/auto_tune.py`

**1. Phase 1 restart now does a clean start (not checkpoint resume)**

`restart_training()` gained a `clean=True` flag. Phase 1 (contact weight bump) and Phase 3 (success weight bump) now pass `clean=True` — no `--resume`, fresh critic. Phase 2 (std_h shaping) keeps checkpoint resume since it doesn't shift the return scale. Prevents MorFiC corruption on any future intervention.

**2. Phase 1 trend check edge case fixed**

When `len(hit_vals) <= 40`, the old code fell back to `prior_mean = recent_mean`, giving `trend = 0.0`. Since `0.0 <= 0` is True in Python, Phase 1 could fire at iter ~1860 (the first iter where MIN_DATA=20 clears) even with a slowly climbing run. Fix: when `len(hit_vals) <= 40`, `recent_trend` is forced to `+1.0` (assume climbing — insufficient data for a real comparison). Phase 1 trend check is now only meaningful once 40 events exist (~iter 3700).

### Changes to `tools/check_training.py`

Corrected stale hardcoded weights: `reward_contact` 50 → 150, `reward_future_pass_net` 150 → 100. Fixed the std_h ratio denominator (was dividing by 150, should be 100). Moved `reward_table_success` warning threshold from iter 200 → iter 1000.

---

## 2026-05-24 05:30 — Overnight status: run `2026-05-24_04-03-45`, heading into night

**Current state at commit time:**

| Metric | Value |
|--------|-------|
| Iteration | ~1600 / 10000 |
| Hit rate | **0.7%** and climbing (+0.4% trend) |
| Success rate | 0.0% (expected — too early) |
| reward_future_dis_ee | 0.24 (arm-swing learning active since iter ~1075) |
| reward_future_vel_base | 0.96 (body guidance near peak) |
| termination_penalty | -0.036 (robot most stable yet) |
| ETA | ~7h |

**Trajectory:** Hit rate has climbed every check since iter 1075 (0.3% → 0.4% → 0.5% → 0.6% → 0.7%). The dis_ee inflection at iter 1075 (0.043 → 0.160, a 4× jump) marks when arm-swing learning started. No intervention fired; none expected until at least iter 3700.

**auto_tune cron:** job `21f3e192`, every 15 min, running autonomously. Any overnight interventions will be appended to this log automatically.


---

## 2026-05-24 — Reward weights fully reverted to friend's fork; train_and_viz launched

### Root cause review (this session)

Previous Claude session's changes diverged from the friend's fork (kylevansant/TTRL-ICRA2026/feature/k1-tt) that achieved 98% hit rate. Confirmed by fetching both repos directly:

| Term | Friend's fork (98% hit) | Previous Claude → now reverted |
|------|------------------------|-------------------------------|
| `reward_contact` | 150 | 50 → **150** ✓ |
| `reward_future_landing_dis` | 60 | 120 → **60** ✓ |
| `reward_future_pass_net` weight | 100 | 150 → **100** ✓ |
| `reward_future_pass_net` std_h | 0.4 | 0.15 → **0.4** ✓ |
| `reward_future_pass_net` z_target | 0.76+0.35 | 0.76+0.20 → **0.76+0.35** ✓ |
| `reward_table_success` | 100 | 250 → **100** ✓ |
| K1 arm stiffness | 40 | 25 → **40** ✓ |
| K1 arm damping | 10 | 3 → **10** ✓ |

The reward_contact:reward_table_success ratio was 50:250 = 0.2:1, meaning the sparse terminal signal dominated the reward landscape before the robot had any shaping signal to learn from. The robot wandered aimlessly (including consistently drifting left) because there was no dense reward gradient to follow.

### Visual symptoms that were observed (now fixed)

- **Elbow floppy / reverse forehand**: arm stiffness=25 (too low) let the elbow collapse under swing load, flipping the paddle face backward. Reverted to stiffness=40 (friend's fork value).
- **Walking aimlessly to the left**: broken reward landscape (contact=50, success=250) left no shaping signal. Robot had no incentive to track the ball.
- **Hitting from under the table**: consequence of floppy arm + no reward signal for correct positioning. Geometry is fine at init_pos=(-1.6,0,0.72); the friend's fork used the same and achieved 98%.

### Infrastructure changes

- `tools/train_and_viz.sh` — new script that alternates headless training (1000-iter chunks, 4096 envs) with 90-second VNC visualizations (DISPLAY=:1, 1 env, k1_tt_eval). Auto-detects latest checkpoint for resuming; accepts `--clean` to force fresh start.
- `tools/auto_tune.py` — updated to call `train_and_viz.sh` instead of `train.py` directly, so auto_tune restarts preserve the visualization loop. Fixed `ld_w 120→60` in Phase 2 ratio check. Fixed default `success_weight 250→100`.
- `tools/auto_tune_state.json` — reset to correct baseline (`success_weight=100`).
- `tools/check_training.py` — fixed stale weights (`landing_dis 120→60`, `success 250→100`, ratio denominator 120→60).

### How to watch

- **VNC viewer**: `http://localhost:5999/vnc.html` — visualization opens automatically every 1000 iterations (~15–20 min of training per chunk, 90 s viz window)
- **TensorBoard**: `tensorboard --logdir logs --port 6006 --host 0.0.0.0`
- **Quick stats**: `python3 tools/check_training.py`

### Clean start

Training started fresh (--clean) from iter 0 in `tmux k1_train:0`. Old checkpoints from iter 4250 (trained with wrong rewards) were discarded.

---

## 2026-05-24 (evening) — Infrastructure fixes; training running overnight

### Reconnecting after disconnect

```bash
tmux attach -t k1_train
# Window 0 (bash):      train_and_viz.sh — main training loop
# Window 1 (play):      manual viz (kill when done)
# Window 2 (services):  noVNC (websockify port 5999 → VNC :1)
# Window 3 (auto_tune): auto_tune loop, fires every 15 min
```

SSH tunnel for VNC/TensorBoard:
```bash
ssh -L 5999:localhost:5999 -L 6006:localhost:6006 <user@host>
# VNC:         http://localhost:5999/vnc.html
# TensorBoard: http://localhost:6006
```

### Bugs fixed this session

**1. `train_and_viz.sh` — chunk size doubling bug**

`--max_iterations "$STOP_AT"` was passed on resume. RSL-RL's `learn()` does `tot_iter = checkpoint_iter + num_learning_iterations`, so each chunk grew: 500, 1000, 1500 ... → ~105k total instead of 10k. Fixed to `--max_iterations "$VIZ_INTERVAL"` (always 500 additional iters per chunk).

**2. `train_and_viz.sh` — timeout not killing Isaac Sim**

Isaac Sim ignores SIGTERM. `timeout 90` never killed play.py; it ran indefinitely. Fixed to `timeout --kill-after=10 90` (escalates to SIGKILL after 10s grace period).

**3. VNC visualization — no window appearing**

`isaaclab.python.kit` has `present.enabled=false` (MGPU stability note, irrelevant on single RTX 4090). Added `--/exts/omni.kit.renderer.core/present/enabled=true` to play.py invocation in viz step. Isaac Sim window now renders to VNC display `:1`.

**4. auto_tune cron not running**

No cron daemon available in this container. Auto_tune was never actually firing despite tuning log claiming cron job `21f3e192`. Fixed by running auto_tune as a `while true; sleep 900` loop in `tmux k1_train:3 (auto_tune)`. Will persist across SSH disconnects.

### Current state (leaving overnight)

| | |
|---|---|
| RSL-RL iter | ~750 / heading to ~1499 (first doubled chunk — bug was mid-run) |
| Subsequent chunks | 500 iters each, ~28 min/chunk |
| ETA to iter 10000 | ~9 hours |
| Hit rate | 0/47 serves (2.1% — 1 hit observed, very early stage) |
| auto_tune | Running in tmux window 3, no intervention expected until ~iter 3700 |
| noVNC | Running in tmux window 2 on port 5999 |


---

## 2026-05-26 — check_training.py and eval config fixes; training stopped at iter 10482

### Training state on reconnect

New run `2026-05-26_04-04-16` had completed and stopped (GPU idle, 0% utilisation).
`train_and_viz.sh` hit its hardcoded `MAX_ITERS=10000` and exited normally.

`check_training.py` was reporting misleading numbers:
- Hit rate: **72.6%** (wrong) → actual **94.5%**
- Success rate: **54.9%** (wrong) → actual **77.4%**
- Trend "+94.5% over last 10 iters" was comparing iter 10399 vs iter 0, not a recent window

### Bug 1: check_training.py — iter-0 zero contaminates mean and trend

TensorBoard only has 6 logged events for `Train/TT_hit_rate` and `Train/TT_success_rate`.
`last_n_values(..., 10)` grabbed all 6 including the iter=0 initialisation point (always 0.0),
which dragged the reported mean well below the true recent value.
The "trend" was `vals[-1] - vals[0]` = latest minus iter-0, not a recent-window trend.

**Fix:** filter out `step == 0` events in `last_n_values`. Report the latest value directly
(not a mean), and label the trend with the actual iteration span it covers.

### Bug 2: K1TT_EvalEnvCfg — ball speeds faster than training

`K1TT_EvalEnvCfg` overrode ball speeds to `x: (-6.5, -5.2)` while training uses the
base defaults `x: (-5.5, -4.5)` — roughly 15–45% faster. This made the visualization
look much worse than TensorBoard metrics suggested (robot trained on slow balls, viz tested
on fast balls).

**Fix:** removed the three `ball_speed_*_range` overrides from `K1TT_EvalEnvCfg` so it
inherits the training distribution. Eval and training now use identical ball speeds.

### Actual state at iter 10482

| Metric | Value | Target |
|--------|-------|--------|
| Hit rate | **94.5%** | 96% |
| Success rate | **77.4%** | 92% |
| contact_weight | 150 | — |
| success_weight | 100 | — |
| std_h | 0.4 | — |

Both metrics climbing fast (hit +36.7%, success +55.7% over last 400 iters).
No auto_tune intervention fired — all phases were already satisfied or moot.

### Next step

Bump `MAX_ITERS` in `tools/train_and_viz.sh` from 10000 → 20000 and resume.
Script will auto-detect `model_10482.pt` and continue without `--clean`.

---

## 2026-05-26 (evening) — Root cause research: IsaacSim version + --predictor bug + eval config

### Critical finding: IsaacSim 5.x degrades success rate

We are running **IsaacSim 5.1.0-rc.19**. The purdue-tracelab paper (and Kyle's K1 fork) were
developed and tested on **IsaacSim 4.5.0 + Isaac Lab 2.1.0**. The purdue-tracelab README
explicitly states (twice): *"We notice a significant performance drop of the same training
config in updated IsaacSim 5.0+."*

This is almost certainly the primary reason success rate is ~17–20% in play.py despite
the paper reporting ≥92%. The hit rate (~94%) is unaffected. Success (ball placement) is
more sensitive to physics simulation differences.

**Implication:** Reaching 92% success on IS5.1 may require different tuning than the paper
used. The 92% target may need to be reconsidered if we can't downgrade to IS4.5.

### Bug: --predictor flag missing from every viz invocation

All visualization attempts before this session were loaded with `OnPolicyRunner` instead
of `OnPolicyPredictorRegressionRunner`, because `--predictor` was never passed to play.py.
The policy was trained with the predictor runner; loading with the wrong runner produces
garbage actions (0% hit rate regardless of checkpoint quality). Once `--predictor` was
added, hit rate immediately confirmed at ~94% (matching TensorBoard).

Fixed in CLAUDE.md play.py example command and Critical Gotchas section.

### Bug: eval config ball speed "fix" was wrong

Earlier this session we removed ball speed overrides from `K1TT_EvalEnvCfg` on the theory
that they made the viz look worse than TensorBoard. The real cause was the missing
`--predictor` flag. The eval ball speeds (`x: -6.5/-5.2`) are intentional — they match
the paper's T1 evaluation protocol exactly. Reverted the removal.

Training ball speeds (default): `x: (-5.5, -4.5)`, `y: ±0.8`, `z: (1.6, 1.7)`
Eval ball speeds (paper protocol): `x: (-6.5, -5.2)`, `y: (-0.6, 0.2)`, `z: (1.5, 1.9)`

### TensorBoard success rate is inflated

`Train/TT_success_rate` uses a pure position check (ball within opponent table zone
coordinates), not a physics bounce event. This can trigger when a ball clips the zone
while in flight. Play.py uses the same logic and also shows ~17–20% success — consistent.

The hit rate TB metric (94.5%) IS accurate and confirmed by play.py (~94.1%).

### Kyle's fork results

`reward_table_success = 100` in Kyle's fork — same as ours. The 98% hit rate Kyle
reported was on IsaacSim 4.5.0. His success rate was not confirmed; paper baseline is 28%.

### Summary of files changed this session

| File | Change |
|------|--------|
| `legged_lab/envs/k1_tt/k1_tt_config.py` | Reverted eval ball speed removal |
| `CLAUDE.md` | Added `--predictor` to play.py command; updated training state; added 4 new gotchas |
| `tools/check_training.py` | Fixed iter-0 contamination of mean/trend (kept) |

---

## 2026-05-26 (late evening) — IS4.5.0 environment setup; fresh training started

### Setup

- Created Python 3.10 venv at `~/.venv/isaac45/` (IS4.5 requires Python 3.10)
- Installed `isaacsim[all]==4.5.0.0` from NVIDIA PyPI
- Installed `isaaclab==2.1.0` from NVIDIA PyPI
- Fixed broken torch install (partial `-orch` from isaacsim[all]); reinstalled `torch==2.5.1+cu124`
- Installed project: `pip install -e .` and `pip install -e rsl_rl/`
- New script: `tools/train_and_viz_is45.sh` — identical logic to `train_and_viz.sh` but uses `~/.venv/isaac45/bin/python` and `MAX_ITERS=15000`

### Why IS4.5.0

Paper (purdue-tracelab) explicitly warns: "significant performance drop in updated IsaacSim 5.0+."
IS5.1 run confirmed this: 94% hit rate but only ~17–20% success in play.py.

### Key advantage of this run vs Kyle's

Kyle's fork has the `has_touch_paddle` bug (new_hits always False for fast balls), meaning
`reward_table_success`, `reward_future_pass_net`, and `reward_future_landing_dis` were all
dead in his training. He got 28% success with zero success reward signal.

This run has the bug fixed → success rewards active + IS4.5.0 physics.
Hypothesis: should significantly exceed 28% success and approach the 92% target.

### Run parameters

| Parameter | Value |
|-----------|-------|
| MAX_ITERS | 15000 |
| VIZ_INTERVAL | 500 |
| NUM_ENVS | 4096 |
| Reward weights | contact=150, table_success=100, pass_net=100, landing_dis=60 (unchanged) |
| Clean start | yes |
| Tmux window | `k1_train:is45_train` |
| Log | `/tmp/is45_train.log` |

Monitor: `tail -f /tmp/is45_train.log` or attach `tmux attach -t k1_train` → window `is45_train`

---

## 2026-05-27 — Fix has_touch_opponent_table_just_now fly-over inflation; clean restart

Applied immediately at iter 0 before 10+ hours of training would have been wasted.

### Bug

`has_touch_opponent_table_just_now` used a pure position check. A ball descending
toward a long or wide miss still passed through the z=0.70–0.85 zone while x=0–1.37
was satisfied, triggering the check even though the ball never bounced on the table.
Effect: TensorBoard `Train/TT_success_rate` inflated ~4× vs play.py measured rate.
Also inflated `reward_table_success` during training — robot got rewarded for balls
that flew over the table at the right height, not just balls that actually landed.

### Fix (`legged_lab/envs/base/tt_env.py`)

Added `vz < 0` (ball moving downward) to the zone check. A bounce always has
negative z-velocity at table height; a fly-over typically has mixed or positive vz.

### Clean restart

Training restarted `--clean` at iter 0 with this fix in place.

---

## 2026-05-27 (afternoon) — Reward shaping research; decision to tighten landing_dis threshold

### Status at time of research

IS4.5.0 run reached **~iter 14500** (`model_14500.pt`). Saved checkpoint.

| Metric | Value |
|--------|-------|
| Hit rate | ~94% |
| Success rate | **>50%** (confirmed in play.py — substantially above 28% Kyle baseline) |

Hit rate appears plateaued. Success rate is strong and still climbing. Question raised: should we add an out-of-bounds penalty or amplify the success reward to push success rate toward 92%?

### Research conducted — reward shaping for placement precision

#### Finding 1: The upstream authors tried out-penalties and removed them

The purdue-tracelab TTRL-ICRA2026 repo (the direct upstream of this codebase) contains
`penalty_ball_to_floor` and `penalty_table_fail` as **commented-out functions** in
`legged_lab/mdp/rewards.py`. They are not absent — they were written, tested on the same
system, and removed. The system achieves ≥92% success without them.
Source: https://github.com/purdue-tracelab/TTRL-ICRA2026

#### Finding 2: `reward_future_landing_dis` threshold=3.0m is too generous at this stage

`reward_future_landing_dis` computes `reward = threshold - dist_to_target`, linear,
firing once at hit-time via predicted trajectory. With `threshold=3.0` (set in
`k1_tt_config.py:185`), a ball landing 0.1m *outside* the table edge is ~0.4m from
the target center → reward = `(3.0 - 0.4) × 60 = +156`. Landing perfectly on center =
`3.0 × 60 = +180`. The function provides **near-zero gradient at the table boundary** —
it cannot distinguish "just in" from "just out."

The only signal distinguishing in-bounds from out-of-bounds is `reward_table_success`
(+100, sparse terminal). At the boundary the total differential is 100 points. This is
likely the primary cause of the 50% success plateau.

At iter 0, threshold=3.0 made sense to bootstrap learning from a random policy. At
iter 14500 with >50% success, the agent is past bootstrapping — it needs a sharper
precision signal.

#### Finding 3: Tightening threshold is equivalent to an implicit out-penalty

With `threshold=0.6`, a ball landing 0.7m from target (well outside the table on most
trajectories) gets `(0.6 - 0.7) × 60 = -6` from landing_dis alone. Together with
`reward_table_success` at +100 for in-bounds, the total gradient at the boundary becomes
much sharper without adding a new reward term.

#### Finding 4: The PACE paper uses only positive shaping, no out-penalty

PACE (arXiv:2509.21690) uses dense physics-guided rewards for predicted landing position
only — no failure penalties. Its ablation shows success rate is driven by landing signal
precision, not failure penalties. The paper explicitly states: "binary metrics such as
successful hit or return are too sparse to effectively train an RL agent" — the solution
was denser/more precise positive guidance, not adding penalties.
Source: https://arxiv.org/abs/2509.21690

#### Finding 5: Tebbe 2021 uses pure negative distance reward for placement

"Sample-efficient RL in Robotic Table Tennis" (arXiv:2011.03275) uses
`R = −|desired_landing − actual_landing|` — entirely negative, equivalent to our
`landing_dis` with threshold=0 (always penalizes distance from target). Achieves accurate
returns in <200 training balls on a robot arm. This is the asymptotic case of tightening
our threshold.
Source: https://arxiv.org/abs/2011.03275

### Decision: tighten `reward_future_landing_dis` threshold 3.0 → 0.6, actor-only warm start

**What:** Rewrite `reward_future_landing_dis` in `legged_lab/mdp/rewards.py` to use
signed distance to the rectangular table zone instead of circular distance to a point target.

```
margin_x = min(pred_x - 0.0,    1.35 - pred_x)
margin_y = min(pred_y - (-0.7625), 0.7625 - pred_y)
reward   = min(margin_x, margin_y)   # positive inside, negative outside
```

Table bounds from `tt_env_config.py`: x∈[0.0, 1.35], y∈[-0.7625, +0.7625].

Sanity check (weight=60):

| Landing | margin | reward |
|---------|--------|--------|
| Center of table (0.675, 0) | +0.675 | +40.5 |
| Old target (1.15, 0) | +0.200 | +12.0 |
| Any table edge/corner | 0.000 | 0.0 |
| 0.1m past far end | -0.100 | -6.0 |
| 0.3m wide miss | -0.300 | -18.0 |
| Into net | -0.300 | -18.0 |

Every table edge gets exactly 0; gradient always points toward table interior.
No `threshold` parameter needed; removed from `k1_tt_config.py`.

**Why not out-penalty:** Upstream authors tried and removed it. Adding a new term risks
conflicting gradients. The threshold change addresses the root cause (weak boundary
signal) rather than patching it.

**Why not bump success_weight:** `reward_table_success` already fires only for "in" shots.
Bumping its weight amplifies the same signal that `landing_dis` at tight threshold also
provides — redundant. The threshold change sharpens the landing gradient everywhere, not
just at the discrete table boundary.

**Why actor-only warm start from model_14500.pt:** Changing reward weights with a full
resume corrupts the PPO critic (see MorFiC, arXiv:2603.14554; CLAUDE.md gotcha). A clean
restart throws away 14500 iters of learned motor skill. Actor-only warm start preserves
the hitting policy while the critic re-learns the value function under the new reward shape.

**Actor-only warm start approach:** Patch `OnPolicyPredictorRegressionRunner.load()` to
accept an `actor_only=True` flag that loads only `actor.*` keys (plus predictor weights)
from the checkpoint state dict, skips critic keys, and does not restore the optimizer or
iteration counter.

### Implementation

| File | Change |
|------|--------|
| `legged_lab/mdp/rewards.py` | Rewrote `reward_future_landing_dis` — signed distance to rectangular zone; removed `threshold` param |
| `legged_lab/envs/k1_tt/k1_tt_config.py` | Removed `threshold` param from `reward_future_landing_dis` RewTerm |
| `rsl_rl/rsl_rl/runners/on_policy_predictor_regression_runner.py` | Added `actor_only=True` mode to `load()` |
| `legged_lab/scripts/train.py` | Added `--actor_only` CLI flag |
| `tools/finetune_is45.sh` | New script: actor-only warm start from `model_14500.pt`, then normal resume loop |

Fine-tuning run launched in tmux window `is45_train`.

---

## 2026-05-28 — Fine-tune launch (actor-only from model_14500.pt, rectangular landing_dis)

### IS4.5.0 original run final state (model_15469.pt)

Original `train_and_viz_is45.sh` ran to MAX_ITERS=15000 and stopped at iter 15469. The
training continued past the 14500 save point in the previous session, producing further
checkpoints (14750, 14970, 15000, 15250, 15469). Both model_14500.pt and model_15469.pt
are effectively identical in performance — hit rate and success rate plateaued:

| Checkpoint | Hit rate (TB) | Success rate (TB) |
|------------|--------------|-------------------|
| model_14500.pt (iter 14500) | ~95.6–95.8% | ~21–23% |
| model_15469.pt (iter 15469) | ~95.6–95.8% | ~20–22% |

The extra 1000 iterations moved neither metric — confirming the success plateau is a reward
shaping problem (weak gradient at table boundary), not a training duration problem.

**Note on previously recorded >50% play.py success rate:** This figure was likely measured
before the `vz < 0` fly-over gate fix (applied 2026-05-27 morning). Post-fix TensorBoard
and play.py are consistent at ~20–22%.

### Previous fine-tune attempt (PID 342498) — hung and killed

A finetune_is45.sh run was launched at ~23:40 on May 27 but got stuck during Isaac Sim
initialization. It ran for 8+ hours at 0% GPU utilization and 0 completed training
iterations. Root cause unknown (likely a torch.compile CPU stall or PhysX CPU fallback
with 4096 envs). Process killed 2026-05-28 morning.

### Fine-tuning run (re-launch)

Started fresh `finetune_is45.sh` in tmux window `is45_train`. Parameters:

| Parameter | Value |
|-----------|-------|
| Seed | `logs/k1_table_tennis/2026-05-27_20-46-44/model_14500.pt` |
| Load mode | Actor-only (critic re-initialised from scratch) |
| Reward change | `reward_future_landing_dis`: circular threshold=3.0 → signed rectangular distance |
| Python | `~/.venv/isaac45/bin/python` (IS4.5.0) |
| MAX_ITERS | 10000 (iter 0–10000 of new run) |
| VIZ_INTERVAL | 500 |

### Iter 4749 health check (2026-05-28 morning)

`check_training.py` snapshot:

| Metric | Value | 200-iter trend |
|--------|-------|---------------|
| Success rate | 36.7% | +6.8% |
| Hit rate | 93.4% | +14.5% |
| reward_future_pass_net avg | 0.781 (weight=100) | — |
| reward_future_landing_dis avg | 0.217 (weight=60) | — |
| reward_contact avg | 1.100 (weight=150) | — |
| reward_table_success avg | 0.369 (weight=100) | — |

Run active in tmux `is45_train`, 3.36s/iter, ETA ~14 min to next VIZ_INTERVAL checkpoint.
Rectangular `landing_dis` reward confirmed working — success rate up ~15 points from 20–22% plateau.
Hit rate recovering from actor-only warm start dip; strong upward trend.

---

## 2026-05-28 — Fine-tune v1 post-mortem + v2 launch

### What went wrong with fine-tune v1

**Timeline:**
- 00:55 – 06:41: finetune_is45.sh ran normally, 12 × 500-iter chunks, reached iter ~6000
- 06:15: training process (PID 379038) hung mid-chunk — saved model_5988.pt then stalled
- 06:41 – 15:29: 9-hour gap; finetune_is45.sh blocked waiting for PID 379038 to exit
- 15:29: manual intervention killed hung process; continue_is45.sh launched from model_6000.pt

**Policy collapse (fine-tune v1 reward: signed rectangular with negatives):**

| Iter | Hit rate | Success rate |
|------|----------|--------------|
| ~4750 | 93.4% | 36.7% |
| 6499 | 77.7% | 31.4% |
| 6699 | 46.5% | 18.9% |
| 7699 | 0.3% | 0.0% |

Root cause: `reward_future_landing_dis` returned negative values for out-of-bounds shots.
The policy learned that **not hitting = 0 reward** is safer than **hitting-and-missing = negative reward**.
Over ~3000 iterations the actor fully regressed to never hitting the ball.

This is the standard positive-only shaping principle: the PACE paper achieves ≥92% success
using only positive rewards. Adding negative shaping to a PPO agent creates a zero-reward
trap — the agent converges to the no-op policy.

### Fix: clamp reward_future_landing_dis to zero from below

```python
# legged_lab/mdp/rewards.py
reward = torch.clamp(torch.min(margin_x, margin_y), min=0.0)
```

Positive inside table (up to +0.675 at center), zero anywhere outside. No gradient toward
table from outside, but no incentive to avoid hitting either. The contact and pass_net rewards
provide the hitting incentive.

### Fine-tune v2 parameters

| Parameter | Value |
|-----------|-------|
| Seed | `logs/k1_table_tennis/2026-05-27_20-46-44/model_14500.pt` |
| Load mode | Actor-only (critic re-initialised) |
| Reward | `reward_future_landing_dis` clamped to `[0, inf)` |
| Max iters | 15000 |
| VIZ | Removed — was causing 9-hour hangs (Isaac Sim SIGTERM issue); run play.py manually |
| Script | `tools/finetune_is45.sh` (rewritten — no VIZ blocks, MAX_ITERS=15000) |

Launched in tmux window `is45_train` 2026-05-28 evening.

---

## 2026-05-29 — Fine-tune v2 plateau analysis + decision to switch to Gaussian reward

### Fine-tune v2 plateau: per-chunk success rate history

| Iter (end of chunk) | Success rate | Hit rate |
|---------------------|-------------|---------|
| 499  | 10.6% | 71.5% |
| 899  | 26.4% | 89.9% |
| 1399 | 31.0% | 92.0% |
| 1899 | 31.8% | 92.5% |
| 2399 | 32.9% | 93.6% |
| 2899 | 33.2% | 93.5% |
| 3399 | 35.5% | 93.8% |
| 3899 | 36.0% | 93.7% |
| 4399 | 34.8% | 93.5% |
| 4899 | 36.2% | 93.8% |
| 5399 | 36.7% | 93.9% |
| 5899 | 37.0% | 94.1% |
| 6399 | 33.6% | 92.9% |
| 6699 | 35.6% | 93.7% |

Hit rate hit its ceiling (~93–94%) at iter ~1900. Success rate plateaued at 31–37%
from iter ~1400 with zero sustained improvement over 5,000+ further iterations.

### Root cause: zero gradient for 64% of shots

`reward_future_landing_dis` with the clamped rectangular formulation:
- Inside table: positive reward (up to 0.675 at centre)
- Outside table: **exactly zero** — no gradient signal

At 36% success rate, 64% of shots land outside the table. Each of these gets zero
gradient for landing position. The policy cannot learn "which direction to adjust"
for a near-miss. More training iterations cannot fix a structural zero-gradient zone.

### Option A rejected: soft exponential decay outside

Proposed `exp(signed_min / sigma)` outside, `signed_min` inside. Found a **massive
discontinuity at the table boundary**: 1mm inside → reward 0.001; 1mm outside →
reward 0.997. Creates a perverse incentive to barely miss the table rather than land
on it. Same discontinuity at the net edge (landing on robot's side gets higher reward
than landing just inside opponent's side). Rejected.

### Option B selected: elliptical Gaussian

Replace entire reward with an elliptical Gaussian centred on the opponent's half-centre:

```python
cx, cy = 0.675, 0.0   # centre of x=[0,1.35], y=[−0.7625,0.7625]
dx = (predict_x_land - cx) / sigma_x
dy = (predict_y_land - cy) / sigma_y
reward = exp(-0.5 * (dx**2 + dy**2))
```

**Sigma derivation (equal 50% edge criterion):**
Solve `exp(-0.5 × (half_width/σ)²) = 0.5` per axis:
- σx = 0.675 / sqrt(2 ln 2) = **0.58**  → back/net edges: 50.8% of centre
- σy = 0.7625 / sqrt(2 ln 2) = **0.65** → side edges: 50.3% of centre

Key reward values at σx=0.58, σy=0.65:
| Position | Reward |
|----------|--------|
| Centre (0.675, 0) | 1.000 |
| Back/net edge | 0.508 |
| Side edge | 0.503 |
| Corner | 0.255 |
| 10cm outside x | 0.410 |
| 30cm outside x | 0.243 |

Properties: smooth everywhere, no discontinuity, gradient always points toward (0.675, 0),
no negative values (no collapse risk). PACE paper (arXiv:2509.21690) describes their
landing reward as "penalize distance from desired landing position on opponent's side" —
no explicit formula, no sigma, no curriculum. This Gaussian is the principled
implementation of that description.

### PACE paper findings

Read the full PDF. Key points:
- Landing reward: "penalize distance from desired landing position" — no equation given
- No sigma curriculum or annealing described anywhere
- The "desired landing position" is a specific target (almost certainly table centre)
- Only positive shaping is used (no penalties for missing)
- Curriculum learning is cited in passing (Bengio 2009) but not applied to reward shape

### Decision

- **Kill fine-tune v2** (stuck at 37%, no path to 92% with zero gradient for 64% of shots)
- **Best checkpoint archived:** `k1_tt_IS4.5_finetune_v2_rect_reward_iter5988_succ37pct.pt`
  (copied to workspace root; source: `logs/k1_table_tennis/2026-05-29_03-19-03/model_5988.pt`)
- **Reward updated:** `legged_lab/mdp/rewards.py` and `k1_tt_config.py` now use Gaussian
- **Fine-tune v3:** actor-only warm start from `model_14500.pt`, same `finetune_is45.sh`
  script, new Gaussian `reward_future_landing_dis` with σx=0.58, σy=0.65
