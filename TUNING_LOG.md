# K1 Table Tennis Tuning Log

## Reconnecting after disconnect

If your SSH session drops (e.g. laptop sleeps), training keeps running on the pod. To reconnect:

```bash
tmux attach -t k1_train                                      # see live training output
python3 /workspace/TTRL-ICRA2026/tools/check_training.py    # quick health check
tensorboard --logdir logs --port 6006 --host 0.0.0.0         # start TensorBoard (expose port 6006 in RunPod dashboard)
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
