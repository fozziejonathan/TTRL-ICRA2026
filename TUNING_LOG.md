# K1 Table Tennis Tuning Log

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
