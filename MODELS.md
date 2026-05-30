# Saved Models

## k1_tt_IS4.5_finetune_v3_gaussian_iter8500_succ37pct.pt — IS4.5.0 fine-tune v3 best

| | |
|---|---|
| Filename | `k1_tt_IS4.5_finetune_v3_gaussian_iter8500_succ37pct.pt` |
| Run | `2026-05-29_15-19-30` |
| Iteration | 8500 |
| IsaacSim version | **4.5.0** |
| Isaac Lab version | 2.1.0 |
| Hit rate | ~92% (TensorBoard) |
| Success rate | **36.8%** (TensorBoard — peak of v3 run) |
| Reward | `landing_dis`: elliptical Gaussian σx=0.58, σy=0.65 |
| Reward weights | contact=150, table_success=100, pass_net=100, landing_dis=60 |
| Notes | Best checkpoint of fine-tune v3. Plateaued 34–37% from iter 2000 onward (8500+ iters); ceiling structural, not reward-shaped. Gaussian provided 2× gradient signal vs v2 rect reward but same ~37% ceiling. Run killed at iter 10750. |

---

## model_14500.pt — IsaacSim 4.5.0 (active best)

| | |
|---|---|
| Filename | `model_14500.pt` |
| Run | `2026-05-27_20-46-44` |
| Iteration | 14500 |
| IsaacSim version | **4.5.0** (paper's version) |
| Isaac Lab version | 2.1.0 |
| Hit rate | **94.5%** (TensorBoard) |
| Success rate | **22.7%** (TensorBoard — accurate; `vz < 0` fly-over fix active) |
| Reward weights | contact=150, table_success=100, pass_net=100, landing_dis=60 |
| Notes | Both bugs fixed: `has_touch_paddle` (success rewards active) + `vz < 0` gate on opponent table contact (no fly-over inflation). Run stopped at iter 14500 of 15000 target. IS4.5.0 matches paper physics. |

---

## model_10482.pt — IsaacSim 5.1.0

| | |
|---|---|
| Filename | `model_10482.pt` |
| Run | `2026-05-26_04-04-16` |
| Iteration | 10482 |
| IsaacSim version | **5.1.0** (paper used 4.5.0) |
| Isaac Lab version | 2.3.2 |
| Hit rate | **~94%** (confirmed in play.py with `--predictor`) |
| Success rate | **~17–20%** in play.py; TensorBoard reports 77.4% but that metric is inflated (position-based zone check, not physics bounce) |
| Reward weights | contact=150, table_success=100, pass_net=100, landing_dis=60 |
| Notes | `has_touch_paddle` bug fixed in this run — success rewards (reward_table_success, reward_future_pass_net) were active during training. Low success rate attributed to IsaacSim 5.x physics regression vs paper's 4.5.0. Retraining on IS4.5.0 in progress. |
