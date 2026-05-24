"""Quick training health check — run standalone or called by the loop monitor."""
import sys
import os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = Path(__file__).resolve().parents[1] / "logs" / "k1_table_tennis"

def latest_run():
    runs = sorted(LOG_ROOT.iterdir()) if LOG_ROOT.exists() else []
    return runs[-1] if runs else None

def last_n_values(ea, tag, n=10):
    try:
        events = ea.Scalars(tag)
        return [(e.step, e.value) for e in events[-n:]]
    except KeyError:
        return []

def mean(vals):
    return sum(v for _, v in vals) / len(vals) if vals else None

def run():
    run_dir = latest_run()
    if run_dir is None:
        print("No training runs found under logs/k1_table_tennis/")
        return

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 50})
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"Run found ({run_dir.name}) but no scalar events yet — still initialising.")
        return

    # --- Iteration progress ---
    iters = last_n_values(ea, "Train/mean_reward", 1) or last_n_values(ea, "Train/TT_success_rate", 1)
    current_iter = iters[-1][0] if iters else "?"

    print(f"\n{'='*60}")
    print(f"Run: {run_dir.name}   |   Iteration: {current_iter} / 10000")
    print(f"{'='*60}")

    # --- Success / hit rates ---
    for tag, label in [
        ("Train/TT_success_rate", "Success rate"),
        ("Train/TT_hit_rate",     "Hit rate   "),
    ]:
        vals = last_n_values(ea, tag, 10)
        if vals:
            recent = mean(vals)
            trend = vals[-1][1] - vals[0][1]
            print(f"  {label}: {recent:.1%}  (trend last 10 iters: {trend:+.1%})")

    # --- Key reward episode totals ---
    rewards = {
        "reward_future_pass_net":   ("Episode_Reward/reward_future_pass_net",   100),
        "reward_future_landing_dis":("Episode_Reward/reward_future_landing_dis", 60),
        "reward_contact":           ("Episode_Reward/reward_contact",            150),
        "reward_table_success":     ("Episode_Reward/reward_table_success",      100),
    }
    print()
    raw = {}
    for name, (tag, weight) in rewards.items():
        vals = last_n_values(ea, tag, 10)
        avg = mean(vals)
        raw[name] = avg
        if avg is not None:
            print(f"  {name:<30} avg={avg:7.3f}  (weight={weight})")
        else:
            print(f"  {name:<30} NO DATA")

    # --- std_h diagnostic ---
    pn  = raw.get("reward_future_pass_net")
    ld  = raw.get("reward_future_landing_dis")
    ts  = raw.get("reward_table_success")
    print()
    if pn is not None and ld is not None and ld > 0:
        ratio = (pn / 100) / (ld / 60)
        flag = "  *** STD_H MAY BE TOO TIGHT — consider widening ***" if ratio < 0.20 else "  OK"
        print(f"  pass_net/landing_dis (normalised by weight): {ratio:.2f}{flag}")
    elif pn == 0 and ld == 0:
        print("  WARNING: both reward_future_pass_net and reward_future_landing_dis are zero.")
        print("  Check that has_touch_paddle fix is in place and training is past iter ~100.")
    if ts is not None and ts == 0 and current_iter not in ("?",) and int(current_iter) > 1000:
        print("  WARNING: reward_table_success still zero after iter 1000 — success reward may be blocked.")

    print()

if __name__ == "__main__":
    run()
