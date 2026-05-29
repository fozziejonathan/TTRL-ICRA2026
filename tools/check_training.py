"""Quick training health check — run standalone or called by the loop monitor."""
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = Path(__file__).resolve().parents[1] / "logs" / "k1_table_tennis"

TREND_WINDOW = 500  # iters to look back for trend computation


def _load_run_events(run_dir, tag):
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 500})
    ea.Reload()
    try:
        return [(e.step, e.value) for e in ea.Scalars(tag) if e.step > 0]
    except KeyError:
        return []


def collect_chained_events(tag, n=200, max_step_gap=200):
    """Collect last n events for tag across the continuous training chain.

    Walks backward through run dirs. A run is included in the chain only if
    its max step is within max_step_gap of the chain's current min step.
    This correctly bridges chunk boundaries (each continue_is45.sh chunk
    creates a new run dir) without pulling in unrelated prior runs (e.g.
    the IS4.5 base run that ended at step 15469 — well outside the fine-tune
    step range).
    """
    runs = sorted(LOG_ROOT.iterdir()) if LOG_ROOT.exists() else []
    chain = {}        # step -> value; later (more recent) runs win on duplicates
    chain_min = None  # lowest step currently in chain

    for run_dir in reversed(runs):
        events = _load_run_events(run_dir, tag)
        if not events:
            if chain_min is None:
                continue   # still searching for the latest non-empty run
            break          # empty run breaks chain continuity

        run_max = max(s for s, _ in events)
        run_min = min(s for s, _ in events)

        if chain_min is None:
            for s, v in events:
                chain[s] = v
            chain_min = run_min
        elif abs(chain_min - run_max) <= max_step_gap:
            for s, v in events:
                if s not in chain:      # newer run's data takes priority
                    chain[s] = v
            chain_min = min(chain_min, run_min)
        else:
            break  # gap too large — different training run

    sorted_events = sorted(chain.items())
    return sorted_events[-n:]


def mean(vals):
    return sum(v for _, v in vals) / len(vals) if vals else None


def latest_run():
    runs = sorted(LOG_ROOT.iterdir()) if LOG_ROOT.exists() else []
    return runs[-1] if runs else None


def run():
    run_dir = latest_run()
    if run_dir is None:
        print("No training runs found under logs/k1_table_tennis/")
        return

    ea_cur = EventAccumulator(str(run_dir), size_guidance={"scalars": 10})
    ea_cur.Reload()
    if not ea_cur.Tags().get("scalars"):
        print(f"Run found ({run_dir.name}) but no scalar events yet — still initialising.")
        return

    # Current iteration — from the latest run only
    iters = []
    for tag in ("Train/mean_reward", "Train/TT_success_rate"):
        try:
            iters = [(e.step, e.value) for e in ea_cur.Scalars(tag) if e.step > 0]
            if iters:
                break
        except KeyError:
            pass
    current_iter = iters[-1][0] if iters else "?"

    print(f"\n{'='*60}")
    print(f"Run: {run_dir.name}   |   Iteration: {current_iter}")
    print(f"{'='*60}")

    # Success / hit rates — chained across all chunks of this fine-tune run
    for tag, label in [
        ("Train/TT_success_rate", "Success rate"),
        ("Train/TT_hit_rate",     "Hit rate   "),
    ]:
        vals = collect_chained_events(tag)
        if not vals:
            print(f"  {label}: NO DATA")
            continue

        latest_val  = vals[-1][1]
        latest_step = vals[-1][0]

        cutoff = latest_step - TREND_WINDOW
        older  = [(s, v) for s, v in vals if s <= cutoff]
        if older:
            ref_val, ref_step = older[-1][1], older[-1][0]
        else:
            ref_val, ref_step = vals[0][1], vals[0][0]

        span = latest_step - ref_step
        trend_str = f"  (trend over {span} iters: {latest_val - ref_val:+.1%})" if span > 0 else ""
        print(f"  {label}: {latest_val:.1%}{trend_str}")

    # Key reward episode totals — latest run only (no cross-chunk needed)
    rewards = {
        "reward_future_pass_net":    ("Episode_Reward/reward_future_pass_net",    100),
        "reward_future_landing_dis": ("Episode_Reward/reward_future_landing_dis",  60),
        "reward_contact":            ("Episode_Reward/reward_contact",            150),
        "reward_table_success":      ("Episode_Reward/reward_table_success",      100),
    }
    print()
    ea_full = EventAccumulator(str(run_dir), size_guidance={"scalars": 50})
    ea_full.Reload()
    raw = {}
    for name, (tag, weight) in rewards.items():
        try:
            events = [(e.step, e.value) for e in ea_full.Scalars(tag) if e.step > 0]
            avg = mean(events[-10:]) if events else None
        except KeyError:
            avg = None
        raw[name] = avg
        print(f"  {name:<30} avg={avg:7.3f}  (weight={weight})" if avg is not None
              else f"  {name:<30} NO DATA")

    pn = raw.get("reward_future_pass_net")
    ld = raw.get("reward_future_landing_dis")
    ts = raw.get("reward_table_success")
    print()
    if pn is not None and ld is not None and ld > 0:
        ratio = (pn / 100) / (ld / 60)
        flag = "  *** STD_H MAY BE TOO TIGHT — consider widening ***" if ratio < 0.20 else "  OK"
        print(f"  pass_net/landing_dis (normalised by weight): {ratio:.2f}{flag}")
    elif pn == 0 and ld == 0:
        print("  WARNING: both pass_net and landing_dis are zero — check has_touch_paddle fix.")
    if ts is not None and ts == 0 and current_iter not in ("?",) and int(current_iter) > 1000:
        print("  WARNING: reward_table_success still zero after iter 1000.")
    print()


if __name__ == "__main__":
    run()
