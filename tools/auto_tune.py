#!/usr/bin/env python3
"""
Autonomous K1 training monitor and tuner.

Called every 15 min by the cron job. Reads TensorBoard metrics, decides whether
to intervene, and if so edits k1_tt_config.py, kills the current training in tmux,
and restarts (resuming from the latest checkpoint for reward-only changes).

Intervention tree
-----------------
Phase 1 — get hits:
  After iter 800, if peak hit_rate in last 500 iters < 3%:
      bump reward_contact weight by 50 (cap 300); resume checkpoint.

Phase 2 — pass-net signal:
  If avg hit_rate > 5% and pass_net/landing_dis ratio < 0.20 and landing_dis fires:
      multiply std_h by 1.5 (cap 0.8); resume checkpoint.

Phase 3 — success stuck:
  If avg hit_rate > 10% and avg success_rate < 5% for 500+ iters:
      bump reward_table_success by 50 (cap 500); resume checkpoint.

Convergence:
  hit_rate >= 96% AND success_rate >= 92% over last 100 iters → mark done.

State is persisted in tools/auto_tune_state.json.
"""

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WORKSPACE  = Path("/workspace/TTRL-ICRA2026")
LOG_ROOT   = WORKSPACE / "logs" / "k1_table_tennis"
K1_CFG     = WORKSPACE / "legged_lab/envs/k1_tt/k1_tt_config.py"
TUNING_LOG = WORKSPACE / "TUNING_LOG.md"
STATE_FILE = WORKSPACE / "tools" / "auto_tune_state.json"

BASE_TRAIN_CMD = "cd /workspace/TTRL-ICRA2026 && bash tools/continue_is45.sh"
BASE_TRAIN_CMD_CLEAN = "cd /workspace/TTRL-ICRA2026 && bash tools/finetune_is45.sh"

# ── helpers ────────────────────────────────────────────────────────────────────

def latest_run():
    runs = sorted(LOG_ROOT.iterdir()) if LOG_ROOT.exists() else []
    return runs[-1] if runs else None

def latest_checkpoint(run_dir):
    pts = sorted(
        [f for f in run_dir.iterdir() if f.name.startswith("model_") and f.suffix == ".pt"],
        key=lambda f: int(f.stem.split("_")[1]),
    )
    return pts[-1] if pts else None

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "contact_weight": 150.0,
        "success_weight": 100.0,
        "std_h": 0.4,
        "last_intervention_iter": 0,
        "last_intervention_run": "",
        "interventions": [],
        "done": False,
    }

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def get_all(ea, tag):
    try:
        return [(e.step, e.value) for e in ea.Scalars(tag)]
    except KeyError:
        return []

def mean_last(vals, n=10):
    if not vals:
        return None
    subset = vals[-n:]
    return sum(v for _, v in subset) / len(subset)

def peak_in_window(vals, current_iter, window=500):
    cutoff = current_iter - window
    recent = [v for s, v in vals if s >= cutoff]
    return max(recent) if recent else 0.0

def iters_since_last_intervention(state, current_iter, current_run):
    if state["last_intervention_run"] != current_run:
        return current_iter  # new run → all iters are "since last"
    return current_iter - state["last_intervention_iter"]

# ── config editing ─────────────────────────────────────────────────────────────

def set_contact_weight(new_w):
    text = K1_CFG.read_text()
    text = re.sub(
        r'(reward_contact\s*=\s*RewTerm\(\s*func=mdp\.reward_contact,\s*weight=)[\d.]+',
        rf'\g<1>{new_w}',
        text,
    )
    K1_CFG.write_text(text)

def set_success_weight(new_w):
    text = K1_CFG.read_text()
    text = re.sub(
        r'(reward_table_success\s*=\s*RewTerm\(\s*func=mdp\.reward_table_success,\s*weight=)[\d.]+',
        rf'\g<1>{new_w}',
        text,
    )
    K1_CFG.write_text(text)

def set_std_h(new_std):
    text = K1_CFG.read_text()
    text = re.sub(r'("std_h":\s*)[\d.]+', rf'\g<1>{new_std:.3f}', text)
    K1_CFG.write_text(text)

# ── training control ───────────────────────────────────────────────────────────

def restart_training(run_dir=None, clean=False):
    """Kill the IS4.5 training loop and restart it.

    clean=True  → run finetune_is45.sh (actor-only warm start from model_14500.pt).
                  Use when reward weights changed (corrupts critic on resume).
    clean=False → run continue_is45.sh (resumes from latest checkpoint).
                  Safe for shaping-only changes that don't corrupt the critic.
    """
    cmd = BASE_TRAIN_CMD_CLEAN if clean else BASE_TRAIN_CMD

    subprocess.run(["tmux", "send-keys", "-t", "k1_train:is45_train", "C-c", ""], check=False)
    time.sleep(8)
    subprocess.run(["tmux", "send-keys", "-t", "k1_train:is45_train", cmd, "Enter"], check=False)
    mode = "CLEAN" if clean else "RESUME"
    print(f"[auto_tune] Restarted training ({mode}): {cmd}")

# ── logging ────────────────────────────────────────────────────────────────────

def log_intervention(state, kind, before, after, iter_, run_name, metrics):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = (
        f"\n## {ts} — auto_tune: {kind}\n\n"
        f"- Run: `{run_name}`  |  Iter: {iter_}\n"
        f"- Change: {before} → {after}\n"
        f"- Hit rate: {metrics['hit']:.1%}  |  Success: {metrics['success']:.1%}\n"
        f"- reward_contact: {metrics['contact']:.4f}  |  "
        f"pass_net: {metrics['pass_net']:.4f}  |  "
        f"landing_dis: {metrics['landing_dis']:.4f}\n\n"
    )
    with open(TUNING_LOG, "a") as f:
        f.write(entry)
    state["interventions"].append({
        "ts": ts, "kind": kind, "before": before, "after": after,
        "iter": iter_, "run": run_name,
    })
    state["last_intervention_iter"] = iter_
    state["last_intervention_run"] = run_name

# ── main ───────────────────────────────────────────────────────────────────────

def run():
    state = load_state()

    if state.get("done"):
        print("[auto_tune] Training previously marked complete — nothing to do.")
        return

    run_dir = latest_run()
    if run_dir is None:
        print("[auto_tune] No training run found under logs/k1_table_tennis/")
        return

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 2000})
    ea.Reload()
    if not ea.Tags().get("scalars"):
        print(f"[auto_tune] Run {run_dir.name}: no scalar events yet.")
        return

    # ── read metrics ──────────────────────────────────────────────────────────
    hit_vals     = get_all(ea, "Train/TT_hit_rate")
    succ_vals    = get_all(ea, "Train/TT_success_rate")
    contact_vals = get_all(ea, "Episode_Reward/reward_contact")
    pn_vals      = get_all(ea, "Episode_Reward/reward_future_pass_net")
    ld_vals      = get_all(ea, "Episode_Reward/reward_future_landing_dis")
    ts_vals      = get_all(ea, "Episode_Reward/reward_table_success")

    current_iter = hit_vals[-1][0] if hit_vals else 0

    # Display averages (10-iter) — for printing only, not used in intervention logic.
    avg_hit      = mean_last(hit_vals,  10) or 0.0
    avg_succ     = mean_last(succ_vals, 10) or 0.0
    avg_contact  = mean_last(contact_vals, 10) or 0.0
    avg_pn       = mean_last(pn_vals, 10) or 0.0
    avg_ld       = mean_last(ld_vals, 10) or 0.0
    avg_ts       = mean_last(ts_vals, 10) or 0.0

    # Smoothed averages (30-iter) — used for ALL intervention threshold decisions.
    # 10-iter means are too noisy at low hit rates: a single bad batch can swing
    # them by ±2%, causing premature intervention.
    smooth_hit  = mean_last(hit_vals,  30) or 0.0
    smooth_succ = mean_last(succ_vals, 30) or 0.0
    smooth_pn   = mean_last(pn_vals,   30) or 0.0
    smooth_ld   = mean_last(ld_vals,   30) or 0.0

    metrics = {
        "hit": avg_hit, "success": avg_succ, "contact": avg_contact,
        "pass_net": avg_pn, "landing_dis": avg_ld, "table_success": avg_ts,
    }

    run_name  = run_dir.name
    gap_iters = iters_since_last_intervention(state, current_iter, run_name)

    # ── print report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[auto_tune] Run: {run_name}  |  Iter: {current_iter}")
    if len(hit_vals) >= 10:
        _r20 = mean_last(hit_vals, 20) or 0.0
        _p20 = mean_last(hit_vals[:-20], 20) or 0.0 if len(hit_vals) > 40 else None
        _trend_str = f"{_r20 - _p20:+.1%}" if _p20 is not None else "n/a (<40 events)"
        print(f"  Hit rate:  {avg_hit:.1%}  (sustained trend last 20 vs prior 20: {_trend_str})")
    else:
        print(f"  Hit rate:  {avg_hit:.1%}")
    print(f"  Success:   {avg_succ:.1%}")
    print(f"  contact_w={state['contact_weight']}  std_h={state['std_h']}  "
          f"success_w={state['success_weight']}")
    print(f"  reward_contact={avg_contact:.4f}  pass_net={avg_pn:.4f}  "
          f"landing_dis={avg_ld:.4f}  table_success={avg_ts:.4f}")
    print(f"  Iters since last intervention: {gap_iters}")

    # ── convergence check ─────────────────────────────────────────────────────
    if current_iter >= 200:
        hit_100  = mean_last(hit_vals,  100) or 0.0
        succ_100 = mean_last(succ_vals, 100) or 0.0
        if hit_100 >= 0.96 and succ_100 >= 0.92:
            state["done"] = True
            save_state(state)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(TUNING_LOG, "a") as f:
                f.write(
                    f"\n## {ts} — TRAINING CONVERGED\n\n"
                    f"Hit rate: {hit_100:.1%}  |  Success: {succ_100:.1%}\n"
                    f"Run: `{run_name}`  |  Iter: {current_iter}\n\n"
                    f"Next step: eval with `--livestream 2`\n"
                )
            print(f"\n*** CONVERGED — hit {hit_100:.1%}, success {succ_100:.1%} ***")
            return

    # Require at least 50 data points before any intervention — avoids acting on
    # a handful of noisy events at the very start of a new run.
    MIN_DATA = 20
    MIN_GAP  = 600  # minimum iters between any two interventions (raised from 400;
                    # after a restart the new run needs runway before we touch it again)

    if len(hit_vals) < MIN_DATA:
        print(f"\n[auto_tune] Only {len(hit_vals)} hit_rate events — need {MIN_DATA} before intervening.")
        save_state(state)
        return

    # ── Phase 1: hit rate stuck ───────────────────────────────────────────────
    # Use mean-of-last-20 vs mean-of-prior-20 for trend (not two individual points).
    # Only fire if the smoothed 30-iter hit rate is also below the peak threshold,
    # so a momentary dip in either window can't trigger alone.
    # Require >= 40 events so both the "recent 20" and "prior 20" windows are
    # populated independently; fewer events → prior_mean falls back to recent_mean
    # giving trend=0.0 which satisfies <=0 and can trigger a false intervention.
    recent_mean  = mean_last(hit_vals, 20) or 0.0
    if len(hit_vals) > 40:
        prior_mean   = mean_last(hit_vals[:-20], 20) or 0.0
        recent_trend = recent_mean - prior_mean
    else:
        # Not enough events for a real prior window — treat as positive trend so
        # Phase 1 can't fire on a false 0.0 tie from the prior_mean fallback.
        recent_trend = 1.0
    if current_iter >= 1500 and gap_iters >= MIN_GAP and recent_trend <= 0:
        peak = peak_in_window(hit_vals, current_iter, window=500)
        if peak < 0.03 and smooth_hit < 0.03 and state["contact_weight"] < 300:
            new_w = min(state["contact_weight"] + 50, 300)
            old_w = state["contact_weight"]
            set_contact_weight(new_w)
            state["contact_weight"] = new_w
            log_intervention(state, "bump reward_contact",
                             f"weight={old_w}", f"weight={new_w}",
                             current_iter, run_name, metrics)
            save_state(state)
            print(f"\n[auto_tune] INTERVENTION: reward_contact {old_w} → {new_w} "
                  f"(peak={peak:.1%}, smooth_hit={smooth_hit:.1%}, trend={recent_trend:+.1%})")
            restart_training(clean=True)  # reward weight change → fresh critic
            return

    # ── Phase 2: pass-net gradient too narrow ────────────────────────────────
    # Use smoothed 30-iter hit/pn/ld averages — 10-iter averages near the 5%
    # threshold are too noisy to distinguish signal from variance.
    if smooth_hit >= 0.05 and gap_iters >= MIN_GAP:
        if smooth_ld > 1e-4:  # landing_dis is actually firing (30-iter smoothed)
            pn_w  = 100.0
            ld_w  = 60.0
            ratio = (smooth_pn / pn_w) / (smooth_ld / ld_w) if smooth_ld > 0 else 0
            if ratio < 0.20 and state["std_h"] < 0.8:
                new_std = min(round(state["std_h"] * 1.5, 3), 0.8)
                old_std = state["std_h"]
                set_std_h(new_std)
                state["std_h"] = new_std
                log_intervention(state, "widen std_h",
                                 f"std_h={old_std}", f"std_h={new_std}",
                                 current_iter, run_name, metrics)
                save_state(state)
                print(f"\n[auto_tune] INTERVENTION: std_h {old_std} → {new_std} "
                      f"(pass_net/landing_dis ratio: {ratio:.2f}, smooth_hit={smooth_hit:.1%})")
                restart_training(run_dir)  # shaping-only change; resume keeps critic valid
                return

    # ── Phase 3: success stuck despite good hit rate ─────────────────────────
    # smooth_hit (30-iter) guards the entry threshold; 100-iter window for
    # success rate to confirm it's a genuine plateau, not a transient dip.
    if smooth_hit >= 0.10 and gap_iters >= MIN_GAP:
        avg_succ_100 = mean_last(succ_vals, 100) or 0.0
        if avg_succ_100 < 0.05 and state["success_weight"] < 500:
            new_w = min(state["success_weight"] + 50, 500)
            old_w = state["success_weight"]
            set_success_weight(new_w)
            state["success_weight"] = new_w
            log_intervention(state, "bump reward_table_success",
                             f"weight={old_w}", f"weight={new_w}",
                             current_iter, run_name, metrics)
            save_state(state)
            print(f"\n[auto_tune] INTERVENTION: reward_table_success {old_w} → {new_w} "
                  f"(success_rate 100-iter avg: {avg_succ_100:.1%}, smooth_hit={smooth_hit:.1%})")
            restart_training(clean=True)  # reward weight change → fresh critic
            return

    print("\n[auto_tune] No intervention needed.")
    save_state(state)


if __name__ == "__main__":
    run()
