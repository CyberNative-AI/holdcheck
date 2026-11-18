#!/usr/bin/env python3
"""Does the hold recipe (README), recorded by the owner with their own leader arm and LeRobot's own
teleoperation, give the estimator enough qualifying holds for a figure on all five positioning joints and
a gripper figure? A simulation: the arm is test_holdcheck.sim's play, scatter, bias and 4-frame lag model;
no physical arm or hand was recorded.

Recipe (what the owner is told): move the leader to a new pose, moving each of the five positioning joints
(shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll) a little, gripper left open, then hold it
still for a slow count of three; repeat N times. The simulated hand: each stop moves each positioning joint
by U(lo, hi) degrees in a random direction (kept within +-30 deg of the start), over 0.5-1.5 s, then holds
for U(2.0, 3.0) s with hand jitter. The gripper target does not move (jitter only). Jitter models:
'step' = test_holdcheck's (within one step), 'sd1' = Gaussian sd 1 step on the leader reading,
'drift' = sd 1 step plus a slow wander clipped to +-1 step.
Per cell: five_joint_fits = runs with a play figure on all five positioning joints; gripper_figures = runs
whose card has a gripper figure (holdcheck.tip_figures, as the command line computes it).
  python recipe_sim.py out.json      (about 10 minutes; 5 seeds per cell, deterministic)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holdcheck as hc, test_holdcheck as th

TIP = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
FPS = 30.0

def commands(n_stops, rng, jitter, lo=2.0, hi=6.0):
    """Leader readings (steps) for 6 joints, one continuous episode."""
    cols = [[] for _ in range(6)]
    x = np.full(6, 2048.0)
    for _ in range(n_stops):
        move = int(rng.uniform(0.5, 1.5) * FPS)
        hold = int(rng.uniform(2.0, 3.0) * FPS)
        tgt = x.copy()
        for c in range(5):
            d = rng.choice([-1, 1]) * rng.uniform(lo, hi) / hc.STEP_DEG
            if abs(tgt[c] + d - 2048) > 30 / hc.STEP_DEG:
                d = -d
            tgt[c] += d
        w = np.cumsum(rng.normal(0, 0.15, hold)) if jitter == "drift" else np.zeros(hold)
        w = np.clip(w, -1.0, 1.0)
        for c in range(6):
            cols[c] += list(np.linspace(x[c], tgt[c], move))
            if jitter == "step":
                j = rng.integers(-1, 2, hold) * 0.5
            else:
                j = rng.normal(0, 1.0, hold) + w
            cols[c] += list(tgt[c] + j)
        x = tgt
    return [np.round(np.array(c)) for c in cols]

def follower(cmd, b, sd, bias, rng, lag=4):
    fol = np.empty_like(cmd); p = cmd[0]; settle = 0.0
    for t in range(cmd.size):
        src = cmd[max(0, t - lag)]
        q = min(max(p, src - b / 2), src + b / 2)
        if q != p or t == 0:
            settle = rng.normal(0, sd)
        p = q
        fol[t] = p + bias + settle
    return np.round(fol)

def one(b, sd, n_stops, jitter, seed):
    rng = np.random.default_rng(seed)
    cmds = commands(n_stops, rng, jitter)
    A = np.stack([(c - 2048) * hc.STEP_DEG for c in cmds], 1)
    S = np.stack([(follower(c, b, sd, 1, rng) - 2048) * hc.STEP_DEG for c in cmds], 1)
    A = A.astype(np.float32).astype(np.float64); S = S.astype(np.float32).astype(np.float64)
    EP = np.zeros(A.shape[0], int)
    card = hc.analyse(th.info6(), A, S, EP, np.random.default_rng(0))
    pe, se = [], []
    for k in TIP:
        j = card["joints"][k]
        if "lost_motion_steps" not in j:
            pe.append(np.inf); se.append(np.inf); continue
        pe.append(abs(j["lost_motion_steps"] - b)); se.append(hc.joint_scatter_deg(j) / hc.STEP_DEG)
    five = all(np.isfinite(pe))
    grip = "one_side_p95_mm" in hc.tip_figures(card["joints"])
    return pe, se, five, grip, A.shape[0] / FPS / 60

out = {}
for jitter in ("step", "sd1", "drift"):
    for n in (40, 60, 80):
        for b in (0, 6, 12, 24):
            for sd in (0.5, 1.0, 2.0, 3.0):
                pe, se, wh, gr, mins = [], [], [], [], []
                for seed in range(5):
                    p, s, w, g, m = one(b, sd, n, jitter, seed); pe += p; se += s; wh.append(w); gr.append(g); mins.append(m)
                pe, se = np.array(pe), np.array(se)
                fin = np.isfinite(se)
                row = {"minutes": round(float(np.mean(mins)), 2), "five_joint_fits": f"{sum(wh)}/{len(wh)}",
                       "gripper_figures": f"{sum(gr)}/{len(gr)}",
                       "play_within_1.5": float((pe <= 1.5).mean()),
                       "play_abs_err_p50_p90": [float(np.percentile(pe, 50)), float(np.percentile(pe, 90))],
                       "scatter_read_p10_p50_p90": [round(float(x), 2) for x in np.percentile(se[fin], [10, 50, 90])] if fin.any() else None}
                key = f"{jitter} n={n} b={b} sd={sd}"
                out[key] = row
                print(key, json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
