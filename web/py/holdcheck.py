#!/usr/bin/env python3
"""holdcheck: measure a low-cost servo arm's lost motion from its own LeRobot recordings.

Where the leader (the command, `action`) holds still, the follower (`observation.state`) should
settle on the command. What it settles on, split by the direction it arrived from, measures the
closed servo loop's lost motion (deadband, friction against a soft P gain, gear play inside the
loop) and its scatter. No hardware is moved; this only reads recordings.

Units. Feetech STS3215 servos (SO-100 / SO-101) read 4096 steps a turn, 0.08789 deg a step.
LeRobot stores joints either in degrees or as -100..100 over each arm's calibrated range. The
follower un-normalises `action` with its own calibration, so `state - action` is the follower's
error in its own frame either way. The step size is recovered from the quantisation of `state`
itself (smallest common difference between recorded values); a joint whose values are not on a
lattice is reported as unit-unknown and left out of the degree figures.

What it cannot see: play between the servo's output shaft and the link (horn screws, printed
parts, link bending). The encoder sits on the output shaft. Figures here are a lower bound on the
lost motion at the tool.

Usage (reads only meta/info.json and data/*.parquet, never videos):
  holdcheck.py --repo lerobot/community_dataset_v3 --subdir <contributor>/<dataset> --out card.json
  holdcheck.py --local <dir with meta/ and data/> --out card.json

The card also carries each joint's play correction; playfix.py applies it to a LeRobot follower.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

VERSION = "0.4.2"
STEP_DEG = 360.0 / 4096.0
HF = "https://huggingface.co"
SO_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LOADED = {"shoulder_lift", "elbow_flex", "wrist_flex"}  # carry gravity torque in most poses

# Hold and approach parameters, in follower steps and frames at the dataset's fps.
HOLD_S = 0.5          # the command must stay within HOLD_BAND for this long
HOLD_BAND = 2.0       # steps; a hand on the leader jitters by about one step
APPROACH_MIN = 12.0   # steps (about 1 deg) of travel that defines the approach direction
APPROACH_S = 2.0      # the approach must end within this long before the hold starts
GRIP_FREE = 20.0      # steps; a gripper error below this means nothing is held in the jaws


def _get(url: str, retries: int = 4) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"holdcheck/{VERSION}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - network retry
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def hf_tree(repo: str, path: str) -> list[dict]:
    q = urllib.parse.quote(path)
    return json.loads(_get(f"{HF}/api/datasets/{repo}/tree/main/{q}?recursive=true"))


def hf_file(repo: str, path: str) -> bytes:
    return _get(f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}")


def info_remote(repo: str, sub: str) -> dict:
    pre = f"{sub}/" if sub else ""
    return json.loads(hf_file(repo, pre + "meta/info.json"))


def info_local(d: str) -> dict:
    return json.load(open(os.path.join(d, "meta", "info.json")))


def load_remote(repo: str, sub: str):
    import io

    import pyarrow.parquet as pq

    pre = f"{sub}/" if sub else ""
    files = sorted(
        x["path"] for x in hf_tree(repo, pre + "data") if x["type"] == "file" and x["path"].endswith(".parquet")
    )
    tables = [pq.read_table(io.BytesIO(hf_file(repo, f)), columns=["action", "observation.state", "episode_index"]) for f in files]
    return tables, files


def load_local(d: str):
    import pyarrow.parquet as pq

    files = []
    for root, _, fs in os.walk(os.path.join(d, "data")):
        files += [os.path.join(root, f) for f in fs if f.endswith(".parquet")]
    files.sort()
    tables = [pq.read_table(f, columns=["action", "observation.state", "episode_index"]) for f in files]
    return tables, files


def to_arrays(tables):
    acts, sts, eps = [], [], []
    for t in tables:
        acts.append(np.asarray(t.column("action").to_pylist(), dtype=np.float64))
        sts.append(np.asarray(t.column("observation.state").to_pylist(), dtype=np.float64))
        eps.append(np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64))
    return np.concatenate(acts), np.concatenate(sts), np.concatenate(eps)


def lattice_step(x: np.ndarray) -> dict:
    """Recover the quantisation step of recorded joint values, or say it is absent."""
    u = np.unique(np.round(x.astype(np.float64), 6))
    if u.size < 20:
        return {"ok": False, "why": f"only {u.size} distinct values"}
    d = np.diff(u)
    d = d[d > 1e-5]
    if d.size < 10:
        return {"ok": False, "why": "too few differences"}
    d0 = np.percentile(d, 2)
    k = np.round(d / d0)
    k[k < 1] = 1
    rel = np.abs(d / k - d0) / d0
    frac = float(np.mean(rel < 0.03))
    if frac < 0.95:
        return {"ok": False, "why": f"values not on a lattice ({frac:.2f} of gaps fit)", "d0": float(d0)}
    step = float(np.median(d / k))
    mode = "degrees" if abs(step - STEP_DEG) / STEP_DEG < 0.01 else ("range_m100_100" if step < 1 else "raw_or_other")
    out = {"ok": True, "step_units": step, "fit": frac, "mode": mode, "deg_per_unit": STEP_DEG / step}
    if mode == "range_m100_100":
        out["calibrated_range_steps"] = 200.0 / step
    if mode == "raw_or_other" and abs(step - 1.0) < 0.01:
        out["mode"] = "raw_steps"
    return out


def holds_for_joint(a: np.ndarray, s: np.ndarray, ep: np.ndarray, fps: float, step: float):
    """One sample per maximal hold: (approach_dir, error_steps at hold end, hold length frames,
    command in steps from the approach point to the hold end, global frame index of the hold end,
    error_steps once the hold had lasted HOLD_S)."""
    W = max(3, int(round(HOLD_S * fps)))
    back = int(round(APPROACH_S * fps))
    out = []
    a_st = a / step
    e_st = (s - a) / step
    n = a.size
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    ends = np.r_[starts[1:], n]
    for b0, b1 in zip(starts, ends):
        x = a_st[b0:b1]
        m = x.size
        if m < W + 2:
            continue
        # running hold: extend a window while the command stays inside the band
        i = 0
        while i < m:
            j = i
            lo = hi = x[i]
            while j + 1 < m:
                v = x[j + 1]
                nlo, nhi = min(lo, v), max(hi, v)
                if nhi - nlo > HOLD_BAND:
                    break
                lo, hi, j = nlo, nhi, j + 1
            L = j - i + 1
            if L >= W and i > 0:
                ref = float(np.median(x[i : j + 1]))
                k = i - 1
                kmin = max(0, i - back)
                while k >= kmin and abs(x[k] - ref) < APPROACH_MIN:
                    k -= 1
                if k >= kmin:
                    d = 1 if ref > x[k] else -1
                    out.append((d, float(e_st[b0 + j]), L, x[k : j + 1].copy(), b0 + j, float(e_st[b0 + i + W - 1])))
            i = j + 1
    return out


def trimmed_mean(v: np.ndarray, cut: float = 0.2) -> float:
    v = np.sort(np.asarray(v, float))
    k = int(cut * v.size)
    return float(v[k : v.size - k].mean())


def robust_sd(v: np.ndarray) -> float:
    if v.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


B_GRID = np.arange(0.0, 60.25, 0.25)  # candidate lost motion, encoder steps
# 0.4.1: a fit or its 90% interval that reaches the top of B_GRID is refitted on B_WIDE (6.6% of the
# corpus cards had a joint pinned at 60 steps while its median gap read 53-187). A fit still at the top
# of B_WIDE (21 deg) is not servo play; the joint is reported as not measured.
B_WIDE = np.arange(0.0, 240.25, 0.25)
# The play correction is given only up to 60 steps (5.3 deg) of play, the range the 0.4 check covered.
# Larger play is a loose horn, screw or gear (elbow p90 is 4.7 deg); aiming far past a target is not a fix.
CORR_MAX_STEPS = 60.0
# 0.4.2: the correction is given only where an encoder step is known to be 1/4096 turn (Feetech STS3215 on
# SO-100/101, Dynamixel on Koch) and, for -100..100 recordings, the calibrated range is at most one turn.
# Elsewhere the step, and so the size of the correction, is not known (Piper's range read 40 turns).
CORR_FAMILIES = ("so100", "so101", "koch", "mixed")
TURN_STEPS = 4096
# A family is placed against other arms only when the corpus has at least this many of them.
COMPARE_MIN_ARMS = 5


def play_predictions(holds, grid: np.ndarray = B_GRID) -> np.ndarray:
    """Predicted settle error (steps) at each hold end for every b in B_GRID: holds x grid.

    A servo loop with lost motion b behaves as a play operator: the output moves only when the
    command leaves the band [out - b/2, out + b/2]. Jitter or a small overshoot of the leader
    inside a hold therefore leaves the output at the arrival-side edge, not at the median command,
    which is why a plain gap of median errors reads short. The operator starts at the approach
    point, where the command was travelling in the approach direction (output trails by b/2).
    """
    half = grid / 2.0
    P = np.empty((len(holds), grid.size))
    for r, h in enumerate(holds):
        d, w = h[0], h[3]
        p = w[0] - d * half
        for v in w[1:]:
            p = np.minimum(np.maximum(p, v - half), v + half)
        P[r] = p - w[-1]
    return P


def fit_play(e: np.ndarray, P: np.ndarray, idx=None, grid: np.ndarray = B_GRID):
    """b minimising the absolute deviation of (observed - predicted) about its median."""
    if idx is not None:
        e, P = e[idx], P[idx]
    R = e[:, None] - P
    cost = np.abs(R - np.median(R, 0)).sum(0)
    g = int(np.argmin(cost))
    return float(grid[g]), R[:, g]


def boot_play(e: np.ndarray, P: np.ndarray, rng, n=400, grid: np.ndarray = B_GRID):
    b = np.empty(n)
    for i in range(n):
        b[i] = fit_play(e, P, rng.integers(0, e.size, e.size), grid)[0]
    return float(np.percentile(b, 5)), float(np.percentile(b, 95))


def family(robot_type) -> str:
    """Arm family from LeRobot's robot_type. SO-100 and SO-101 followers share the STS3215 servo;
    Koch (Dynamixel), Moss, Piper and mixed rigs are other hardware and stay out of the SO table."""
    rt = str(robot_type or "").lower().replace("-", "")
    if "koch" in rt and "so10" in rt:
        return "mixed"
    if rt.startswith("so101"):
        return "so101"
    if rt.startswith("so100") or rt.startswith("so_100"):
        return "so100"
    if rt.startswith("koch"):
        return "koch"
    return "other:" + (rt or "none")


def joint_key(name: str) -> str | None:
    n = name.lower()
    for k in SO_JOINTS:
        if n.endswith(k) or n.endswith(k + ".pos"):
            return k
    return None


def feature_names(info: dict, key: str) -> list:
    """Joint names of one feature in meta/info.json, in the layouts LeRobot recordings use."""
    names = info["features"][key].get("names") or []
    if isinstance(names, dict):
        names = next(iter(names.values()))
    # some recordings store names one level deeper ([[...]]); flatten one level
    if names and isinstance(names[0], list):
        names = [x for n in names for x in n]
    return names


def precheck(info: dict) -> dict | None:
    """The skip decision from meta/info.json alone, before any parquet is read.

    Returns a skip card when the declared features are not one six-joint SO-style arm, else None.
    analyse() repeats the same checks on the data, so a dataset whose files disagree with its
    info.json is still caught there. A skip card from here carries the declared frame and episode
    totals and no file list."""
    names, snames = feature_names(info, "action"), feature_names(info, "observation.state")
    card = {
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "fps": float(info.get("fps", 30)),
        "frames": int(info.get("total_frames") or 0),
        "episodes": int(info.get("total_episodes") or 0),
        "action_names": names,
        "joints": {},
        "files": [],
        "skipped_from": "meta/info.json",
    }
    shp = [list(info["features"][k].get("shape") or []) for k in ("action", "observation.state")]
    if shp[0] != shp[1] or list(names) != list(snames):
        card["skip"] = "action and state differ in shape or names"
        return card
    keys = [joint_key(n) for n in names]
    if (shp[0] and shp[0] != [6]) or len(names) != 6 or sorted(k for k in keys if k) != sorted(SO_JOINTS):
        card["skip"] = "not a single six-joint SO-style arm"
        return card
    return None


def meta() -> dict:
    """The method constants stamped on every card."""
    return {"version": VERSION, "grip_free_steps": GRIP_FREE, "hold_s": HOLD_S, "hold_band_steps": HOLD_BAND,
            "approach_min_steps": APPROACH_MIN, "approach_s": APPROACH_S}


def analyse(info: dict, A: np.ndarray, S: np.ndarray, EP: np.ndarray, rng) -> dict:
    fps = float(info.get("fps", 30))
    names, snames = feature_names(info, "action"), feature_names(info, "observation.state")
    card = {
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "fps": fps,
        "frames": int(A.shape[0]),
        "episodes": int(np.unique(EP).size),
        "action_names": names,
        "joints": {},
    }
    if A.shape != S.shape or list(names) != list(snames):
        card["skip"] = "action and state differ in shape or names"
        return card
    keys = [joint_key(n) for n in names]
    if A.shape[1] != 6 or sorted(k for k in keys if k) != sorted(SO_JOINTS):
        card["skip"] = "not a single six-joint SO-style arm"
        return card
    # Gripper squeeze: with an object in the jaws the follower gripper cannot reach the leader's
    # closed command, so a large gripper error marks a hold where the arm may carry or press on
    # something. Holds with a small gripper error and at least twice HOLD_S are "free" holds.
    gc = keys.index("gripper")
    glat = lattice_step(S[:, gc])
    g_err = np.abs(S[:, gc] - A[:, gc]) / glat["step_units"] if glat["ok"] else None
    for c, k in enumerate(keys):
        lat = lattice_step(S[:, c])
        jr = {"lattice": lat}
        if lat["ok"] and k != "gripper":
            h = holds_for_joint(A[:, c], S[:, c], EP, fps, lat["step_units"])
            d = np.array([x[0] for x in h])
            e = np.array([x[1] for x in h])
            pos, neg = e[d > 0], e[d < 0]
            jr.update({"holds_pos": int(pos.size), "holds_neg": int(neg.size)})
            if pos.size >= 8 and neg.size >= 8:
                naive = float(np.median(neg) - np.median(pos))  # median-gap estimate, reads short under jitter
                P = play_predictions(h)
                gap, r = fit_play(e, P)
                st = rng.bit_generator.state
                lo, hi = boot_play(e, P, rng)
                if hi >= B_GRID[-1]:
                    # same bootstrap draws on the wide grid, so the rng ends where it would have
                    rng.bit_generator.state = st
                    P = play_predictions(h, B_WIDE)
                    gap, r = fit_play(e, P, grid=B_WIDE)
                    lo, hi = boot_play(e, P, rng, grid=B_WIDE)
                    if gap >= B_WIDE[-1]:
                        jr.update({"lost_motion_median_gap_steps": naive,
                                   "why": f"the fit reached {B_WIDE[-1]:.0f} steps ({B_WIDE[-1] * STEP_DEG:.0f} deg), "
                                          "more than servo play; check the calibration and the recording"})
                        card["joints"][k] = jr
                        continue
                sd = float(np.nanmedian([robust_sd(r[d > 0]), robust_sd(r[d < 0])]))
                bias = float(np.median(r))
                jr.update(
                    {
                        "median_err_pos_steps": float(np.median(pos)),
                        "median_err_neg_steps": float(np.median(neg)),
                        "lost_motion_steps": gap,
                        "lost_motion_ci90_steps": [lo, hi],
                        "lost_motion_median_gap_steps": naive,
                        "scatter_sd_steps": sd,
                        "bias_steps": bias,
                        "lost_motion_deg": gap * STEP_DEG,
                        "scatter_sd_deg": sd * STEP_DEG,
                        "bias_deg": bias * STEP_DEG,
                        "scatter_over_lost_motion": (sd / abs(gap)) if abs(gap) >= 1 else None,
                    }
                )
                # scatter source check (0.2): long holds, gripper not squeezing
                L = np.array([x[2] for x in h])
                endi = np.array([x[4] for x in h])
                eW = np.array([x[5] for x in h])
                long_ = L >= 2 * max(3, int(round(HOLD_S * fps)))
                free = long_ & (g_err[endi] < GRIP_FREE if g_err is not None else long_)
                # creep: signed change of the error after the first hold time, in the arrival direction
                # (>0: the output is still closing on the command). A servo that re-settles on each
                # one-step jitter of the leader changes its error at random; that averages out here and
                # shows only in settle_change_abs_steps, which is not a settling measure.
                if long_.sum() >= 8:
                    cr = d[long_] * (e[long_] - eW[long_])
                    jr["settle_creep_steps"] = trimmed_mean(cr)  # whole-step data: a median jumps a step
                    jr["settle_creep_se_steps"] = float(1.2 * robust_sd(cr) / np.sqrt(cr.size))
                    jr["holds_long"] = int(cr.size)
                    jr["settle_creep_deg"] = jr["settle_creep_steps"] * STEP_DEG
                    jr["settle_change_abs_steps"] = float(np.median(np.abs(e[long_] - eW[long_])))
                jr["holds_free_pos"], jr["holds_free_neg"] = int((free & (d > 0)).sum()), int((free & (d < 0)).sum())
                if jr["holds_free_pos"] >= 8 and jr["holds_free_neg"] >= 8:
                    sdf = float(np.nanmedian([robust_sd(r[free & (d > 0)]), robust_sd(r[free & (d < 0)])]))
                    jr["scatter_sd_free_steps"] = sdf
                    jr["scatter_sd_free_deg"] = sdf * STEP_DEG
        card["joints"][k] = jr
    card["correction"] = play_correction(card)
    return card


UNIT_NAMES = {"degrees": "deg", "range_m100_100": "range units (-100..100)", "raw_steps": "steps"}


def correction_skip(card: dict) -> list:
    """Tip joints with a play fit that get no correction, and why: [(joint, sentence)], one line each.

    A family whose encoder step is not known gets one line for the whole arm (joint None)."""
    fam = family(card.get("robot_type"))
    js = card.get("joints") or {}
    fitted = [k for k in SO_JOINTS[:5] if "lost_motion_steps" in js.get(k, {}) and js[k].get("lattice", {}).get("ok")]
    if fam not in CORR_FAMILIES:
        if not fitted:
            return []
        rt = card.get("robot_type") or "none"
        return [(None, f"No correction for this arm: holdcheck's encoder step is that of SO-100/101 and Koch servos "
                       f"(1/{TURN_STEPS} turn), and for robot_type {rt} the step, so the size of a correction, is not known.")]
    out = []
    for k in fitted:
        j, lat = js[k], js[k]["lattice"]
        if lat.get("calibrated_range_steps", 0.0) > TURN_STEPS * 1.01:
            out.append((k, f"No correction for {k}: its recorded -100..100 range spans {lat['calibrated_range_steps']:.0f} "
                           f"encoder steps, more than one servo turn ({TURN_STEPS}). Check that joint's calibration."))
        elif j["lost_motion_steps"] > CORR_MAX_STEPS:
            out.append((k, f"No correction for {k}: its {j['lost_motion_deg']:.1f} deg of play is more than software should "
                           f"cancel (it is given up to {CORR_MAX_STEPS * STEP_DEG:.1f} deg). Check that joint's horn screws and servo."))
    return out


def play_correction(card: dict) -> dict:
    """What playfix.py adds to each command: half the fitted play, in the recording's own action units.

    The closed servo loop settles half its play short of the target, on the side it came from.
    Aiming that far past the target, in the direction the joint is moving, lands it on the target
    from either side. A negative fitted play is taken as none. Joints in correction_skip() get none.
    """
    names = card.get("action_names") or []
    skip = correction_skip(card)
    if any(k is None for k, _ in skip):
        return {}
    skipped = {k for k, _ in skip}
    out = {}
    for k in SO_JOINTS[:5]:
        j = card["joints"].get(k, {})
        lat = j.get("lattice", {})
        if "lost_motion_steps" not in j or not lat.get("ok") or k in skipped:
            continue
        half = max(float(j["lost_motion_steps"]), 0.0) / 2.0
        out[k] = {
            "action_key": next((n for n in names if joint_key(n) == k), f"{k}.pos"),
            "half_play_steps": half,
            "half_play_units": half * float(lat["step_units"]),
            "half_play_deg": half * STEP_DEG,
            "units": UNIT_NAMES.get(lat.get("mode"), lat.get("mode")),
        }
    return out


TIPMAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tipmap.json")
TIP_JOINTS = SO_JOINTS[:5]


def joint_scatter_deg(j: dict) -> float:
    """The scatter that is the arm's: free holds (1 s or more, empty gripper) where measured."""
    return j["scatter_sd_free_deg"] if j.get("scatter_sd_free_deg") is not None else j["scatter_sd_deg"]


def tip_figures(joints: dict, tipmap: dict | None = None, per_pose: int = 200, seed: int = 0) -> dict:
    """Joint figures at the servo shafts -> sideways error of the gripper tip, in mm.

    Each joint settles with Gaussian scatter of its measured sd; arriving from either side it also
    sits at one edge of its play band (half the lost motion), either edge equally often. The joint
    errors go through the SO-101 Jacobian at 64 table poses (`tipmap.json`, made by tipmap.py); the
    figure is the 95th percentile of the sideways (x-y) distance over poses and draws. Seeded, so the
    same card always gives the same figure.
    """
    missing = [k for k in TIP_JOINTS if "lost_motion_deg" not in joints.get(k, {})]
    if missing:
        return {"not_computed": "not measured: " + ", ".join(missing)}
    if tipmap is None:
        with open(TIPMAP) as f:
            tipmap = json.load(f)
    tm = tipmap
    J = np.asarray(tm["jacobian_mm_per_rad"], float)  # (P,3,5)
    sd = np.radians([joint_scatter_deg(joints[k]) for k in TIP_JOINTS])
    hw = np.radians([max(joints[k]["lost_motion_deg"], 0.0) / 2 for k in TIP_JOINTS])
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal((J.shape[0], per_pose, 5)) * sd
    e2 = e1 + rng.choice([-1.0, 1.0], e1.shape) * hw
    lat = [np.hypot(*np.einsum("pij,pmj->ipm", J[:, :2, :], e)) for e in (e1, e2)]
    return {
        "one_side_p95_mm": round(float(np.percentile(lat[0], 95)), 2),
        "one_side_p50_mm": round(float(np.percentile(lat[0], 50)), 2),
        "either_side_p95_mm": round(float(np.percentile(lat[1], 95)), 2),
        "scatter_from": {k: ("free holds" if joints[k].get("scatter_sd_free_deg") is not None else "all holds") for k in TIP_JOINTS},
        "geometry": "SO-101 tool point, pointing down, 64 poses 13-25 cm in front of the base (SO-100 links differ slightly)",
    }


def _place(v: float, p: dict) -> str:
    if not p:
        return ""
    w = ("tighter than 9 in 10" if v < p["p10"] else "tighter than most" if v < p["p25"] else
         "typical" if v <= p["p75"] else "looser than most" if v <= p["p90"] else "looser than 9 in 10")
    return f"{w} (median {p['p50']:.2f})"


def compare_table(summary: dict | None, fam: str) -> dict:
    """The per-joint corpus table a card of this family is placed against; empty when the corpus has
    fewer than COMPARE_MIN_ARMS arms of the family (an arm is not compared with itself)."""
    if not summary:
        return {}
    if fam in ("so100", "so101"):
        return summary.get("joints", {})
    if summary.get("arms_by_family", {}).get(fam, 0) < COMPARE_MIN_ARMS:
        return {}
    return summary.get("by_family", {}).get(fam, {})


def report_text(card: dict, summary: dict | None = None) -> str:
    """The card as a person reads it; with a corpus summary, each joint is placed against it."""
    if "error" in card:
        return f"could not read this dataset: {card['error']}"
    if "skip" in card:
        return f"not measured: {card['skip']}"
    L = [
        f"{card.get('robot_type')}  {card['episodes']} episodes, {card['frames']} frames at {card['fps']:g} fps",
        "Lost motion: how far apart the follower settles when it arrives at the same command from above",
        "and from below. Scatter: how much it varies arriving from one side. Both at the servo's output",
        "shaft (horn, screws and printed links add more at the tool). 1 encoder step = 0.088 deg.",
        "",
        f"{'joint':<14}{'holds':>7}{'lost motion deg':>17}{'90% interval':>16}{'scatter deg':>13}{'bias deg':>10}  vs other arms (lost motion; scatter)",
    ]
    fam = family(card.get("robot_type"))
    table = compare_table(summary, fam)
    for k in SO_JOINTS[:-1]:
        j = card["joints"].get(k, {})
        if "lost_motion_deg" not in j:
            why = j.get("why") or j.get("lattice", {}).get("why") or f"too few holds ({j.get('holds_pos', 0)} up, {j.get('holds_neg', 0)} down)"
            L.append(f"{k:<14}{'':>7}  not measured: {why}")
            continue
        lo, hi = (x * STEP_DEG for x in j["lost_motion_ci90_steps"])
        sc = joint_scatter_deg(j)
        where = ""
        if k in table:
            free = j.get("scatter_sd_free_deg") is not None and table[k].get("scatter_sd_free_deg")
            where = "; ".join(x for x in (_place(j["lost_motion_deg"], table[k]["lost_motion_deg"]),
                                          _place(sc, table[k]["scatter_sd_free_deg" if free else "scatter_sd_deg"])) if x)
        L.append(
            f"{k:<14}{j['holds_pos'] + j['holds_neg']:>7}{j['lost_motion_deg']:>17.2f}{f'{lo:.2f} to {hi:.2f}':>16}"
            f"{sc:>13.2f}{j['bias_deg']:>10.2f}  {where}"
        )
    L.append("Scatter is from holds of 1 s or more with an empty gripper where the recording has enough of them.")
    t = card.get("tip") or {}
    L.append("")
    if "one_side_p95_mm" not in t:
        L.append(f"At the gripper: {t.get('not_computed', 'not computed')}")
        return "\n".join(L + correction_text(card))
    ts = (summary or {}).get("tip", {}) if fam in ("so100", "so101") else {}
    L += [
        "At the gripper (sideways error of the tool point, table poses 13-25 cm in front of the base):",
        f"  arriving from one side      95% within {t['one_side_p95_mm']:.1f} mm   {_place(t['one_side_p95_mm'], ts.get('one_side_p95_mm', {}))}",
        f"  arriving from any side      95% within {t['either_side_p95_mm']:.1f} mm   {_place(t['either_side_p95_mm'], ts.get('either_side_p95_mm', {}))}",
        "A placement whose lead-in or tolerance is smaller than these will miss about 1 time in 20 or more from",
        "this arm's servos alone, before camera, calibration and printed-part play are added. Approaching every",
        *("target from the same side removes the lost-motion part" + same_side_clause(card)).split("\n"),
    ]
    return "\n".join(L + correction_text(card))


def correction_left(card: dict) -> list:
    """Tip joints with play (fitted lost motion above 0) that the card's correction does not cover.

    The one-side gripper figure leaves out every tip joint's play, so a correction reaches it only when
    this list is empty; with a joint left, that joint's play still adds to arrivals from either side."""
    c = card.get("correction") or {}
    js = card.get("joints") or {}
    return [k for k in TIP_JOINTS if js.get(k, {}).get("lost_motion_deg", 0.0) > 0 and k not in c]


def same_side_clause(card: dict) -> str:
    """How report_text's same-side sentence ends: whether the play correction also removes the lost-motion part.
    A newline marks report_text's line break; the page joins it with a space."""
    if not card.get("correction"):
        return "."
    left = correction_left(card)
    if not left:
        return "; so does the play correction below."
    return (f"; the play correction below removes it only on the joints it lists, not on {', '.join(left)}.\n"
            "With that correction, the one-side figure is not established for this arm.")


def correction_withheld(card: dict) -> list:
    """Tip joints (or the whole arm, joint None) with a play fit that get no correction: [(joint, line)].
    correction_text() ends with these lines, one each, in this order."""
    return correction_skip(card)


def correction_text(card: dict) -> list:
    c = card.get("correction") or {}
    w = [line for _, line in correction_withheld(card)]
    if not c:
        return ([""] + w) if w else []
    left = correction_left(card)
    L = [
        "",
        "Play correction (playfix.py in the source repository): aim this far past each target, in the",
        "direction the joint is moving. On the recordings of 122 arms, corrected arrivals from any side",
    ] + ([
        "landed as close as arrivals from one side. We have not tried it on a physical arm. Fit it from a",
        "recent recording of this arm, and again after re-assembly, re-calibration or a servo swap.",
    ] if not left else [
        "landed as close as arrivals from one side. This card corrects only the joints listed here; the",
        f"play of {', '.join(left)} is left, so the one-side figure is not established with this correction.",
        "We have not tried it on a physical arm. Fit it from a recent recording of this arm, and again",
        "after re-assembly, re-calibration or a servo swap.",
    ]) + [
        f"  {'joint':<14}{'add':>8}  {'units':<25}{'deg':>6}",
    ]
    for k, v in c.items():
        L.append(f"  {k:<14}{v['half_play_units']:>8.2f}  {v['units']:<25}{v['half_play_deg']:>6.2f}")
    return L + w


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo")
    p.add_argument("--subdir", default="")
    p.add_argument("--local")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print", action="store_true", help="print the card for a person")
    p.add_argument("--compare", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_summary.json"),
                   help="corpus summary.json to place each joint against (default: the bundled 371-arm table)")
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    try:
        src = {"local": a.local} if a.local else {"repo": a.repo, "subdir": a.subdir}
        info = info_local(a.local) if a.local else info_remote(a.repo, a.subdir)
        card = precheck(info)
        if card is None:
            tables, files = load_local(a.local) if a.local else load_remote(a.repo, a.subdir)
            A, S, EP = to_arrays(tables)
            card = analyse(info, A, S, EP, rng)
            card["files"] = files
            if card.get("joints"):
                card["tip"] = tip_figures(card["joints"])
    except Exception as e:  # noqa: BLE001 - a bad dataset is a result, not a crash
        card, src = {"error": f"{type(e).__name__}: {e}"}, {"repo": a.repo, "subdir": a.subdir, "local": a.local}
    card["source"] = src
    card["holdcheck"] = meta()
    card["seconds"] = round(time.time() - t0, 2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(card, open(a.out, "w"), indent=1)
    if a.print:
        summary = None
        if a.compare and os.path.exists(a.compare):
            with open(a.compare) as f:
                summary = json.load(f)
        print(report_text(card, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
