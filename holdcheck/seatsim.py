#!/usr/bin/env python3
"""Run the simulated pick-and-place cell (`seatcell.py`) for one arm, or redo the corpus runs behind
`seatmap.json` and rebuild the table. Numpy only. Simulation of a stated model, not a measurement.

Your own arm, from the card the page lets you download (SO-100/101, five tip joints measured):
    python seatsim.py arm card.json                       # 3 and 5 mm, camera 0.5 mm, 10 units x 500 placements
    python seatsim.py arm card.json --lead-in 2 3 4 5 8 --camera-mm 0.3 --grasp-capture-mm 4

The median-joint and p90-joint arms from the shipped corpus table (seconds; compares with METHOD.md):
    python seatsim.py anchor

The table itself, from your own corpus run (`corpus.py run ... --cards runs/cards`, see README.md):
    python seatsim.py corpus --cards runs/cards --out runs/seat.jsonl     # about 70 CPU-minutes, resumable
    python seatsim.py map --runs runs/seat.jsonl --out runs/seatmap.json
    cmp runs/seatmap.json seatmap.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seatcell as sc  # noqa: E402

SUMMARY = os.path.join(HERE, "corpus_summary.json")
TIPMAP = os.path.join(HERE, "tipmap.json")
def _load(path):
    with open(path) as f:
        return json.load(f)


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def run_unit(seed, label, half_width_deg, scatter_sd_deg, lead_in_mm=(2.0, 3.0, 5.0), camera_mm=0.5,
             placements=500, task=sc.Task()):
    """One simulated unit (arm build + cell) of `placements` transfers; seats counted per lead-in.
    The random stream is seeded by (seed, 9002, crc32(label)), so a label and a seed give one result."""
    u = sc.Unit(seed, half_width_deg, scatter_sd_deg)
    cam = sc.Cam(camera_mm)
    rng = np.random.default_rng([seed, 9002, zlib.crc32(label.encode())])
    cal = sc.calibrate(u, cam, rng, k=12)
    rec = sc.run_transfers(u, cal, cam, placements, rng, task)
    return {c: int(sc.success_at(rec, c).sum()) for c in lead_in_mm}


def run_arm(job):
    """`units` units of one arm (unit seeds 2000, 2001, ...): seats and placements per lead-in."""
    label, hw, sd, units, lead_in, camera_mm, placements, task = job
    tot = {c: 0 for c in lead_in}
    per_unit = []
    for seed in range(2000, 2000 + units):
        r = run_unit(seed, label, hw, sd, lead_in, camera_mm, placements, task)
        per_unit.append(r)
        for c in tot:
            tot[c] += r[c]
    return label, tot, units * placements, per_unit


def card_params(card):
    """A card's five tip joints -> the cell's inputs: play half-width (half the lost motion) and settle
    scatter (free holds where measured), degrees, rounded to 4 decimals as in the corpus runs."""
    import holdcheck as hc
    if hc.family(card.get("robot_type")) not in hc.SO_FAMILIES:
        sys.exit("the cell simulates an SO-101 arm; this card is not an SO-100 or SO-101 arm")
    J = card.get("joints", {})
    miss = [k for k in JOINTS if "lost_motion_deg" not in J.get(k, {})]
    if miss:
        sys.exit("not measured on this card: " + ", ".join(miss))
    return ([round(max(J[k]["lost_motion_deg"], 0.0) / 2, 4) for k in JOINTS],
            [round(hc.joint_scatter_deg(J[k]), 4) for k in JOINTS])


def one_side_within(scatter_sd_deg, lead_in_mm, tipmap):
    """The card's predictor: share of one-side tip errors (scatter only, through the SO-101 Jacobian at the
    64 table poses) within each lead-in. Same draws as `tip_figures` in holdcheck.py, unrounded."""
    J = np.asarray(tipmap["jacobian_mm_per_rad"], float)
    e1 = np.random.default_rng(0).standard_normal((J.shape[0], 200, 5)) * np.radians(scatter_sd_deg)
    lat = np.hypot(*np.einsum("pij,pmj->ipm", J[:, :2, :], e1))
    return {c: float((lat <= c).mean()) for c in lead_in_mm}


def cmd_arm(a):
    card = _load(a.card)
    hw, sd = card_params(card)
    lead_in = tuple(float(c) for c in a.lead_in)
    task = sc.Task(grasp_capture_mm=a.grasp_capture_mm, tilt_max_deg=a.tilt_max_deg)
    label = a.label or "arm"
    _, tot, n, per_unit = run_arm((label, hw, sd, a.units, lead_in, a.camera_mm, a.placements, task))
    print(f"this arm: play half-width {hw} deg, settle scatter {sd} deg (base to wrist roll); random stream '{label}'")
    print(f"cell: camera {a.camera_mm} mm, nest capture {a.grasp_capture_mm} mm, tilt limit {a.tilt_max_deg} deg, "
          f"{a.units} simulated units x {a.placements} placements")
    for c in lead_in:
        u = [r[c] / a.placements for r in per_unit]
        print(f"  lead-in {c:g} mm: simulated seat success {tot[c] / n:.1%} (units {min(u):.1%} to {max(u):.1%})")
    print("Simulation of a stated model with this card's figures; no physical cell was run.")


CORPUS_UNITS = 10  # unit seeds 2000-2009, 500 placements each, camera 0.5 mm, lead-ins 2, 3 and 5 mm


def _job(label, hw, sd, units=CORPUS_UNITS):
    return (label, hw, sd, units, (2.0, 3.0, 5.0), 0.5, 500, sc.Task())


ANCHOR_RESULT = {"p50": (0.638, 0.8688, 0.9912), "p90": (0.234, 0.4128, 0.6782)}  # seat success at 2, 3, 5 mm


def anchor_params(summary, q):
    """The arm with every joint at the corpus quantile q (free-hold scatter): not a real arm, a reference."""
    J = summary["joints"]
    return ([round(max(J[k]["lost_motion_deg"][q], 0.0) / 2, 4) for k in JOINTS],
            [round(J[k]["scatter_sd_free_deg"][q], 4) for k in JOINTS])


def cmd_anchor(a):
    summary = _load(a.summary)
    jobs = [_job(f"anchor:{q}", *anchor_params(summary, q)) for q in ("p50", "p90")]
    ok = True
    with get_context("spawn").Pool(2) as pool:
        for (label, tot, n, _), q in zip(pool.map(run_arm, jobs), ("p50", "p90")):
            got = tuple(tot[c] / n for c in (2.0, 3.0, 5.0))
            same = got == ANCHOR_RESULT[q]
            ok &= same
            print(f"{q} joint arm: seat success {got[0]:.1%} / {got[1]:.1%} / {got[2]:.1%} at 2 / 3 / 5 mm "
                  f"({'equal to' if same else 'DIFFERS from'} the run behind seatmap.json)")
    return 0 if ok else 1


def corpus_cards(cards_dir):
    """The SO-100/101 cards with five tip joints measured, in name order; label = file name without .json."""
    import holdcheck as hc
    out, skipped = [], 0
    for f in sorted(os.listdir(cards_dir)):
        if not f.endswith(".json"):
            continue
        c = _load(os.path.join(cards_dir, f))
        J = c.get("joints", {})
        fam = hc.family(c.get("robot_type"))
        if fam not in hc.SO_FAMILIES or any("lost_motion_deg" not in J.get(k, {}) for k in JOINTS):
            skipped += 1
            continue
        hw, sd = card_params(c)
        out.append({"key": f[:-5], "family": fam, "half_width_deg": hw, "scatter_sd_deg": sd})
    return out, skipped


def cmd_corpus(a):
    rows, skipped = corpus_cards(a.cards)
    rows = rows[: a.limit] if a.limit else rows
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(line)["key"] for line in f}
    todo = [r for r in rows if r["key"] not in done]
    print(f"{len(rows)} SO-100/101 cards with five tip joints ({skipped} other cards skipped), {len(todo)} to run, "
          f"{a.procs} processes", flush=True)
    meta = {r["key"]: r for r in todo}
    with get_context("spawn").Pool(a.procs) as pool, open(a.out, "a") as fh:
        jobs = [_job(r["key"], r["half_width_deg"], r["scatter_sd_deg"]) for r in todo]
        for i, (key, tot, n, _) in enumerate(pool.imap_unordered(run_arm, jobs), 1):
            fh.write(json.dumps({**meta[key], "n": n, **{f"success_c{int(c)}": tot[c] / n for c in tot}}) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print("done; next: seatsim.py map --runs", a.out)


def pava_increasing(x, y):
    """Isotonic (non-decreasing) least-squares fit of y on x; sorted x knots and fitted values."""
    o = np.argsort(-np.asarray(x, float), kind="mergesort")
    xs, ys = -np.asarray(x, float)[o], -np.asarray(y, float)[o]
    v, w, n = [], [], []
    for yi in ys:
        v.append(yi); w.append(1.0); n.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            tw = w[-2] + w[-1]
            v[-2] = (v[-2] * w[-2] + v[-1] * w[-1]) / tw
            w[-2] = tw; n[-2] += n[-1]
            v.pop(); w.pop(); n.pop()
    return -xs[::-1], -np.repeat(v, n)[::-1]


def cmd_map(a):
    rows = {}
    with open(a.runs) as f:
        for line in f:
            r = json.loads(line)
            rows[r["key"]] = r
    rows = [rows[k] for k in sorted(rows, key=lambda k: k + ".json")]  # card file-name order, as `corpus` reads them
    tm = _load(TIPMAP)
    out = {"version": 1,
           "what": "simulated seat success of the arm in a calibrated pick-and-place cell that approaches every target "
                   "from one side (each joint: its own play band and settle scatter; camera 0.5 mm), looked up from "
                   "the card's P(one-side tip error <= c)",
           "fitted_on": f"{len(rows)} SO-100/101 corpus cards, {CORPUS_UNITS} simulated units of 500 placements each",
           "lead_in_mm": {}}
    # linear in P up to 0.9, then on -log10(1 - P) up to 1 - 1e-5
    gx = np.unique(np.round(np.concatenate([np.arange(0.0, 0.9, 0.01), 1.0 - 10.0 ** (-np.arange(1.0, 5.0001, 0.05))]), 8))
    within = [one_side_within(r["scatter_sd_deg"], (3, 5), tm) for r in rows]
    for c in (3, 5):
        x = np.array([w[c] for w in within])
        y = np.array([r[f"success_c{c}"] for r in rows])
        xk, fk = pava_increasing(x, y)
        gy = np.interp(gx, xk, fk)
        full = np.interp(x, xk, fk)
        grid_err = float(np.max(np.abs(np.interp(x, gx, gy) - full)))
        # band: 5-fold cross-validated error of the lookup as shipped (fit on 4 folds, gridded, read on the 5th),
        # folds split by recording (card), not by arm
        idx = np.random.default_rng(0).permutation(len(x))
        cv, cv_raw = np.empty(len(x)), np.empty(len(x))
        for f in range(5):
            te = idx[f::5]
            tr = np.setdiff1d(idx, te)
            tk, tf = pava_increasing(x[tr], y[tr])
            cv[te] = np.abs(np.interp(x[te], gx, np.interp(gx, tk, tf)) - y[te])
            cv_raw[te] = np.abs(np.interp(x[te], tk, tf) - y[te])
        bound = {3: 0.05, 5: 0.03}[c]
        out["lead_in_mm"][str(c)] = {
            "p_one_side": [round(float(v), 6) for v in gx],
            "success": [round(float(v), 4) for v in gy],
            "p_min": round(float(x.min()), 6),
            "band_p90_abs": round(float(np.percentile(cv, 90)), 4),
            "band_median_abs": round(float(np.median(cv)), 4),
            "share_within_plan_bound": round(float((cv <= bound).mean()), 4),
            "plan_bound": bound,
            "ungridded_cv_p90_abs": round(float(np.percentile(cv_raw, 90)), 4),
            "grid_vs_fit_max_abs": round(grid_err, 4),
        }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=0)
    print(f"wrote {a.out} from {len(rows)} cards")
    # one value per arm: the median over the recordings of one contributor's SO-100 or SO-101 arm
    by = {}
    for r in rows:
        by.setdefault((r["key"].split("__")[0], r["family"]), []).append(r)
    for c in (3, 5):
        arm = np.array([np.median([r[f"success_c{c}"] for r in v]) for v in by.values()])
        p = np.percentile(arm, [10, 50, 90])
        print(f"  {c} mm: band {out['lead_in_mm'][str(c)]['band_p90_abs']:.4f}; over {len(by)} arms median "
              f"{p[1]:.1%} (p10 {p[0]:.1%}, p90 {p[2]:.1%})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arm", help="simulate your own arm from its card")
    p.add_argument("card")
    p.add_argument("--lead-in", nargs="+", default=["3", "5"], help="lead-ins in mm (default 3 5)")
    p.add_argument("--camera-mm", type=float, default=0.5, help="camera noise per axis, mm (default 0.5)")
    p.add_argument("--grasp-capture-mm", type=float, default=6.0, help="nest: sideways miss the gripper still picks (default 6)")
    p.add_argument("--tilt-max-deg", type=float, default=4.0, help="tool tilt the lead-in tolerates (default 4)")
    p.add_argument("--units", type=int, default=10)
    p.add_argument("--placements", type=int, default=500)
    p.add_argument("--label", help="names the random stream (default: arm); a corpus arm reproduces with its corpus key")
    p = sub.add_parser("anchor", help="the median-joint and p90-joint arms from the shipped corpus table")
    p.add_argument("--summary", default=SUMMARY)
    p = sub.add_parser("corpus", help="run every SO-100/101 card of a corpus run through the cell")
    p.add_argument("--cards", required=True, help="the --cards directory of corpus.py run")
    p.add_argument("--out", required=True, help="results, one JSON line per card (resumable)")
    p.add_argument("--limit", type=int, default=0, help="only the first N cards")
    p.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p = sub.add_parser("map", help="build seatmap.json from corpus results")
    p.add_argument("--runs", required=True, help="results from `corpus`")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"arm": cmd_arm, "anchor": cmd_anchor, "corpus": cmd_corpus, "map": cmd_map}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
