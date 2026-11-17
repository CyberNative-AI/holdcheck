#!/usr/bin/env python3
"""Run holdcheck over every dataset of a two-level corpus repo (contributor/dataset), resumably.

  corpus.py list   --repo lerobot/community_dataset_v3 --out runs/index.json
  corpus.py run    --index runs/index.json --cards runs/cards [--limit N] [--workers 6]
  corpus.py summary --cards runs/cards --out runs/summary.json --md runs/summary.md

A card already on disk is not fetched again. Each card names the dataset; the contributor (the
first path part) stands in for one physical arm, since a person usually records on one arm.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import holdcheck as hc  # noqa: E402


def cmd_list(a):
    top = json.loads(hc._get(f"{hc.HF}/api/datasets/{a.repo}/tree/main"))
    subs = []
    for t in top:
        if t["type"] != "directory":
            continue
        for x in json.loads(hc._get(f"{hc.HF}/api/datasets/{a.repo}/tree/main/{t['path']}")):
            if x["type"] == "directory":
                subs.append(x["path"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"repo": a.repo, "datasets": sorted(subs)}, open(a.out, "w"), indent=1)
    print(f"{len(subs)} datasets from {len(top)} contributors")


def _one(repo, sub, cards):
    out = os.path.join(cards, sub.replace("/", "__") + ".json")
    if os.path.exists(out):
        return out, "cached"
    hc.main(["--repo", repo, "--subdir", sub, "--out", out])
    return out, "done"


def cmd_run(a):
    idx = json.load(open(a.index))
    subs = idx["datasets"][: a.limit] if a.limit else idx["datasets"]
    os.makedirs(a.cards, exist_ok=True)
    n = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for out, st in ex.map(lambda s: _one(idx["repo"], s, a.cards), subs):
            n += 1
            if n % 25 == 0:
                print(f"{n}/{len(subs)} {st} {out}", flush=True)
    print(f"finished {n}")


def pct(v, qs=(10, 25, 50, 75, 90)):
    v = np.asarray(v, float)
    return {f"p{q}": float(np.percentile(v, q)) for q in qs} if v.size else {}


def _table(per_contrib):
    joints = [k for k in hc.SO_JOINTS if k != "gripper"]
    out = {}
    for k in joints:
        rows = [(c, js) for c, d in per_contrib.items() for kk, js in d.items() if kk == k]
        # one value per contributor: the median over that contributor's datasets
        lm = [np.median([j["lost_motion_deg"] for j in js]) for _, js in rows]
        sd = [np.median([j["scatter_sd_deg"] for j in js]) for _, js in rows]
        bias = [np.median([j["bias_deg"] for j in js]) for _, js in rows]
        ratio = [np.median([j["scatter_over_lost_motion"] for j in js if j["scatter_over_lost_motion"] is not None] or [np.nan]) for _, js in rows]
        ratio = [r for r in ratio if not np.isnan(r)]
        holds = [sum(j["holds_pos"] + j["holds_neg"] for j in js) for _, js in rows]
        sdf = [np.median(v) for v in ([j["scatter_sd_free_deg"] for j in js if j.get("scatter_sd_free_deg") is not None] for _, js in rows) if v]
        drift = [np.median(v) for v in ([j["settle_creep_deg"] for j in js if j.get("settle_creep_deg") is not None] for _, js in rows) if v]
        out[k] = {
            "loaded": k in hc.LOADED,
            "contributors": len(rows),
            "datasets": sum(len(js) for _, js in rows),
            "holds": int(sum(holds)),
            "lost_motion_deg": pct(lm),
            "scatter_sd_deg": pct(sd),
            "bias_deg": pct(bias),
            "scatter_over_lost_motion": pct(ratio),
            "scatter_sd_free_deg": pct(sdf),
            "arms_with_free_holds": len(sdf),
            "settle_creep_deg": pct(drift),
            "share_lost_motion_over_1deg": float(np.mean(np.asarray(lm) > 1.0)) if lm else None,
            "share_lost_motion_over_0p5deg": float(np.mean(np.asarray(lm) > 0.5)) if lm else None,
        }
    return out


def _md(title, table):
    L = [
        f"### {title}",
        "",
        "| joint | loaded | arms | datasets | holds | lost motion p10 / p50 / p90 (deg) | scatter sd p50 / p90 (deg) | free-hold scatter p50 / p90 (deg, arms) | creep toward command after 0.5 s p50 / p90 (deg) | >0.5 deg | >1 deg |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, j in table.items():
        if not j["contributors"]:
            continue
        lm, sd = j["lost_motion_deg"], j["scatter_sd_deg"]
        L.append(
            f"| {k} | {'yes' if j['loaded'] else 'no'} | {j['contributors']} | {j['datasets']} | {j['holds']} | "
            f"{lm['p10']:.2f} / {lm['p50']:.2f} / {lm['p90']:.2f} | {sd['p50']:.2f} / {sd['p90']:.2f} | "
            + (f"{j['scatter_sd_free_deg']['p50']:.2f} / {j['scatter_sd_free_deg']['p90']:.2f} ({j['arms_with_free_holds']}) | " if j["scatter_sd_free_deg"] else "not measured | ")
            + (f"{j['settle_creep_deg']['p50']:.2f} / {j['settle_creep_deg']['p90']:.2f} | " if j["settle_creep_deg"] else "- | ")
            + f"{j['share_lost_motion_over_0p5deg']:.0%} | {j['share_lost_motion_over_1deg']:.0%} |"
        )
    return L + [""]


def cmd_summary(a):
    cards = []
    for f in sorted(os.listdir(a.cards)):
        if f.endswith(".json"):
            cards.append((f, json.load(open(os.path.join(a.cards, f)))))
    tally = {"cards": len(cards), "error": 0, "skip": 0, "measured": 0}
    # family -> contributor -> joint -> [joint records]
    per_fam: dict[str, dict[str, dict[str, list]]] = {}
    modes: dict[str, int] = {}
    robot_types: dict[str, int] = {}
    for f, c in cards:
        if "error" in c:
            tally["error"] += 1
            continue
        if "skip" in c:
            tally["skip"] += 1
            continue
        tally["measured"] += 1
        rt = str(c.get("robot_type"))
        robot_types[rt] = robot_types.get(rt, 0) + 1
        fam = hc.family(rt)
        contrib = c["source"]["subdir"].split("/")[0]
        for k, j in c["joints"].items():
            m = j["lattice"].get("mode") if j["lattice"].get("ok") else "unlattice"
            modes[m] = modes.get(m, 0) + 1
            if "lost_motion_deg" in j:
                per_fam.setdefault(fam, {}).setdefault(contrib, {}).setdefault(k, []).append(j)
    so = {}
    for fam in hc.SO_FAMILIES:
        for contrib, d in per_fam.get(fam, {}).items():
            for k, js in d.items():
                # the same person on SO-100 and SO-101 counts as two arms
                so.setdefault(f"{contrib}#{fam}", {}).setdefault(k, []).extend(js)
    # one gripper-tip figure per arm, from that arm's per-joint medians (the same rule a card uses)
    tm = json.load(open(hc.TIPMAP))
    tips = []
    for arm, d in so.items():
        jj = {}
        for k, js in d.items():
            fr = [j["scatter_sd_free_deg"] for j in js if j.get("scatter_sd_free_deg") is not None]
            jj[k] = {"lost_motion_deg": float(np.median([j["lost_motion_deg"] for j in js])),
                     "scatter_sd_deg": float(np.median([j["scatter_sd_deg"] for j in js])),
                     "scatter_sd_free_deg": float(np.median(fr)) if fr else None}
        t = hc.tip_figures(jj, tm)
        if "one_side_p95_mm" in t:
            tips.append(t)
    out = {
        "tip": {"arms": len(tips), "rule": "per arm: median of each joint figure over its datasets, then holdcheck.tip_figures",
                "one_side_p95_mm": pct([t["one_side_p95_mm"] for t in tips]),
                "either_side_p95_mm": pct([t["either_side_p95_mm"] for t in tips])},
        "tally": tally,
        "lattice_modes": modes,
        "robot_types": robot_types,
        "family_rule": "so100: robot_type starts so100; so101: starts so101; koch; mixed (koch with so); other",
        "joints": _table(so),  # SO-100 and SO-101 together: the table holdcheck --compare uses
        "by_family": {fam: _table(pc) for fam, pc in sorted(per_fam.items())},
        "arms_by_family": {fam: len(pc) for fam, pc in sorted(per_fam.items())},
    }
    json.dump(out, open(a.out, "w"), indent=1)
    if a.md:
        L = _md("SO-100 and SO-101 (STS3215 servos), one value per arm", out["joints"])
        for fam in ("so100", "so101", "koch"):
            if fam in out["by_family"]:
                L += _md(f"{fam} only", out["by_family"][fam])
        tp = out["tip"]
        if tp["arms"]:
            L += ["### At the gripper (SO-100 and SO-101, one value per arm with all five joints measured)", "",
                  "| figure | arms | p10 | p25 | p50 | p75 | p90 |", "|---|---|---|---|---|---|---|"]
            for key, lab in (("one_side_p95_mm", "95% sideways radius, arriving from one side (mm)"),
                             ("either_side_p95_mm", "95% sideways radius, arriving from any side (mm)")):
                v = tp[key]
                L.append(f"| {lab} | {tp['arms']} | " + " | ".join(f"{v[q]:.2f}" for q in ("p10", "p25", "p50", "p75", "p90")) + " |")
            L.append("")
        L.append(f"cards {tally['cards']}, measured {tally['measured']}, skipped {tally['skip']}, errors {tally['error']}")
        L.append(f"arms by family {out['arms_by_family']}; lattice modes {modes}")
        open(a.md, "w").write("\n".join(L) + "\n")
    print(json.dumps(out["tally"]))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("list")
    x.add_argument("--repo", default="lerobot/community_dataset_v3")
    x.add_argument("--out", required=True)
    x = sp.add_parser("run")
    x.add_argument("--index", required=True)
    x.add_argument("--cards", required=True)
    x.add_argument("--limit", type=int, default=0)
    x.add_argument("--workers", type=int, default=6)
    x = sp.add_parser("summary")
    x.add_argument("--cards", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--md")
    a = p.parse_args()
    {"list": cmd_list, "run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
