"""Browser glue for holdcheck.py: the same steps as its main(), fed with bytes the page fetched.

holdcheck.py is bundled unchanged. The page fetches meta/info.json and data/*.parquet from Hugging
Face (holdcheck's own urllib fetch cannot run in a browser) and hands the bytes here. Everything
the visitor reads comes from the card and from holdcheck's own wording functions; this module only
formats the card's numbers the way report_text prints them and sorts them into rows for the page.
"""
from __future__ import annotations

import io
import json
import os
import re
import time

import numpy as np

import holdcheck as hc

COLS = ["action", "observation.state", "episode_index"]
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "corpus_summary.json")) as _f:
    SUMMARY = json.load(_f)


def _bytes(x) -> bytes:
    """A JS Uint8Array (Pyodide proxy) or bytes-like, as bytes."""
    return x.to_bytes() if hasattr(x, "to_bytes") else bytes(x)


def measure(repo: str, sub: str, info_bytes, files, blobs, seed: int = 0) -> dict:
    """holdcheck.main() for --repo/--subdir, without the download; returns the card."""
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    t0 = time.time()
    try:
        info = json.loads(_bytes(info_bytes))
        pairs = sorted(zip([str(f) for f in files], [_bytes(b) for b in blobs]), key=lambda p: p[0])
        files = [p[0] for p in pairs]
        tables = [pq.read_table(io.BytesIO(b), columns=COLS) for _, b in pairs]
        A, S, EP = hc.to_arrays(tables)
        card = hc.analyse(info, A, S, EP, rng)
        card["files"] = files
        if card.get("joints"):
            card["tip"] = hc.tip_figures(card["joints"])
            card["seat"] = hc.seat_figures(card)
        src = {"repo": repo, "subdir": sub}
    except Exception as e:  # noqa: BLE001 - a bad dataset is a result, not a crash
        card, src = {"error": f"{type(e).__name__}: {e}"}, {"repo": repo, "subdir": sub, "local": None}
    return _stamp(card, src, t0)


def _stamp(card: dict, src: dict, t0: float) -> dict:
    """What main() adds to every card: source, the method stamp, elapsed seconds."""
    card["source"] = src
    card["holdcheck"] = hc.meta()
    card["seconds"] = round(time.time() - t0, 2)
    return card


def precheck(repo: str, sub: str, info_bytes) -> str:
    """main()'s first step: the skip decision from meta/info.json alone. Returns the page view as a
    JSON string when the dataset skips (or info.json cannot be read the way main() reads it), else ""
    and the page goes on to fetch the parquet."""
    t0 = time.time()
    try:
        card = hc.precheck(json.loads(_bytes(info_bytes)))
        if card is None:
            return ""
        src = {"repo": repo, "subdir": sub}
    except Exception as e:  # noqa: BLE001 - same as main(): a bad dataset is a result
        card, src = {"error": f"{type(e).__name__}: {e}"}, {"repo": repo, "subdir": sub, "local": None}
    return json.dumps(view(_stamp(card, src, t0)))


def _pct(p: dict | None) -> dict | None:
    return {k: p[k] for k in ("p10", "p25", "p50", "p75", "p90")} if p else None


def view(card: dict) -> dict:
    """The card laid out for the page: report_text's selections, formats and wording, as rows."""
    out = {"text": hc.report_text(card, SUMMARY), "json": json.dumps(card, indent=1)}
    if "error" in card:
        out["state"] = "error"
        out["message"] = card["error"]
        return out
    out.update({"robot_type": card.get("robot_type"), "episodes": card["episodes"], "frames": card["frames"],
                "fps": f"{card['fps']:g}", "codebase_version": card.get("codebase_version")})
    if "skip" in card:
        out["state"] = "skip"
        out["message"] = card["skip"]
        return out
    fam = hc.family(card.get("robot_type"))
    out["family"] = fam
    so = fam in hc.SO_FAMILIES
    table = hc.compare_table(SUMMARY, fam)
    # No table: the corpus has too few arms of this family to place one among (fewer than COMPARE_MIN_ARMS).
    out["compared"] = bool(table)
    out["family_arms"] = SUMMARY.get("arms_by_family", {}).get(fam, 0)
    out["compare_min"] = hc.COMPARE_MIN_ARMS
    rows = []
    for k in hc.SO_JOINTS[:-1]:
        j = card["joints"].get(k, {})
        r = {"joint": k}
        if "lost_motion_deg" not in j:
            r["why"] = j.get("why") or j.get("lattice", {}).get("why") or f"too few holds ({j.get('holds_pos', 0)} up, {j.get('holds_neg', 0)} down)"
            r["few"] = not (j.get("why") or j.get("lattice", {}).get("why"))
            rows.append(r)
            continue
        lo, hi = (x * hc.STEP_DEG for x in j["lost_motion_ci90_steps"])
        sc = hc.joint_scatter_deg(j)
        free_card = j.get("scatter_sd_free_deg") is not None
        r.update({
            "holds": j["holds_pos"] + j["holds_neg"],
            "lost_motion": f"{j['lost_motion_deg']:.2f}", "lost_motion_v": j["lost_motion_deg"],
            "ci90": f"{lo:.2f} to {hi:.2f}",
            "scatter": f"{sc:.2f}", "scatter_v": sc, "scatter_from": "free holds" if free_card else "all holds",
            "bias": f"{j['bias_deg']:.2f}",
        })
        if k in table:
            free = free_card and table[k].get("scatter_sd_free_deg")
            pl, ps = table[k]["lost_motion_deg"], table[k]["scatter_sd_free_deg" if free else "scatter_sd_deg"]
            r.update({"lost_motion_place": hc._place(j["lost_motion_deg"], pl), "lost_motion_pct": _pct(pl),
                      "scatter_place": hc._place(sc, ps), "scatter_pct": _pct(ps),
                      "arms": table[k].get("contributors")})
        rows.append(r)
    out["rows"] = rows
    out["measured"] = sum(1 for r in rows if "lost_motion" in r)
    t = card.get("tip") or {}
    if "one_side_p95_mm" not in t:
        out["tip"] = {"not_computed": t.get("not_computed", "not computed")}
    else:
        ts = SUMMARY.get("tip", {}) if so else {}
        # same_side: report_text's closing clause of the same-side sentence (the page joins its line break)
        tip = {"arms": ts.get("arms"), "geometry": t.get("geometry"),
               "same_side": hc.same_side_clause(card).replace("\n", " ")}
        for key in ("one_side_p95_mm", "either_side_p95_mm"):
            tip[key] = {"text": f"{t[key]:.1f}", "v": t[key], "place": hc._place(t[key], ts.get(key, {})), "pct": _pct(ts.get(key))}
        out["tip"] = tip
    # report_text prints the cell figures only under the gripper figures; so does the page
    out["seat"] = seat(card) if "one_side_p95_mm" in t else None
    out["correction"] = correction(card)
    out["state"] = "measured" if out["measured"] else "few"
    return out


def seat(card: dict) -> dict | None:
    """The card's pick-and-place figures for the page: seat_text()'s own lines, sorted into blocks.

    seat_text() prints prose lines and indented rows ("  3 mm lead-in   ..."; a row's continuation is
    indented further). The prose runs before the first row group go in `lead`, between the groups in
    `ask`, after them in `close`; each row's words after its lead-in go in `rows[c]` or `calls[c]`.
    A seated row's line splits into `success`, `what` and `band` for the figure (`v`, `band_v` raw, for its bar).
    None when the card carries no seat block; {"not_computed": reason} when no lead-in has a figure."""
    lines = hc.seat_text(card)
    if not lines:
        return None
    st = card.get("seat") or {}
    rows = st.get("lead_in_mm") or {}
    if not any("success" in r for r in rows.values()):
        return {"not_computed": lines[1].split(": not computed: ", 1)[1]}
    blocks, cur = [], None
    for x in lines[1:]:
        kind = "row" if x.startswith("  ") else "prose"
        if cur is None or cur[0] != kind:
            cur = (kind, [])
            blocks.append(cur)
        cur[1].append(x)

    def join(ls):
        return " ".join(x.strip() for x in ls)

    def keyed(ls):
        out, last = {}, None
        for x in ls:
            m = re.match(r"  (\d+) mm lead-in   (.*)$", x)
            if m:
                last = m.group(1)
                out[last] = m.group(2).strip()
            else:
                out[last] += " " + x.strip()
        return out

    prose = [join(b) for k, b in blocks if k == "prose"]
    groups = [keyed(b) for k, b in blocks if k == "row"]
    figs = []
    for c, r in rows.items():
        line = groups[0][c]
        # call_none: no threshold for this lead-in (the arm misses more than the rule was tested on)
        f = {"lead_in": c, "line": line, "call": groups[1].get(c) if len(groups) > 1 else None,
             "call_none": "success" in r and "escalate_misses_in_60" not in r}
        if "success" in r:
            # "66% of placements seat (within 3 points for 9 in 10 held-out recordings)": figure, what it counts, its band
            m = re.match(r"(\S+) (.+?) \((.+)\)$", line)
            f.update({"success": m.group(1), "what": m.group(2), "band": m.group(3), "v": r["success"], "band_v": r["band"]}
                     if m else {"success": f"{r['success']:.0%}", "what": line, "band": "", "v": r["success"], "band_v": r["band"]})
        else:
            f["not_computed"] = line.split("not computed: ", 1)[-1]
        figs.append(f)
    return {"lead": prose[0], "ask": prose[1] if len(prose) > 1 else "", "close": prose[2] if len(prose) > 2 else "",
            "figs": figs, "model": st.get("model"), "band_note": st.get("band")}


def correction(card: dict) -> dict | None:
    """The card's play correction for the page: correction_text()'s own sentences and its rows' formats.

    The sentences up to the one saying it is untried go in `lead`; that one and the rest go in
    `caution`, which the page sets apart. `withheld` holds correction_text()'s closing lines, one per
    joint whose play is too large to correct; they can come with no rows at all. None when the card has
    neither rows nor withheld joints (the section hides). `full` is False when a tip joint with play
    is left uncorrected; the page then does not offer the one-side figure with the correction."""
    c = card.get("correction") or {}
    lines = hc.correction_text(card)
    n = len(hc.correction_withheld(card))
    withheld = [x.strip() for x in lines[len(lines) - n:]] if n else []
    if not c:
        return {"lead": [], "caution": [], "rows": [], "withheld": withheld} if withheld else None
    lines = lines[:len(lines) - n]
    head = next(i for i, x in enumerate(lines) if x.strip().startswith("joint"))
    said = re.split(r"(?<=\.) (?=[A-Z])", " ".join(x.strip() for x in lines[:head] if x.strip()))
    cut = next((i for i, x in enumerate(said) if "not tried" in x), len(said))
    rows = [{"joint": k, "add": f"{v['half_play_units']:.2f}", "units": v["units"], "deg": f"{v['half_play_deg']:.2f}"}
            for k, v in c.items()]
    # full: every tip joint with play is corrected, so the correction reaches the one-side figure
    return {"lead": said[:cut], "caution": said[cut:], "rows": rows, "withheld": withheld,
            "full": not hc.correction_left(card)}


def run(repo, sub, info_bytes, files, blobs) -> str:
    """Card plus page view as one JSON string (the card JSON inside is holdcheck's own json.dump format)."""
    card = measure(repo, sub, info_bytes, files, blobs)
    return json.dumps(view(card))
