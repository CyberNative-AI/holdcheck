#!/usr/bin/env python3
"""Instrument check: holdcheck must recover known lost motion and scatter from a simulated arm.

The simulated follower is a loop with lost motion b (steps): when the command moves up, the
follower stops b/2 below it; down, b/2 above; plus a constant bias and Gaussian scatter; then
quantised to whole encoder steps and stored either in degrees or in -100..100 over a calibrated
range, as LeRobot does. A teleop-like command (ramps, holds, hand jitter) drives it.
"""
import json
import math
import sys
import unittest

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import holdcheck as hc  # noqa: E402


def sim(b_steps, sd_steps, bias_steps, mode, rng, eps=40, fps=30.0, rng_steps=2873, lag=4, ep_len=700, creep=0.0, tau=15.0):
    A_all, S_all, EP_all = [], [], []
    for ep in range(eps):
        cmd = []
        x = 2048.0
        while len(cmd) < ep_len:  # fixed episode length so every joint column lines up
            tgt = 2048 + rng.uniform(-600, 600)
            n = int(rng.uniform(8, 40))
            cmd += list(np.linspace(x, tgt, n))
            x = tgt
            hold = int(rng.uniform(10, 50))
            cmd += list(x + rng.integers(-1, 2, hold) * 0.5)  # hand jitter within one step
        cmd = np.round(np.array(cmd[:ep_len]))  # leader reads whole steps
        fol = np.empty_like(cmd)
        p = cmd[0]
        settle = 0.0
        tm, dm = 0, 0.0
        for t in range(cmd.size):
            src = cmd[max(0, t - lag)]
            # play operator: the output moves only when the command leaves the band around it
            q = min(max(p, src - b_steps / 2), src + b_steps / 2)
            if q != p or t == 0:
                settle = rng.normal(0, sd_steps)  # where the loop settles, drawn once per move
                tm, dm = t, float(np.sign(q - p))
            p = q
            # optional slow settling: after a move the output still trails by creep, decaying with tau frames
            fol[t] = p + bias_steps + settle - dm * creep * math.exp(-(t - tm) / tau)
        fol = np.round(fol)
        if mode == "degrees":
            a = (cmd - 2048) * hc.STEP_DEG
            s = (fol - 2048) * hc.STEP_DEG
        else:
            lo = 2048 - rng_steps / 2
            a = (cmd - lo) / rng_steps * 200 - 100
            s = (fol - lo) / rng_steps * 200 - 100
        A_all.append(a)
        S_all.append(s)
        EP_all.append(np.full(a.size, ep))
    a = np.concatenate(A_all).astype(np.float32).astype(np.float64)
    s = np.concatenate(S_all).astype(np.float32).astype(np.float64)
    return a, s, np.concatenate(EP_all)


def info6():
    names = [f"{k}.pos" for k in hc.SO_JOINTS]
    return {"fps": 30, "robot_type": "so101_follower", "features": {"action": {"names": names}, "observation.state": {"names": names}}}


class T(unittest.TestCase):
    def run_case(self, b, sd, bias, mode, **kw):
        rng = np.random.default_rng(7)
        cols_a, cols_s = [], []
        ep = None
        for _ in hc.SO_JOINTS:
            a, s, ep = sim(b, sd, bias, mode, rng, **kw)
            cols_a.append(a)
            cols_s.append(s)
        A = np.stack(cols_a, 1)
        S = np.stack(cols_s, 1)
        card = hc.analyse(info6(), A, S, ep, np.random.default_rng(0))
        return card

    def check(self, card, b, sd, bias, mode):
        for k in hc.SO_JOINTS[:-1]:
            j = card["joints"][k]
            self.assertTrue(j["lattice"]["ok"], j["lattice"])
            self.assertEqual(j["lattice"]["mode"], "degrees" if mode == "degrees" else "range_m100_100")
            self.assertGreater(j["holds_pos"], 50)
            self.assertAlmostEqual(j["lost_motion_steps"], b, delta=1.01, msg=(k, j))
            self.assertAlmostEqual(j["bias_steps"], bias, delta=1.01)
            if sd >= 2:
                self.assertAlmostEqual(j["scatter_sd_steps"], sd, delta=0.35 * sd + 0.6)
                # 0.2: long holds with an empty gripper see the same scatter (nothing differs here)
                self.assertAlmostEqual(j["scatter_sd_free_steps"], sd, delta=0.45 * sd + 0.6, msg=(k, j))
            if j.get("settle_creep_steps") is not None:
                # the fixture settles at once: re-settling on jitter (b = 0) must not read as creep
                tol = max(0.5, 3 * j["settle_creep_se_steps"])
                self.assertLessEqual(abs(j["settle_creep_steps"]), tol, msg=(k, j))

    def test_degrees_lost_motion_12(self):
        self.check(self.run_case(12, 2.0, -3, "degrees"), 12, 2.0, -3, "degrees")

    def test_range_lost_motion_6(self):
        c = self.run_case(6, 1.0, 2, "range")
        self.check(c, 6, 1.0, 2, "range")
        self.assertAlmostEqual(c["joints"]["shoulder_pan"]["lattice"]["calibrated_range_steps"], 2873, delta=3)

    def test_no_lost_motion(self):
        self.check(self.run_case(0, 3.0, 0, "degrees"), 0, 3.0, 0, "degrees")

    def test_small_lost_motion_2(self):
        # lost motion smaller than the hand's jitter must not read high
        self.check(self.run_case(2, 0.5, 1, "range"), 2, 0.5, 1, "range")

    def test_large_lost_motion_90(self):
        # 0.4.1: play above the 60-step base grid is refitted on the wide grid, not pinned at 60
        # (free-hold scatter is not checked: with 90 steps of play the fixture leaves too few free holds)
        c = self.run_case(90, 2.0, 0, "degrees")
        for k in hc.SO_JOINTS[:-1]:
            self.assertAlmostEqual(c["joints"][k]["lost_motion_steps"], 90, delta=1.01, msg=k)
            lo, hi = c["joints"][k]["lost_motion_ci90_steps"]
            self.assertTrue(lo <= 90 <= hi and hi < hc.B_WIDE[-1], msg=(k, lo, hi))
        self.assertEqual(c["correction"], {})  # 90 steps is beyond CORR_MAX_STEPS: measured, not corrected
        t = hc.report_text(c)
        self.assertIn("No correction for elbow_flex: its 7.9 deg of play is more than software should cancel", t)

    def test_correction_up_to_60_steps(self):
        c = self.run_case(50, 2.0, 0, "degrees")
        for k in hc.SO_JOINTS[:-1]:
            self.assertAlmostEqual(c["correction"][k]["half_play_steps"], 25, delta=0.51, msg=k)
        self.assertNotIn("No correction", hc.report_text(c))

    def test_play_beyond_wide_grid_not_measured(self):
        c = self.run_case(300, 2.0, 0, "degrees")
        j = c["joints"]["elbow_flex"]
        self.assertNotIn("lost_motion_steps", j)
        self.assertIn("more than servo play", j["why"])
        self.assertEqual(c["correction"], {})
        self.assertIn("not_computed", hc.tip_figures(c["joints"]))
        self.assertIn("more than servo play", hc.report_text(c))

    def test_correction_needs_known_step(self):
        # 0.4.2: no correction for a family whose encoder step is not known, nor for a joint whose
        # -100..100 range spans more than one turn; each gets one stated line, and the tip sentence
        # offers the correction only when there is one
        c = self.run_case(20, 2.0, 0, "range")
        c["tip"] = hc.tip_figures(c["joints"])
        self.assertEqual(len(c["correction"]), 5)
        self.assertIn("so does the play correction below.", hc.report_text(c))
        foreign = json.loads(json.dumps(c))
        foreign["robot_type"] = "piper_follower"
        foreign["correction"] = hc.play_correction(foreign)
        self.assertEqual(foreign["correction"], {})
        self.assertEqual([k for k, _ in hc.correction_withheld(foreign)], [None])
        t = hc.report_text(foreign)
        self.assertIn("No correction for this arm", t)
        self.assertNotIn("so does the play correction", t)
        self.assertIn("removes the lost-motion part.", t)
        wide = json.loads(json.dumps(c))
        wide["joints"]["elbow_flex"]["lattice"]["calibrated_range_steps"] = 165746.0
        wide["correction"] = hc.play_correction(wide)
        self.assertNotIn("elbow_flex", wide["correction"])
        self.assertEqual(len(wide["correction"]), 4)
        self.assertIn("No correction for elbow_flex: its recorded -100..100 range spans 165746", hc.report_text(wide))
        # a partial correction does not promise the one-side figure: the same-side sentence and the
        # correction lead both say elbow_flex's play is left
        t = hc.report_text(wide)
        self.assertEqual(hc.correction_left(wide), ["elbow_flex"])
        self.assertNotIn("so does the play correction", t)
        self.assertIn("removes it only on the joints it lists, not on elbow_flex.", t)
        self.assertIn("With that correction, the one-side figure is not established for this arm.", t)
        self.assertIn("This card corrects only the joints listed here; the\nplay of elbow_flex is left", t)
        self.assertIn("We have not tried it on a physical arm.", t)
        self.assertEqual(hc.correction_left(c), [])
        self.assertNotIn("not established", hc.report_text(c))

    def test_small_family_not_compared(self):
        c = self.run_case(20, 2.0, 0, "degrees")
        summ = {"joints": {}, "by_family": {"koch": {"elbow_flex": {}}, "other:piper_follower": {"elbow_flex": {}}},
                "arms_by_family": {"koch": 12, "other:piper_follower": 1}}
        self.assertEqual(hc.compare_table(summ, "other:piper_follower"), {})
        self.assertEqual(hc.compare_table(summ, "koch"), {"elbow_flex": {}})
        self.assertEqual(hc.compare_table(None, "koch"), {})

    def test_wide_refit_leaves_other_joints_unchanged(self):
        # the pinned joint is refitted with the same bootstrap draws, so the result equals a fit on the
        # wide grid from the start and every later joint keeps the figures it had before the refit
        rng = np.random.default_rng(7)
        cols = [sim(90 if k == "shoulder_pan" else 12, 2.0, 0, "degrees", rng) for k in hc.SO_JOINTS]
        A, S, ep = np.stack([x[0] for x in cols], 1), np.stack([x[1] for x in cols], 1), cols[0][2]
        new = hc.analyse(info6(), A, S, ep, np.random.default_rng(0))
        base = hc.B_GRID
        try:
            hc.B_GRID = hc.B_WIDE  # play_predictions/fit_play defaults are bound at import: pass through analyse's names
            hc.play_predictions.__defaults__ = (hc.B_WIDE,)
            hc.fit_play.__defaults__ = (None, hc.B_WIDE)
            hc.boot_play.__defaults__ = (400, hc.B_WIDE)
            wide = hc.analyse(info6(), A, S, ep, np.random.default_rng(0))
        finally:
            hc.B_GRID = base
            hc.play_predictions.__defaults__ = (base,)
            hc.fit_play.__defaults__ = (None, base)
            hc.boot_play.__defaults__ = (400, base)
        self.assertAlmostEqual(new["joints"]["shoulder_pan"]["lost_motion_steps"], 90, delta=1.01)
        self.assertEqual(new["joints"], wide["joints"])

    def test_gripper_squeeze_excludes_holds(self):
        rng = np.random.default_rng(7)
        cols_a, cols_s = [], []
        for _ in hc.SO_JOINTS:
            a, s, ep = sim(12, 2.0, 0, "degrees", rng)
            cols_a.append(a)
            cols_s.append(s)
        A = np.stack(cols_a, 1)
        S = np.stack(cols_s, 1)
        S[:, -1] = A[:, -1] - 50 * hc.STEP_DEG  # jaws held open 50 steps by an object
        card = hc.analyse(info6(), A, S, ep, np.random.default_rng(0))
        j = card["joints"]["shoulder_pan"]
        self.assertEqual(j["holds_free_pos"] + j["holds_free_neg"], 0)
        self.assertNotIn("scatter_sd_free_steps", j)
        self.assertAlmostEqual(j["lost_motion_steps"], 12, delta=1.01)  # the main figure is unchanged

    def test_slow_settling_reads_as_creep(self):
        # output still trails by 8 steps at the move, decaying over 0.5 s: creep must show
        c = self.run_case(6, 1.0, 0, "degrees", creep=8.0, tau=15.0)
        for k in hc.SO_JOINTS[:-1]:
            j = c["joints"][k]
            self.assertGreaterEqual(j["settle_creep_steps"], 1.0, msg=(k, j))
            self.assertGreater(j["settle_creep_steps"], 3 * j["settle_creep_se_steps"], msg=(k, j))

    def test_tip_figures_known_map(self):
        # one pose, only shoulder_pan moves the tip sideways, 1000 mm per rad
        J = np.zeros((1, 3, 5)); J[0, 0, 0] = 1000.0
        tm = {"jacobian_mm_per_rad": J.tolist()}
        z = {"lost_motion_deg": 0.0, "scatter_sd_deg": 0.0}
        js = {k: dict(z) for k in hc.TIP_JOINTS}
        js["shoulder_pan"] = {"lost_motion_deg": 0.0, "scatter_sd_deg": 1.0}
        t = hc.tip_figures(js, tm, per_pose=40000)
        want = 1.95996 * math.radians(1.0) * 1000.0  # p95 of |N(0, 17.45 mm)|
        self.assertAlmostEqual(t["one_side_p95_mm"], want, delta=0.03 * want)
        # free-hold scatter is used in preference to all-hold scatter
        js["shoulder_pan"]["scatter_sd_free_deg"] = 0.5
        self.assertAlmostEqual(hc.tip_figures(js, tm, per_pose=40000)["one_side_p95_mm"], want / 2, delta=0.03 * want)
        # pure play: one side reads 0, any side reads the half-width at the tip exactly
        js["shoulder_pan"] = {"lost_motion_deg": 2.0, "scatter_sd_deg": 0.0}
        t = hc.tip_figures(js, tm)
        self.assertEqual(t["one_side_p95_mm"], 0.0)
        self.assertAlmostEqual(t["either_side_p95_mm"], math.radians(1.0) * 1000.0, delta=0.01)

    def test_tip_not_computed_without_all_joints(self):
        js = {k: {"lost_motion_deg": 1.0, "scatter_sd_deg": 0.2} for k in hc.TIP_JOINTS[:-1]}
        self.assertIn("not_computed", hc.tip_figures(js))

    def test_card_reports_tip(self):
        c = self.run_case(12, 1.5, 0, "degrees")
        c["tip"] = hc.tip_figures(c["joints"])
        self.assertGreater(c["tip"]["either_side_p95_mm"], c["tip"]["one_side_p95_mm"])
        self.assertIn("At the gripper", hc.report_text(c))

    def test_off_lattice_refused(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 30, 5000)
        self.assertFalse(hc.lattice_step(x)["ok"])

    def test_not_so_arm_skipped(self):
        info = {"fps": 30, "features": {"action": {"names": list("abcdefg")}, "observation.state": {"names": list("abcdefg")}}}
        z = np.zeros((100, 7))
        self.assertIn("skip", hc.analyse(info, z, z, np.zeros(100, int), np.random.default_rng(0)))

    def test_precheck_from_info(self):
        # an SO arm declared in info.json passes on to the data; others skip before any parquet
        self.assertIsNone(hc.precheck(info6()))
        seven = {"fps": 30, "total_frames": 900, "total_episodes": 3,
                 "features": {"action": {"names": list("abcdefg"), "shape": [7]}, "observation.state": {"names": list("abcdefg"), "shape": [7]}}}
        c = hc.precheck(seven)
        self.assertEqual(c["skip"], "not a single six-joint SO-style arm")
        self.assertEqual((c["frames"], c["episodes"], c["files"]), (900, 3, []))
        two = {"fps": 30, "features": {"action": {"names": [f"left_{k}.pos" for k in hc.SO_JOINTS] + [f"right_{k}.pos" for k in hc.SO_JOINTS], "shape": [12]},
                                       "observation.state": {"names": [f"left_{k}.pos" for k in hc.SO_JOINTS] + [f"right_{k}.pos" for k in hc.SO_JOINTS], "shape": [12]}}}
        self.assertIn("skip", hc.precheck(two))
        odd = info6(); odd["features"]["observation.state"] = {"names": list("abcdef")}
        self.assertEqual(hc.precheck(odd)["skip"], "action and state differ in shape or names")
        # analyse agrees with precheck on the same declaration
        z = np.zeros((100, 7))
        self.assertEqual(hc.analyse(seven, z, z, np.zeros(100, int), np.random.default_rng(0))["skip"], c["skip"])


    # 0.5: simulated seat success and the supervisor thresholds that follow
    def _seat_card(self, lm, sd, rt="so101_follower"):
        js = {k: {"lost_motion_deg": lm, "scatter_sd_deg": sd} for k in hc.TIP_JOINTS}
        c = {"robot_type": rt, "joints": js}
        c["tip"] = hc.tip_figures(js)
        return c

    def test_escalation_matches_m3(self):
        # values from the simulated fault campaign: p0 0.0102 -> 5, 3; 0.1279 -> 18, 5; 0.3268 -> 32, 9; p0 0 -> 4, 3
        for p0, want in ((0.0, (4, 3)), (0.0102, (5, 3)), (0.1279, (18, 5)), (0.3268, (32, 9))):
            self.assertEqual(hc.escalation(p0), want)

    def test_seat_tight_arm_seats_and_keeps_published_rule(self):
        st = hc.seat_figures(self._seat_card(0.5, 0.05))
        r5 = st["lead_in_mm"]["5"]
        self.assertGreaterEqual(r5["success"], 0.99)
        self.assertEqual((r5["escalate_misses_in_60"], r5["escalate_in_a_row"]), hc.escalation(1 - r5["success"]))
        self.assertIn("not measured on a physical cell", st["model"])

    def test_seat_monotone_in_scatter(self):
        prev = None
        for sd in (0.1, 0.2, 0.3, 0.45):
            r = hc.seat_figures(self._seat_card(1.0, sd))["lead_in_mm"]["3"]
            if prev is not None:
                self.assertLessEqual(r["success"], prev)
            prev = r["success"]

    def test_seat_loose_arm_gets_no_threshold(self):
        st = hc.seat_figures(self._seat_card(2.0, 0.55))
        r3 = st["lead_in_mm"]["3"]
        self.assertTrue("not_computed" in r3 or ("success" in r3 and 1 - r3["success"] > hc.SEAT_P0_TESTED))
        self.assertNotIn("escalate_misses_in_60", r3)
        self.assertNotIn(" of the last 60 placements miss", "\n".join(l for l in hc.seat_text({"seat": st}) if l.startswith("  3 mm")))

    def test_seat_outside_fitted_range_not_computed(self):
        st = hc.seat_figures(self._seat_card(2.0, 3.0))
        self.assertIn("looser than every arm", st["lead_in_mm"]["3"]["not_computed"])

    def test_seat_only_so_arms_with_tip(self):
        self.assertIn("not_computed", hc.seat_figures(self._seat_card(1.0, 0.2, rt="koch_follower")))
        js = {k: {"lost_motion_deg": 1.0, "scatter_sd_deg": 0.2} for k in hc.TIP_JOINTS[:-1]}
        c = {"robot_type": "so100_follower", "joints": js, "tip": hc.tip_figures(js)}
        self.assertIn("not_computed", hc.seat_figures(c))

    def test_seat_text_says_model(self):
        c = self.run_case(12, 1.5, 0, "degrees")
        c["robot_type"] = "so101_follower"
        c["tip"] = hc.tip_figures(c["joints"])
        c["seat"] = hc.seat_figures(c)
        t = hc.report_text(c)
        self.assertIn("not measured on a physical cell", t)
        self.assertIn("Count the cell's own misses when it is set up", t)
        # the band is over held-out recordings, split by recording: never a held-out-arm figure
        self.assertIn("held-out recordings", t)
        self.assertIn("split recordings, not arms", " ".join(t.split()))
        self.assertNotIn("9 in 10 arms", t)
        self.assertIn("not by arm", c["seat"]["band"])

    def test_seat_share_never_reads_certain(self):
        self.assertEqual([hc.share(p) for p in (0.999, 0.995, 0.994, 0.5, 0.005, 0.004, 0.0)],
                         [">99%", ">99%", "99%", "50%", "1%", "<1%", "<1%"])
        st = hc.seat_figures(self._seat_card(0.5, 0.05))
        t = "\n".join(hc.seat_text({"seat": st}))
        self.assertNotIn("100%", t)
        self.assertNotIn("(a supervisor", t)

    def test_small_family_note_in_text(self):
        c = self.run_case(20, 2.0, 0, "degrees")
        c["robot_type"] = "piper_follower"
        summ = {"joints": {}, "by_family": {}, "arms_by_family": {"other:piper_follower": 1}}
        t = hc.report_text(c, summ)
        if hc.family("piper_follower") not in ("so100", "so101"):
            self.assertIn("Not placed among other arms", t)
            self.assertIn(f"needs at least {hc.COMPARE_MIN_ARMS}", t)

    # 0.5.2: LeRobot writes robot_type "so_follower" for SO-100 and SO-101 since its 2026 SO consolidation
    def test_so_follower_is_an_so_arm(self):
        import json, os
        import playfix
        self.assertEqual(hc.family("so_follower"), "so")
        self.assertEqual(playfix._family("so_follower"), "so")
        self.assertFalse(hc.family("bi_so_follower") in hc.SO_FAMILIES)
        # only the canonical name: an unknown suffix stays out of the SO table and the correction
        self.assertEqual(hc.family("SO_Follower"), "so")
        for rt in ("so_followerx", "so_follower_custom", "so_followerpiper"):
            self.assertFalse(hc.family(rt) in hc.SO_FAMILIES, rt)
            self.assertEqual(playfix._family(rt), "other", rt)
        summ = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_summary.json")))
        cards = {}
        for rt in ("so101_follower", "so_follower"):
            c = self.run_case(20, 2.0, 0, "degrees")
            c["robot_type"] = rt
            c["tip"] = hc.tip_figures(c["joints"])
            c["seat"] = hc.seat_figures(c)
            c["correction"] = hc.play_correction(c)
            cards[rt] = c
        old, new = cards["so101_follower"], cards["so_follower"]
        self.assertNotIn("not_computed", new["seat"])
        self.assertEqual(new["seat"], old["seat"])
        self.assertEqual(new["correction"], old["correction"])
        self.assertTrue(new["correction"].get("joints") or new["correction"])
        t_old = hc.report_text(old, summ).replace("so101_follower", "RT")
        t_new = hc.report_text(new, summ).replace("so_follower", "RT")
        self.assertNotIn("Not placed among other arms", t_new)
        self.assertEqual(t_new, t_old)

    def test_large_play_text_does_not_locate_the_cause(self):
        c = self.run_case(20, 2.0, 0, "degrees")
        c["robot_type"] = "so101_follower"
        k = hc.TIP_JOINTS[2]
        c["joints"][k]["lost_motion_steps"] = hc.CORR_MAX_STEPS + 10
        c["joints"][k]["lost_motion_deg"] = (hc.CORR_MAX_STEPS + 10) * hc.STEP_DEG
        t = " ".join(m for j, m in hc.correction_skip(c) if j == k)
        self.assertIn("position_p_coefficient", t)
        self.assertIn("cannot say where", t)
        self.assertIn("not measured at all", t)
        for claim in ("horn", "inside the servo", "try 32", "32", "shaking"):
            self.assertNotIn(claim, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
