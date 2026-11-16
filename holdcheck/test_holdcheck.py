#!/usr/bin/env python3
"""Instrument check: holdcheck must recover known lost motion and scatter from a simulated arm.

The simulated follower is a loop with lost motion b (steps): when the command moves up, the
follower stops b/2 below it; down, b/2 above; plus a constant bias and Gaussian scatter; then
quantised to whole encoder steps and stored either in degrees or in -100..100 over a calibrated
range, as LeRobot does. A teleop-like command (ramps, holds, hand jitter) drives it.
"""
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
