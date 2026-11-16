#!/usr/bin/env python3
"""Closed-loop check of playfix on a simulated follower (play operator, settle scatter, 4-frame lag,
whole-step encoder), in degrees as LeRobot stores them.

1. effect   : b = 24 steps. Fit a card on an uncorrected recording, drive a new recording through
              PlayCorrection; holdcheck on it must find the play gone (|b| <= 2 steps) and the
              direction gap closed, with scatter unchanged.
2. no effect: b = 0. The card's correction is at most one step; nothing gets worse.
3. jitter   : a held command with one-step hand jitter never flips the correction.
4. wrapper  : wrap_send_action sends the corrected command and returns what the robot sent.
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import holdcheck as hc  # noqa: E402
import playfix as pf  # noqa: E402
import test_holdcheck as th  # noqa: E402


def commands(rng, eps=40, ep_len=700):
    out = []
    for _ in range(eps):
        cmd, x = [], 2048.0
        while len(cmd) < ep_len:
            tgt = 2048 + rng.uniform(-600, 600)
            cmd += list(np.linspace(x, tgt, int(rng.uniform(8, 40))))
            x = tgt
            cmd += list(x + rng.integers(-1, 2, int(rng.uniform(10, 50))) * 0.5)
        out.append(np.round(np.array(cmd[:ep_len])))
    return out


def follower(cmd_steps, b, sd, bias, rng, lag=4):
    fol = np.empty_like(cmd_steps)
    p, settle = cmd_steps[0], 0.0
    for t in range(cmd_steps.size):
        src = cmd_steps[max(0, t - lag)]
        q = min(max(p, src - b / 2), src + b / 2)
        if q != p or t == 0:
            settle = rng.normal(0, sd)
        p = q
        fol[t] = p + bias + settle
    return np.round(fol)


def record(b, sd, seed, corr=None):
    """Six joints; returns A (intended command, degrees), S (follower, degrees), EP."""
    rng = np.random.default_rng(seed)
    names = [f"{k}.pos" for k in hc.SO_JOINTS]
    A, S, EP = [], [], []
    eps = [commands(rng) for _ in names]  # per joint: list of episodes
    for e in range(len(eps[0])):
        a_deg = np.stack([(eps[j][e] - 2048) * hc.STEP_DEG for j in range(6)], 1)
        sent = a_deg.copy()
        if corr is not None:
            corr.reset()
            for t in range(a_deg.shape[0]):
                o = corr({n: a_deg[t, j] for j, n in enumerate(names)})
                sent[t] = [o[n] for n in names]
        s = np.stack([follower(sent[:, j] / hc.STEP_DEG + 2048, b, sd, 3.0, rng) for j in range(6)], 1)
        A.append(a_deg)
        S.append((s - 2048) * hc.STEP_DEG)
        EP.append(np.full(a_deg.shape[0], e))
    A = np.concatenate(A).astype(np.float32).astype(np.float64)
    S = np.concatenate(S).astype(np.float32).astype(np.float64)
    return A, S, np.concatenate(EP)


def card(A, S, EP):
    c = hc.analyse(th.info6(), A, S, EP, np.random.default_rng(0))
    c["holdcheck"] = hc.meta()
    return c


class T(unittest.TestCase):
    def check(self, b):
        c0 = card(*record(b, 2.0, 1))
        corr = pf.PlayCorrection.from_card(c0)
        A, S, EP = record(b, 2.0, 2, corr)
        c1 = card(A, S, EP)
        for c, k in enumerate(pf.TIP):
            j0, j1 = c0["joints"][k], c1["joints"][k]
            gap1 = j1["median_err_neg_steps"] - j1["median_err_pos_steps"]
            # spread about each side's median, plain sd: the MAD-based scatter jumps a whole step on
            # integer data (1.48 / 2.22 / 2.97), so it cannot test a small change
            h = hc.holds_for_joint(A[:, c], S[:, c], EP, 30.0, hc.STEP_DEG)
            e, d = np.array([x[1] for x in h]), np.array([x[0] for x in h])
            r = np.concatenate([e[d > 0] - np.median(e[d > 0]), e[d < 0] - np.median(e[d < 0])])
            print(f"b={b} {k}: fitted {j0['lost_motion_steps']:.1f} -> after correction {j1['lost_motion_steps']:.1f}, "
                  f"gap {gap1:+.1f}, settle sd {r.std():.2f} (true 2.0), |r| > 6 steps {np.mean(np.abs(r) > 6):.1%}")
            self.assertLessEqual(abs(j1["lost_motion_steps"]), 2.0)
            self.assertLessEqual(abs(gap1), 2.0)
            self.assertLess(r.std(), 2.5)  # whole-step rounding adds about 0.3 to sd 2.0
            self.assertLess(np.mean(np.abs(r) > 6), 0.015)  # no holds corrected the wrong way (+-24 steps)
        return corr

    def test_effect(self):
        corr = self.check(24)
        self.assertTrue(all(abs(v / hc.STEP_DEG - 12) <= 1 for v in corr.half.values()))

    def test_no_effect(self):
        corr = self.check(0)
        self.assertTrue(all(v / hc.STEP_DEG <= 1.0 for v in corr.half.values()))

    def test_card_block_is_the_correction(self):
        # the wrapper applies the card's correction block as it is: five rows here
        c = card(*record(12, 2.0, 3))
        a = pf.PlayCorrection.from_card(c)
        self.assertEqual(len(a.half), 5)
        for k, v in c["correction"].items():
            self.assertAlmostEqual(a.half[v["action_key"]], v["half_play_units"], places=12)
        self.assertIn("Play correction", hc.report_text(c))

    def test_empty_block_corrects_nothing(self):
        # 0.4.2: a card whose every joint is withheld has correction {}; the wrapper must not refit the joints
        c = card(*record(12, 2.0, 3))
        c["correction"] = {}
        self.assertEqual(pf.PlayCorrection.from_card(c).half, {})
        c["robot_type"] = "piper_follower"
        c["correction"] = hc.play_correction(c)
        self.assertEqual(c["correction"], {})
        self.assertEqual(pf.PlayCorrection.from_card(c).half, {})

    def test_old_cards_refused(self):
        # cards without the 0.4.2 per-arm checks (no block, or 0.4.0 / 0.4.1) are refused, not refitted
        c = card(*record(12, 2.0, 3))
        for ver in ("0.3.1", "0.4.0", "0.4.1"):
            old = json.loads(json.dumps(c))
            old["holdcheck"]["version"] = ver
            with self.assertRaises(ValueError):
                pf.PlayCorrection.from_card(old)
        old = json.loads(json.dumps(c))
        old.pop("correction")
        with self.assertRaises(ValueError):
            pf.PlayCorrection.from_card(old)

    def test_oversized_correction_refused(self):
        # a hand-edited or foreign card cannot make the wrapper aim more than 30 steps past a target
        c = card(*record(12, 2.0, 3))
        bad = json.loads(json.dumps(c))
        bad["correction"]["elbow_flex"]["half_play_steps"] = 100.0
        with self.assertRaises(ValueError):
            pf.PlayCorrection.from_card(bad)

    def test_malformed_block_refused(self):
        # what the wrapper applies is bound by the checks: the unit size must be the checked steps times
        # the joint's checked encoder step, the key must be that joint's own, and nothing else is accepted
        c = card(*record(12, 2.0, 3))
        self.assertEqual(len(pf.PlayCorrection.from_card(c).half), 5)  # control: the genuine card passes

        def edit(f):
            bad = json.loads(json.dumps(c))
            f(bad)
            return bad

        def corr(k, f, v):
            return lambda b: b["correction"][k].__setitem__(f, v)

        def lat(k, f, v):
            return lambda b: b["joints"][k]["lattice"].__setitem__(f, v)

        cases = {
            "units 10000": corr("shoulder_pan", "half_play_units", 10000.0),
            "units x2": lambda b: b["correction"]["elbow_flex"].__setitem__(
                "half_play_units", 2 * b["correction"]["elbow_flex"]["half_play_units"]),
            "units negative": lambda b: b["correction"]["elbow_flex"].__setitem__(
                "half_play_units", -b["correction"]["elbow_flex"]["half_play_units"]),
            "units inf": corr("elbow_flex", "half_play_units", float("inf")),
            "units nan": corr("elbow_flex", "half_play_units", float("nan")),
            "units text": corr("elbow_flex", "half_play_units", "0.5"),
            "deg wrong": corr("elbow_flex", "half_play_deg", 5.0),
            "steps negative": corr("elbow_flex", "half_play_steps", -1.0),
            "steps nan": corr("elbow_flex", "half_play_steps", float("nan")),
            "steps bool": corr("elbow_flex", "half_play_steps", True),
            "step_units big": lat("elbow_flex", "step_units", 1000.0),
            "step_units zero": lat("elbow_flex", "step_units", 0.0),
            "step_units nan": lat("elbow_flex", "step_units", float("nan")),
            "mode unknown": lat("elbow_flex", "mode", "raw_or_other"),
            "no lattice": lambda b: b["joints"]["elbow_flex"].pop("lattice"),
            "key of gripper": corr("shoulder_pan", "action_key", "gripper.pos"),
            "key of another joint": corr("shoulder_pan", "action_key", "elbow_flex.pos"),
            "key missing": lambda b: b["correction"]["shoulder_pan"].pop("action_key"),
            "key not recorded": lambda b: b.__setitem__("action_names", [n for n in b["action_names"] if "elbow" not in n]),
            "gripper row": lambda b: b["correction"].__setitem__("gripper", dict(b["correction"]["elbow_flex"], action_key="gripper.pos")),
            "block a list": lambda b: b.__setitem__("correction", [1]),
            "foreign arm": lambda b: b.__setitem__("robot_type", "piper_follower"),
        }
        for name, f in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                pf.PlayCorrection.from_card(edit(f))
        for hs in (float("nan"), -1.0, 1000.0):
            with self.subTest(hysteresis=hs), self.assertRaises(ValueError):
                pf.PlayCorrection.from_card(c, hysteresis_steps=hs)

    def test_range_units(self):
        # a -100..100 recording: the step is 200 / calibrated range; at most one servo turn
        c = card(*record(12, 2.0, 3))
        for k, v in c["correction"].items():
            rs = 3000.0
            c["joints"][k]["lattice"].update(mode="range_m100_100", step_units=200.0 / rs, calibrated_range_steps=rs)
            v["half_play_units"] = v["half_play_steps"] * 200.0 / rs
        a = pf.PlayCorrection.from_card(c)
        self.assertEqual(len(a.half), 5)
        wide = json.loads(json.dumps(c))
        for k, v in wide["correction"].items():
            rs = 175000.0
            wide["joints"][k]["lattice"].update(step_units=200.0 / rs, calibrated_range_steps=rs)
            v["half_play_units"] = v["half_play_steps"] * 200.0 / rs
        with self.assertRaises(ValueError):
            pf.PlayCorrection.from_card(wide)

    def test_jitter_no_flip(self):
        corr = pf.PlayCorrection({"x": 1.0}, {"x": 2.0})
        seq = [0, 3, 6, 9, 10] + [10 + j for j in np.random.default_rng(0).integers(-1, 2, 200)]
        outs = [corr({"x": float(v)})["x"] - float(v) for v in seq]
        self.assertTrue(all(o == 1.0 for o in outs[2:]), outs[:12])
        self.assertEqual(corr({"x": 7.0})["x"], 6.0)  # a real reversal (> hysteresis) flips it

    def test_wrapper(self):
        class R:
            sent = None

            def send_action(self, a):
                R.sent = a
                return a

        r = pf.wrap_send_action(R(), pf.PlayCorrection({"x": 1.0}, {"x": 0.5}))
        r.send_action({"x": 0.0, "g": 5.0})
        back = r.send_action({"x": 2.0, "g": 5.0})
        self.assertEqual(back, {"x": 3.0, "g": 5.0})
        self.assertEqual(R.sent, {"x": 3.0, "g": 5.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
