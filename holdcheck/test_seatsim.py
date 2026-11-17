"""Tests for the seat cell and its runner: python test_seatsim.py (from this directory), about 20 seconds."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import holdcheck as hc  # noqa: E402
import numpy as np  # noqa: E402
import seatsim as ss  # noqa: E402

JOINTS = ss.JOINTS


def card(robot_type="so101_follower", lm=(0.6, 0.8, 2.0, 1.0, 0.5), sd=(0.12, 0.3, 0.35, 0.2, 0.1), free=True):
    j = {}
    for k, l, s in zip(JOINTS, lm, sd):
        j[k] = {"lost_motion_deg": l, "scatter_sd_deg": s + 0.05}
        if free:
            j[k]["scatter_sd_free_deg"] = s
    return {"robot_type": robot_type, "joints": j}


class TestSeatSim(unittest.TestCase):
    def test_card_params(self):
        hw, sd = ss.card_params(card(lm=(0.6, -0.1, 2.00004, 1.0, 0.5)))
        self.assertEqual(hw, [0.3, 0.0, 1.0, 0.5, 0.25])  # half the lost motion, negative clipped, 4 decimals
        self.assertEqual(sd, [0.12, 0.3, 0.35, 0.2, 0.1])  # free-hold scatter where measured
        _, sd_all = ss.card_params(card(free=False))
        self.assertEqual(sd_all, [0.17, 0.35, 0.4, 0.25, 0.15])  # else all holds

    def test_card_refused(self):
        with self.assertRaises(SystemExit):
            ss.card_params(card(robot_type="aloha"))
        c = card()
        del c["joints"]["wrist_flex"]["lost_motion_deg"]
        with self.assertRaises(SystemExit):
            ss.card_params(c)

    def test_unit_is_deterministic(self):
        hw, sd = ss.card_params(card())
        got = ss.run_unit(2000, "test", hw, sd, placements=100)
        self.assertEqual(got, {2.0: 74, 3.0: 93, 5.0: 100})
        self.assertEqual(got, ss.run_unit(2000, "test", hw, sd, placements=100))

    def test_tighter_arm_seats_more(self):
        hw, sd = ss.card_params(card())
        loose = ss.run_unit(2001, "test", [2 * v for v in hw], [2 * v for v in sd], placements=200)
        tight = ss.run_unit(2001, "test", hw, sd, placements=200)
        self.assertGreater(tight[3.0], loose[3.0])
        self.assertLessEqual(tight[2.0], tight[3.0])
        self.assertLessEqual(tight[3.0], tight[5.0])

    def test_median_joint_arm_reproduces(self):
        summary = ss._load(ss.SUMMARY)
        _, tot, n, _ = ss.run_arm(ss._job("anchor:p50", *ss.anchor_params(summary, "p50")))
        self.assertEqual(tuple(tot[c] / n for c in (2.0, 3.0, 5.0)), ss.ANCHOR_RESULT["p50"])

    def test_predictor_is_the_engines(self):
        tm = ss._load(ss.TIPMAP)
        c = card()
        eng = hc.tip_figures(c["joints"], tm)["one_side_within"]
        mine = ss.one_side_within(ss.card_params(c)[1], (3, 5), tm)
        for k in (3, 5):
            self.assertAlmostEqual(eng[str(k)], mine[k], places=6)

    def test_map_from_runs(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            runs, out = os.path.join(d, "seat.jsonl"), os.path.join(d, "seatmap.json")
            with open(runs, "w") as f:
                for i in range(40):
                    s = float(rng.uniform(0.05, 0.6))
                    f.write(json.dumps({"key": f"c{i % 7}__d{i}", "family": "so101", "half_width_deg": [0.3] * 5,
                                        "scatter_sd_deg": [s] * 5, "success_c3": 1 - s, "success_c5": 1 - s / 2}) + "\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                ss.cmd_map(SimpleNamespace(runs=runs, out=out))
            m = ss._load(out)
            ship = ss._load(os.path.join(HERE, "seatmap.json"))
            self.assertEqual(list(m), list(ship))
            for c in ("3", "5"):
                t = m["lead_in_mm"][c]
                self.assertEqual(list(t), list(ship["lead_in_mm"][c]))
                self.assertEqual(t["p_one_side"], ship["lead_in_mm"][c]["p_one_side"])
                self.assertTrue(np.all(np.diff(t["success"]) >= 0))  # monotone in the card's predictor
            self.assertIn("over 7 arms", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
