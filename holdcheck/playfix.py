#!/usr/bin/env python3
"""playfix: cancel an SO-100/101 arm's measured play in software, from its holdcheck card.

holdcheck fits each joint's play b (the closed servo loop settles b/2 short of the target, on the
side it came from). playfix aims b/2 past the target in the direction the joint is moving, so the
follower settles on the target from either side. On the recordings of 122 arms this gave the same
95% placement radius as approaching every target from one side; we have not tried it on a
physical arm. Fit the card from a recent recording of the same arm, and fit again
after re-assembly, re-calibration or a servo swap.

  corr = PlayCorrection.from_card(json.load(open("card.json")))
  robot = wrap_send_action(robot, corr)      # LeRobot follower: corrects every command it sends

The direction of motion is tracked with a small hysteresis (default 2 encoder steps), so jitter in
a held command does not flip the correction. LeRobot's record loop saves the action before it is
sent (lerobot_record.py at huggingface/lerobot fe1c5459), so recordings made through the wrapper
keep the uncorrected target in `action`, and holdcheck on them measures the play that is left.
"""
from __future__ import annotations

import json
import sys

TIP = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MAX_HALF_STEPS = 30.0  # holdcheck gives a correction only up to 60 steps of play; a larger one is refused
STEP_DEG = 360.0 / 4096.0  # one encoder step of an SO-100/101 (STS3215) or Koch (Dynamixel) servo
TURN_STEPS = 4096
FAMILIES = ("so100", "so101", "koch", "mixed")  # the arms holdcheck gives a correction for


def _family(robot_type) -> str:
    rt = str(robot_type or "").lower().replace("-", "")
    if "koch" in rt and "so10" in rt:
        return "mixed"
    for f, pre in (("so101", ("so101",)), ("so100", ("so100", "so_100")), ("koch", ("koch",))):
        if rt.startswith(pre):
            return f
    return "other"


def _num(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{what} is not a number: {x!r}")
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{what} is not finite: {x!r}")
    return v


def _step_units(k: str, lat: dict) -> float:
    """One encoder step in the recording's action units, checked against the lattice mode it was read as."""
    step = _num(lat.get("step_units"), f"{k}: step_units")
    mode = lat.get("mode")
    if mode == "degrees":
        ok = abs(step / STEP_DEG - 1.0) < 0.01
    elif mode == "raw_steps":
        ok = abs(step - 1.0) < 0.01
    elif mode == "range_m100_100":  # -100..100 over the calibrated range, at most one servo turn
        rng = _num(lat.get("calibrated_range_steps", 200.0 / step if step > 0 else 0.0), f"{k}: calibrated_range_steps")
        ok = 200.0 / (TURN_STEPS * 1.01) <= step < 1.0 and abs(rng * step / 200.0 - 1.0) < 1e-6
    else:
        ok = False
    if not ok:
        raise ValueError(f"{k}: an encoder step of {step!r} is not what holdcheck reads for {mode!r} recordings")
    return step


class PlayCorrection:
    def __init__(self, half_play: dict[str, float], hysteresis: dict[str, float]):
        """half_play, hysteresis: per action key (e.g. 'elbow_flex.pos'), in the action's own units."""
        self.half = dict(half_play)
        self.hyst = dict(hysteresis)
        self.anchor: dict[str, float] = {}
        self.dir: dict[str, int] = {}

    @classmethod
    def from_card(cls, card: dict, hysteresis_steps: float = 2.0, joints=TIP):
        """From a holdcheck card's `correction` block (0.4.2 and later). An empty block corrects nothing.

        A card without the block, or from 0.4.0 or 0.4.1 (before the per-arm checks of 0.4.2), is refused:
        make a new card from the recording. So is a block that is not what holdcheck writes: a joint other
        than the five arm joints, an action key that is not that joint's or is used twice, a size outside
        0..30 encoder steps, or a size in the recording's units that is not those steps times the joint's
        encoder step. Nothing is applied unless the whole block passes."""
        ver = str((card.get("holdcheck") or {}).get("version", "0"))
        if "correction" not in card or tuple(int(x) for x in ver.split(".")[:3] if x.isdigit()) < (0, 4, 2):
            raise ValueError(f"card from holdcheck {ver}: make a new card with holdcheck 0.4.2 or later")
        corr = card["correction"] or {}
        if not isinstance(corr, dict):
            raise ValueError("the card's correction block is not a table of joints")
        hs = _num(hysteresis_steps, "hysteresis_steps")
        if not 0.0 <= hs <= MAX_HALF_STEPS:
            raise ValueError(f"hysteresis of {hs:g} steps is outside 0..{MAX_HALF_STEPS:g}")
        other = sorted(str(k) for k in corr if k not in TIP)
        if other:
            raise ValueError(f"correction for {', '.join(other)}: playfix corrects only the five arm joints")
        if corr and _family(card.get("robot_type")) not in FAMILIES:
            raise ValueError(f"correction for robot_type {card.get('robot_type')!r}: holdcheck gives one only for SO-100/101 and Koch arms")
        half, hyst = {}, {}
        for k in joints:
            if k not in corr:
                continue
            c = corr[k]
            key = c.get("action_key") if isinstance(c, dict) else None
            if not isinstance(key, str) or not (key.lower().endswith(k) or key.lower().endswith(k + ".pos")):
                raise ValueError(f"{k}: action key {key!r} is not this joint's")
            if key in half:
                raise ValueError(f"{k}: action key {key!r} is corrected twice")
            names = card.get("action_names") or []
            if names and key not in names:
                raise ValueError(f"{k}: action key {key!r} is not one of the recording's actions")
            steps = _num(c.get("half_play_steps"), f"{k}: half_play_steps")
            if not 0.0 <= steps <= MAX_HALF_STEPS:
                raise ValueError(f"{k}: correction of {steps:g} steps is outside 0..{MAX_HALF_STEPS:g}")
            step = _step_units(k, ((card.get("joints") or {}).get(k) or {}).get("lattice") or {})
            units = steps * step  # what is applied: the checked steps in the recording's units
            for f, want in (("half_play_units", units), ("half_play_deg", steps * STEP_DEG)):
                if abs(_num(c.get(f), f"{k}: {f}") - want) > 1e-9 * max(1.0, abs(want)):
                    raise ValueError(f"{k}: {f} {c.get(f)!r} is not {steps:g} steps ({want:.6g})")
            half[key] = units
            hyst[key] = hs * step
        return cls(half, hyst)

    def reset(self):
        self.anchor.clear()
        self.dir.clear()

    def __call__(self, action: dict) -> dict:
        out = dict(action)
        for key, h in self.half.items():
            if key not in action:
                continue
            v = float(action[key])
            a = self.anchor.get(key)
            d = self.dir.get(key, 0)
            if a is None:
                a = v
            elif v > a + (self.hyst[key] if d <= 0 else 0.0):
                d, a = 1, v  # moving up (or still moving up: follow the extreme)
            elif v < a - (self.hyst[key] if d >= 0 else 0.0):
                d, a = -1, v
            self.anchor[key], self.dir[key] = a, d
            out[key] = v + d * h
        return out


def wrap_send_action(robot, corr: PlayCorrection):
    """Correct every command a LeRobot robot sends. Returns what the robot sent, per Robot.send_action."""
    orig = robot.send_action

    def send_action(action):
        return orig(corr(action))

    robot.send_action = send_action
    return robot


if __name__ == "__main__":
    c = PlayCorrection.from_card(json.load(open(sys.argv[1])))
    print(json.dumps({"half_play": c.half, "hysteresis": c.hyst}, indent=1))
