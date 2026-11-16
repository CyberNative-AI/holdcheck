#!/usr/bin/env python3
"""Bake the SO-101 tool-point Jacobians that turn joint figures (deg) into gripper-tip figures (mm).

holdcheck measures each joint at the servo shaft. An owner wants to know what that does at the
gripper. This script samples working poses on a table (a fixtured pick cell's work area: x 0.13 to
0.25 m, y -0.10 to 0.10 m, tool 0.03 m above the table, pointing straight down), solves IK on the
SO-101 model read from its MJCF (`model/so101.xml`, MuJoCo Menagerie, Apache-2.0, read by
`so101kin.py`) and stores the 3x5 positional Jacobian of the `gripperframe` point at each pose, in
mm per radian.

holdcheck then needs only numpy and `tipmap.json`. SO-100 links differ slightly from SO-101; the
same map is used for both and the card says so.

  python tipmap.py --out tipmap.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import so101kin  # noqa: E402

N_POSES = 64
SEED = 23982
AREA = dict(x=(0.13, 0.25), y=(-0.10, 0.10))  # m, in front of the base
Z_WORK = 0.03  # m, tool point above the table
XML_SHA256 = "982c02c027ec0c62d3459d0a6eb00dfb7dc13a451992e72ce99addc71f209805"


def sample_targets(n, rng):
    x = rng.uniform(*AREA["x"], n)
    y = rng.uniform(*AREA["y"], n)
    return np.stack([x, y, np.full(n, Z_WORK)], 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=so101kin.DEFAULT_XML, help="MuJoCo Menagerie robotstudio_so101/so101.xml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    arm = so101kin.load_arm(a.xml)
    rng = np.random.default_rng(SEED)
    T = sample_targets(4 * N_POSES, rng)
    q, ok = arm.ik_down(T)
    q = q[ok][:N_POSES]
    assert len(q) == N_POSES, f"only {len(q)} reachable poses"
    J = arm.jac5(q)[:, :3, :] * 1000.0  # (N,3,5) mm per rad
    p, _ = arm.fk(q)
    out = {
        "what": "SO-101 gripperframe positional Jacobian, mm per rad, joints in order",
        "joints": list(so101kin.ARM_JOINTS),
        "model": f"MuJoCo Menagerie robotstudio_so101/so101.xml (Apache-2.0, sha256 {XML_SHA256[:16]}), read by so101kin.py",
        "poses": f"{N_POSES} IK poses, tool pointing down, x {AREA['x']} m, y {AREA['y']} m, z {Z_WORK} m, seed {SEED}",
        "tool_mm": np.round(p * 1000.0, 2).tolist(),
        "jacobian_mm_per_rad": np.round(J, 3).tolist(),
    }
    json.dump(out, open(a.out, "w"), indent=0)
    lever = np.sqrt((J[:, :2, :] ** 2).sum(1).mean(0))
    print("rms lateral lever arm per joint (mm per rad):", dict(zip(so101kin.ARM_JOINTS, np.round(lever, 1))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
