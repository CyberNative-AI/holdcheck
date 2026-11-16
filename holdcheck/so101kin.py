"""Batched kinematics and gravity torques for the SO-101 arm, read from the upstream MJCF.

numpy only. Reads body frames, joint axes and ranges, inertial masses and centres of mass and the
`gripperframe` site from `model/so101.xml` (MuJoCo Menagerie `robotstudio_so101/so101.xml`,
Apache-2.0, see `model/NOTICE.md`). No meshes, no dynamics: this is a quasi-static model.

Conventions follow MuJoCo: a body's frame is its parent's frame composed with (pos, quat), then
rotated about the joint axis (through the body origin) by the joint angle. Quaternions are w x y z.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "model", "so101.xml")
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
G = 9.81


def _vec(s, n):
    v = np.array([float(x) for x in s.split()], dtype=float)
    assert v.shape == (n,), (s, n)
    return v


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_batch(axis, ang):
    """Rodrigues for one fixed axis and a batch of angles -> (N,3,3)."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s = np.sin(ang)[:, None, None]
    c = np.cos(ang)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


@dataclass
class Link:
    name: str
    pos: np.ndarray          # body origin in parent frame
    rot: np.ndarray          # body orientation in parent frame (3,3)
    axis: np.ndarray         # joint axis in body frame
    qrange: tuple
    mass: float
    com: np.ndarray          # centre of mass in body frame


@dataclass
class Arm:
    """One arm instance. `links` are the five actuated bodies in chain order."""
    links: list
    base_pos: np.ndarray
    base_rot: np.ndarray
    tool_pos: np.ndarray     # gripperframe site in the last body frame
    tool_rot: np.ndarray
    extra_tip_mass: float = 0.0          # moving jaw (lumped) + payload, at the tool point
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.links)

    def qmin(self):
        return np.array([l.qrange[0] for l in self.links])

    def qmax(self):
        return np.array([l.qrange[1] for l in self.links])

    def frames(self, q):
        """q (N,5) -> lists of body origins (N,3) and rotations (N,3,3), after each joint."""
        q = np.atleast_2d(q)
        N = q.shape[0]
        R = np.broadcast_to(self.base_rot, (N, 3, 3)).copy()
        p = np.broadcast_to(self.base_pos, (N, 3)).copy()
        origins, rots = [], []
        for j, l in enumerate(self.links):
            p = p + np.einsum("nij,j->ni", R, l.pos)
            R = R @ l.rot[None]
            R = R @ axis_angle_batch(l.axis, q[:, j])
            origins.append(p.copy())
            rots.append(R.copy())
        return origins, rots

    def fk(self, q):
        """Tool point (N,3) and unit approach axis (N,3) in the base/world frame."""
        origins, rots = self.frames(q)
        R = rots[-1]
        p = origins[-1] + np.einsum("nij,j->ni", R, self.tool_pos)
        Rt = R @ self.tool_rot[None]
        return p, Rt[:, :, APPROACH_COL] * APPROACH_SIGN

    def gravity_torque(self, q, payload=0.0):
        """Generalised gravity torque about each joint axis (N,5), N·m.

        tau_j = axis_j . sum_i (c_i - o_j) x (m_i g), over bodies i at or after joint j.
        """
        origins, rots = self.frames(q)
        N = origins[0].shape[0]
        coms = [origins[i] + np.einsum("nij,j->ni", rots[i], self.links[i].com) for i in range(self.n)]
        masses = [l.mass for l in self.links]
        tip = origins[-1] + np.einsum("nij,j->ni", rots[-1], self.tool_pos)
        coms.append(tip)
        masses.append(self.extra_tip_mass + payload)
        g = np.array([0.0, 0.0, -G])
        tau = np.zeros((N, self.n))
        for j in range(self.n):
            axis_w = np.einsum("nij,j->ni", rots[j], self.links[j].axis)
            tot = np.zeros((N, 3))
            for i in range(j, self.n + 1):
                r = coms[i] - origins[j]
                tot += np.cross(r, masses[i] * g[None])
            tau[:, j] = np.einsum("ni,ni->n", axis_w, tot)
        return tau

    def pose5(self, q):
        """5-vector task coordinate: tool position (3) and approach-axis x,y components (2)."""
        p, a = self.fk(q)
        return np.concatenate([p, a[:, :2]], axis=1)

    def jac5(self, q, eps=1e-6):
        """Numeric Jacobian of pose5 wrt q: (N,5,5)."""
        q = np.atleast_2d(q)
        f0 = self.pose5(q)
        J = np.zeros((q.shape[0], 5, self.n))
        for j in range(self.n):
            dq = q.copy()
            dq[:, j] += eps
            J[:, :, j] = (self.pose5(dq) - f0) / eps
        return J

    def ik_down(self, target, q0=None, iters=60, damp=1e-3, tol=2e-6):
        """Damped least squares IK for tool position `target` (N,3) with the approach axis pointing
        straight down. Returns (q, ok). `ok` is False where the residual exceeds 0.05 mm, the axis
        is more than 0.1 deg off vertical or a joint limit is hit."""
        target = np.atleast_2d(target)
        N = target.shape[0]
        if q0 is None:
            pan = np.arctan2(target[:, 1], target[:, 0] - self.links[0].pos[0])
            q = np.tile(np.array([0.0, -0.3, 0.6, 1.27, 0.0]), (N, 1))
            q[:, 0] = PAN_SIGN * pan
        else:
            q = np.array(q0, dtype=float, copy=True)
        want = np.concatenate([target, np.zeros((N, 2))], axis=1)
        lo, hi = self.qmin(), self.qmax()
        for _ in range(iters):
            f = self.pose5(q)
            e = want - f
            # weight the axis terms so 1 mrad counts like 0.1 mm
            e_w = e * np.array([1, 1, 1, 0.1, 0.1])
            if np.max(np.abs(e_w)) < tol:
                break
            J = self.jac5(q) * np.array([1, 1, 1, 0.1, 0.1])[None, :, None]
            JT = np.transpose(J, (0, 2, 1))
            H = JT @ J + damp * np.eye(self.n)[None]
            dq = np.linalg.solve(H, (JT @ e_w[:, :, None]))[:, :, 0]
            dq = np.clip(dq, -0.3, 0.3)
            q = np.clip(q + dq, lo, hi)
        p, a = self.fk(q)
        pos_err = np.linalg.norm(p - target, axis=1)
        tilt = np.arccos(np.clip(-a[:, 2], -1, 1))
        at_limit = np.any((q <= lo + 1e-9) | (q >= hi - 1e-9), axis=1)
        ok = (pos_err < 5e-5) & (tilt < np.deg2rad(0.1)) & (~at_limit)
        return q, ok


# The gripperframe site's approach direction and the pan sign are fixed by the model; they are
# checked in tests/test_so101.py (tool moves toward +x when the arm reaches forward).
APPROACH_COL = 0
APPROACH_SIGN = 1.0
PAN_SIGN = -1.0


def load_arm(xml_path=DEFAULT_XML, jaw_angle=0.0) -> Arm:
    root = ET.parse(xml_path).getroot()
    wb = root.find("worldbody")
    base = wb.find("body[@name='base']")
    assert base is not None
    chain_names = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
    links = []
    cur = base
    for bn, jn in zip(chain_names, ARM_JOINTS):
        b = cur.find(f"body[@name='{bn}']")
        assert b is not None, bn
        j = b.find("joint")
        assert j.get("name") == jn, (j.get("name"), jn)
        inert = b.find("inertial")
        links.append(Link(
            name=bn,
            pos=_vec(b.get("pos"), 3),
            rot=quat_to_mat(_vec(b.get("quat", "1 0 0 0"), 4)),
            axis=_vec(j.get("axis"), 3),
            qrange=tuple(_vec(j.get("range"), 2)),
            mass=float(inert.get("mass")),
            com=_vec(inert.get("pos"), 3),
        ))
        cur = b
    site = None
    for s in cur.iter("site"):
        if s.get("name") == "gripperframe":
            site = s
    assert site is not None
    jaw = None
    for b in cur.findall("body"):
        if b.get("name", "").startswith("moving_jaw"):
            jaw = b
    jaw_mass = float(jaw.find("inertial").get("mass")) if jaw is not None else 0.0
    base_inert = base.find("inertial")
    return Arm(
        links=links,
        base_pos=_vec(base.get("pos", "0 0 0"), 3),
        base_rot=quat_to_mat(_vec(base.get("quat", "1 0 0 0"), 4)),
        tool_pos=_vec(site.get("pos"), 3),
        tool_rot=quat_to_mat(_vec(site.get("quat", "1 0 0 0"), 4)),
        extra_tip_mass=jaw_mass,
        meta={"xml": os.path.relpath(xml_path, HERE), "base_mass": float(base_inert.get("mass")),
              "moving_mass": sum(l.mass for l in links) + jaw_mass},
    )
