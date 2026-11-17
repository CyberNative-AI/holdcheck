"""The simulated pick-and-place cell behind the card's seat figures (`seat` in holdcheck.py).

A quasi-static model of an SO-101 arm in a bench cell with an overhead camera. It is the cell that
produced `seatmap.json`; `seatsim.py` runs it. Numpy only; kinematics and gravity torques come from
`so101kin.py` and `model/so101.xml`.

What is modelled (every parameter is explicit below):
  - unit-to-unit variation: joint zero offsets, link-origin tolerances, base mounting position and yaw;
  - proportional-only servo position control: steady-state sag = gravity torque / kp;
  - per joint, the arm's own play band (half-width = half the card's lost motion): the link stays
    where it was if that is inside the band around the motor, otherwise it is dragged to the band's
    edge; then Gaussian settle scatter of the card's sd is added;
  - the 12-bit encoder's command resolution;
  - a camera that reports world positions with isotropic Gaussian noise;
  - automatic calibration (12 looks, 9 fitted parameters), then open-loop placement from it, every
    target approached from one side (each joint finishes its move in the + direction).
What is not modelled: dynamics, vibration, contact forces, stick-slip, load-dependent play, wear,
temperature, cable drag, correlation between joints, vision failures. A lead-in (capture radius)
stands in for the whole insertion contact problem. Simulation of a stated model: not a measurement.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

import so101kin

DEG = np.pi / 180.0


@dataclass(frozen=True)
class Mech:
    kp: float = 17.8              # N m/rad, joint stiffness of the STS3215 servo class (TheRobotStudio joint file)
    enc_counts: int = 4096        # STS3215 12-bit encoder
    offset_sd_deg: float = 1.0    # per-unit joint zero offset sd (assumption)
    link_sd_mm: float = 0.3       # per-unit link-origin tolerance sd, each axis (assumption)
    base_sd_mm: float = 2.0       # per-unit base mounting position sd in x and y (assumption)
    base_yaw_sd_deg: float = 0.5  # per-unit base mounting yaw sd (assumption)
    payload: float = 0.02         # kg at the tool point


@dataclass(frozen=True)
class Cam:
    sd_mm: float = 0.5            # per-axis position noise of one look (assumption)


@dataclass(frozen=True)
class Task:
    grasp_capture_mm: float = 6.0  # lateral miss the gripper still closes on the part (the nest's capture)
    carry: float = 0.5             # fraction of the grasp miss that stays as in-hand offset
    tilt_max_deg: float = 4.0      # tool tilt the socket's lead-in tolerates


class Unit:
    """One physical arm in one cell: true kinematics, hidden per-unit offsets, and per-joint play and
    settle scatter (five tip joints, base to wrist roll, in degrees at the servo output)."""

    def __init__(self, seed: int, half_width_deg, scatter_sd_deg, mech: Mech = Mech(), nominal=None):
        self.seed = seed
        self.mech = mech
        self.nominal = nominal or so101kin.load_arm()
        self.premove_deg = float(max(half_width_deg))
        self.B = np.asarray(half_width_deg) * DEG
        self.S = np.asarray(scatter_sd_deg) * DEG
        rng = np.random.default_rng([seed, 7919])
        arm = copy.deepcopy(self.nominal)
        for l in arm.links:
            l.pos = l.pos + rng.normal(0, mech.link_sd_mm * 1e-3, 3)
        dxy = rng.normal(0, mech.base_sd_mm * 1e-3, 2)
        yaw = rng.normal(0, mech.base_yaw_sd_deg * DEG)
        arm.base_pos = arm.base_pos + np.array([dxy[0], dxy[1], 0.0])
        c, s = np.cos(yaw), np.sin(yaw)
        arm.base_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ arm.base_rot
        self.true = arm
        self.offsets = rng.normal(0, mech.offset_sd_deg * DEG, arm.n)
        self.res = 2 * np.pi / mech.enc_counts

    def _play(self, u, rng, y_prev):
        """Link angle for motor-side equilibrium u (N,5): play band, then settle scatter. `y_prev` is the
        link state before the move (None = unknown history, drawn uniformly inside the band)."""
        if y_prev is None:
            y = u + self.B * rng.uniform(-1, 1, u.shape)
        else:
            y = np.clip(y_prev, u - self.B, u + self.B)
        return y + self.S * rng.standard_normal(u.shape)

    def settle(self, q_cmd, rng, y_prev=None):
        m = self.mech
        q_c = np.round(np.atleast_2d(q_cmd) / self.res) * self.res
        q = q_c + self.offsets
        for _ in range(3):
            tau = self.true.gravity_torque(q, m.payload)
            q = q_c + self.offsets + tau / m.kp
        return self._play(q, rng, y_prev)

    def reach(self, q_cmd, rng, y_prev=None):
        """After settling: true tool position (N,3), tilt from vertical (N,), link state (N,5)."""
        y = self.settle(q_cmd, rng, y_prev)
        p, a = self.true.fk(y)
        tilt = np.arccos(np.clip(-a[:, 2], -1, 1))
        return p, tilt, y


def look(p, cam: Cam, rng):
    return p + rng.normal(0, cam.sd_mm * 1e-3, p.shape)


AREA = dict(x=(0.13, 0.25), y=(-0.10, 0.10))  # work area in front of the base, m
Z_WORK = 0.03
HOME = np.array([[0.19, 0.0, 0.07]])


def sample_targets(n, rng, area=AREA, z=Z_WORK):
    x = rng.uniform(*area["x"], n)
    y = rng.uniform(*area["y"], n)
    return np.stack([x, y, np.full(n, z)], 1)


def home_state(unit: Unit, n, rng):
    """Link state after power-up and a move to the home pose (play history unknown)."""
    q, _ = unit.nominal.ik_down(HOME)
    return unit.settle(np.repeat(q, n, 0), rng, None)


def move(unit: Unit, q, rng, y_prev):
    """Approach from one side: first to q minus three of the widest play half-widths on every joint,
    then to q, so each joint finishes moving in the + direction."""
    d = 3.0 * unit.premove_deg * DEG
    _, _, y_prev = unit.reach(q - d, rng, y_prev)
    p, tilt, y = unit.reach(q, rng, y_prev)
    return p, tilt, y


@dataclass
class Calib:
    """Fitted per-unit correction: joint offsets, joint compliance 1/kp, base shift and yaw."""
    theta: np.ndarray  # (9,) = offsets(5), compliance(1), base dx, dy, dyaw
    rms_mm: float
    n_points: int

    def model_arm(self, nominal):
        arm = copy.copy(nominal)
        dx, dy, yaw = self.theta[6:9]
        c, s = np.cos(yaw), np.sin(yaw)
        arm.base_pos = nominal.base_pos + np.array([dx, dy, 0.0])
        arm.base_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ nominal.base_rot
        return arm


def _calib_predict(nominal, theta, q_cmd, payload):
    arm = Calib(theta, 0.0, 0).model_arm(nominal)
    q = q_cmd + theta[:5]
    for _ in range(2):
        q = q_cmd + theta[:5] + theta[5] * arm.gravity_torque(q, payload)
    p, _ = arm.fk(q)
    return p


def calib_poses(k, rng):
    side = int(np.ceil(np.sqrt(k / 2)))
    xs = np.linspace(AREA["x"][0], AREA["x"][1], side)
    ys = np.linspace(AREA["y"][0], AREA["y"][1], side)
    pts = np.array([(x, y, z) for z in (Z_WORK, Z_WORK + 0.03) for x in xs for y in ys])
    return pts[rng.permutation(len(pts))[:k]]


def calibrate(unit: Unit, cam: Cam, rng, k=12):
    """Automatic calibration, no human input: visit k poses over the work area at two heights, one
    camera look at the tool per pose, fit 9 parameters by regularised Gauss-Newton."""
    tgt = calib_poses(k, rng)
    q_cmd, ok = unit.nominal.ik_down(tgt)
    q_cmd = q_cmd[ok]
    y = home_state(unit, 1, rng)
    p_true = []
    for i in range(len(q_cmd)):
        p, _, y = move(unit, q_cmd[i:i + 1], rng, y)
        p_true.append(p[0])
    p_obs = look(np.array(p_true), cam, rng)
    theta = np.zeros(9)
    # weak zero-mean prior keeps the wrist-roll offset (barely observable from a tool point) tame
    w_prior = 1e-3 / np.array([3 * DEG] * 5 + [0.2, 0.01, 0.01, 2 * DEG])
    payload = unit.mech.payload
    for _ in range(12):
        r0 = (_calib_predict(unit.nominal, theta, q_cmd, payload) - p_obs).ravel()
        J = np.zeros((r0.size, 9))
        for i in range(9):
            d = np.zeros(9)
            d[i] = 1e-6
            J[:, i] = ((_calib_predict(unit.nominal, theta + d, q_cmd, payload) - p_obs).ravel() - r0) / 1e-6
        A = np.vstack([J, np.diag(w_prior)])
        b = np.concatenate([-r0, -w_prior * theta])
        step, *_ = np.linalg.lstsq(A, b, rcond=None)
        theta = theta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    r = _calib_predict(unit.nominal, theta, q_cmd, payload) - p_obs
    return Calib(theta, float(np.sqrt(np.mean(np.sum(r ** 2, 1))) * 1e3), int(len(q_cmd)))


def calibrated_command(unit: Unit, cal: Calib, target_obs):
    """Command that the calibrated model predicts will put the tool at target_obs."""
    arm = cal.model_arm(unit.nominal)
    q_true, ok = arm.ik_down(target_obs)
    tau = arm.gravity_torque(q_true, unit.mech.payload)
    return q_true - cal.theta[:5] - cal.theta[5] * tau, ok


def place(unit: Unit, cal: Calib, cam: Cam, target, rng, y_prev):
    """One open-loop placement on the calibrated model: one look at the target, one-sided approach."""
    obs = look(target, cam, rng)
    q, ok = calibrated_command(unit, cal, obs)
    p, tilt, y = move(unit, q, rng, y_prev)
    return dict(p=p, tilt=tilt, valid=ok, y=y)


def run_transfers(unit: Unit, cal: Calib, cam: Cam, n, rng, task: Task = Task()):
    """n independent pick-and-seat transfers: home -> source nest -> destination socket, both at uniformly
    drawn positions in the work area. Every transfer is returned; none is dropped. miss_mm is the
    sideways distance between the part's axis and the socket's axis at insertion."""
    src = sample_targets(n, rng)
    dst = sample_targets(n, rng)
    y0 = home_state(unit, n, rng)
    a = place(unit, cal, cam, src, rng, y0)
    b = place(unit, cal, cam, dst, rng, a["y"])
    e1 = (a["p"] - src)[:, :2] * 1e3
    e2 = (b["p"] - dst)[:, :2] * 1e3
    grasp_miss = np.linalg.norm(e1, axis=1)
    return dict(miss_mm=np.linalg.norm(e2 + task.carry * e1, axis=1),
                grasp_ok=a["valid"] & (grasp_miss <= task.grasp_capture_mm),
                tilt_ok=np.rad2deg(b["tilt"]) <= task.tilt_max_deg,
                valid=a["valid"] & b["valid"])


def success_at(rec, c_mm):
    """Per-transfer success for lead-in c_mm. A transfer whose pose had no solution is a failure."""
    return rec["valid"] & rec["grasp_ok"] & rec["tilt_ok"] & (rec["miss_mm"] <= c_mm)
