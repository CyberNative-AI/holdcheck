# holdcheck: method, corpus results and limits

Version 0.3.1, November 2026. Evidence kind: published teleoperation recordings read by this instrument. No arm was moved for this work, and nothing here is a bench trial.

## What it measures

A LeRobot teleoperation recording stores the leader arm (`action`, the command) and the follower arm (`observation.state`) at every frame. When the leader is held still for half a second, the follower should settle on the command. holdcheck takes the follower's settled error at the end of every such hold and splits it by the direction the joint arrived from.

- **Lost motion**: the width b of a play operator (the output moves only when the command leaves the band out ± b/2), fitted to the settled error at the end of every hold by replaying the command from the approach point. It is the play of the closed servo loop: deadband, friction against LeRobot's soft position gain and gear play inside the loop. A plain gap of median errors (above minus below, reported as `lost_motion_median_gap_steps`) reads about one step short per side, because hand jitter or a small leader overshoot inside a hold leaves the output at the arrival-side edge. The instrument check caught this (12 steps read as 10, 6 as 4), which is why the fit is used.
- **Scatter**: the spread of the settled error when arriving from one side (robust standard deviation).
- **Bias**: the median settled error over all holds (gravity sag plus a constant offset).
- **Free-hold scatter**: scatter on holds at least twice the hold time with a small gripper error, so nothing is in the jaws.
- **Settle creep**: the trimmed mean of how far the error still closes on the command, in the arrival direction, after the first 0.5 s of a long hold, with its standard error. It is signed on purpose. A servo that re-settles on each one-step jitter of the leader changes its error at random, and an unsigned drift would misread that as unfinished settling.

The step size is read from the recording itself. Joint values sit on the 4096-step encoder lattice whether LeRobot stored degrees or -100..100 over the calibrated range, so every figure is in encoder steps (0.088 deg) and degrees. A joint that is not on a lattice is reported as not measured.

A dataset whose `meta/info.json` does not declare one six-joint SO-style arm is skipped before any parquet is fetched. The same checks run again on the data.

### At the gripper

Each joint settles with Gaussian scatter of its measured sd (free-hold scatter where the recording has enough long empty-gripper holds). Arriving from any side, each joint also sits at one edge of its play band, either edge equally often. These go through the SO-101 tool-point Jacobian at 64 table poses, 13 to 25 cm in front of the base with the tool pointing down (`tipmap.json`, from the MuJoCo Menagerie SO-101 model). The card reports the 95% sideways radius of the tip arriving from one side and from any side. SO-100 links differ slightly from SO-101; the same map is used for both.

## Instrument check

`test_holdcheck.py` drives a simulated follower (a play operator of known width, a bias, Gaussian settling scatter, a 4-frame lag, whole-step quantisation, stored in degrees or in -100..100) with a teleop-like command. holdcheck must recover the lost motion and bias within one step and the scatter within about a third, including lost motion smaller than the hand's jitter (b = 0 and 2 steps). Further cases: squeezed jaws are excluded from free holds, a slow 8-step settling creep must be read and a no-creep arm must not show creep, the tip map gives an analytic 95% radius on a known map, and the `meta/info.json` skip. 12 cases, all pass on 0.3.1. The corpus was read only after the check passed.

## Corpus

`lerobot/community_dataset_v3`: 1,755 datasets from 492 contributors (ids in `corpus/datasets.txt`). 1,574 were measured, 180 skipped (not a single six-joint SO-style arm, or action and state differ) and 1 failed to read. Each contributor counts as one arm per robot family (median of each figure over their datasets), so a prolific recorder does not dominate. 371 SO-100/101 arms result.

### Per joint, SO-100 and SO-101 together, one value per arm

| joint | gravity load | arms | lost motion p10 / p50 / p90 (deg) | scatter sd p50 / p90 (deg) | free-hold scatter p50 / p90 (deg, arms) | creep after 0.5 s p50 / p90 (deg) |
|---|---|---|---|---|---|---|
| shoulder_pan | no | 348 | 0.44 / 0.75 / 1.22 | 0.20 / 0.39 | 0.19 / 0.33 (219) | 0.01 / 0.05 |
| shoulder_lift | yes | 308 | 0.00 / 0.79 / 2.87 | 0.42 / 1.04 | 0.36 / 0.75 (126) | 0.03 / 0.09 |
| elbow_flex | yes | 329 | 1.84 / 2.59 / 4.58 | 0.42 / 0.91 | 0.34 / 0.75 (161) | 0.02 / 0.07 |
| wrist_flex | yes | 346 | 0.70 / 1.14 / 2.11 | 0.26 / 0.59 | 0.20 / 0.67 (215) | 0.03 / 0.07 |
| wrist_roll | no | 339 | 0.53 / 0.73 / 1.23 | 0.13 / 0.33 | 0.13 / 0.30 (224) | 0.02 / 0.06 |

SO-100 and SO-101 are close. The SO-101 shoulder_lift is looser (median lost motion 1.14 against 0.44 deg).

### What the corpus says

1. **Settling is finished at 0.5 s.** Creep after the first 0.5 s of a long hold is under one encoder step (0.088 deg) on every joint at p50 and p90.
2. **The scatter is the arm, not the recording.** Contact and a payload in the gripper would make scatter read high. On long holds with nothing in the jaws, loaded-joint scatter drops by about a fifth (elbow 0.42 to 0.34 deg) and free-joint scatter not at all.
3. **The largest play is on loaded joints.** A common assumption is that gravity takes up the play on a loaded joint. The corpus shows the largest gap between approach directions on the elbow (median 2.6 deg).
4. **Approach direction matters more than anything else an owner controls.** At the gripper (280 arms with all five joints measured):

| 95% sideways radius at the tip | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| arriving from one side (mm) | 1.82 | 2.49 | **3.63** | 5.11 | 7.04 |
| arriving from any side (mm) | 6.58 | 7.50 | **9.23** | 11.82 | 14.89 |

Approaching every target from the same side removes the play from the error and leaves the scatter. For a typical arm that shrinks the 95% placement radius from about 9 mm to about 3.6 mm.

## Limits

- The STS3215's encoder sits on the output shaft. Play between the shaft and the link (horn screws, printed parts, link bending) does not appear. All figures are a lower bound on error at the tool.
- A hold where the arm presses on the table or an object reads as error. Medians limit this; they do not remove it. Free-hold figures are therefore still an upper bound on the servo's own scatter.
- Free-hold scatter covers fewer arms (126 to 224 per joint) because not every recording has long empty-gripper holds; those arms may differ from the rest.
- One contributor is taken as one arm per family. Some record with several arms; some arms appear under several names.
- The tip figures use one SO-101 map for both arms, at table poses only, and assume joint errors are independent.
- Recordings were made by teleoperation. An arm under a learned policy or a scripted controller may settle differently.

Prepared by CyberNative AI LLC, an AI-run company, using AI agents.
