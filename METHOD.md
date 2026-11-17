# holdcheck: method, corpus results and limits

Version 0.4.2, November 2026. Evidence kind: published teleoperation recordings read by this instrument. No arm was moved for this work, and nothing here is a bench trial.

## What it measures

A LeRobot teleoperation recording stores the leader arm (`action`, the command) and the follower arm (`observation.state`) at every frame. When the leader is held still for half a second, the follower should settle on the command. holdcheck takes the follower's settled error at the end of every such hold and splits it by the direction the joint arrived from.

- **Lost motion**: the width b of a play operator (the output moves only when the command leaves the band out ± b/2), fitted to the settled error at the end of every hold by replaying the command from the approach point. It is the play of the closed servo loop: deadband, friction against LeRobot's soft position gain and gear play inside the loop. A plain gap of median errors (above minus below, reported as `lost_motion_median_gap_steps`) reads about one step short per side, because hand jitter or a small leader overshoot inside a hold leaves the output at the arrival-side edge. The instrument check caught this (12 steps read as 10, 6 as 4), which is why the fit is used. The fit searches 0 to 60 steps (5.3 deg); a fit or its 90% interval that reaches 60 is refitted on 0 to 240 steps. Before 0.4.1 the search stopped at 60, so 6.6% of corpus recordings showed a joint at 60 when it had more (the median gap on those joints read 53 to 187 steps). A fit that still reaches 240 steps (21 deg) is not servo play; that joint is reported as not measured.
- **Scatter**: the spread of the settled error when arriving from one side (robust standard deviation).
- **Bias**: the median settled error over all holds (gravity sag plus a constant offset).
- **Free-hold scatter**: scatter on holds at least twice the hold time with a small gripper error, so nothing is in the jaws.
- **Settle creep**: the trimmed mean of how far the error still closes on the command, in the arrival direction, after the first 0.5 s of a long hold, with its standard error. It is signed on purpose. A servo that re-settles on each one-step jitter of the leader changes its error at random, and an unsigned drift would misread that as unfinished settling.

The step size is read from the recording itself. Joint values sit on the 4096-step encoder lattice whether LeRobot stored degrees or -100..100 over the calibrated range, so every figure is in encoder steps (0.088 deg) and degrees. A joint that is not on a lattice is reported as not measured.

A dataset whose `meta/info.json` does not declare one six-joint SO-style arm is skipped before any parquet is fetched. The same checks run again on the data.

### At the gripper

Each joint settles with Gaussian scatter of its measured sd (free-hold scatter where the recording has enough long empty-gripper holds). Arriving from any side, each joint also sits at one edge of its play band, either edge equally often. These go through the SO-101 tool-point Jacobian at 64 table poses, 13 to 25 cm in front of the base with the tool pointing down (`tipmap.json`, from the MuJoCo Menagerie SO-101 model). The card reports the 95% sideways radius of the tip arriving from one side and from any side. SO-100 links differ slightly from SO-101; the same map is used for both.

## Instrument check

`test_holdcheck.py` drives a simulated follower (a play operator of known width, a bias, Gaussian settling scatter, a 4-frame lag, whole-step quantisation, stored in degrees or in -100..100) with a teleop-like command. holdcheck must recover the lost motion and bias within one step and the scatter within about a third, including lost motion smaller than the hand's jitter (b = 0 and 2 steps). Further cases: squeezed jaws are excluded from free holds, a slow 8-step settling creep must be read and a no-creep arm must not show creep, the tip map gives an analytic 95% radius on a known map, and the `meta/info.json` skip. 0.4.1 adds three: 90 steps of play must read 90 (not the old ceiling of 60), play beyond 240 steps must read as not measured, and the refit must leave every other joint's figures unchanged. 0.4.2 adds two: no correction where the encoder step is not known, and no comparison with a family of fewer than 5 arms. 18 cases, all pass on 0.4.2. The corpus was read only after the check passed.

## Corpus

`lerobot/community_dataset_v3`: 1,755 datasets from 492 contributors (ids in `corpus/datasets.txt`). 1,574 were measured, 180 skipped (not a single six-joint SO-style arm, or action and state differ) and 1 failed to read. Each contributor counts as one arm per robot family (median of each figure over their datasets), so a prolific recorder does not dominate. 371 SO-100/101 arms result.

### Per joint, SO-100 and SO-101 together, one value per arm

| joint | gravity load | arms | lost motion p10 / p50 / p90 (deg) | scatter sd p50 / p90 (deg) | free-hold scatter p50 / p90 (deg, arms) | creep after 0.5 s p50 / p90 (deg) |
|---|---|---|---|---|---|---|
| shoulder_pan | no | 348 | 0.44 / 0.75 / 1.22 | 0.20 / 0.39 | 0.19 / 0.33 (219) | 0.01 / 0.05 |
| shoulder_lift | yes | 308 | 0.00 / 0.79 / 2.90 | 0.42 / 1.04 | 0.36 / 0.75 (126) | 0.03 / 0.09 |
| elbow_flex | yes | 329 | 1.84 / 2.59 / 4.66 | 0.42 / 0.91 | 0.34 / 0.75 (161) | 0.02 / 0.07 |
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
| arriving from any side (mm) | 6.58 | 7.50 | **9.23** | 11.92 | 16.10 |

Approaching every target from the same side removes the play from the error and leaves the scatter. For a typical arm that shrinks the 95% placement radius from about 9 mm to about 3.6 mm.

## Play correction (0.4)

The play can be cancelled in software instead of by path. The closed servo loop settles half its play short of the target, on the side it arrived from. `playfix.py` aims half the fitted play past each target, in the direction the joint is moving, so the follower settles on the target from either side. Since 0.4 the card carries the amount per joint (`correction`, in the recording's own units). The direction is tracked with 2 steps of hysteresis so that jitter in a held command does not flip it. A joint with more than 60 steps (5.3 deg) of play gets no correction: play that large is a loose horn, screw or gear, and aiming that far past a target is not a fix. `playfix.py` checks the whole block before it applies any of it: at most 30 steps per joint; a size in the recording's units equal to those steps times the joint's encoder step, as holdcheck reads that step for degrees, raw steps or a -100..100 range of at most one turn; each joint's own recorded action key, used once; the five arm joints only; an SO-100/101 or Koch arm. A card that fails any check is refused and nothing is applied.

Since 0.4.2 the correction is also withheld where the size of an encoder step is not known. The step holdcheck uses (1/4096 turn) is that of the SO-100/101's STS3215 and of Koch's Dynamixel servos. Another arm (Piper, Moss, other robot types) gets no correction, and neither does a joint whose -100..100 range spans more than one servo turn (4,096 steps), which only a wrong calibration produces. Each withheld joint or arm gets a line saying why. In the corpus this withholds the correction from 4 of 1,110 recordings (2 Moss, 1 Piper, 1 SO-100) and from 5 more SO-100 joints. `playfix.py` 0.4.2 refuses a card made before 0.4.2 and applies only the card's own `correction` block: an empty block corrects nothing. When a joint with play is withheld, the correction covers only the other joints; the card says so and does not offer the one-side figure with it (108 of the 1,106 corrected corpus cards). The table below was computed the same way (no correction above 60 steps).

```python
import json
from playfix import PlayCorrection, wrap_send_action
robot = wrap_send_action(robot, PlayCorrection.from_card(json.load(open("card.json"))))
```

**Does a correction fitted on one recording hold on another?** On recordings, replayed: the play was fitted on one part of an arm's data and the held-out holds were corrected with it, then the 95% sideways tip radius was compared. One value per contributor. The decision rule was fixed before the run.

| per arm, median | within one recording (211 arms) | across two recordings of one contributor (122 arms) |
|---|---|---|
| corrected, any side / offset only, any side | 0.70 | 0.73 |
| corrected, any side / one side only | 0.99 | 1.02 |
| correction beats offset only | 93% of arms | 89% of arms |

So a corrected arm arriving from any side lands about as close as the same arm arriving from one side. The fit does not last forever: between two recordings the play moved by more than 8 steps on 30% of shoulder_lift and 27% of elbow_flex pairs (some of these pairs are probably two arms recorded by one person). Fit the correction from a recent recording of the same arm, and again after re-assembly, re-calibration or a servo swap.

`test_playfix.py` closes the loop on a simulated follower (play operator, settle scatter, 4-frame lag, whole-step encoder). With 24 steps of play, holdcheck reads 24 before the correction and 1 after on every joint, with settle scatter unchanged and no hold corrected the wrong way. With no play, the correction is at most half a step and nothing gets worse. That simulated follower uses the same play model as the correction, so it checks the code, not the physics. The same file checks that edited or foreign blocks (sizes that disagree with their steps, non-finite numbers, another joint's action key, unknown encoder steps) are refused.

**Not tried on a physical arm.** Overshoot or hunting when a joint reverses, and the interaction with LeRobot's position gain, are unknown. LeRobot's record loop saves `action` before it is sent, so a recording made through the wrapper keeps the intended target, and holdcheck on that recording measures the play that is left. That second recording is the check on a real arm.

## Seat success in a pick-and-place cell, and supervisor thresholds (0.5)

A gripper figure says how far this arm lands from a target. A cell owner asks something else: in a cell that picks a part and seats it in a socket, how often does the part go in, and when should the cell call a person? Since 0.5 the card answers both for SO-100/101 arms, as a simulation.

**The cell.** A simulated SO-101 picks a part from a nest and seats it in a socket, both at random places in the work area, with automatic calibration, a camera with 0.5 mm of noise, and every target approached from one side. Each joint gets this arm's own play band (half the fitted lost motion) and settle scatter. A placement seats when the part lands within the lead-in (3 or 5 mm) of the socket. The simulator for this cell is not in this repository yet, so the seat figures cannot be reproduced from it. What is here is the table the 686 runs produced (`seatmap.json`) and the code that looks a card up in it (`seat_figures` in `holdcheck.py`); the threshold rule below is plain arithmetic you can check.

**From card to figure.** Running the cell takes about 6 seconds of desktop processor time per arm, more in a browser. Instead, all 686 SO-100/101 corpus recordings with five tip joints measured were run through it (10 simulated units of 500 placements each), and the card looks its arm up in a monotone table (`seatmap.json`). The key is the card's own chance that one-side tip error stays within the lead-in, from the gripper figure's draws. It ranks arms almost exactly as the cell does (rank correlation 0.995 at 3 mm, 0.990 at 5 mm). Cross-validated on five folds, the table is within 3.3 points of the recording's own simulation for 9 in 10 held-out recordings at 3 mm and within 2.7 points at 5 mm. The card prints that band. The folds split recordings, not arms: the 686 recordings come from about 277 arms, many with several recordings (up to 49), so recordings of one arm can sit in both the fitted and the held-out part. The band is an error over held-out recordings; how well the table does on an arm it has never seen was not measured. An arm looser than every fitted one gets no figure.

**Over the corpus.** One value per arm (277 arms): median 72% at 3 mm (p10 to p90: 36% to 96%) and 94% at 5 mm (61% to 100%). An arm built from the median of each joint separately does better (87% and 99%), because the loose joints of a typical arm are not all median joints.

**Supervisor thresholds.** A cell supervisor that treats a run of misses as "something changed" keeps calling a person when the arm simply misses on its own, and mostly for the wrong fix. In a simulated fault campaign on arms with corpus play and scatter, a supervisor set for an arm with no misses of its own (a seat rule of 4 misses in the last 60 placements or 3 in a row, and fixed trend-alarm limits) made 18.8 wrong calls per persistent fault on an arm with the median play and scatter on every joint, at 3 mm. With two things changed together, the seat rule set from the arm's own miss rate p0 and the trend alarm's limits scaled from the arm's own baseline scatter, it made 1.03. That run does not separate the two, so how much of the drop the seat rule gives on its own was not measured. The card gives only the seat rule, not the trend-alarm change: k misses in the last 60, k = max(4, smallest k with P(Binomial(60, p0) >= k) < 0.001), or s in a row, s = max(3, smallest s with p0^s < 1e-4). The cost is that some small faults are noticed later. The rule was tested up to p0 = 0.33; for an arm that misses more often the card offers no threshold, because a larger lead-in or a tighter arm comes first. Count the cell's own misses when it is set up and use that count instead: the card's p0 is a model figure.

## Limits

- The STS3215's encoder sits on the output shaft. Play between the shaft and the link (horn screws, printed parts, link bending) does not appear. All figures are a lower bound on error at the tool.
- A hold where the arm presses on the table or an object reads as error. Medians limit this; they do not remove it. Free-hold figures are therefore still an upper bound on the servo's own scatter.
- Free-hold scatter covers fewer arms (126 to 224 per joint) because not every recording has long empty-gripper holds; those arms may differ from the rest.
- Every degree figure assumes the 1/4096-turn step of SO-100/101 and Koch servos. For other arms the figures are in that unit, which may not be theirs. A family is placed against other arms only when the corpus has at least 5 of them (SO-100/101 together, 371; Koch, 12), so a lone Piper or Moss arm is not compared with itself.
- One contributor is taken as one arm per family. Some record with several arms; some arms appear under several names.
- The tip figures use one SO-101 map for both arms, at table poses only, and assume joint errors are independent.
- Seat success and thresholds are simulation. The cell has no load-dependent play, wear, temperature drift or correlation between joints, and its part, nest and camera are one design. No physical cell was run.
- Recordings were made by teleoperation. An arm under a learned policy or a scripted controller may settle differently.

Prepared by CyberNative AI LLC, an AI-run company, using AI agents.
