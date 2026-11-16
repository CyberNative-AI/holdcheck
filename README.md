# holdcheck

How much play and settle scatter does your SO-100 or SO-101 arm have, and what does that do at the gripper?

holdcheck reads a LeRobot teleoperation recording you already published on Hugging Face and measures, for each joint, how far the follower settles from the leader's command when the leader holds still. It splits that by the direction the joint arrived from, then turns the joint figures into the 95% sideways radius at the gripper tip, and places your arm against 371 other SO-100/101 arms measured the same way.

**Try it in the browser:** https://huggingface.co/spaces/CyberNative-AI/holdcheck

The analysis runs in your browser and the dataset is read directly from Hugging Face. We receive nothing you paste. Hugging Face hosts the page and sees the visit. No analytics. The page's scripts, the open-source Python runtime it uses and its fonts all come from Hugging Face, which hosts the page; none come from a third-party CDN.

## What 371 arms show

One value per arm, SO-100 and SO-101 together, measured at the servo output shaft:

| joint | lost motion p50 (deg) | settle scatter sd p50 (deg) |
|---|---|---|
| shoulder_pan | 0.75 | 0.20 |
| shoulder_lift | 0.79 | 0.42 |
| elbow_flex | 2.59 | 0.42 |
| wrist_flex | 1.14 | 0.26 |
| wrist_roll | 0.73 | 0.13 |

At the gripper (280 arms with all five joints measured), the 95% sideways placement radius of a typical arm is **3.6 mm when every target is approached from the same side** and **9.2 mm from any side** (p10 to p90: 1.8 to 7.0 mm, and 6.6 to 16.1 mm). If you place things with an SO arm, approaching from one side is the cheapest accuracy you can buy.

If your path or policy can't approach from one side, `holdcheck/playfix.py` cancels the measured play in software: it aims half each joint's play past the target in the direction the joint is moving. On the recordings of 122 arms, a correction fitted on one recording and applied to another landed about as close from any side as one-side approaches (median ratio 1.02). If your card withholds a joint with play, the correction covers only the joints it lists, and the card says the one-side figure is not established with it. We have not tried it on a physical arm. Your recording made through it is that check, and holdcheck on that recording shows the play that is left. Refit after re-assembly, re-calibration or a servo swap. Details: [METHOD.md](METHOD.md#play-correction-04).

These are the servos alone, a lower bound at the tool: horns, screws and printed links add more. Full method, tables and limits: [METHOD.md](METHOD.md).

## Run it yourself

Needs Python 3.11+, `numpy` and `pyarrow`. It reads only `meta/info.json` and `data/*.parquet`, never videos.

```sh
cd holdcheck
python holdcheck.py --repo <user>/<dataset> --out card.json --print
python holdcheck.py --local ~/.cache/huggingface/lerobot/<user>/<dataset> --out card.json --print
python test_holdcheck.py      # instrument check on a simulated arm with known play
python test_playfix.py        # the correction on a simulated follower, closed loop
```

Apply the correction to a LeRobot follower (any robot object with `send_action`). `card.json` is the card written by `--out` above, or the file saved by the page's "Download card (JSON)" (named `holdcheck-<dataset>.json`); give its name:

```python
import json
from playfix import PlayCorrection, wrap_send_action
robot = wrap_send_action(robot, PlayCorrection.from_card(json.load(open("card.json"))))
```

Rebuild the corpus table:

```sh
python corpus.py list --repo lerobot/community_dataset_v3 --out runs/index.json
python corpus.py run --index runs/index.json --cards runs/cards --workers 8
python corpus.py summary --cards runs/cards --out runs/summary.json --md runs/summary.md
```

The input dataset ids are in [corpus/datasets.txt](corpus/datasets.txt). The comparison table that ships with the tool, `holdcheck/corpus_summary.json`, holds distributions only. Per-dataset results are not published; you get a card for the dataset you run.

The browser page is in `web/`. It runs the same `holdcheck.py`, unchanged, in Pyodide, and its card equals the command line's.

## Contents

- `holdcheck/holdcheck.py`: the measurement (numpy and pyarrow)
- `holdcheck/test_holdcheck.py`: instrument check on a simulated arm with known play, scatter and creep
- `holdcheck/playfix.py`, `holdcheck/test_playfix.py`: the play correction from a card, a LeRobot `send_action` wrapper and its closed-loop check
- `holdcheck/corpus.py`: list, run and summarise a corpus
- `holdcheck/tipmap.json`, `holdcheck/tipmap.py`, `holdcheck/so101kin.py`, `holdcheck/model/`: SO-101 tool-point Jacobians at 64 table poses, and how to rebuild them. `python holdcheck/tipmap.py --out tipmap.json` reads the bundled MuJoCo Menagerie SO-101 model (`model/so101.xml`, Apache-2.0, see `model/NOTICE.md`) with numpy only and reproduces the shipped Jacobians exactly.
- `holdcheck/corpus_summary.json`: the comparison table (percentiles only)
- `web/`: the static page, Pyodide and the fonts it serves itself
- `METHOD.md`: method, corpus results and limits

## Licence

Code and tables: Apache-2.0 (see `LICENSE`). Bundled third-party files: see `THIRD-PARTY.md`. The corpus figures are derived from `lerobot/community_dataset_v3` (Apache-2.0).

Prepared by CyberNative AI LLC, an AI-run company, using AI agents.
