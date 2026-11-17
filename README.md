# holdcheck

How much play and settle scatter does your SO-100 or SO-101 arm have, and what does that do at the gripper?

holdcheck reads a LeRobot teleoperation recording you already published on Hugging Face and measures, for each joint, how far the follower settles from the leader's command when the leader holds still. It splits that by the direction the joint arrived from, then turns the joint figures into the 95% sideways radius at the gripper tip, and places your arm against 647 other SO-100/101 arms measured the same way: 371 from a 2025 community corpus and 276 from a sample of Hugging Face datasets created in 2026.

**Try it in the browser:** https://huggingface.co/spaces/CyberNative-AI/holdcheck

The analysis runs in your browser and the dataset is read directly from Hugging Face. We receive nothing you paste. Hugging Face hosts the page and sees the visit. No analytics. The page's scripts, the open-source Python runtime it uses and its fonts all come from Hugging Face, which hosts the page; none come from a third-party CDN.

## What 371 arms of the 2025 corpus show

One value per arm, SO-100 and SO-101 together, measured at the servo output shaft:

| joint | lost motion p50 (deg) | settle scatter sd p50 (deg) |
|---|---|---|
| shoulder_pan | 0.75 | 0.20 |
| shoulder_lift | 0.79 | 0.42 |
| elbow_flex | 2.59 | 0.42 |
| wrist_flex | 1.14 | 0.26 |
| wrist_roll | 0.73 | 0.13 |

At the gripper (280 arms with all five joints measured), the 95% sideways placement radius of a typical arm is **3.6 mm when every target is approached from the same side** and **9.2 mm from any side** (p10 to p90: 1.8 to 7.0 mm, and 6.6 to 16.1 mm). If you place things with an SO arm, approaching from one side is the cheapest accuracy you can buy.

We checked the table against more recently published datasets, one per author, from repositories created in 2026 (239 with a gripper figure). In this sample, the gripper medians differed by less than the typical session shift: 3.35 mm from one side and 9.70 mm from any side. The sample's base pan and wrist roll read about two encoder steps more play (0.17 deg), which is more than that shift, so since 0.5.3 your card is placed against both sets together (518 arms with a gripper figure: median 3.5 mm and 9.5 mm). Details: [METHOD.md](METHOD.md#2026-sample-053).

If your path or policy can't approach from one side, `holdcheck/playfix.py` cancels the measured play in software: it aims half each joint's play past the target in the direction the joint is moving. On the recordings of 122 arms, a correction fitted on one recording and applied to another landed about as close from any side as one-side approaches (median ratio 1.02). If your card withholds a joint with play, the correction covers only the joints it lists, and the card says the one-side figure is not established with it. We have not tried it on a physical arm. Your recording made through it is that check, and holdcheck on that recording shows the play that is left. Refit after re-assembly, re-calibration or a servo swap. Details: [METHOD.md](METHOD.md#play-correction-04).

For SO-100/101 arms the card also gives a simulated figure: how often a pick-and-place cell with 3 or 5 mm of lead-in would seat a part with this arm, and how many misses a supervisor should allow before it calls a person. Across 277 corpus arms the median is 72% at 3 mm and 94% at 5 mm. It is a model figure from a lookup table (`holdcheck/seatmap.json`), and no physical cell was run. The cell simulator is in `holdcheck/seatsim.py`: run your own card through it, with your own lead-in, camera noise and nest, or rebuild the table. Details: [METHOD.md](METHOD.md#seat-success-in-a-pick-and-place-cell-and-supervisor-thresholds-05).

These are the servos alone, a lower bound at the tool: horns, screws and printed links add more. Full method, tables and limits: [METHOD.md](METHOD.md).

## Run it yourself

Needs Python 3.11+, `numpy` and `pyarrow`. It reads only `meta/info.json` and `data/*.parquet`, never videos.

```sh
cd holdcheck
python holdcheck.py --repo <user>/<dataset> --out card.json --print
python holdcheck.py --local ~/.cache/huggingface/lerobot/<user>/<dataset> --out card.json --print
python test_holdcheck.py      # instrument check on a simulated arm with known play
python test_playfix.py        # the correction on a simulated follower, closed loop
python seatsim.py arm card.json --lead-in 3 5 --camera-mm 0.5   # simulated seat success of your arm (SO-100/101)
python seatsim.py anchor       # the median-joint and p90-joint arms, against METHOD.md
python test_seatsim.py         # the seat cell and its runner
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
mkdir -p runs/cards2026 && grep -v '^#' ../corpus/datasets-2026.txt | while read id; do python holdcheck.py --repo "$id" --out "runs/cards2026/$(echo "$id" | tr / _).json"; done
python corpus.py summary --cards runs/cards --cards runs/cards2026 --exclude ../corpus/left-out-2026.txt --out runs/summary.json --md runs/summary.md
python corpus.py summary --cards runs/cards --out runs/summary2025.json   # the 2025 corpus alone (corpus_summary_2025.json)
python seatsim.py corpus --cards runs/cards --out runs/seat.jsonl    # every SO-100/101 card through the cell
python seatsim.py map --runs runs/seat.jsonl --out runs/seatmap.json  # the seat lookup; compare with seatmap.json
```

The input dataset ids are in [corpus/datasets.txt](corpus/datasets.txt) and [corpus/datasets-2026.txt](corpus/datasets-2026.txt). The seat lookup was fitted on the 2025 corpus only. The comparison table that ships with the tool, `holdcheck/corpus_summary.json`, holds distributions only. Per-dataset results are not published; you get a card for the dataset you run.

The browser page is in `web/`. It runs the same `holdcheck.py`, unchanged, in Pyodide, and its card equals the command line's.

## Contents

- `holdcheck/holdcheck.py`: the measurement (numpy and pyarrow)
- `holdcheck/test_holdcheck.py`: instrument check on a simulated arm with known play, scatter and creep
- `holdcheck/playfix.py`, `holdcheck/test_playfix.py`: the play correction from a card, a LeRobot `send_action` wrapper and its closed-loop check
- `holdcheck/corpus.py`: list, run and summarise a corpus
- `holdcheck/tipmap.json`, `holdcheck/tipmap.py`, `holdcheck/so101kin.py`, `holdcheck/model/`: SO-101 tool-point Jacobians at 64 table poses, and how to rebuild them. `python holdcheck/tipmap.py --out tipmap.json` reads the bundled MuJoCo Menagerie SO-101 model (`model/so101.xml`, Apache-2.0, see `model/NOTICE.md`) with numpy only and reproduces the shipped Jacobians exactly.
- `holdcheck/corpus_summary.json`: the comparison table (percentiles only), the 2025 corpus and the 2026 sample together; `holdcheck/corpus_summary_2025.json`: the 2025 corpus alone, behind the figures in METHOD.md and `seatsim.py anchor`
- `holdcheck/seatmap.json`: the lookup from a card's one-side tip figure to simulated seat success at 3 and 5 mm
- `holdcheck/seatcell.py`, `holdcheck/seatsim.py`, `holdcheck/test_seatsim.py`: the simulated pick-and-place cell behind it, a runner for one arm or the whole corpus, and their tests
- `web/`: the static page, Pyodide and the fonts it serves itself
- `METHOD.md`: method, corpus results and limits

## Licence

Code and tables: Apache-2.0 (see `LICENSE`). Bundled third-party files: see `THIRD-PARTY.md`. The corpus figures are derived from `lerobot/community_dataset_v3` (Apache-2.0).

Prepared by CyberNative AI LLC, an AI-run company, using AI agents.
