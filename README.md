# More Than a Hack 2027 — Tracking-Only Possession Prediction

This repository contains the final submission pipeline for the Metrica Sports **More Than a Hack
2027 — Phase 1: Ball Possession** challenge. For every frame of the FC Barcelona–Athletic Club
match, the system predicts Home possession (`H`), Away possession (`A`), dead ball (`D`), or `X`
outside the official scored periods.

The selected solution uses only automatic tracking at inference time. Metrica Smart Tags,
broadcast video, homography, manual corrections, and public ground-truth labels are not used to
produce the submission.

## Final approach

### Training data and labels

The model is trained on all 20 matches from
[SkillCorner Open Data](https://github.com/SkillCorner/opendata), fixed at commit
`4340d274572876239c154c90bc507a9b3250a656`.

The deterministic `skillcorner_final_v1` policy builds four frame labels:

- `H` and `A` from valid Phases of Play and player-possession evidence;
- `D` from strictly supported interruption gaps;
- `U` where the available annotations do not support a reliable label.

The frozen aggregate label counts are:

| H | A | D | U |
|---:|---:|---:|---:|
| 319,399 | 308,895 | 288,850 | 327,670 |

`U` frames are excluded from supervised loss but remain available as temporal context. Official
P1/P2 metadata isolates the two halves; halftime and out-of-period frames never enter model
sequences.

### Features

SkillCorner and Metrica are converted to the same `common_tracking_v1` schema at 10 Hz. Coordinates
are expressed in canonical pitch-centred metres with Home always attacking in the positive x
direction. Inputs include:

- per-player x/y position, speed, position validity and speed validity;
- team centroids, spread, compactness, valid-player counts and speed statistics;
- relative Home/Away geometry;
- ball position, displacement, missingness duration and player proximity;
- one-second team temporal changes and explicit validity indicators.

Invalid measurements remain neutral zero and are accompanied by validity features. Normalization
is fitted on the 20 SkillCorner matches and stored with the model.

### Model and training

Each team is encoded independently with a Deep Sets player MLP (`5 → 32 → 64`) and masked mean
pooling. The two team embeddings are concatenated with 51 engineered features and processed by a
64-channel noncausal temporal convolutional network with residual dilation blocks
`[1, 2, 4, 8, 16]`. The receptive field is 125 samples, or 12.5 seconds at 10 Hz.

The final model is trained for exactly 16 epochs on all 20 matches with AdamW, learning rate
`0.001`, weight decay `0.0001`, batch size `8`, core length `512`, dropout `0.1`, and seed `2027`.
There is no validation split or checkpoint selection in this final refit; epoch 16 is saved.

Training uses possession-duration weighted cross-entropy. Complete interior H/A spells receive:

```text
clip(8 / duration_seconds, 0.25, 4.0)
```

H/A spells touching `U` or a period boundary and all `D` frames receive weight 1. `U` and
out-of-period frames receive weight 0.

### Metrica inference

Each official Metrica half is sampled independently from 25 Hz to 10 Hz using nearest timestamps,
with the earlier frame selected on an exact tie. The model probabilities are decoded independently
inside P1 and P2 with Viterbi switch penalty `1.5` and probability epsilon `1e-8`, then mapped back
to the original 199,319-frame tactical timeline. Official half blocks receive H/A/D and all other
frames receive X.

## Environment

The final environment is Python `3.12.10`. Create a clean environment and install the single
pinned dependency file.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data setup

The competition data is gated and must not be committed or redistributed. Enrol through the
[official competition page](https://landing.mobileworldcapital.com/more-than-a-hack-2027), accept
the terms, and use the participant-specific download link supplied by the organisers.

Extract the supplied tactical and broadcast directories under `data/metrica/`:

```text
data/
├── metrica/
│   ├── tactical/
│   │   ├── BAR-ATH_tactical.mp4
│   │   ├── BAR-ATH_tactical_atd.xml
│   │   ├── BAR-ATH_tactical_atd.txt
│   │   ├── BAR-ATH_tactical_homography.csv
│   │   └── BAR-ATH_tactical_smart_tagging.xml
│   └── broadcast/
│       ├── BAR-ATH_broadcast.mp4
│       ├── BAR-ATH_broadcast_atd.xml
│       ├── BAR-ATH_broadcast_atd.txt
│       └── BAR-ATH_broadcast_homography.csv
├── external/
│   └── skillcorner-opendata/
└── derived/
```

The final pipeline consumes only the tactical ATD XML and TXT files. Broadcast assets are retained
for completeness but are not used. No playback application or derived video is created.

Clone the training data and check out the frozen source revision:

```powershell
git clone https://github.com/SkillCorner/opendata.git data/external/skillcorner-opendata
git -C data/external/skillcorner-opendata checkout 4340d274572876239c154c90bc507a9b3250a656
```

Raw and derived data under `data/` are ignored by Git.

## Reproduce the submitted CSV

The committed `artifacts/final_model/` directory contains the frozen epoch-16 checkpoint,
normalization statistics and decoder definition. With the Metrica data installed, regenerate the
exact submission with:

```powershell
python -m possession.preprocessing.feature_extraction metrica
python -m possession.inference
```

This writes:

```text
outputs/final_model/submission/metrica_possession.csv
outputs/final_model/submission/prediction_statistics.json
```

The expected SHA-256 of `metrica_possession.csv` is:

```text
171882f7fc390421fd4fe35f65f2652eb78892eea876694d50cec445a787da48
```

The expected full-match statistics are H=60,117, A=32,657, D=66,083, X=40,462 and 357 predicted
possession spells.

## Rebuild labels, features and model

To reproduce the complete training pipeline instead of using the committed checkpoint:

```powershell
python -m possession.preprocessing.labels
python -m possession.preprocessing.feature_extraction skillcorner
python -m possession.preprocessing.feature_extraction metrica
python -m possession.modeling.training
python -m possession.inference --model-dir outputs/final_model/training
```

Generated data and training outputs are written under `data/derived/` and `outputs/final_model/`.
Retraining is documented for methodological reproducibility; the committed frozen artifacts are
the authoritative path for byte-identical submission regeneration across different hardware.

## Validate, evaluate and inspect

Validate the exact CSV before submission:

```powershell
python evaluation/validate_submission.py outputs/final_model/submission/metrica_possession.csv
```

The file must contain exactly `frame,label`, 199,319 ordered frames `0…199318`, and only H/A/D/X.
There are 158,857 scored H/A/D frames and 40,462 X frames.

Run the unchanged official evaluator on the provided public five-minute ground truth:

```powershell
python evaluation/run_evaluation.py --gt public `
  --results outputs/final_model/submission `
  --out outputs/final_model/submission/evaluation
```

Generate the optional public diagnostic timeline:

```powershell
python -m possession.plot_timeline --gt public `
  --submission outputs/final_model/submission/metrica_possession.csv
```

The public interval gives mean spell F1 `0.685714`, spell F1 at IoU 0.5 `0.742857`, macro-F1
`0.882434`, and accuracy `0.894533`. These figures are diagnostic only: public labels were not
used to train, tune, select or alter the final model.

For the official upload, rename the generated file to `<team_name>.csv` as required by
[SUBMISSION.md](SUBMISSION.md).

## Frozen artifact hashes

| File | SHA-256 |
|---|---|
| `artifacts/final_model/model.pt` | `9e3b0633365d5b86fe1a61754b2f81c52fed8d20418410ca274855e102a16c14` |
| `artifacts/final_model/normalization.npz` | `20dd6ac19a175fbb9280475c488a70568743073f42356f6ec6d8c991ec96aeb6` |
| `artifacts/final_model/final_decoder.json` | `6e97cae84baea0967c1afb0c36ce5f80c3972fd8eda04a12f975bd2f0d4c020` |

Additional official documentation remains available in [CHALLENGE.md](CHALLENGE.md),
[DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md), and
[RULES.md](RULES.md).

## Licence

Repository code is covered by [LICENSE](LICENSE). Metrica match assets use separate competition
terms and are not redistributable. SkillCorner Open Data retains its upstream licence and
attribution requirements.
