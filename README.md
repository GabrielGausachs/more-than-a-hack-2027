# Tracking-only possession prediction

## Summary

This repository contains our solution for the **Metrica Sports More Than a Hack 2027** possession
challenge. The goal is to label every frame of FC Barcelona vs Athletic Club as Home possession
(`H`), Away possession (`A`), dead ball (`D`), or outside the scored part of the match (`X`).

The organisers provide tactical and broadcast video, automatic player and ball tracking,
homography, and tactical Smart Tags. Our final system uses only the tactical tracking feed. It is a
supervised temporal model trained on 20 matches from SkillCorner Open Data, followed by a Viterbi
decoder that turns frame probabilities into coherent possession spells.

On the provided five-minute public diagnostic window, the final model achieves:

| Spell F1 | Spell F1 at IoU 0.5 | Macro-F1 | Frame accuracy |
|---:|---:|---:|---:|
| **0.6857** | **0.7429** | **0.8824** | **0.8945** |

The public window is used only to check the finished pipeline. It was not used for training,
parameter tuning, or model selection.

This README describes the approach, label and feature construction, pipeline, evaluation results,
data setup, training, inference, and submission commands.

## Task and approach

The submission is a CSV with one row for each of the 199,319 frames in the tactical video. The
official metric is possession-spell F1: a good model must recover complete possessions, not only
classify many individual frames correctly.

We train a tracking-only **Deep Sets + noncausal TCN** model at 10 Hz:

1. Build H/A/D/U supervision from SkillCorner Phases of Play and Dynamic Events.
2. Convert SkillCorner and Metrica tracking to the same provider-independent feature schema.
3. Encode the two teams with shared player-level Deep Sets.
4. Model 12.5 seconds of temporal context with a noncausal temporal convolutional network.
5. Decode H/A/D probabilities with a period-aware Viterbi decoder.
6. Map the 10 Hz predictions back to the original 25 Hz Metrica timeline.

Smart Tags, broadcast video, homography, and hand-written corrections are not part of the final
prediction pipeline.

### How we created the training labels

Training uses all 20 matches in
[SkillCorner Open Data](https://github.com/SkillCorner/opendata), fixed at commit
`4340d274572876239c154c90bc507a9b3250a656`.

The `skillcorner_final_v1` policy assigns:

- `H` or `A` from valid Phases of Play and player-possession intervals;
- `D` only when consecutive possession records contain compatible interruption evidence;
- `U` when annotations are missing, contradictory, or do not support a reliable label.

All original match indices are preserved. Official P1 and P2 metadata separates the halves, so
halftime and out-of-period frames never enter model sequences. `U` frames remain available as
temporal context but receive no supervised loss.

| H frames | A frames | D frames | U frames |
|---:|---:|---:|---:|
| 319,399 | 308,895 | 288,850 | 327,670 |

Training uses possession-duration weighted cross-entropy. Complete interior H/A spells receive:

```text
weight = clip(8 / duration_seconds, 0.25, 4.0)
```

H/A spells touching `U` or a period boundary, and all `D` frames, receive weight 1. `U` and
out-of-period frames receive weight 0.

### Features extracted

Both providers are converted to the same `common_tracking_v1` representation. Coordinates are in
canonical pitch-centred metres, with Home always attacking in the positive x direction.

The model receives:

- per-player x/y position, speed, position validity, and speed validity;
- team centroids, spread, compactness, player counts, and speed statistics;
- relative Home/Away geometry;
- ball position, displacement, missingness duration, and player proximity;
- one-second changes in team position and compactness;
- explicit validity indicators for missing measurements.

Invalid measurements are represented by neutral zeros together with their validity flags. The
normalization statistics are fitted on the SkillCorner training matches and saved with the model.

### Temporal model and decoder

Each player is embedded with a shared MLP (`5 → 32 → 64`). Masked mean pooling produces one
64-dimensional representation for each team. The two team embeddings are concatenated with 51
engineered frame features and passed through a 64-channel residual TCN with dilations
`[1, 2, 4, 8, 16]`.

The receptive field is 125 samples, equivalent to 12.5 seconds at 10 Hz. The output layer predicts
H/A/D probabilities for every in-period frame.

The final decoder is Viterbi with switch penalty `1.5` and probability epsilon `1e-8`. Each half is
decoded independently, so neither the model context nor the decoder can cross halftime.

## Pipeline description

```text
SkillCorner tracking + events
        │
        ├── label generation ──> H / A / D / U supervision
        │
        └── feature extraction ──> common_tracking_v1 archives
                                      │
                                      ├── normalization
                                      └── Deep Sets + TCN training
                                                   │
                                                   └── model + decoder artifacts

Metrica tactical tracking
        │
        └── feature extraction at 25 Hz
                    │
                    └── period-local nearest sampling to 10 Hz
                                  │
                                  └── model probabilities
                                              │
                                              └── Viterbi decoding per half
                                                           │
                                                           └── 25 Hz reconstruction
                                                                       │
                                                                       └── submission CSV
```

The final refit uses all 20 SkillCorner matches for exactly 16 epochs. It uses AdamW with learning
rate `0.001`, weight decay `0.0001`, batch size `8`, dropout `0.1`, and seed `2027`. With no
validation partition in the final refit, epoch 16 is saved directly.

For Metrica, each official half is sampled independently from 25 Hz to 10 Hz using nearest
timestamps, choosing the earlier frame on an exact tie. Predictions are reconstructed onto the
199,319-frame tactical timeline. The two official half blocks receive H/A/D; all other frames
receive X.

## Evaluation

### SkillCorner development validation

We used a fixed 16-match training / 4-match validation split while developing the final training
recipe. The table below reports the duration-weighted model with the same Viterbi 1.5 decoder used
by the final system.

| Split | Spell F1 | Macro-F1 | F1 H | F1 A | F1 D | Predicted / GT spells | Fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 train / 4 validation | **0.7298** | **0.9214** | 0.9131 | 0.9177 | 0.9336 | 847 / 733 | 1.1555 |

Spell F1 is the match-macro, U-aware validation metric. The four validation matches were not part
of this development run's training set. After fixing the training recipe, the model was refitted on
all 20 matches for the submission.

### Metrica public five-minute window

The official repository includes four nearest-player-to-ball reference submissions. They differ
only in their temporal smoothing window. The comparison below uses the unchanged official
evaluator on the same public five-minute interval.

| Method | Macro-F1 | Mean spell F1 | Predicted possessions | Mean duration (s) |
|---|---:|---:|---:|---:|
| **Our final model** | **0.8824** | **0.6857** | 18 | 13.82 |
| Nearest player + smoothing 125 | 0.5812 | 0.4242 | 16 | 18.5 |
| Nearest player + smoothing 50 | 0.6207 | 0.3256 | 26 | 10.8 |
| Nearest player + smoothing 25 | 0.6200 | 0.2868 | 36 | 7.6 |
| Nearest player, raw | 0.5972 | 0.0378 | 131 | 2.0 |

The public labels cover only 7,500 frames and 17 possession spells. These numbers are useful for
checking alignment and the complete inference path, but the full-match ground truth remains
private.

## How to run it

All commands below are run from the repository root.

### 1. Create the Conda environment

The final environment uses Python 3.12.10 and pinned package versions.

```powershell
conda create --name mtah-possession python=3.12.10 pip -y
conda activate mtah-possession
python -m pip install -r requirements.txt
```

### 2. Download the data

The Metrica match data is distributed under the competition terms and must not be committed or
redistributed. After accepting the terms, download the tactical and broadcast assets with:

```powershell
python setup_data.py
```

The script installs the expected assets under `data/metrica/tactical/` and
`data/metrica/broadcast/`. It does not crop video or create playback files. If Google Drive
requires browser sign-in, download the supplied archive manually and run:

```powershell
python setup_data.py --bundle path\to\downloaded_bundle.zip
```

Clone the SkillCorner training data and check out the exact revision used in this project:

```powershell
git clone https://github.com/SkillCorner/opendata.git data/external/skillcorner-opendata
git -C data/external/skillcorner-opendata checkout --detach 4340d274572876239c154c90bc507a9b3250a656
```

Raw and derived data under `data/` are ignored by Git.

### 3. Prepare the training data

Generate the SkillCorner labels and the common feature archives:

```powershell
python -m possession.preprocessing.labels
python -m possession.preprocessing.feature_extraction skillcorner
python -m possession.preprocessing.feature_extraction metrica
```

The generated files are written under `data/derived/`.

### 4. Train the model

```powershell
python -m possession.modeling.training
```

This runs the fixed 20-match, 16-epoch training configuration and writes the model, normalization,
history, and decoder definition to `outputs/final_model/training/`.

### 5. Run inference with the newly trained model

```powershell
python -m possession.inference --model-dir outputs/final_model/training
```

### 6. Run inference with the saved model

The repository includes the final model under `artifacts/final_model/`. To regenerate the exact
submitted CSV without retraining, only the Metrica feature archive is required:

```powershell
python -m possession.preprocessing.feature_extraction metrica
python -m possession.inference
```

The expected output is:

```text
outputs/final_model/submission/metrica_possession.csv
```

Its expected SHA-256 is:

```text
171882f7fc390421fd4fe35f65f2652eb78892eea876694d50cec445a787da48
```

### 7. Validate and evaluate the submission

Always validate the exact file that will be submitted:

```powershell
python evaluation/validate_submission.py outputs/final_model/submission/metrica_possession.csv
```

The validator checks the exact `frame,label` header, all 199,319 ordered frame indices, and the
allowed H/A/D/X labels.

The public diagnostic evaluation is run with the official evaluator:

```powershell
python evaluation/run_evaluation.py --gt public `
  --results outputs/final_model/submission `
  --out outputs/final_model/submission/evaluation
```

An optional horizontal comparison of predictions and public ground truth can be generated with:

```powershell
python -m possession.plot_timeline --gt public `
  --submission outputs/final_model/submission/metrica_possession.csv
```

For the official upload, rename the CSV to `<team_name>.csv` as described in
[SUBMISSION.md](SUBMISSION.md).

## Repository layout

```text
artifacts/final_model/       saved model, normalization and decoder
configs/                     fixed training and inference configurations
data/                        local raw and derived data (not committed)
evaluation/                  official validator, evaluator and public ground truth
possession/
├── preprocessing/           SkillCorner labels and cross-provider feature extraction
├── modeling/                data preparation, Deep Sets-TCN and training
├── decoding.py              period-aware Viterbi decoder
├── inference.py             Metrica inference and submission generation
└── plot_timeline.py         optional public diagnostic plot
starter/                     original competition examples and utilities
setup_data.py                Metrica tactical/broadcast data installer
requirements.txt             pinned Python dependencies
```

The original challenge documentation remains available in [CHALLENGE.md](CHALLENGE.md),
[DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md), and
[RULES.md](RULES.md).

## Licence

Repository code is covered by [LICENSE](LICENSE). The Metrica match assets are governed by the
competition terms and cannot be redistributed. SkillCorner Open Data retains its upstream licence
and attribution requirements.
