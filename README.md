# Decoding Possession
### From Player Tracking to Football Possession Spells

**Gabriel Gausachs | Team GG**

*Metrica Sports — More Than a Hack 2027*

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
4. Model about 12.4 seconds of temporal context with a noncausal temporal convolutional network.
5. Decode H/A/D probabilities with a period-aware Viterbi decoder.
6. Map the 10 Hz predictions back to the original 25 Hz Metrica timeline.

Smart Tags, broadcast video, homography, and hand-written corrections are not part of the final
prediction pipeline.

### How we created the training labels

Training uses all 20 matches in
[SkillCorner Open Data](https://github.com/SkillCorner/opendata).

SkillCorner does not provide a ready-made Home/Away/Dead-ball label for every tracking frame. It
does, however, provide two useful sources of information:

- **Phases of Play** describe intervals when the ball is in play and identify the team considered
  to be in possession.
- **Dynamic Events** include player-possession intervals, the player and team controlling the
  ball, and interruption information recorded before or after a possession.

We use the team named by a valid Phase of Play as the main source for `H` and `A`. The overlapping
player-possession event acts as a controller check: if the phase and the player controller point to
different teams, we do not guess and leave that frame unknown.

We assign `D` only to gaps between phases that have strong evidence of a real stoppage. The player
possession immediately before the gap must report the same interruption type as the possession
immediately after it. Their `for`/`against` perspectives must also agree once the two teams are
taken into account. Both possession records must be close to the corresponding gap boundary, and
there must be no Home or Away control evidence inside the gap. If any of these conditions fails,
the gap is not labelled dead ball.

Frames that are not supported reliably become `U` (unknown). This includes missing periods,
conflicting phase/controller information, unsupported gaps, and areas outside phase coverage. The
purpose of `U` is to avoid teaching the model labels that may be wrong. These frames remain in the
sequence as temporal context, but they do not contribute to the supervised loss.

All original match indices are preserved. Official P1 and P2 metadata separates the halves, so
halftime and out-of-period frames never enter model sequences.

| H frames | A frames | D frames | U frames |
|---:|---:|---:|---:|
| 319,399 | 308,895 | 288,850 | 327,670 |

Training uses possession-duration weighted cross-entropy. Without weighting, long possessions
contribute many more frames to the loss and can dominate learning, even though recovering short
possessions is equally important for spell-based evaluation. We therefore give complete short H/A
spells more importance per frame and long spells less, with lower and upper limits to prevent
extreme weights. Possessions touching an unknown region or a period boundary keep a neutral weight
because their true duration is uncertain. Dead-ball frames also keep a neutral weight, while `U`
and out-of-period frames are excluded from the loss.

### Features extracted

Both providers are converted to the same common tracking representation. Coordinates are in
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

The model first applies the same small neural network to every tracked player (`5 → 32 → 64`). It
then averages the valid player representations separately for Home and Away. This produces one
summary of each team without relying on a particular player identity or tracking slot. The two team
summaries are combined with 51 frame-level features and passed to a temporal convolutional network
with five layers operating at different time scales.

For each prediction, the temporal network can use 125 consecutive samples. At 10 samples per
second, this covers about 12.4 seconds from the earliest to the latest observation—roughly 6.2
seconds before and after the frame being classified. This context helps the model tell the
difference between a momentary tracking fluctuation, a sustained possession, and a genuine
stoppage. The output is a probability for H, A, and D at every in-period frame.

Choosing the most likely class independently on every frame would create many one- or two-frame
changes and fragment otherwise continuous possessions. The Viterbi decoder finds the complete
H/A/D sequence that best balances the model probabilities with a fixed cost for changing state.
The selected switch penalty is `1.5` (with probability epsilon `1e-8`): it discourages unlikely
flickering without preventing a well-supported turnover or restart. Each half is decoded
independently, so model context and decoding never cross halftime.

## Pipeline description

- **Create the training labels.** SkillCorner Phases of Play provide the main H/A ownership
  intervals. Player-possession and interruption events are used to reject contradictions and to
  identify reliable dead-ball gaps. Uncertain frames remain U.
- **Build the common features.** SkillCorner and Metrica tracking are placed in the same coordinate
  system and converted into the same player, team, ball, movement, and validity features.
- **Fit normalization and train the model.** Normalization is learned from SkillCorner only. The
  Deep Sets encoder summarizes each team, and the temporal network predicts H/A/D probabilities
  using the surrounding sequence.
- **Prepare the Metrica timeline.** Each official half is sampled independently from 25 Hz to 10
  Hz using the nearest timestamp. Exact ties select the earlier frame.
- **Run model inference.** The saved model and normalization convert each 10 Hz frame into H/A/D
  probabilities.
- **Decode complete possessions.** Viterbi decoding is applied independently to each half to
  reduce isolated label changes and preserve coherent spells.
- **Create the submission.** Decoded predictions are mapped back to the original 199,319-frame
  tactical timeline. Official half blocks receive H/A/D and every other frame receives X.

## Evaluation

### SkillCorner development validation

We used a fixed 16-match training / 4-match validation split while developing the final training
recipe. The table below reports the duration-weighted model with the same Viterbi 1.5 decoder used
by the final system.

| Split | Spell F1 | Macro-F1 | F1 H | F1 A | F1 D | Predicted / GT spells | Fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 train / 4 validation | **0.7298** | **0.9214** | 0.9131 | 0.9177 | 0.9336 | 847 / 733 | 1.1555 |

Spell F1 measures whether complete predicted H/A possessions overlap the reference possessions,
rather than scoring frames independently. Here, "U-aware" means that unknown frames are not
treated as a fourth class or counted as model errors. They are omitted from the frame metrics, and
reference possessions touching U are excluded from spell matching because their true boundary is
not known reliably. For the remaining possessions, temporal overlaps are scored per match and then
averaged across the four validation matches. These matches were not part of this development run's
training set. After fixing the training recipe, the model was refitted on all 20 matches for the
submission.

### Metrica public five-minute window

The official repository includes four nearest-player-to-ball reference submissions. They differ
only in their temporal smoothing window. The comparison below uses the unchanged official
evaluator on the same public five-minute interval.

| Method | Mean spell F1 | Macro-F1 | Predicted possessions |
|---|---:|---:|---:|
| **Our final model** | **0.6857** | **0.8824** | 18 |
| Nearest player, ±5 s smoothing | 0.4242 | 0.5812 | 16 |
| Nearest player, ±2 s smoothing | 0.3256 | 0.6207 | 26 |
| Nearest player, ±1 s smoothing | 0.2868 | 0.6200 | 36 |
| Nearest player, no smoothing | 0.0378 | 0.5972 | 131 |

The public labels cover only 7,500 frames and 17 possession spells. These numbers are useful for
checking alignment and the complete inference path, but the full-match ground truth remains
private.

## How to run it

All commands below are run from the repository root.

### 1. Create the Conda environment

The final environment uses Python 3.12.10 and pinned package versions.

```powershell
conda create --name metrica-hackathon python=3.12.10 pip -y
conda activate metrica-hackathon
python -m pip install -r requirements.txt
```

### 2. Download the data

The Metrica match data is distributed under the competition terms and must not be committed or
redistributed. After accepting the terms, download the tactical and broadcast assets with:

```powershell
python setup_data.py
```

The script installs the expected assets under `data/metrica/tactical/` and
`data/metrica/broadcast/`. If Google Drive requires browser sign-in, download the supplied archive
manually and run:

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
python -m possession.inference `
  --model-dir outputs/final_model/training `
  --output-dir outputs/final_model
```

This explicitly loads the model, normalization and decoder produced by the new training run. It
writes the submission to `outputs/final_model/GG.csv`.

### 6. Run inference with the saved model

The repository includes the final model under `artifacts/final_model/`. To regenerate the exact
submitted CSV without retraining, only the Metrica feature archive is required:

```powershell
python -m possession.preprocessing.feature_extraction metrica
python -m possession.inference
```

Without `--model-dir`, inference uses `artifacts/final_model/`, as specified in
`configs/inference.json`. This is the frozen model included with the repository, and its prediction
is written alongside it. The default filename is `GG.csv`; use `--submission-name` only when a
different name is needed.

The expected output is:

```text
artifacts/final_model/GG.csv
```

### 7. Validate and evaluate the submission

Before running the public diagnostic evaluation, validate the exact file with this command:

```powershell
python evaluation/validate_submission.py artifacts/final_model/GG.csv
```

The validator checks the exact `frame,label` header, all 199,319 ordered frame indices, and the
allowed H/A/D/X labels.

The public diagnostic evaluation is run with the official evaluator:

```powershell
python evaluation/run_evaluation.py --gt public `
  --results artifacts/final_model `
  --out artifacts/final_model/evaluation
```

## Repository layout

```text
artifacts/final_model/       saved model, normalization and decoder
configs/                     fixed training and inference configurations
data/                        local raw and derived data (not committed)
evaluation/                  official validator, evaluator and public ground truth
possession/
|-- preprocessing/           SkillCorner labels and cross-provider feature extraction
|-- modeling/                data preparation, Deep Sets-TCN and training
|-- decoding.py              period-aware Viterbi decoder
|-- inference.py             Metrica inference and submission generation
starter/                     original competition examples and utilities
setup_data.py                Metrica tactical/broadcast data installer
requirements.txt             pinned Python dependencies
```

The original challenge documentation remains available in [CHALLENGE.md](CHALLENGE.md),
[DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md), and
[RULES.md](RULES.md).

## Licence

Repository code is covered by [LICENSE](LICENSE). The Metrica match assets are governed by the
competition terms and cannot be redistributed.

[SkillCorner Open Data](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656)
is released under the
[MIT licence](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/LICENSE),
© 2020 SkillCorner. Credit to SkillCorner and its joint open-data initiative with
[PySport](https://pysport.org/).
