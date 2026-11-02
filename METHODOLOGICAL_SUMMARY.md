# WCSF Temporal Classification Methodology

## 1. Purpose and scope

This document describes the implemented WCSF Groningen pipeline used to train and evaluate
temporal football-state classifiers. It follows the code as it currently exists in
`feature_engineering/` and `modeling/alive_dead/`.

Two supervised tasks share the same tracking representation and temporal modeling infrastructure:

1. **Binary ALIVE/DEAD classification**
   - `0 = DEAD`
   - `1 = ALIVE`
2. **Direct H/A/D classification**
   - `0 = H`: Home possession while the ball is alive
   - `1 = A`: Away possession while the ball is alive
   - `2 = D`: Dead ball

The main motivation is to learn temporal tracking patterns from the externally labelled WCSF
match and later transfer them to Metrica tracking. WCSF contains only one match, so its held-out
half results measure within-match temporal generalization, not independent-match generalization.

The baseline excludes ball features, while an optional ball-enabled path adds only ball X/Y,
speed and independent validity masks. Both paths exclude `team_game_phase`, acceleration,
player-ID embeddings and competition Smart Tags. In particular, `team_game_phase` is strongly
label-derived in WCSF and would create target leakage.

## 2. End-to-end data flow

```text
WCSF half CSV files
    |
    +--> player/team feature builder
    |       `wcsf_player_team_features.npz`
    |       optional: `wcsf_player_team_ball_features.npz`
    |
    +--> strict binary label builder
    |       `wcsf_alive_dead_labels_strict.npz`
    |
    +--> strict H/A/D label builder
            `wcsf_had_labels.npz`

Aligned archive loading and validation
    |
Outer half split + three purged inner validation blocks
    |
Training-only normalization
    |
Center-supervised temporal windows
    |
Team TCN or Deep Sets + Team TCN
    |
Three inner models selected by full-segment validation loss
    |
Median best epoch
    |
Fresh full-half refit
    |
Held-out-half evaluation and two-fold OOF reconstruction
```

The original 934.2-second halftime timestamp gap remains in metadata. No artificial halftime
frames are created, and no temporal window or inference sequence connects the two halves.

## 3. WCSF source data

The source files are:

```text
data/open_data_groningen/positions_FIRST_HALF.csv
data/open_data_groningen/positions_SECOND_HALF.csv
```

WCSF is already sampled at 10 Hz. The processed timeline contains:

| Half | Frames | Storage interval |
|---|---:|---|
| First half | 28,014 | `[0, 28014)` |
| Second half | 29,360 | `[28014, 57374)` |
| Total | 57,374 | `[0, 57374)` |

Within each half, timestamps increase by exactly 100 ms. Half identifiers, rather than adjacency
in the NPZ array, define temporal continuity.

## 4. Player and team feature creation

Implementation: `feature_engineering/wcsf_features.py`.

### 4.1 Raw fields used

The feature builder reads only:

- `timestamp_ms`
- `group_id`
- `player_id`
- `x`
- `y`
- `speed`

`group_id=1` is Home and `group_id=2` is Away. `player_id` is used only to reject duplicate
observations and to create deterministic within-frame storage. It is not saved as a model feature
and is never used as a temporal identity.

The builder rejects:

- Missing required columns.
- Invalid identifiers.
- Team identifiers outside `{1,2}`.
- Duplicate `(half, timestamp, team, player_id)` rows.
- More than 24 observations for either team at one timestamp.

### 4.2 Coordinate orientation and normalization

The WCSF coordinates are team-relative. They are converted into the shared home-centric convention:

- Home is retained.
- Away is rotated 180 degrees by changing the sign of both X and Y.

The normalized coordinates are:

```text
home: x_norm =  x / 105 + 0.5, y_norm =  y / 68 + 0.5
away: x_norm = -x / 105 + 0.5, y_norm = -y / 68 + 0.5
```

This creates the convention that Home attacks toward positive X and Away toward negative X. The
105 x 68 m playing field is the normalization reference. Coordinates are not clipped, so valid
provider observations outside the nominal pitch remain outside `[0,1]`.

### 4.3 Unordered player representation

The primary array is:

```text
players: (N, 2, 24, 3), float32
```

The last dimension is ordered as:

1. Normalized X.
2. Normalized Y.
3. Provider-reported speed in m/s.

Rows are sorted by player ID within each frame and team and densely packed from slot zero. This is
only deterministic storage. The model treats the 24 slots as an unordered set and does not assume
that slot `j` at time `t` is the same person as slot `j` at time `t+1`.

Two independent Boolean masks are saved:

```text
player_location_valid: (N, 2, 24)
player_speed_valid:    (N, 2, 24)
```

A location is valid only when both X and Y are finite. Speed validity depends only on whether the
reported speed is finite. Therefore, a player may have a valid location and an invalid speed, or
vice versa. Invalid and padded numeric values are filled with zero, but zero is never used to infer
missingness; the masks are authoritative.

### 4.4 Team activity features

For every frame and team, the builder independently calculates:

```text
team_features: (N, 2, 6), float32
```

Feature order:

1. `mean_speed`
2. `median_speed`
3. `speed_std`
4. `tracked_player_count`
5. `valid_speed_count`
6. `speed_coverage`

Definitions:

- Speed statistics use only slots where `player_speed_valid=True`.
- Standard deviation is the population standard deviation (`ddof=0`).
- Tracked-player count is the number of valid X/Y locations.
- Valid-speed count is independent of location validity.
- Speed coverage is `valid_speed_count / tracked_player_count`.

The accompanying `team_feature_valid` array has shape `(N,2,6)`:

- Mean, median and standard deviation are valid when at least one speed exists.
- Both count features are always valid for an existing frame/team.
- Coverage is valid when at least one tracked location exists.
- Invalid values are zero-filled.

For the current WCSF archive, all recorded observations have valid positions and speeds. The
24-slot padding means overall slot validity is approximately 45.83%. At team level, 114,745
frame/team observations contain 11 tracked players and three contain 12; speed coverage is always
1.0. This near-complete coverage is an important domain difference from Metrica.

### 4.5 Feature archive

Output:

```text
data/open_data_groningen/processed/wcsf_player_team_features.npz
```

The parallel ball-enabled archive is
`data/open_data_groningen/processed/wcsf_player_team_ball_features.npz`; the historical archive is
not overwritten.

Core arrays:

| Array | Shape | Dtype |
|---|---|---|
| `players` | `(57374,2,24,3)` | `float32` |
| `player_location_valid` | `(57374,2,24)` | `bool` |
| `player_speed_valid` | `(57374,2,24)` | `bool` |
| `team_features` | `(57374,2,6)` | `float32` |
| `team_feature_valid` | `(57374,2,6)` | `bool` |
| `ball` (optional) | `(57374,3)` | `float32` |
| `ball_location_valid` (optional) | `(57374,)` | `bool` |
| `ball_speed_valid` (optional) | `(57374,)` | `bool` |
| `timestamp_ms` | `(57374,)` | `int64` |
| `half_id` | `(57374,)` | `int8` |
| `sample_index` | `(57374,)` | `int64` |

The archive also records feature names, team order, source frequency, pitch dimensions, schema
version and orientation. Validation reconstructs team features from player arrays and masks and
checks shapes, dtypes, finite values, zero-filling, chronology and metadata alignment.

## 5. Strict label creation

The source flags are repeated on player rows. They are treated as timestamp-level votes. Duplicate
player votes are rejected, and labels must align exactly with feature `half_id`, `timestamp_ms` and
`sample_index`.

The current authoritative policy is strict unanimity. Ambiguous timestamps are retained as tracking
context but marked invalid for supervised loss and metrics.

### 5.1 Strict binary ALIVE/DEAD labels

Implementation: `feature_engineering/wcsf_labels.py`.

Encoding:

```text
0  = DEAD
1  = ALIVE
-1 = UNKNOWN
```

Rules:

- All player rows report `AliveFlag=0` -> DEAD.
- All player rows report `AliveFlag=1` -> ALIVE.
- Any mixture of zero and one -> UNKNOWN with `label_valid=False`.

Output:

```text
data/open_data_groningen/processed/wcsf_alive_dead_labels_strict.npz
```

Current counts:

| Class | Frames |
|---|---:|
| DEAD | 26,776 |
| ALIVE | 30,559 |
| UNKNOWN/invalid | 39 |

The archive stores vote counts, vote fractions and conflict provenance for auditing. These arrays
are not exposed to the model loader.

The older `wcsf_alive_dead_labels.npz` used majority resolution and contains 57,374 valid binary
labels. It is retained only so historical checkpoints remain reproducible. New experiments default
to the strict archive.

### 5.2 Strict H/A/D labels

Implementation: `feature_engineering/wcsf_had_labels.py`.

The verified `PossessionFlag` mapping is:

- `1 = Home`
- `2 = Away`

Encoding:

```text
0  = H
1  = A
2  = D
-1 = UNKNOWN
```

Rules:

1. Unanimous `AliveFlag=0` produces D. Possession may persist or conflict during dead-ball periods,
   but DEAD has precedence.
2. Unanimous `AliveFlag=1` plus unanimous `PossessionFlag=1` produces H.
3. Unanimous `AliveFlag=1` plus unanimous `PossessionFlag=2` produces A.
4. Conflicting ALIVE flags or unresolved possession on an ALIVE frame produce UNKNOWN.

Output:

```text
data/open_data_groningen/processed/wcsf_had_labels.npz
```

Current counts:

| Class | Frames |
|---|---:|
| H | 18,249 |
| A | 12,239 |
| D | 26,776 |
| UNKNOWN/invalid | 110 |

`team_game_phase` is never consulted when constructing either target.

## 6. Archive loading and integrity checks

Implementation: `modeling/alive_dead/dataset.py`.

`load_wcsf_data` deliberately loads only approved predictor arrays plus labels and alignment
metadata. Vote fractions and conflict provenance never enter a model batch.

Before returning a `WCSFData` object, the loader verifies:

- Exact feature/label timeline alignment.
- Required array shapes and dtypes.
- Finite numeric model inputs.
- Contiguous zero-based `sample_index`.
- Exact 100 ms chronology within each half.
- Valid class values at `label_valid=True` frames.
- `-1` at every invalid label.
- Declared `[H,A,D]` class order for H/A/D archives.

For backward compatibility, a legacy binary archive without `label_valid` receives an in-memory
all-true mask. H/A/D archives must explicitly provide `label_valid`.

## 7. Outer folds and purged inner validation

Implementation: `modeling/alive_dead/splits.py`.

There are two outer folds:

| Fold | Outer training half | Untouched outer test half |
|---|---|---|
| 1 | First half | Entire second half |
| 2 | Second half | Entire first half |

The outer test half is not used for normalization, learning-rate scheduling, early stopping,
checkpoint selection or epoch selection.

Within each training half, three deterministic validation blocks are used. Every block occupies
`round(15% of the half)` and begins at 17.5%, 42.5% or 67.5% of that half.

Exact validation intervals in global archive indices:

| Fold | Block 1 | Block 2 | Block 3 | Validation length |
|---|---|---|---|---:|
| 1 | `[4902,9104)` | `[11906,16108)` | `[18909,23111)` | 4,202 |
| 2 | `[33152,37556)` | `[40492,44896)` | `[47832,52236)` | 4,404 |

An exclusion buffer is placed on both sides of the validation interval:

```text
TRAIN | BUFFER | VALIDATION | BUFFER | TRAIN
```

- Default and configurations B/C: 100 frames = 10 seconds per side.
- Configurations D/E: 150 frames = 15 seconds per side.

Buffer frames contribute to neither training nor validation loss. The two remaining training
regions stay as independent temporal segments; they are never joined. Split validation asserts
that train, validation, buffer and test are non-overlapping and that every frame belongs to exactly
one partition.

For the 100-frame buffer setting, each Fold 1 inner run has 23,612 training frames, 4,202
validation frames, 200 buffer frames and 29,360 test frames. Fold 2 has 24,756 training, 4,404
validation, 200 buffer and 28,014 test frames. D/E use 300 buffer frames total, reducing inner
training to 23,512 and 24,656 frames respectively.

## 8. Training-fold normalization

Implementation: `modeling/alive_dead/preprocessing.py`.

Normalization is fitted independently for each split using only `split.train_mask`. Validation,
buffer and outer test frames never contribute.

The transformations are:

- Player X/Y: unchanged because they are already normalized pitch coordinates.
- Player speed: one shared mean and population standard deviation across valid player speeds from
  both teams and all training frames.
- Team mean speed, median speed and speed standard deviation: standardized independently per
  feature, pooling Home and Away valid observations from the training frames.
- Tracked-player count, valid-speed count and speed coverage: retained in their original scales.
- Validity masks: unchanged.
- Invalid numeric positions: reset to zero after transformation.
- Ball-enabled runs standardize valid ball speed with training-only statistics while leaving
  normalized ball X/Y unchanged.

Each inner block therefore has its own normalizer. The final refit normalizer is fitted again from
scratch on the complete outer training half. Normalizer parameters are saved inside each
checkpoint and restored exactly during evaluation or transfer; they are never refitted on test or
Metrica data.

## 9. Receptive-field-aware training windows

Implementation: `TemporalWindowDataset` in `modeling/alive_dead/dataset.py`.

The model is sequence-to-sequence: a window of `T` tracking frames produces `T` frame logits.
Windows never cross a half, buffer, validation boundary or disconnected training segment.

### 9.1 Receptive field

For stride-one temporal convolutions, the implemented receptive field is:

```text
RF = 1 + convolutions_per_block * (kernel_size - 1) * sum(dilations)
context_radius = (RF - 1) / 2
```

The network is noncausal, so each fully contextualized output uses an equal amount of past and
future context.

### 9.2 Center-only supervision

Feeding a finite training window creates artificial edges. An output near an artificial edge does
not have the same context it would receive during full-segment inference. The dataset therefore
separates two masks:

- `output_mask`: real input positions rather than right-padding.
- `supervision_mask`: output positions allowed to affect loss.

Interior windows supervise only:

```text
[context_radius, sequence_length - context_radius)
```

The surrounding frames are input context only. At a genuine segment boundary, however, no legal
real context exists beyond the boundary. The first and last windows therefore supervise those true
edge frames using the same zero padding as full-segment TCN inference.

The final input window is anchored to the segment end instead of creating progressively shorter
tail windows. Configuration validation requires:

```text
training_stride <= sequence_length - 2 * context_radius
```

This guarantees that every training-partition frame appears in at least one supervision region.
Where supervision regions overlap, `loss_weight = 1 / supervision_coverage`, so the weights for
all appearances of an original frame sum to one across the dataset.

UNKNOWN labels remain available inside input context but are excluded by `label_valid`.

### 9.3 Temporal configurations

| Config | Sequence | Stride | Dilations | RF | Radius | Interior supervised width | Buffer/side |
|---|---:|---:|---|---:|---:|---:|---:|
| B | 100 (10 s) | 40 (4 s) | `[1,2,4,8]` | 61 | 30 | 40 | 100 |
| C | 150 (15 s) | 50 (5 s) | `[1,2,4,8]` | 61 | 30 | 90 | 100 |
| D | 200 (20 s) | 50 (5 s) | `[1,2,4,8,16]` | 125 | 62 | 76 | 150 |
| E | 250 (25 s) | 75 (7.5 s) | `[1,2,4,8,16]` | 125 | 62 | 126 | 150 |

The buffer check is conservative: the full receptive-field length, not only its radius, must fit
inside the configured buffer. Training windows are shuffled between batches, but frame order inside
each window is never changed.

## 10. Model inputs and architectures

Implementations:

- `modeling/alive_dead/player_encoder.py`
- `modeling/alive_dead/tcn.py`

Both architectures use the same split, normalization, temporal backbone, training loop, losses and
evaluation protocol.

When `model.use_ball=true`, both architectures concatenate five additional per-frame values:
normalized ball X/Y, normalized ball speed, location validity and speed validity. WCSF ball speed
is backward displacement at 10 Hz and resets at halftime. Metrica keeps its provider-reported
speed from the selected 25 Hz frame; missing location or speed is never interpolated.

### 10.1 Team statistics TCN

At each frame, Home/Away team features `(2 x 6)` and their Boolean validity masks `(2 x 6)` are
flattened while preserving Home/Away order. The resulting 24 values are projected to the configured
temporal channel width, normally 64, before the TCN.

### 10.2 Deep Sets + team statistics TCN

Each player slot receives five inputs:

```text
[x, y, normalized_speed, location_valid, speed_valid]
```

A shared MLP is applied to every slot and both teams:

```text
5 -> 32 -> 32
```

with GELU activations. A player is considered present when either location or speed is valid.
Masked mean pooling over the player dimension produces separate 32-dimensional Home and Away
embeddings. This makes the player branch permutation-invariant: reordering player slots cannot
change the pooled representation.

The Home embedding, Away embedding, team features and team validity masks are concatenated and
projected to the shared temporal channel width. No player IDs, slot embeddings or player-specific
temporal links are used.

### 10.3 Temporal convolutional network

The TCN consists of residual blocks. Each block uses:

- Two same-length noncausal `Conv1d` layers by default.
- The block's configured dilation for both convolutions.
- Channel-wise LayerNorm independently at every timestamp.
- GELU activation.
- Dropout, default 0.15.
- Residual connection.
- Temporal masking after convolutional operations.

The default backbone has 64 channels, kernel size 3 and dilations `[1,2,4,8]`, producing RF 61.
Configurations D/E add dilation 16 and produce RF 125.

Output heads:

- Binary mode: one logit per frame, shape `(B,T)`.
- H/A/D mode: three logits `[H,A,D]` per frame, shape `(B,T,3)`.

Because the task is offline, using both past and future context is intentional. The model carries no
hidden state between windows or disconnected sequences.

## 11. Optimization and training loss

Implementation: `modeling/alive_dead/train.py`.

Default optimization settings:

- AdamW.
- Learning rate `1e-3`.
- Weight decay `1e-4`.
- Batch size 16.
- Maximum 100 epochs.
- Gradient clipping norm 1.0.
- Dropout 0.15.
- Seed 42.
- CPU/CUDA automatic device selection.

The binary task uses unweighted `BCEWithLogitsLoss` semantics. H/A/D uses unweighted
cross-entropy. Class weighting is intentionally not applied in the current baseline.

For a training window, effective per-frame weight is:

```text
supervision_mask * label_valid * inverse_supervision_coverage
```

Padding, artificial window edges and UNKNOWN targets therefore contribute neither loss nor
gradient. The reported epoch loss is the accumulated weighted numerator divided by the accumulated
effective weight.

Reproducibility seeds Python, NumPy and PyTorch. CUDA deterministic mode is enabled and cuDNN
benchmarking is disabled.

## 12. Full-segment validation and checkpoint selection

Early stopping does not use validation windows. After each training epoch:

1. The model enters evaluation mode and gradients are disabled.
2. Each complete contiguous validation segment is inferred independently.
3. Every original validation frame receives exactly one logit.
4. UNKNOWN labels are removed only from the loss; their tracking remains available as temporal
   context during inference.
5. BCE or cross-entropy is summed across all valid validation frames and divided by the total valid
   frame count.

This full-segment loss matches the intended evaluation-time inference behavior and avoids duplicate
validation frames or artificial transitions caused by overlapping windows.

If a full segment is too large for memory, the shared inference function can divide outputs into
chunks. Each chunk is expanded with a receptive-field halo, and only central logits are retained.
The result is equivalent to whole-segment inference within numerical tolerance.

The full-segment validation loss controls:

- `ReduceLROnPlateau`, with patience 3 and factor 0.5 by default.
- Best-checkpoint replacement.
- Early stopping, with patience 8 by default.

There is no minimum-delta parameter: any strictly smaller validation loss is an improvement.
Validation macro-F1 and segment/spell metrics are recorded after training, but they do not select
the checkpoint within an inner run.

## 13. Three-block inner CV and full-half refit

Within each outer fold requested by the training command, the pipeline performs:

1. Train inner block 1 from a fresh initialization.
2. Save its minimum-validation-loss checkpoint.
3. Evaluate that checkpoint only on its validation block.
4. Repeat independently for blocks 2 and 3.
5. Record each block's best epoch and validation metrics.
6. Choose the integer median of the three best epochs.
7. Reinitialize the model and normalizer.
8. Fit normalization on the entire outer training half.
9. Train for exactly the median epoch count.
10. Save the final full-half refit checkpoint without evaluating the outer test half.

The refit uses AdamW at the configured constant learning rate. It has no validation scheduler and no
early stopping because no validation frames are withheld from the full training half.

The code does not automatically select between Team TCN, Deep Sets TCN or temporal configurations
B-E. Those methodological choices must be compared using aggregate inner-validation results in
`cv_summary.json`. Outer-test or Metrica performance must not be used to choose them.

## 14. Saved artifacts and model identity

One experiment now has a timestamped identity shared by both outer folds. Examples:

```text
alive_dead_rf125_seq20_20261102_122033
had_rf125_seq20_20261102_122033
```

The architecture, seed, complete resolved configuration, temporal dimensions and methodological
decisions are stored in `run_metadata.json` rather than encoded repeatedly in child paths:

```text
<run>/
  run_metadata.json
  fold1/
    inner_block1/
    inner_block2/
    inner_block3/
    cv_summary.json
    refit/history.json, final_checkpoint.pt
  fold2/
    ...
  evaluation/
```

`history.json` contains epoch losses and learning rate. `cv_summary.json` contains block boundaries,
best losses, frame macro-F1, binary segment F1 or H/A/D spell F1, best epochs, aggregate means and
the selected refit epoch count.

Checkpoints contain:

- Model state.
- Optimizer state.
- Scheduler state for inner runs.
- Full experiment configuration.
- Fold and seed.
- Fitted normalizer.
- Epoch and best validation loss.
- Source frequency and receptive field.
- Training mode (`inner_cv` or `full_half_refit`).
- Validation block identity where applicable.
- Inner best epochs for a final refit.

Evaluation reads run identity from the manifest and validates checkpoint fold, architecture, seed,
training mode and block identity before loading weights. Historical checkpoint payloads remain
readable, although new CLI evaluation requires a manifest-backed run directory.

## 15. Binary evaluation

Implementation: `modeling/alive_dead/evaluate.py`.

For every requested partition, evaluation:

1. Loads the checkpoint and its saved configuration.
2. Loads and validates the aligned feature/label archives.
3. Reconstructs the saved normalizer without fitting it again.
4. Rebuilds the correct inner or refit split.
5. Infers every genuinely contiguous partition segment independently.
6. Applies `sigmoid(logit) >= 0.5` to predict ALIVE; otherwise DEAD.
7. Splits scoring runs at UNKNOWN labels after inference.

Inference is allowed to use tracking on an UNKNOWN frame as context, but that frame is not scored.
No scored segment can cross an UNKNOWN gap, buffer, disconnected training interval or halftime.

### 15.1 Frame metrics

Binary reports contain:

- Accuracy.
- Per-class precision, recall, F1 and support.
- Macro-F1 across DEAD and ALIVE.
- `2 x 2` confusion matrix, with ground truth in rows and predictions in columns.

### 15.2 Binary temporal segment metrics

Ground-truth and predicted sequences are run-length encoded independently inside every valid
contiguous run. Same-class segments from the same run are greedily matched one-to-one in descending
temporal IoU order.

Precision, recall and F1 are reported at IoU thresholds:

```text
0.3, 0.4, 0.5, 0.6, 0.7
```

The headline threshold is 0.5, and `mean_f1` is the mean across all five thresholds. Both ALIVE and
DEAD segments participate in this binary segment score. This is an adaptation for the binary task;
it is not the official three-class challenge possession-spell score.

### 15.3 OOF reconstruction

After both refit models exist:

- Fold 2 predicts held-out H1.
- Fold 1 predicts held-out H2.

The two sets of held-out predictions are combined chronologically. Coverage checks require every
valid labelled frame to appear exactly once. This combined OOF result is the main WCSF estimate for
comparing an already-selected architecture while exposing variation between halves.

An optional timeline PNG plots ground truth and predictions for each half. Disconnected runs remain
visually separated.

## 16. H/A/D evaluation

Implementation: `modeling/alive_dead/evaluate_had.py`.

H/A/D predictions use `argmax` over logits `[H,A,D]`. Frame evaluation reports:

- Accuracy.
- Per-class H/A/D precision, recall, F1 and support.
- Macro-F1 across H, A and D.
- `3 x 3` confusion matrix.

Possession spell evaluation reuses `evaluation/possession_metrics.py`, the repository's official
competition metric implementation. A possession spell is a maximal H or A interval; D intervals are
not possession spells. Same-team predicted and ground-truth spells are matched one-to-one at IoU
thresholds 0.3-0.7, with 0.5 as the headline threshold.

The report also includes:

- Ground-truth and predicted possession counts.
- Mean spell duration.
- Fragmentation ratio.
- Transition diagnostics at 5 and 10 frames, corresponding to approximately 0.5 and 1.0 seconds at
  WCSF's 10 Hz rate.

Valid runs are explicitly separated at halftime and UNKNOWN labels. Gap separators and run bounds
ensure that neither spells nor transitions can cross those discontinuities.

## 17. Reproducible commands

Build WCSF features:

```powershell
python -m feature_engineering.wcsf_features
```

Build strict binary labels:

```powershell
python -m feature_engineering.wcsf_labels
```

Build strict H/A/D labels:

```powershell
python -m feature_engineering.wcsf_had_labels
```

Train binary Deep Sets configuration D for both outer folds:

```powershell
python -m modeling.alive_dead.train --config configs/alive_dead/temporal_d_rf125_seq20.json --fold all --model-type deepsets_tcn --classification-mode binary
```

Train H/A/D Deep Sets configuration D:

```powershell
python -m modeling.alive_dead.train --config configs/alive_dead/temporal_d_rf125_seq20.json --fold all --model-type deepsets_tcn --classification-mode had
```

Evaluate binary combined OOF results from the timestamped run printed by training:

```powershell
python -m modeling.alive_dead.evaluate --dataset wcsf --run-dir <alive-dead-run-folder> --fold all --save-timeline
```

Evaluate binary inner validation:

```powershell
python -m modeling.alive_dead.evaluate --dataset wcsf --run-dir <alive-dead-run-folder> --fold 1 --partition validation --inner-block all
```

Evaluate H/A/D combined OOF results:

```powershell
python -m modeling.alive_dead.evaluate_had --run-dir <had-run-folder> --fold all
```

Training and evaluation display `tqdm` progress by default. `--no-progress` disables terminal bars
without changing losses, predictions or metrics.

## 18. Important methodological decisions and limitations

### Decisions that protect validity

- No random frame split; all splits are temporal and contiguous.
- The opposite half remains untouched during inner model selection.
- Buffers purge context around validation boundaries.
- Normalization is fitted only on training frames.
- Training loss excludes artificial window edges.
- Validation and evaluation use full contiguous-segment inference.
- UNKNOWN labels remain temporal context but contribute no loss or metric.
- No sequence crosses halftime or a disconnected partition.
- Player encoding is permutation-invariant and does not invent identity continuity.
- Test/Metrica data never select checkpoint epochs.

### Limitations

- WCSF is a single match, so the number of independent samples is much smaller than 57,374 frames.
- Home/Away possession labels may encode provider-specific semantics, especially around stoppages.
- WCSF player and speed coverage is nearly complete, unlike Metrica.
- WCSF reported speed is provider-smoothed and may not be measurement-equivalent to Metrica speed.
- Ball-free H/A/D models lack direct ball evidence. Ball-enabled models address this but face a
  strong domain shift: WCSF ball location is complete, while Metrica location and especially
  provider speed are frequently missing.
- Noncausal inference is appropriate for offline reconstruction but not real-time deployment.
- Fixed binary threshold 0.5 is not calibrated or tuned.
- No augmentation, temporal smoothing, structured decoding, pseudo-labeling or confidence calibration
  is included in the clean baseline.
- Binary segment F1 and official H/A/D possession-spell F1 answer different questions and should not
  be compared as if they were the same metric.

## 19. Source-code map

| Responsibility | File |
|---|---|
| WCSF player/team/optional ball features | `feature_engineering/wcsf_features.py` |
| Metrica compatible player/team/optional ball features | `feature_engineering/metrica_features.py` |
| Strict binary labels | `feature_engineering/wcsf_labels.py` |
| Strict H/A/D labels | `feature_engineering/wcsf_had_labels.py` |
| Configuration and RF validation | `modeling/alive_dead/config.py` |
| Archive loading and temporal windows | `modeling/alive_dead/dataset.py` |
| Purged folds and valid subsegments | `modeling/alive_dead/splits.py` |
| Train-only normalization | `modeling/alive_dead/preprocessing.py` |
| Deep Sets encoder | `modeling/alive_dead/player_encoder.py` |
| TCN architectures | `modeling/alive_dead/tcn.py` |
| Shared full/chunked inference | `modeling/alive_dead/inference.py` |
| Training, checkpoint selection and refit | `modeling/alive_dead/train.py` |
| Timestamped run metadata and paths | `modeling/alive_dead/runs.py` |
| Binary evaluation and Metrica transfer | `modeling/alive_dead/evaluate.py` |
| H/A/D evaluation | `modeling/alive_dead/evaluate_had.py` |
| Binary timeline plotting | `modeling/alive_dead/plotting.py` |
| Official challenge metrics | `evaluation/possession_metrics.py` |

