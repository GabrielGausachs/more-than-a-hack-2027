# WCSF ALIVE/DEAD modeling

This package is the training foundation for a binary temporal classifier trained on the external
WCSF Groningen match. It does not use Metrica possession ground truth and does not implement the
final challenge evaluation or H/A ownership model.

## Package responsibilities

- `config.py` centralizes data, split, model and training settings.
- `splits.py` builds the deterministic half-held-out folds.
- `preprocessing.py` fits mask-aware normalization on training frames only.
- `dataset.py` loads aligned NPZ archives and creates contiguous temporal windows.
- `player_encoder.py` implements the permutation-invariant player-set encoder.
- `tcn.py` contains the shared noncausal residual TCN and both model frontends.
- `train.py` contains future training, validation, checkpointing and history logging.

PyTorch is required but is intentionally not installed by this package. Install a build appropriate
for the target CPU/CUDA environment before importing or running the modeling code.

## Inputs

The default paths are:

```text
data/open_data_groningen/processed/wcsf_player_team_features.npz
data/open_data_groningen/processed/wcsf_alive_dead_labels.npz
```

Only `labels` is read from the label archive. Vote counts and vote fractions are never exposed to
the model. Player IDs, `PossessionFlag` and `team_game_phase` are not model inputs.

Each batch contains:

```text
players                  (B,T,2,24,3)
player_location_valid    (B,T,2,24)
player_speed_valid       (B,T,2,24)
team_features            (B,T,2,6)
team_feature_valid       (B,T,2,6)
labels                   (B,T)
output_mask              (B,T)
loss_weight              (B,T)
global_index             (B,T)
```

## Folds

All intervals are half-open global archive indices.

| Fold | Train | Buffer | Validation | Test |
|---|---|---|---|---|
| 1 | `[0,11806)`, `[16208,28014)` | `[11806,11906)`, `[16108,16208)` | `[11906,16108)` | `[28014,57374)` |
| 2 | `[28014,40392)`, `[44996,57374)` | `[40392,40492)`, `[44896,44996)` | `[40492,44896)` | `[0,28014)` |

The validation interval is the centered `round(15%)` block of the training half. The two adjacent
buffers are each 100 frames (10 seconds). Windows are constructed independently inside each
segment and can never cross a half, partition, buffer or disconnected training interval.

The default 100-frame training windows use stride 50. Inverse-coverage loss weights ensure each
original frame has total weight one per epoch despite overlap. Validation and future test windows
use stride 100, with right padding and an output mask on the final window.

## Normalization

Normalization is fitted separately for every fold using only its training mask:

- Player X/Y remain in their existing normalized pitch coordinates.
- Valid player speeds share one training-fold mean and standard deviation across both teams.
- Team mean speed, median speed and speed standard deviation are standardized per feature, with
  statistics shared across Home and Away.
- Tracked-player count, valid-speed count and speed coverage remain in their original scales.
- Invalid and padded values remain zero and masks are unchanged.

The fitted parameters are stored in future checkpoints. Validation, buffer and test frames never
contribute to fitting.

## Models

`team_tcn` flattens Home/Away team statistics and masks, projects them to 64 temporal channels and
applies the TCN.

`deepsets_tcn` additionally applies one shared `5 -> 32 -> 32` MLP to every player. Its five inputs
are X, Y, speed, location validity and speed validity. Masked mean pooling separately creates Home
and Away embeddings, making the branch invariant to player-slot permutations. These embeddings are
fused with the same team inputs before the shared TCN type.

Both produce one logit per frame with shape `(B,T)`.

The default TCN has four residual blocks, dilations `[1,2,4,8]`, two noncausal kernel-3
convolutions per block and 64 channels. Its receptive field is:

```text
1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 61 frames = 6.1 seconds
```

Its context radius is 30 frames on either side. Configuration validation rejects a receptive field
larger than either the sequence or validation buffer.

## Future training

Defaults can be used directly:

```powershell
python -m modeling.alive_dead.train --fold 1 --model-type team_tcn
python -m modeling.alive_dead.train --fold 2 --model-type deepsets_tcn
```

Or provide a JSON file containing any nested `data`, `split`, `model` or `training` dataclass
fields:

```powershell
python -m modeling.alive_dead.train --config path/to/config.json
```

Training uses AdamW, unweighted masked BCE, ReduceLROnPlateau, gradient clipping and validation-loss
early stopping. Runs are written below `data/open_data_groningen/model_runs/`, which is already
excluded by the repository's `data/` ignore rule. The training function constructs a test loader
but never evaluates it.

## Evaluation

Evaluation is a separate checkpoint-based command. It restores the normalizer saved in the best
checkpoint and never refits statistics. A single fold and partition can be evaluated with:

```powershell
python -m modeling.alive_dead.evaluate --fold 1 --model-type team_tcn --partition test
python -m modeling.alive_dead.evaluate --fold 2 --model-type deepsets_tcn --partition validation
```

`--partition` accepts `train`, `validation` or `test`. Each genuinely contiguous partition segment
is inferred as one sequence, so temporal metrics are never calculated independently per training
window and no segment can cross a buffer, disconnected interval or halftime.

After both folds have been trained, combine their held-out predictions with:

```powershell
python -m modeling.alive_dead.evaluate --fold all --model-type team_tcn
python -m modeling.alive_dead.evaluate --fold all --model-type deepsets_tcn
```

The combined out-of-fold result uses Fold 2 to predict the first half and Fold 1 to predict the
second half. Every scored frame therefore comes from a model that did not train on that half.

Predictions use a fixed threshold of `sigmoid(logit) >= 0.5`. Reports contain accuracy, binary
confusion matrix, per-class precision/recall/F1, macro F1, and per-class plus macro segment F1 at
temporal IoU 0.50 and 0.75. Confusion-matrix rows are ground truth, columns are predictions, and
the class order is `DEAD`, `ALIVE`. Segment matching is one-to-one and performed independently for
both classes.

Individual reports are saved as `evaluation_<partition>.json` in the run directory. Combined
reports are saved as `oof_<model_type>_seed<seed>.json` in the model-runs directory. Evaluation
does not modify checkpoints and does not run automatically during training.

## Not implemented yet

- Threshold selection or probability calibration.
- Metrica feature transformation and inference.
- Tracking degradation augmentation or domain adaptation.
- H/A possession classification and final spell decoding.
