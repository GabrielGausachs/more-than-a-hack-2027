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
- `train.py` runs the three inner trainings, validation summaries and full-half refit.

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

Each outer fold keeps the opposite half untouched as test data. Three inner runs place a
`round(15%)` validation block at fixed locations within the training half:

| Fold | Block 1 | Block 2 | Block 3 | Outer test |
|---|---|---|---|---|
| 1 | `[4902,9104)` | `[11906,16108)` | `[18909,23111)` | `[28014,57374)` |
| 2 | `[33152,37556)` | `[40492,44896)` | `[47832,52236)` | `[0,28014)` |

Every inner validation interval has a 100-frame (10-second) buffer on both sides. All remaining
parts of the training half are independent training segments. Windows can never cross a half,
partition, buffer or disconnected training interval.

The default 100-frame training windows use stride 50. Inverse-coverage loss weights ensure each
original frame has total weight one per epoch despite overlap. Validation and future test windows
use stride 100, with right padding and an output mask on the final window.

## Normalization

Normalization is fitted separately for each inner block using only that block's training mask.
The final refit normalizer uses the complete training half:

- Player X/Y remain in their existing normalized pitch coordinates.
- Valid player speeds share one training-fold mean and standard deviation across both teams.
- Team mean speed, median speed and speed standard deviation are standardized per feature, with
  statistics shared across Home and Away.
- Tracked-player count, valid-speed count and speed coverage remain in their original scales.
- Invalid and padded values remain zero and masks are unchanged.

The fitted parameters are stored in checkpoints. Inner validation/buffer/test frames never enter
inner normalization, and outer test frames never enter refit normalization.

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

## Training

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

One command runs all three inner blocks and then the full-half refit. Inner runs use AdamW,
unweighted masked BCE, ReduceLROnPlateau, gradient clipping and validation-loss early stopping.
Their median best epoch becomes the fixed refit length. The refit starts from new weights, uses
the complete training half, constant configured learning rate and no validation-dependent
scheduler. It never iterates the outer test loader.

Artifacts are kept separate from legacy single-block runs:

```text
fold1_team_tcn_seed42/
  inner_block1/history.json, best_checkpoint.pt, evaluation_validation.json
  inner_block2/history.json, best_checkpoint.pt, evaluation_validation.json
  inner_block3/history.json, best_checkpoint.pt, evaluation_validation.json
  cv_summary.json
  refit/history.json, final_checkpoint.pt
```

## Evaluation

Evaluation is checkpoint-based and restores the saved normalizer without refitting. Train and test
evaluation use the final full-half refit checkpoint:

```powershell
python -m modeling.alive_dead.evaluate --fold 1 --model-type team_tcn --partition test
python -m modeling.alive_dead.evaluate --fold 2 --model-type deepsets_tcn --partition train
```

Refit models have no validation partition. Evaluate one or all inner validation checkpoints with:

```powershell
python -m modeling.alive_dead.evaluate --fold 1 --model-type team_tcn --partition validation --inner-block 1
python -m modeling.alive_dead.evaluate --fold 1 --model-type team_tcn --partition validation --inner-block all
```

Each genuinely contiguous partition segment is inferred as one sequence, so temporal metrics are
never calculated independently per training window and no segment crosses a buffer or halftime.

After both folds have been trained, combine their held-out predictions with:

```powershell
python -m modeling.alive_dead.evaluate --fold all --model-type team_tcn
python -m modeling.alive_dead.evaluate --fold all --model-type deepsets_tcn
```

The combined out-of-fold result uses Fold 2 to predict the first half and Fold 1 to predict the
second half. Every scored frame therefore comes from a model that did not train on that half.

Add `--save-timeline` to any evaluation command to save a diagnostic PNG with aligned ground-truth
and prediction segments. For combined OOF evaluation, the first half is shown above the second:

```powershell
python -m modeling.alive_dead.evaluate --fold all --model-type deepsets_tcn --save-timeline
```

The combined figure is saved as `oof_<model_type>_seed<seed>_timeline.png` in the selected
model-runs directory. Single-fold figures are saved beside the checkpoint they evaluate. Blank
timeline areas indicate disconnected or excluded intervals and are never joined artificially.

Predictions use a fixed threshold of `sigmoid(logit) >= 0.5`. Reports contain accuracy, binary
confusion matrix, per-class precision/recall/F1 and macro F1. Confusion-matrix rows are ground
truth, columns are predictions, and the class order is `DEAD`, `ALIVE`.

The binary segment score follows the competition's `spellF1` structure: ALIVE and DEAD runs are
pooled, same-label segments are matched one-to-one by descending temporal IoU, and F1 is averaged
over thresholds `0.3, 0.4, 0.5, 0.6, 0.7`. F1@0.5 is also reported as the headline threshold.
Unlike the official possession score, this binary diagnostic includes DEAD segments and cannot
separate consecutive Home and Away possessions inside one uninterrupted ALIVE interval.

Individual reports are saved beside the checkpoint they evaluate. Combined reports are saved as
`oof_<model_type>_seed<seed>.json` in the model-runs directory. Inner validation evaluation runs
during cross-validation; outer train/test evaluation remains a separate explicit command.

## Not implemented yet

- Threshold selection or probability calibration.
- Metrica feature transformation and inference.
- Tracking degradation augmentation or domain adaptation.
- H/A possession classification and final spell decoding.
