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
for the target CPU/CUDA environment, then install the remaining modeling dependencies with
`pip install -r requirements-modeling.txt`.

## Inputs

The default paths are:

```text
data/open_data_groningen/processed/wcsf_player_team_features.npz
data/open_data_groningen/processed/wcsf_alive_dead_labels.npz
```

Ball-enabled experiments instead use
`data/open_data_groningen/processed/wcsf_player_team_ball_features.npz` and set
`model.use_ball=true`. The historical player/team archive remains unchanged.

```json
{
  "data": {
    "feature_path": "data/open_data_groningen/processed/wcsf_player_team_ball_features.npz"
  },
  "model": {
    "use_ball": true
  }
}
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
ball                     (B,T,3)       # optional: x, y, speed
ball_location_valid      (B,T)         # optional
ball_speed_valid         (B,T)         # optional
labels                   (B,T)
output_mask              (B,T)
supervision_mask         (B,T)
loss_weight              (B,T)
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

The default 100-frame training windows use stride 40. Only outputs with their complete available
TCN context contribute to loss: artificial window edges are context-only, while genuine temporal
segment edges remain supervised with the same natural zero padding used by full-segment inference.
The receptive-field radius and supervised center are derived from the model configuration.
Inverse supervised-coverage weights ensure each original frame has total weight one despite
overlap, and configuration validation rejects strides that would leave supervision gaps.

Validation does not use temporal windows. Each contiguous validation segment is inferred once as
a complete sequence, UNKNOWN labels are excluded, and BCE/CE is averaged over all valid frames.
This full-segment loss drives scheduling, checkpoint selection and early stopping. Evaluation uses
the same shared segment-inference function; optional halo chunking is available for constrained
memory without changing logits.

## Normalization

Normalization is fitted separately for each inner block using only that block's training mask.
The final refit normalizer uses the complete training half:

- Player X/Y remain in their existing normalized pitch coordinates.
- Valid player speeds share one training-fold mean and standard deviation across both teams.
- Team mean speed, median speed and speed standard deviation are standardized per feature, with
  statistics shared across Home and Away.
- For ball-enabled runs, valid ball speeds use one additional training-fold mean and standard
  deviation; normalized ball X/Y and both ball masks remain unchanged.
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

With `use_ball=true`, both frontends also concatenate
`[ball_x, ball_y, normalized_ball_speed, location_valid, speed_valid]` before the temporal
projection. No ball distances, interpolation or separate ball encoder are used.

The default TCN has four residual blocks, dilations `[1,2,4,8]`, two noncausal kernel-3
convolutions per block and 64 channels. Its receptive field is:

```text
1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 61 frames = 6.1 seconds
```

Its context radius is 30 frames on either side. Configuration validation rejects a receptive field
larger than either the sequence or validation buffer.

## Training

Train both outer folds into one timestamped run:

```powershell
python -m modeling.alive_dead.train --config configs/alive_dead/temporal_d_rf125_seq20.json --fold all --model-type deepsets_tcn --classification-mode had
```

Or provide a JSON file containing any nested `data`, `split`, `model` or `training` dataclass
fields:

```powershell
python -m modeling.alive_dead.train --config path/to/config.json
```

One command runs all three inner blocks and then the full-half refit. Inner runs use AdamW,
masked BCE or cross-entropy, ReduceLROnPlateau, gradient clipping and full-segment validation-loss
early stopping.
Their median best epoch becomes the fixed refit length. The refit starts from new weights, uses
the complete training half, constant configured learning rate and no validation-dependent
scheduler. Training constructs no test DataLoader and never performs outer-test inference.

The command prints the created run path. Run names contain the task, receptive field, sequence
duration and local creation time. `run_metadata.json` records the complete resolved configuration
and methodological decisions:

```text
had_rf125_seq20_YYYYMMDD_HHMMSS/
  run_metadata.json
  fold1/
    inner_block1/history.json, best_checkpoint.pt, evaluation_validation.json
    inner_block2/history.json, best_checkpoint.pt, evaluation_validation.json
    inner_block3/history.json, best_checkpoint.pt, evaluation_validation.json
    cv_summary.json
    refit/history.json, final_checkpoint.pt
  fold2/
    ...
  evaluation/
```

Training shows fold, block, epoch and batch progress with running loss. Pass `--no-progress` for
redirected output or automation.

## Evaluation

Evaluation is checkpoint-based and restores the saved normalizer without refitting. Train and test
evaluation use the final full-half refit checkpoint:

```powershell
python -m modeling.alive_dead.evaluate --run-dir <run-folder> --fold 1 --partition test
python -m modeling.alive_dead.evaluate --run-dir <run-folder> --fold 2 --partition train
```

Refit models have no validation partition. Evaluate one or all inner validation checkpoints with:

```powershell
python -m modeling.alive_dead.evaluate --run-dir <run-folder> --fold 1 --partition validation --inner-block 1
python -m modeling.alive_dead.evaluate --run-dir <run-folder> --fold 1 --partition validation --inner-block all
```

Each genuinely contiguous partition segment is inferred as one sequence, so temporal metrics are
never calculated independently per training window and no segment crosses a buffer or halftime.

After both folds have been trained, combine their held-out predictions with:

```powershell
python -m modeling.alive_dead.evaluate --run-dir <run-folder> --fold all
```

The combined out-of-fold result uses Fold 2 to predict the first half and Fold 1 to predict the
second half. Every scored frame therefore comes from a model that did not train on that half.

Add `--save-timeline` to any evaluation command to save a diagnostic PNG with aligned ground-truth
and prediction segments. For combined OOF evaluation, the first half is shown above the second:

```powershell
python -m modeling.alive_dead.evaluate --run-dir <run-folder> --fold all --save-timeline
```

The combined figure is saved as `evaluation/oof_timeline.png` in the run directory. Blank
timeline areas indicate disconnected or excluded intervals and are never joined artificially.

Predictions use a fixed threshold of `sigmoid(logit) >= 0.5`. Reports contain accuracy, binary
confusion matrix, per-class precision/recall/F1 and macro F1. Confusion-matrix rows are ground
truth, columns are predictions, and the class order is `DEAD`, `ALIVE`.

The binary segment score follows the competition's `spellF1` structure: ALIVE and DEAD runs are
pooled, same-label segments are matched one-to-one by descending temporal IoU, and F1 is averaged
over thresholds `0.3, 0.4, 0.5, 0.6, 0.7`. F1@0.5 is also reported as the headline threshold.
Unlike the official possession score, this binary diagnostic includes DEAD segments and cannot
separate consecutive Home and Away possessions inside one uninterrupted ALIVE interval.

Evaluation reports and figures are saved under the run's `evaluation/` directory. Inner validation evaluation runs
during cross-validation; outer train/test evaluation remains a separate explicit command.

### Metrica transfer evaluation

The same evaluator can apply both frozen WCSF refit checkpoints to the label-free Metrica 10 Hz
features and score the public five-minute annotation at its original 25 Hz rate:

```powershell
python -m modeling.alive_dead.evaluate --dataset metrica --run-dir <alive-dead-run-folder> --fold all
```

Each checkpoint uses its own saved WCSF normalizer; no statistics are fitted on Metrica. The two
halves are inferred independently, and Fold 1/Fold 2 ALIVE probabilities are averaged with fixed
equal weights. The saved source-to-sample index reconstructs a prediction for every scored 25 Hz
frame without interpolation. Public `H` and `A` labels are collapsed to ALIVE, `D` to DEAD, and
`X` is excluded from scoring.

Ball-enabled transfer uses `metrica_player_team_ball_features.npz`. WCSF ball speed is derived
from backward 10 Hz displacement, whereas Metrica ball speed is the sparse provider value from
the selected 25 Hz frame. Missing Metrica locations and speeds remain independently masked.

The report contains the binary frame and segment metrics used for WCSF, plus the competition
scorer's exact `alive_dead_f1`, transition metrics at 12 and 25 frames, and binary-adapted
fragmentation. Official H/A/D macro F1 and possession spell F1 are deliberately not reported,
because an ALIVE/DEAD model does not predict the team in possession.

The command saves one JSON report, one aligned prediction NPZ and one public-window timeline PNG
in the selected runs directory. Invalid 25 Hz predictions outside the scored tracking intervals
are zero-filled and must be interpreted through `prediction_valid_25hz`.

## Not implemented yet

- Threshold selection or probability calibration.
- Tracking degradation augmentation or domain adaptation.
- Learned or hard-duration temporal decoding.

## Strict labels and direct H/A/D mode

New binary experiments use strict unanimity labels. The 39 timestamps with conflicting
`AliveFlag` votes remain in every input sequence but have `label_valid=False`, label `-1`, and
zero supervised loss. They also split evaluation runs, so a scored segment never crosses an
ambiguous timestamp. The original majority-voted archive and historical run directories are
unchanged. Rebuild the two strict archives with:

```powershell
python -m feature_engineering.wcsf_labels
python -m feature_engineering.wcsf_had_labels
```

The direct three-class task uses `H=0`, `A=1`, `D=2`, masked cross-entropy, and the same features,
normalization, purged folds, temporal windows and TCN. DEAD takes precedence whenever the alive
vote is unanimously dead. An alive timestamp is supervised only when both the alive and
possession votes are unanimous. This leaves 110 H/A/D timestamps invalid.

Use a JSON config containing the following fields, or use the command-line mode (which selects
the H/A/D archive when no JSON config is supplied):

```json
{
  "data": {
    "label_path": "data/open_data_groningen/processed/wcsf_had_labels.npz"
  },
  "model": {
    "classification_mode": "had",
    "model_type": "deepsets_tcn"
  }
}
```

```powershell
python -m modeling.alive_dead.train --fold all --model-type deepsets_tcn --classification-mode had
python -m modeling.alive_dead.evaluate_had --run-dir <had-run-folder> --fold 1 --partition test
python -m modeling.alive_dead.evaluate_had --run-dir <had-run-folder> --fold all
```

The evaluator reads task, architecture and seed from `run_metadata.json`; it does not reconstruct
checkpoint paths from command-line naming arguments.

H/A/D evaluation reports frame accuracy, macro F1, per-class precision/recall/F1, a 3x3
confusion matrix, and the official one-to-one possession-spell F1 at IoUs 0.3 through 0.7.
Fragmentation and transition diagnostics use explicit valid runs and 10 Hz tolerances of 5 and
10 frames. Neither spells nor transitions cross halftime or an invalid-label gap.

### H/A/D temporal postprocessing

Temporal postprocessing can be tuned without retraining or using an outer test half:

```powershell
python -m modeling.alive_dead.postprocess_had --run-dir <completed-had-run-folder>
```

For each outer fold, the command selects a centered probability-smoothing window and a symmetric
Viterbi transition penalty using only that fold's three inner-validation checkpoints. It then
applies the selected values unchanged to the corresponding refit model's held-out half and
compares raw argmax, smoothing, Viterbi and the validation-selected overall decoder in combined
OOF results.

Decoding restarts at halftime, partition boundaries and invalid-label gaps. The checkpoint-bound
probability cache is stored at `postprocessing/probabilities.npz`; timestamped experiment reports
are stored under `postprocessing/decoding_<datetime>/`. Existing checkpoints and evaluation
reports are not modified. Use `--rebuild-cache` only when the probabilities should be inferred
again.
