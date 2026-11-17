# Decoding Possession
### From Player Tracking to Football Possession Spells

**Gabriel Gausachs | Team GG**

*Metrica Sports — More Than a Hack 2027*

## 1. Introduction and Approach Overview

### 1.1 Task Definition

Phase 1 of the Metrica Sports *More Than a Hack 2027* challenge aims to reconstruct team possession in the FC Barcelona vs Athletic Club match using the provided data. The objective is to assign one of four labels to each of the 199,319 frames: Home possession (`H`), Away possession (`A`), Dead ball (`D`), or Outside the scored match periods (`X`).

Determining possession from tracking data is not as straightforward as identifying the player closest to the ball. During a match, the ball frequently travels between players, becomes temporarily uncontrolled after challenges or deflections, or disappears from the tracking feed. The model must therefore consider the surrounding play and distinguish temporary situations from actual changes in possession.

The challenge's primary evaluation metric is **possession-spell F1**, which measures how well complete Home and Away possession intervals are reconstructed.

### 1.2 Proposed Approach

We developed a tracking-only approach based on a **Deep Sets encoder combined with a noncausal Temporal Convolutional Network (TCN)**, followed by Viterbi decoding. The model is trained on external SkillCorner Open Data using conservatively generated possession labels and applied to Metrica tracking without using the provided game's ground-truth labels for training.

![Pipeline diagram](figures/pipeline.png)

The approach consists of four main components:

1. **Tracking preprocessing and feature extraction.** SkillCorner and Metrica tracking are converted into a common representation containing player positions, speeds, team geometry, ball information, and observation-validity indicators.

2. **Deep Sets player encoding.** Players are represented as unordered sets rather than fixed identities. Tracking observations may be incomplete or inconsistently ordered, so the encoder learns team representations independently of player ordering.

3. **Temporal modeling with a TCN.** Possession is a continuous state rather than an independent decision at every frame. The noncausal TCN uses observations before and after each frame to recognize sustained control, turnovers, and stoppages. Future context is appropriate because the task reconstructs a completed match offline.

4. **Viterbi decoding.** Frame-level probabilities can fluctuate during passes, challenges, or tracking errors. The decoder penalizes unnecessary state changes to produce more coherent possession intervals.

The final prediction pipeline uses only the provided tactical player and ball tracking data. Automatic Smart Tags were explored as additional inference evidence but are not used in the final predictions. Broadcast video and homographies are not used.

## 2. Data Sources and Label Generation

### 2.1 Data Sources and Assets

The model is trained using **[SkillCorner Open Data](https://github.com/SkillCorner/opendata)**. We use 20 matches from the 2024/25 Australian A-League season, with player and ball tracking at 10 Hz, match metadata, and SkillCorner's tactical and event annotations. Four files are used for each match:

- **Match metadata (`_match.json`):** Team and player identities, pitch dimensions, attacking direction, and official match periods.
- **Tracking (`_tracking_extrapolated.jsonl`):** Frame-level player and ball positions, including observation-quality information.
- **Phases of Play (`_phases_of_play.csv`):** Tactical intervals and the team identified as being in possession.
- **Dynamic Events (`_dynamic_events.csv`):** Player-possession events and interruption information used to verify possession labels and identify supported stoppages.

The tracking files provide model inputs, while Phases of Play and selected Dynamic Events are used to construct training labels. These labels are derived from existing provider annotations; no additional manual annotations are created for the training matches.

For inference, we use the **Metrica tactical tracking data** for FC Barcelona vs Athletic Club. It provides player and ball observations at 25 Hz, together with team and match-period metadata. Metrica's coordinates, player speeds, and observation-validity information are converted into the same feature representation used for SkillCorner.

The supplied broadcast videos were neither explored nor used. The automatic **Smart Tags**, describing situations such as build-up, progression, and attacking transitions, were explored as possible additional possession evidence but are not included in the final system.

### 2.2 Training Data and Label Construction

SkillCorner does not provide the exact frame-level Home (`H`), Away (`A`), and Dead-ball (`D`) targets needed by our model. We construct **conservative H/A/D/U annotations** by combining Phases of Play with selected player-possession Dynamic Events. The fourth annotation, Unknown (`U`), represents frames for which the available evidence is insufficient or contradictory; it is **not a class predicted by the model**.

Phases of Play provide the main possession evidence. Each phase covers a frame interval and identifies the possessing team through `team_in_possession_id`, which is mapped to Home or Away. Dynamic Events provide player-level possession and interruption information that allows us to check these assignments and distinguish supported dead-ball intervals from uncertain gaps.

The labeling procedure follows these rules:

1. **Home and Away:** Assign `H` or `A` to frames covered by a valid Phase of Play for the corresponding team. If a player-possession event identifies the opposite team during the same frame, mark the conflicting frame as `U` rather than forcing a label. Phase/period inconsistencies are also left unknown.
2. **Dead ball:** Consider gaps between consecutive phases within the same match period. Assign `D` only when surrounding player-possession events provide matching interruption evidence, their team-relative `for`/`against` perspectives are consistent, both events are sufficiently close to the gap boundaries (within 30 frames), and there is no contradictory evidence of player or team control inside the gap.
3. **Unknown:** Assign `U` to unsupported phase gaps, conflicting annotations, and frames without sufficiently reliable possession or stoppage evidence. These frames are not used as supervised classification targets.
4. **Match periods:** Restrict training to the official first and second halves. Frames outside those periods are not supervised, and temporal sequences do not cross halftime.

This strategy avoids assuming that every frame without a Phase of Play must be dead ball. Such gaps can also reflect missing or incomplete tactical annotations. The resulting labels prioritize confidence over coverage, at the cost of leaving some frames without supervision.

Across the 20 matches, the generated annotations have the following distribution:

| Annotation | Frames |
|---|---:|
| Home (`H`) | 319,399 |
| Away (`A`) | 308,895 |
| Dead ball (`D`) | 288,850 |
| Unknown (`U`) | 327,670 |

During training, `U` frames remain in the input sequences so they can provide temporal context, but they are ignored by the classification loss. The model itself learns only the three states `H`, `A`, and `D`. Because these targets are inferred from provider annotations rather than independently checked for every frame, they are treated as **weak supervision**. No labels from the Metrica match are used to train the model.

## 3. Methodology

### 3.1 Cross-Provider Feature Extraction

Since the model is trained on SkillCorner data and applied to Metrica tracking, both providers are converted into a shared representation. The pipeline uses player positions and speeds, team geometry, ball information, and explicit validity indicators to account for incomplete observations.

Player and ball coordinates are expressed in a pitch-centered coordinate system measured in metres, with the Home team always attacking towards the positive x-direction. SkillCorner coordinates are oriented using the match metadata. Metrica's normalized coordinates are first converted into pitch dimensions in metres and then oriented consistently for each half. SkillCorner player speeds are estimated from position differences over 0.5 seconds, whereas Metrica's available player-speed measurements are used directly.

The input features are divided into two groups:

- **Player-level features:** Each player is represented by five values: x coordinate, y coordinate, speed, position validity, and speed validity. Observations are grouped by team, with capacity for up to 24 players per team and frame. This accommodates missing and duplicated tracking observations without requiring consistent slot identities.
- **Frame-level features:** A total of 51 engineered values summarize team centroids, longitudinal and lateral dispersion, compactness, tracked-player counts, speed statistics, relative positioning between teams, one-second changes in team geometry, ball movement, ball-to-player distances, and observation availability.

Missing values use neutral zeros alongside validity indicators, enabling the model to distinguish missing observations from measured values. Normalization statistics are fitted on the SkillCorner training matches and reused for Metrica inference, without fitting normalization on the target match.

SkillCorner tracking operates at 10 Hz, while Metrica provides 25 Hz frames. For inference, each Metrica half is sampled at 10 Hz using nearest timestamps without interpolation. The resulting predictions are then mapped back onto the original 25 Hz frame timeline for submission.

### 3.2 Deep Sets + TCN Architecture

The model combines a **Deep Sets encoder** for spatial representations with a **noncausal Temporal Convolutional Network (TCN)** for temporal modeling.

**Deep Sets encoder.** At each frame, the same multilayer perceptron (MLP) processes each player's five input features, with layer sizes `5 → 32 → 64`. Embeddings of players with valid positions are aggregated by masked mean pooling, separately for Home and Away. This produces team-level representations without relying on player ordering or stable player identities.

The Home and Away embeddings are concatenated with the 51 engineered frame-level features, combining information from individual players, team structure, and ball movement.

**Temporal Convolutional Network.** The combined representation is processed by five residual TCN blocks with dilations `[1, 2, 4, 8, 16]` and kernel size 3. The model has 64 hidden channels and a receptive field of 125 frames, spanning approximately 12.4 seconds at 10 Hz.

This temporal context helps distinguish sustained possession from short-lived proximity changes, loose-ball situations, and tracking fluctuations. Because inference is performed offline, the TCN uses both preceding and subsequent observations.

The final classification layer produces three logits per frame, corresponding to `H`, `A`, and `D`, which are converted to probabilities using softmax. **Unknown is used only for training-label masking and is not a predicted state.**

### 3.3 Training Strategy

The model is trained with the conservative SkillCorner annotations described in Section 2.2. Only `H`, `A`, and `D` frames contribute to the loss; `U` frames are ignored while remaining available as temporal input context.

We use **possession-duration weighted cross-entropy** to reduce the dominance of long possessions during training. Standard cross-entropy assigns equal importance to every frame, meaning that longer possessions contribute more to the training loss simply because they contain more frames. However, the challenge's primary metric evaluates possession spells regardless of their duration. Possession-duration weighted cross-entropy aligns training with this objective, assigning higher weights to frames belonging to shorter Home and Away possessions and lower weights to those belonging to longer possessions. Dead-ball frames retain a weight of 1, while H/A spells touching official period boundaries also receive a neutral weight because their complete duration may be unknown.

Training uses 512-frame central sequences, with extra temporal context on either side corresponding to the TCN's receptive-field radius. The model sees these surrounding observations, but the loss is evaluated only on the central frames. This avoids learning from predictions whose context is truncated by artificial sequence boundaries; at official half boundaries, missing context is padded.

During development, the 20 SkillCorner matches were divided into **16 training matches and 4 validation matches**, with no overlap between the sets. Checkpoints were assessed using possession-spell F1 **after Viterbi decoding**. The switch penalty of `1.5` was fixed during development using SkillCorner validation and was not tuned on Metrica ground truth. This emphasizes the challenge's primary metric rather than only frame accuracy.

For the final model, the selected configuration was trained on **all 20 matches for 16 epochs**, with the architecture, loss parameters, and decoder settings fixed. No Metrica ground-truth labels are used for training or checkpoint selection.

### 3.4 Temporal Decoding and Possession Rules

Although the TCN uses temporal context, it outputs probabilities independently for each frame position. Taking the highest-probability class at every frame can produce unnecessary short possession changes during challenges, tracking gaps, or ambiguous passages of play.

We therefore apply **Viterbi decoding** to find a coherent sequence of `H`, `A`, and `D` states, balancing the model's frame probabilities against a fixed penalty of `1.5` for switching states. This encourages continuous possessions while permitting changes when the predicted evidence is strong enough.

Our approach does not explicitly detect each football event or apply separate rules for tackles, passes, fouls, or restarts. Instead, possession and dead-ball states are inferred from tracking features and temporal context, followed by sequence decoding.

Each official half is processed separately, avoiding possession sequences crossing halftime. For the final submission, decoded predictions are transferred back to the original 199,319-frame Metrica timeline, and frames outside the scored match periods are assigned `X`.

## 4. Experiments and Evaluation

### 4.1 Conservative vs. Complete Labeling

One of the main challenges in constructing training labels from SkillCorner Open Data was determining the possession state in frames not covered by Phases of Play. Although these annotations describe situations when the ball is in play, their absence does not necessarily indicate a stoppage. Assigning Dead ball (`D`) to every uncovered frame could therefore introduce incorrect supervision.

During exploratory analysis, we observed substantial gaps in phase coverage and disagreements between Phases of Play and tracking-derived possession information. In several matches, Phases of Play covered only around half of the official-period tracking frames. These observations motivated a **conservative H/A/D/U labeling strategy**, where uncertain frames are marked as Unknown (`U`) rather than assigned a potentially incorrect possession state.

As described in Section 2.2, the conservative strategy uses Phases of Play to identify Home and Away possession, while Dynamic Events provide additional evidence to validate possession and identify dead-ball intervals. Frames with contradictory information or insufficient interruption evidence remain Unknown and are excluded from the training loss.

We also investigated a complete H/A/D labeling strategy that assigned Dead ball to all frames not covered by a valid Phase of Play. While this increased supervision coverage, it relied on the assumption that every uncovered interval corresponded to a stoppage.

Both strategies were evaluated using four SkillCorner validation matches. The conservative labeling experiment achieved a decoded spell F1 of 0.7298, compared with 0.6884 for the complete-labeling experiment. However, these values are not directly comparable because their evaluation references differ. The conservative evaluation excluded 56,681 Unknown frames and 286 possession spells adjacent to uncertain intervals, resulting in 733 evaluable ground-truth spells compared with 974 under complete labeling.

We selected conservative labeling to prioritize the reliability of possession annotations, particularly around stoppages and possession boundaries. This is relevant to the challenge's primary metric, where incorrect boundaries can reduce the overlap between predicted and ground-truth possession spells, even when most individual frames are classified correctly.

The main trade-off is reduced supervision coverage. Some frames marked Unknown may correspond to valid possession or dead-ball situations, but without sufficient evidence, we preferred excluding them from the loss rather than introducing potentially incorrect labels.

### 4.2 Standard vs. Possession-Duration Weighted Cross-Entropy

We compared standard cross-entropy (CE) with possession-duration weighted cross-entropy using the conservative H/A/D/U labeling strategy. Both approaches were evaluated on the same four SkillCorner validation matches, with the remaining 16 matches used for training. Predictions were decoded using Viterbi with a fixed transition penalty of 1.5.

| Loss function | Decoded spell F1 | Frame macro-F1 | Predicted possessions |
|---|---:|---:|---:|
| Standard CE | 0.6906 | 0.9105 | 746 |
| Duration-weighted CE | **0.7298** | **0.9214** | 847 |

Duration-weighted CE improved possession-spell F1 by approximately 0.039 while also improving frame macro-F1.

These results supported our decision to use possession-duration weighted cross-entropy in the final training configuration.

### 4.3 Smart Tag Fusion

We also explored whether Metrica's automatic tactical Smart Tags could provide useful auxiliary possession evidence. These annotations describe broad tactical phases such as build-up, progression, attacking transitions, and chance creation. Direct attacking tags can support the named team's possession, while complementary defensive tags can provide evidence for the opposing team. They are therefore informative about the general state of play, but they are not exact frame-level possession annotations.

To account for this distinction, we tested a conservative probability-fusion procedure rather than replacing the model's predictions directly. Smart Tag support was strongest near the centre of each annotated interval and reduced towards its boundaries, where the precise start or end of possession was less certain. The resulting evidence was combined with the tracking model's H/A/D probabilities before applying the same temporal decoder. Overlapping evidence for different teams was retained as competing evidence rather than resolved with a hand-written correction.

The fused sequence remained very similar to the tracking-only sequence. Most Smart Tag evidence agreed with the possession state already inferred from tracking, and the remaining differences were concentrated around ambiguous transitions and interval boundaries. Consequently, the fusion changed only a limited part of the predicted sequence and did not materially alter the overall possession reconstruction.

This comparison indicated that the player and ball tracking, together with the model's temporal context, already captured most of the possession information contributed by the Smart Tags. We therefore retained the simpler tracking-only pipeline: it avoids an additional annotation-alignment step and makes fewer assumptions about the temporal precision of tactical intervals. This decision was based on the limited change to the prediction sequence and the redundancy of the added evidence.

### 4.4 Final Evaluation on Metrica

We evaluated our final model using the official evaluator on the provided five-minute public ground-truth interval, containing 7,500 graded frames and 17 possession spells. This evaluation was used to assess the complete inference pipeline, including the transfer from SkillCorner to Metrica tracking, temporal decoding, and reconstruction of the original 25 Hz timeline.

| Metric | Result |
|---|---:|
| **Possession-spell F1** | **0.6857** |
| Frame macro-F1 | 0.8824 |
| Frame accuracy | 0.8945 |
| Spell F1 at IoU 0.5 | 0.7429 |
| Transition F1 (±25 frames) | 0.5116 |
| Predicted / ground-truth possessions | 18 / 17 |
| Mean possession duration (predicted / GT) | 13.8 s / 14.2 s |

The model achieved a frame-level F1 of **0.9255 for Home possession**, **0.8828 for Away possession**, and **0.8390 for Dead ball**. Dead-ball detection remained the most challenging class, with a precision of 0.9002 and recall of 0.7856, indicating that some stoppage frames were incorrectly classified as active possession.

At an IoU threshold of 0.5, the model matched 13 of the 17 ground-truth possessions. It predicted 18 possessions overall, with a mean duration close to the ground truth. However, the lower transition F1 indicates that accurately identifying possession boundaries remains more difficult than classifying individual frames.

These results provide an initial indication of the model's ability to transfer from SkillCorner to Metrica tracking. Nevertheless, the public ground truth covers only a small portion of the match, and performance over the complete match cannot be reliably established from this evaluation alone.

## 5. Limitations and Future Work

The main limitation of our approach is the **domain shift between SkillCorner and Metrica tracking data**. Although both providers are converted into the same feature representation, their underlying data distributions are not necessarily equivalent. Differences in tracking quality, player availability, speed estimation, and missing observations can affect how well the model generalizes to the target match.

Metrica tracking also contains missing or inaccurate player and ball observations, including occasional duplicated players and unavailable speed measurements. Although our feature representation includes validity indicators and the Deep Sets encoder can handle variable numbers of players, these mechanisms do not completely eliminate the effects of unreliable tracking data.

Another limitation is the **lack of sufficient labeled Metrica data for evaluating generalization**. The complete match ground truth is private, and the public evaluation covers only 7,500 frames and 17 possession spells. Therefore, even if the model performs well on external SkillCorner validation, we cannot reliably establish how well it reconstructs possession across the entire Metrica match.

Finally, our training labels are derived from SkillCorner's Phases of Play and Dynamic Events rather than independently verified frame-level annotations. The conservative labeling strategy reduces uncertainty by excluding unsupported frames, but it also reduces supervision coverage and may omit useful possession situations.

Future improvements could focus on stronger cross-provider robustness, including realistic tracking-missingness augmentation and additional labeled matches from different providers. More accurate possession-boundary modeling and training objectives that directly consider possession spells could also help improve the primary evaluation metric.

## 6. Reproducibility and Submission

This section describes how to reproduce the pipeline, either by training the model from SkillCorner Open Data or by running inference with the saved model. All commands are executed from the repository root.

### 6.1 Requirements and Data

Create the Conda environment and install the required dependencies:

```powershell
conda create --name metrica-hackathon python=3.12.10 pip -y
conda activate metrica-hackathon
python -m pip install -r requirements.txt
```

Download the Metrica competition data after accepting its usage terms:

```powershell
python setup_data.py
```

If the download requires manual authentication, the downloaded archive can be installed using:

```powershell
python setup_data.py --bundle path\to\downloaded_bundle.zip
```

Clone SkillCorner Open Data and check out the revision used for training:

```powershell
git clone https://github.com/SkillCorner/opendata.git data/external/skillcorner-opendata
git -C data/external/skillcorner-opendata checkout --detach 4340d274572876239c154c90bc507a9b3250a656
```

The Metrica data is stored under `data/metrica/`, while SkillCorner data is stored under `data/external/skillcorner-opendata/`. Raw and derived data are excluded from version control.

### 6.2 Training and Inference

Generate the SkillCorner supervision labels and extract the common features for both providers:

```powershell
python -m possession.preprocessing.labels
python -m possession.preprocessing.feature_extraction skillcorner
python -m possession.preprocessing.feature_extraction metrica
```

The generated labels and feature archives are saved under `data/derived/`.

To reproduce the final training procedure, run:

```powershell
python -m possession.modeling.training
```

This uses the fixed configuration in `configs/training.json`, training the model on all 20 SkillCorner matches for 16 epochs. The model checkpoint, normalization statistics, training history, and decoder configuration are written to `outputs/final_model/training/`.

To generate predictions using this newly trained model:

```powershell
python -m possession.inference `
  --model-dir outputs/final_model/training `
  --output-dir outputs/final_model
```

Alternatively, the repository includes the saved final model under `artifacts/final_model/`, allowing the submission to be regenerated without retraining. After preparing the Metrica data, run:

```powershell
python -m possession.preprocessing.feature_extraction metrica
python -m possession.inference
```

By default, inference uses the model and settings specified in `configs/inference.json`. The generated submission is saved as:

```text
artifacts/final_model/GG.csv
```

### 6.3 Final Artifact and Validation

The final submission is a CSV containing the columns `frame,label`, with one prediction for each of the 199,319 original Metrica frames. Predictions within the scored match periods are `H`, `A`, or `D`, while frames outside these periods are assigned `X`.

Validate the submission using the official validator:

```powershell
python evaluation/validate_submission.py artifacts/final_model/GG.csv
```

To evaluate the submission against the available public ground truth:

```powershell
python evaluation/run_evaluation.py --gt public `
  --results artifacts/final_model `
  --out artifacts/final_model/evaluation
```

The validator checks the submission format, frame ordering, number of frames, and allowed labels. The public evaluator produces the available frame-level and possession-spell metrics.
