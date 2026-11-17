# Football Possession Reconstruction from Tracking Data

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

During development, the 20 SkillCorner matches were divided into **16 training matches and 4 validation matches**, with no overlap between the sets. Checkpoints were assessed using possession-spell F1 **after Viterbi decoding**, with a fixed switch penalty of `1.5` selected in earlier external experiments. This emphasizes the challenge's primary metric rather than only frame accuracy.

For the final model, the selected configuration was trained on **all 20 matches for 16 epochs**, with the architecture, loss parameters, and decoder settings fixed. No Metrica ground-truth labels are used for training or checkpoint selection.

### 3.4 Temporal Decoding and Possession Rules

Although the TCN uses temporal context, it outputs probabilities independently for each frame position. Taking the highest-probability class at every frame can produce unnecessary short possession changes during challenges, tracking gaps, or ambiguous passages of play.

We therefore apply **Viterbi decoding** to find a coherent sequence of `H`, `A`, and `D` states, balancing the model's frame probabilities against a fixed penalty of `1.5` for switching states. This encourages continuous possessions while permitting changes when the predicted evidence is strong enough.

Our approach does not explicitly detect each football event or apply separate rules for tackles, passes, fouls, or restarts. Instead, possession and dead-ball states are inferred from tracking features and temporal context, followed by sequence decoding.

Each official half is processed separately, avoiding possession sequences crossing halftime. For the final submission, decoded predictions are transferred back to the original 199,319-frame Metrica timeline, and frames outside the scored match periods are assigned `X`.

## 4. Experiments and Evaluation

### 4.1 Conservative vs. Complete Labeling

One of the main challenges in constructing training labels from SkillCorner Open Data was determining the possession state in frames not covered by Phases of Play. Although these annotations describe situations when the ball is in play, their absence does not necessarily indicate a stoppage. Assigning Dead ball (`D`) to every uncovered frame could therefore introduce incorrect supervision.

During exploratory analysis, we observed substantial gaps in phase coverage and disagreements between Phases of Play and tracking-derived possession information. For example, in one match, Phases of Play covered only 54.4% of the tracking frames. These observations motivated a **conservative H/A/D/U labeling strategy**, where uncertain frames are marked as Unknown (`U`) rather than assigned a potentially incorrect possession state.

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

We also explored incorporating Metrica's automatic tactical Smart Tags into the possession predictions. These annotations identify tactical phases such as build-up, progression, attacking transitions, and chance creation, which can provide additional evidence about the team in possession.

During exploratory analysis, we observed that the Smart Tags were generally more reliable in the middle of annotated intervals, while their start and end boundaries were less precise. We therefore experimented with combining the model's class probabilities with confidence-weighted Smart Tag evidence, increasing the probability assigned to the team associated with the active tactical phase.

However, the additional information did not provide sufficiently consistent evidence of improved possession reconstruction, particularly around transitions and possession boundaries. Since the model already incorporated spatial and temporal information from tracking, we retained the simpler tracking-only approach without Smart Tag fusion.