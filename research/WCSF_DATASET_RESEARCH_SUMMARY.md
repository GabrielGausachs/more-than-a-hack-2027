# WCSF / Groningen Tracking Dataset — Research Reference

**Source:** [WCSF Solution Room, University of Groningen / SciSports](https://github.com/MatKempeGroningen/WCSF_SolutionRoom)  
**Prepared:** 30 October 2026.  
**Purpose:** Preserve data understanding, labels, alignment, preprocessing requirements and modeling caveats for a clean repository clone.  
**Evidence:** Source README plus local Codex EDA of both tracking halves. No model has been trained as part of that analysis.

## Executive summary

WCSF is an **anonymized single professional football match** recorded with optical tracking originally at 25 Hz, processed to **10 Hz**. It supplies per-player positional tracking, reported/smoothed speed, acceleration, and explicit `AliveFlag` (0 dead, 1 live), `PossessionFlag` (team identifier), and `team_game_phase`. Its key benefit over Metrica Sample Game 3 is nearly complete framewise **external supervision** for live/dead state. It does **not** establish identical timing conventions to the Metrica hackathon; its labels have a small number of row-level disagreements and its tracking is much cleaner than competition tracking.

## A. Files, timebase and entities

Essential inputs: `positions_FIRST_HALF.csv.zip`, `positions_SECOND_HALF.csv.zip`, `README.md` (the repository README references a generic `positions.csv`; actual distribution is two ZIP files). Optional: `centroid_positions.csv`, two physical KPI tables and research presentation PDF. All labels needed for first experiments are already in the position files.

| Property | First half | Second half |
|---|---:|---:|
| Player rows | 616,308 | 645,923 |
| Unique 10 Hz timestamps | 28,014 | 29,360 |
| Timestamp range (ms) | 0–2,801,300 | 3,735,500–6,671,400 |
| Representation duration | 2,801.4 s | 2,936.0 s |
| Player rows per frame | always 22 | 22 in 29,357 frames; 23 in 3 substitution-boundary frames |
| Unique home/away players | 11 / 11 | 13 / 14 |

**Total:** 57,374 frames / approximately 95.62 minutes sampled football. Timestamps progress at exactly **100 ms** within each half. The ~934.2 s halftime gap is **not part of sampled data**; do not insert fabricated halftime DEAD frames or let sequence windows cross it. Home `group_id=1`, away `group_id=2`. `player_id` is anonymized but usable for within-source EDA; it should **not** be used as a model input or as correspondence to competition slots.

Fields: `x`, `y` metres; `speed` m/s; `acceleration` m/s²; `group_id`; `timestamp_ms`; `AliveFlag`; `PossessionFlag`; `x_ball`, `y_ball`; `team_game_phase`; `player_id`; plus an undocumented row index. Data is one row **per player per timestamp**, not one row per frame.

## B. Label audit: trustworthy enough for experiments, not infallible

| Half | ALIVE | DEAD | Conflicted `AliveFlag` | Home possession on all timestamps | Away possession on all timestamps | Conflicted `PossessionFlag` |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 15,821 | 12,193 | 0 | 10,002 | 18,012 | 0 |
| H2 | 14,738 | 14,583 | 39 | 8,740 | 20,530 | 90 |
| Total | **30,559** | **26,776** | **39** | **18,742** | **38,542** | **90** |

**Do not majority-vote disagreements.** For binary ALIVE/DEAD, store `alive_label=1/0`, with `-1` at the 39 conflicted timestamps, and `label_valid` boolean flag for whether a unanimous label exists. During training, masked loss excludes invalid targets (replace `-1` before computing BCE or index only valid frames); during evaluation exclude invalid frames and break segments rather than bridging by assumption. `label_valid` is **not** a predictor input and is distinct from player position/speed validity masks.

For a provisional combined three-state target: if `AliveFlag=0`, state `D` irrespective of the persisted possession; if `AliveFlag=1` and unanimous `PossessionFlag=1/2`, state `H/A`; otherwise `UNKNOWN`. Result: **18,249 H, 12,239 A, 26,776 D and 110 UNKNOWN frames**. Provider possession fields persist through dead-ball periods, and their meaning during dead time is not documented well enough to treat as current controlling team.

**Semantic caveat:** WCSF README: `AliveFlag=1` means ball in play and `0` means dead. This is conceptually aligned with Metrica's live/dead distinction, but **there is no evidence that the two providers place precise stop/restart boundaries on identical frames or follow identical annotation timing procedures**. Provider labels ≠ independently verified referee-event ground truth.

## C. State duration and movement diagnostics

ALIVE intervals: 104 segments, median **19.85 s**, mean **29.38 s**, P95 **85.33 s**, maximum **131.90 s**. DEAD: 104 segments, median **24.15 s**, mean **25.75 s**, P95 **54.71 s**, maximum **164.30 s**. Conflicted timestamps break episodes and are not bridged. Some very short segments persist, including two DEAD episodes shorter than 0.5 s: not proof of annotation errors.

Possession on ALIVE frames only: Home **18,249 (59.86%)**, Away **12,239 (40.14%)**, 71 live frames with invalid/conflicting possession. H/A spells: Home 207 (median 4.2 s), Away 204 (median 3.3 s), including short one-frame assignments. Direct H→A changes = 163, A→H = 142; cross-dead transitions H→D→A = 30, A→D→H = 52. These are provider-label descriptors, not proof of transfer consistency.

**Strong but imperfect movement signal:** Per-frame mean across all tracked player speeds, median across ALIVE frames ≈**2.29 m/s**, DEAD ≈**1.12 m/s**. However, **351** ALIVE frames have combined mean speed below 1 m/s and **2,072** DEAD frames exceed 2 m/s. A speed threshold alone will fail on important edge cases. Away-team fraction of players running >3 m/s: mean ≈31.6% live vs 5.2% dead; away-team mean speed ≈2.43 vs 1.15 m/s. Descriptive 0.5/1/2/5-second centered activity windows reduce distribution overlap only modestly, supporting temporal models rather than replacing them with hard thresholds.

## D. Player, ball and coordinate interpretation

**Complete tracking bias:** Ball observed on **every one of 57,374 frames** (including all 26,776 DEAD frames); there are **no missing-ball ALIVE examples**, unlike the challenge. Players are present at every frame (normally 11+11). The second half contains three 23-row substitution-boundary frames. Reported player speed and acceleration are finite throughout. This makes WCSF comparatively easy and creates severe domain shift for a challenge model with missing ball, missing/duplicated player tracks and unstable slots.

**Mirrored coordinate system:** Each team's own goal is at x=−52.5 m and opponent goal x=+52.5 m, with center-pitch origin, pitch **105 × 68 m**. WCSF group-2 Away observations are in the opposite frame to Home. The EDA verified that leaving Home unchanged and rotating Away **both axes** `(x,y)→(-x,-y)` aligns the two provider ball copies *exactly on all 57,374 frames*. This is a shared **home-centric** pitch, not necessarily absolute stadium/camera orientation. **Do not flip X only**, and do not treat separately team-relative x/y as globally aligned.

To normalize shared WCSF metres to conventional pitch coordinates: `x_norm=x_m/105+0.5` and `y_norm=y_m/68+0.5`, after applying Away rotation; retain genuine positions outside [0,1] if observed. Then WCSF Home always attacks +X. For the Metrica challenge, **home-direction normalization remains a proposed preprocessing operation that must be verified separately by half**: rotate **the full pitch including both teams** using `(x,y)→(1-x,1-y)` when Home actually attacks left. Do not use possession ground truth to decide orientations.

Player reported `speed` is extraordinarily well explained by a **trailing one-second average of position-derived speed**: Pearson correlation ~1.000000, RMSE ~2×10⁻⁸ m/s across ~1.26M measurements. The detailed acceleration convention is not established. Competition reported speed is also m/s, but its smoothing/provenance differs. **For initial shared features, use each provider's speed values as supplied. Never derive competition velocities from changes in unstable slot coordinates, and never smooth per-slot competition speeds across frames.** Optional smoothing of label-independent **team aggregate** speed statistics can be assessed separately and must preserve masks.

## E. Target leakage

`team_game_phase` is **not a safe input** for predicting ALIVE or possession. README values: 2 attacking, 3 defending, 4 defensive transition, 5 offensive transition, 6 ball dead. In the EDA, **all 26,776 unanimously DEAD frames** have both teams in phase 6; no unanimously ALIVE frame has either team phase 6. Most possession changes coincide with phase changes. Including it would let the model predict supplied state labels from another state label, not from tracking physics. Exclude `AliveFlag`, `PossessionFlag`, `team_game_phase`, row index and any label-conditioned aggregate KPI from predictors.

## F. Intended model-ready feature schema and processing

**Core 10 Hz input** (same structure to be built for Metrica after deterministic time-based 25→10 Hz downsampling):

| Array | Shape | Meaning |
|---|---|---|
| `players` | `(N,2,24,3)` | normalized x, y, **provided** speed; player slots are unordered storage positions |
| `player_location_valid` | `(N,2,24)` | observed coordinate validity |
| `player_speed_valid` | `(N,2,24)` | observed speed validity (independent from position) |
| `team_features` | `(N,2,6)` | mean speed, median speed, speed std, tracked player count, valid-speed count, speed coverage |
| `team_feature_valid` | `(N,2,6)` | validity of aggregate measures |
| `alive_label` | `(N,)` WCSF only | 0 DEAD, 1 ALIVE, −1 conflict |
| `label_valid` | `(N,)` WCSF only | usable supervised target mask; **never model input** |
| `half`, timestamps, alignment | metadata | avoid timebase errors, padding confusion and half-spanning windows |

Pad WCSF from 11 players per side to 24 slots, mark padding invalid, and never treat its stable anonymized IDs as model input identities. Recompute six team statistics independently at each frame from valid player speeds; invalid numeric entries can be filled with zero **only with explicit masks**. Avoid optional run-speed-threshold fractions in the core schema (deliberately excluded by project decision).

For competition 25 Hz→10 Hz: define target timestamps in seconds, deterministically sample nearest *same-frame observation* without looking far away for valid tracks, do not bridge missing spans or halftime, and retain a map from model timestamps back to 25 Hz tactical frames. Restore eventual probability predictions to 25 Hz at inference rather than interpolating missing raw player observations.

### Cross-validation design and leakage control

- **Fold 1:** first half train+validation; entire second half held-out test.
- **Fold 2:** second half train+validation; entire first half held-out test.
- Within training half, select a **contiguous ~15% validation interval**, with an initial **10-second buffer on each side**, or longer if the temporal architecture's receptive field requires it. No randomized per-frame train/validation split or overlap of temporal model windows across partitions.
- Fit feature scalers only on the training indices of the relevant fold; preserve masks and avoid independent full-match normalization of Metrica.
- Use both full halves for the final external training **only after** hyperparameters/model selection are fixed based on allowed validation. These two halves are one match, so half-held-out performance measures **within-match temporal transfer**, not independent provider or match generalization.

### Experimental order (not yet implemented)

1. Complete and validate shared WCSF/Metrica feature extraction and half-held-out split manifest.
2. Simple framewise linear or gradient-boosted baseline using team-activity features.
3. Temporal baseline (windowed trees or a small bidirectional TCN); eventually a mask-aware **Deep Sets player encoder + TCN** combining unordered player x/y/speed and team activity. Offline inference can use future context.
4. Compare clean vs realistic player-dropout/speed perturbation, derived from label-independent competition coverage; track behavior around restarts, slow live play and moving dead periods.
5. Output `P(ALIVE)` rather than prematurely hard-labeling the challenge; later combine it with H/A ownership evidence and temporal spell decoding. Metrica Game 3 can add lower-confidence weak labels only as a separate experiment.

Evaluation: accuracy/macro-F1/per-class precision-recall/F1, Brier score and calibration, ALIVE↔DEAD transition errors at 0.5/1/2 s tolerances, spurious transition counts and segment tIoU/F1. Later apply official challenge `spellF1` only to complete H/A/D outputs. **Never use challenge public GT for training, architecture selection, threshold tuning or manual correction.**

## G. Open limitations / decisions to revisit

1. WCSF and Metrica both say ball *in play* vs *dead*, but exact transition timestamp semantics are not shown to match.
2. WCSF contains **one match** only and virtually perfect tracking: performance can be misleading for challenge transfer, especially live play with missing ball.
3. `PossessionFlag` persists across dead periods and may mean last possession or expected restart there; ignore it for labels when dead.
4. The 39 binary and 90 possession conflict timestamps should remain UNKNOWN; consider transition-neighborhood exclusion as a sensitivity analysis, not an automatic repair.
5. For permutation-invariant input, **do not save player IDs as model channels**. Internal WCSF IDs may still be used for source-data audits and reproducible row grouping.
6. The reported-speed smoothing is source-dependent; the physical units match but measured features may differ by provider. Validate distribution shift before interpreting transfer failures.
7. Some chart/image paths below refer to the old local analysis folder; the original image assets are not embedded in this Markdown.

---

# Archived full WCSF EDA report

Below is the complete original report. It preserves source-documented facts, measured statistics, model feasibility discussion, supporting-file names and study limitations; relative links to figures and scripts refer to the previous workspace.



---

## Archived source: `WCSF_DATA_AND_SUPERVISION_REPORT.md`

# WCSF Data and Supervision Report

**Analysis boundary:** local processing of the two WCSF position CSVs and repository documentation. No model training, frame-level target export, interpolation, label correction, competition ground truth, or source-file modification was performed.

**Evidence terminology:** **Verified source fact** is stated by the WCSF README; **calculated** is measured from the local CSVs; **hypothesis** is an interpretation that cannot be semantically verified without event/video ground truth.

## 1. Executive summary and key conclusions

WCSF is a valuable explicit-supervision source, but it is not a turnkey proof of transfer reliability. Calculated frame labels are complete and unanimous throughout the first half. The second half contains **39 conflicting `AliveFlag` timestamps** and **90 conflicting `PossessionFlag` timestamps**. Primary statistics treat these timestamps as unknown rather than majority-voting them.

The strongest result is that WCSF can support a supervised **tracking-only ALIVE/DEAD experiment**, after excluding label-conflict frames and all label-derived inputs. It cannot by itself establish generalization to the challenge match: this is one unusually complete 10 Hz match, the ball is observed on **57,374 of 57,374 frames**, and it contains **0 ALIVE-with-ball-missing frames**. The competition data has substantially more missing players and ball observations.

Coordinate mirroring is resolvable for within-match spatial analysis. After leaving group 1 unchanged and rotating group 2 by `(x, y) -> (-x, -y)`, the two supplied ball copies agree exactly on **57,374 frames** with maximum disagreement **0.000000 m**. This reconstructs a shared home-centric tactical pitch, not absolute stadium/camera orientation.

The most defensible first modeling experiment is a two-stage design: a bidirectional TCN for ALIVE/DEAD using only cross-domain tracking activity, coverage/masks and optional geometry, followed later by a separate H/A control model on ALIVE frames. Validate by entire half in both directions and require an independently annotated challenge-window test before using predictions as pseudo-labels.

## 2. Dataset structure and temporal alignment

The local README documents original 25 Hz optical tracking, downsampling and smoothing to 10 Hz, one row per player per timestamp, a 105 × 68 m pitch, and team-relative mirroring. The observed schema contains the twelve documented fields plus an undocumented leading index column. The source repository lists the same split-half archives: <https://github.com/MatKempeGroningen/WCSF_SolutionRoom>.

| Half | Rows | Frames | Timestamp range (ms) | Represented duration (s) | Timestamp increments | Rows/frame | Unique H/A players |
|---|---|---|---|---|---|---|---|
| 1 | 616,308 | 28,014 | 0–2,801,300 | 2,801.4 | 100 ms: 28,013 | {22: 28014} | 11/11 |
| 2 | 645,923 | 29,360 | 3,735,500–6,671,400 | 2,936.0 | 100 ms: 29,359 | {22: 29357, 23: 3} | 13/14 |

Both halves are internally contiguous at exactly 10 Hz. Timestamps are continuous since kickoff rather than restarting at zero: the second half starts at 3,735.5 s, leaving a 934.2 s halftime gap after the last first-half frame. Half 1 has 22 rows at every timestamp. Half 2 has three 23-row substitution-boundary frames and otherwise 22.

The first half uses 11 identities per team throughout. The second half contains 13 home and 14 away identities because of substitutions; identity spans and exact boundary frames are recorded in the JSON summary and anomaly CSV.

## 3. Label integrity and frame-level distributions

| Half | Frames | ALIVE | DEAD | Alive conflicts | Possession H | Possession A | Possession conflicts |
|---|---|---|---|---|---|---|---|
| 1 | 28,014 | 15,821 (56.48%) | 12,193 (43.52%) | 0 | 10,002 | 18,012 | 0 |
| 2 | 29,360 | 14,738 (50.20%) | 14,583 (49.67%) | 39 | 8,740 | 20,530 | 90 |
| all | 57,374 | 30,559 (53.26%) | 26,776 (46.67%) | 39 | 18,742 | 38,542 | 90 |

| Half | First timestamp | First AliveFlag | First possession | Last timestamp | Last AliveFlag | Last possession |
|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 2 | 2801300 | 0 | 2 |
| 2 | 3735500 | 1 | 1 | 6671400 | 0 | 2 |

Both halves begin on an ALIVE frame. Their final-frame states and persisted possession values are shown above; the halftime gap itself is not represented by tracking rows and is not counted as a DEAD interval.

No missing or out-of-domain numeric label values were observed. All first-half rows agree within a timestamp. The second-half conflicts are row-level disagreements: most affected timestamps contain a dominant value plus one or a few dissenting player rows. Dissenting-player frequencies, neighboring frame labels and identity-boundary proximity are preserved in `wcsf_frame_anomalies.csv` rather than silently repaired.

Calculated conflict context: 39 Alive conflicts and 89 possession conflicts lie between different unanimous neighbor labels; 1 Alive-conflict and 1 possession-conflict anomalies fall within 0.1 s of an identity-span boundary. This pattern is consistent with a row-level synchronization/join artifact, but that interpretation remains a hypothesis.

![ALIVE/DEAD timeline](figures/alive_dead_timeline.svg)

## 4. ALIVE/DEAD interval statistics

| State | Segments | Mean | Min | P5 | P25 | Median | P75 | P95 | Max | <0.5/<1/<2/<5 s | >10/>20/>30/>60 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ALIVE | 104 | 29.38 | 1.80 | 2.53 | 8.10 | 19.85 | 40.30 | 85.33 | 131.90 | 0/0/1/18 | 72/52/37/15 |
| DEAD | 104 | 25.75 | 0.10 | 5.33 | 16.25 | 24.15 | 29.62 | 54.71 | 164.30 | 2/2/3/5 | 89/67/24/3 |

Transition counts are **81 ALIVE→DEAD** and **86 DEAD→ALIVE** across adjacent unanimous frames. Unknown conflict frames break intervals and are not bridged. Segment duration uses exact frame count × 0.1 s.

![Segment-duration distributions](figures/state_segment_durations.svg)

## 5. PossessionFlag analysis

On unanimous ALIVE frames, Home controls **18,249 frames (59.86%)** and Away controls **12,239 frames (40.14%)**. There are **71 ALIVE frames** with conflicting or invalid possession and therefore no primary H/A assignment.

Possession is populated during DEAD frames and usually persists through stoppages. The data therefore supports only the documented statement that it is the team in possession on that timeframe. The CSVs alone cannot distinguish current control, last possession, expected restart, or a broader attacking assignment during DEAD periods. For modeling H/A control, possession supervision should be used only on ALIVE frames unless external semantics are obtained.

| Half | Team state | Spells | Mean (s) | Median (s) | Min (s) | Max (s) | Spells <1 s |
|---|---|---|---|---|---|---|---|
| 1 | H | 91 | 10.64 | 5.20 | 0.50 | 57.40 | 13 |
| 1 | A | 91 | 6.75 | 3.40 | 0.40 | 46.90 | 2 |
| 2 | H | 116 | 7.39 | 4.05 | 0.10 | 43.20 | 13 |
| 2 | A | 113 | 5.40 | 3.30 | 0.10 | 37.50 | 17 |
| all | H | 207 | 8.82 | 4.20 | 0.10 | 57.40 | 26 |
| all | A | 204 | 6.00 | 3.30 | 0.10 | 46.90 | 19 |

There are **163 direct H→A**, **142 direct A→H**, **30 H→D→A**, and **52 A→D→H** patterns. Four live-possession spells last one frame. Very short spells are retained, not smoothed.

## 6. Candidate combined H/A/D state

| Half | H | A | D | UNKNOWN |
|---|---|---|---|---|
| 1 | 9,679 | 6,142 | 12,193 | 0 |
| 2 | 8,570 | 6,097 | 14,583 | 110 |
| all | 18,249 | 12,239 | 26,776 | 110 |

The mapping is internally usable if conflicts remain unknown. DEAD takes precedence over the persisted possession value. ALIVE requires a unanimous possession value of 1 or 2. No target file was created.

![Candidate state timeline](figures/possession_state_timeline.svg)

## 7. Team game phase and target leakage

Phase 6 is almost an explicit DEAD encoding: there are **26,776 DEAD frames**, **26,792 frames where both teams are phase 6**, **0 DEAD frames without both teams in phase 6**, and **0 unanimous ALIVE frames with either team in phase 6**.

Attacking/defending and offensive/defensive transition phases are overwhelmingly complementary. Of **326 direct unanimous possession changes**, **314 (96.32%)** coincide with at least one team phase change on the same frame. These relationships show that `team_game_phase` encodes target information and must not be used as an ordinary input feature.

| Home \ Away | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| 2 | 0 | 13,327 | 0 | 0 | 0 |
| 3 | 7,174 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 5,115 | 0 |
| 5 | 0 | 0 | 4,966 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 | 26,792 |

![Team game phase cross-tabulation](figures/team_game_phase_crosstabs.svg)

## 8. Player and ball tracking quality

| State | Ball observed | Ball missing |
|---|---|---|
| ALIVE | 30,559 | 0 |
| DEAD | 26,776 | 0 |
| CONFLICT OR INVALID | 39 | 0 |

| Half | Team | X range (m) | Y range (m) | Speed median | Speed max | Acceleration min | Acceleration max | Outside field | Outside README extended bounds |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Home | -54.88 to 54.30 | -35.60 to 35.47 | 1.47 | 8.81 | -6.19 | 4.37 | 2,993 | 325 |
| 1 | Away | -54.24 to 54.77 | -35.45 to 36.80 | 1.44 | 9.27 | -5.46 | 4.65 | 2,724 | 966 |
| 2 | Home | -54.02 to 55.00 | -35.55 to 36.58 | 1.41 | 9.15 | -6.93 | 4.36 | 4,832 | 272 |
| 2 | Away | -54.58 to 54.28 | -35.18 to 35.38 | 1.41 | 9.18 | -6.08 | 4.67 | 1,472 | 116 |

Player locations and reported speed/acceleration are finite throughout. No negative speeds, speeds above 12.5 m/s, or absolute accelerations above 15 m/s² occur. The actual player and ball coordinates sometimes exceed the README’s nominal extended ranges of X ±55 m and Y ±35 m; those rows are retained as observed out-of-bounds tracking rather than clipped.

All finite-value, range, speed and acceleration diagnostics are retained in the JSON summary. WCSF provides complete or nearly complete player and ball observations, so it does not contain the ALIVE-with-ball-missing examples needed to prevent a transferred model from using visibility as a shortcut. Conversely, it provides **26,776 DEAD-with-ball-visible frames**, which is useful evidence against the rule ‘visible ball means live’.

Player speeds above 12.5 m/s and absolute accelerations above 15 m/s² are diagnostic flags, not automatic deletions. Ball speeds above 50 m/s are also reported without correction. Position counts outside the on-field and documented extended bounds are separated.

![Ball visibility](figures/ball_visibility_by_state.svg)

## 9. Coordinate mirroring and physical interpretation

The numerical evidence supports a 180° rotation for group 2: both X and Y change sign. The reversible shared-frame mapping is `home: (x, y)` and `away: (-x, -y)`. The two team-specific ball copies become identical after this transform on every finite frame.

Within-team player-to-ball distances are already correct because each row’s player and ball share the same team-relative frame. Cross-team centroids, pairwise distances, graph edges and raw-coordinate neural networks require de-mirroring. Flipping X alone would leave Y inconsistent and corrupt inter-team geometry. Across halves, the home-centric tactical frame remains usable, but absolute stadium orientation cannot be reconstructed from these anonymous files alone.

## 10. Movement patterns by football state

Combined frame mean speed has ALIVE median **2.29 m/s** and DEAD median **1.12 m/s**. The distributions overlap materially: **351 ALIVE frames** have combined mean speed below 1.0 m/s, while **2,072 DEAD frames** exceed 2.0 m/s. Movement is informative but not a deterministic label.

| Half | Team | ALIVE median mean-speed | DEAD median mean-speed | ALIVE P5 | DEAD P95 |
|---|---|---|---|---|---|
| 1 | Home | 2.24 | 1.16 | 1.26 | 2.51 |
| 1 | Away | 2.22 | 1.12 | 1.19 | 2.26 |
| 2 | Home | 2.33 | 1.10 | 1.24 | 2.40 |
| 2 | Away | 2.36 | 1.04 | 1.19 | 2.25 |
| all | Home | 2.29 | 1.13 | 1.26 | 2.44 |
| all | Away | 2.29 | 1.08 | 1.19 | 2.25 |

| Centered window (s) | ALIVE median (m/s) | DEAD median (m/s) | Histogram overlap |
|---|---|---|---|
| 0.5 | 2.28 | 1.12 | 0.332 |
| 1.0 | 2.28 | 1.12 | 0.329 |
| 2.0 | 2.28 | 1.12 | 0.322 |
| 5.0 | 2.28 | 1.14 | 0.308 |

Centered temporal aggregation changes separation but does not eliminate overlap. This supports a temporal model and explicit uncertainty rather than a framewise speed threshold.

![Speed distributions](figures/speed_distributions_by_state.svg)

![Transition activity profiles](figures/transition_activity_profiles.svg)

![Selected intervals](figures/selected_interval_examples.svg)

## 11. Comparison with competition features

| Stage | Feature | Reproducibility | Required handling |
|---|---|---|---|
| 1 | player x/y | Reproducible after transformation | Rotate group 2 by 180 degrees for a shared home-centric frame; preserve validity masks. |
| 1 | player speed | Not measurement-equivalent | Available and complete, but WCSF documents a 10-frame smoothed provider feature; competition speed has different provenance/coverage. |
| 1 | player/ball validity masks | Directly reproducible | Finite-value masks are available, but WCSF has little or no missingness diversity. |
| 1 | tracked-player and speed counts | Directly reproducible | Nearly constant 11 per team; weak preparation for competition fragmentation. |
| 1 | ball x/y | Reproducible after transformation | Select group-1 ball or rotate group-2 ball; retain ball mask. |
| 2 | nearest_home_dist | Reproducible after transformation | Metres already; use ball and players in the same coordinate system. |
| 2 | nearest_away_dist | Reproducible after transformation | Metres already; within-team mirrored distances are invariant. |
| 2 | nearest_dist_diff | Reproducible after transformation | Requires both team distances and explicit validity. |
| 2 | home_players_within_5m | Reproducible after transformation | Direct metric-distance count. |
| 2 | away_players_within_5m | Reproducible after transformation | Direct metric-distance count. |
| 2 | home mean/median/std speed | Not measurement-equivalent | Formula is reproducible; source speed smoothing and coverage differ. |
| 2 | away mean/median/std speed | Not measurement-equivalent | Formula is reproducible; source speed smoothing and coverage differ. |
| 2 | home tracked/valid-speed counts and coverage | Directly reproducible | Values are much less variable than competition features. |
| 2 | away tracked/valid-speed counts and coverage | Directly reproducible | Values are much less variable than competition features. |
| 3 | four tactical categories and Gaussian weights | Unavailable | WCSF team_game_phase is label-derived and must not substitute for Smart Tags. |
| 3 | eleven context categories and timing | Unavailable | No measurement-equivalent WCSF Smart Tag source exists. |

Geometry is the most transferable feature family after coordinate reconstruction. Activity features are formula-compatible but not measurement-equivalent because WCSF speed is documented as a one-second moving-window mean. Counts and coverage have severe distribution shift because WCSF is almost always 11-versus-11. Smart Tags are unavailable; `team_game_phase` cannot substitute because it leaks the targets.

## 12. Domain shift and preprocessing risks

WCSF is 10 Hz, smoothed and almost complete. The challenge tracking is 25 Hz with fragmented players, unstable slots and long ball gaps. Temporal windows must be defined in seconds, and transfer experiments should either resample challenge features to 10 Hz or use a time-aware encoder. Synthetic player dropout, mask augmentation, speed scale/noise augmentation and ball masking are necessary stress tests.

| Candidate speed alignment | Compared points | Correlation | RMSE (m/s) |
|---|---|---|---|
| trailing | 1,261,741 | 1.0000 | 0.0000 |
| centered | 1,261,741 | 0.9627 | 0.3814 |
| forward | 1,261,741 | 0.8983 | 0.6300 |

The candidate reconstruction compares reported speed with trailing, centered and forward ten-sample averages of coordinate-derived speed. The trailing window matches essentially exactly: correlation **1.000000** and RMSE **0.00000002 m/s** over 1,261,741 observations. This calculated result supports a causal ten-sample speed window using current and preceding displacement samples, with no evidence that reported speed itself uses future frames. Acceleration is best approximated by a backward speed difference (correlation **0.6802**, RMSE **0.6151 m/s²**) but does not match exactly, so its precise derivation remains unresolved. Bidirectional offline inference may use future observations, but folds must still be separated before any data-dependent preprocessing or normalization.

Safe inputs are raw/finite player and ball coordinates, reported speed/acceleration with provenance warnings, masks, counts and label-independent spatial/activity summaries. Unsafe inputs are `AliveFlag`, `PossessionFlag`, `team_game_phase`, any statistics conditioned on them, and the undocumented source row index.

## 13. Supervised and weakly supervised feasibility

| Approach | Assessment | Requirement |
|---|---|---|
| Supervised ALIVE/DEAD | Feasible as an external-data experiment | Exclude conflict frames and leaky phase/possession fields; use tracking-only inputs and independent challenge validation. |
| Supervised H/A possession | Feasible but semantically less secure | Train only on unanimous ALIVE frames; do not interpret DEAD persistence as current control without external documentation. |
| Single H/A/D model | Possible but not preferred first | Risks coupling an easier DEAD task to a harder, semantically uncertain possession task. |
| Two-stage model | Recommended | ALIVE/DEAD first; H/A only on ALIVE frames, each with separate calibration and error analysis. |
| WCSF + Game 3 | Complementary | WCSF labels are explicit but imperfect; Game 3 anchors remain lower-confidence weak labels with separate provenance and weights. |
| Pseudo-labeling | Later-stage only | Calibrate on independently labeled challenge windows before accepting high-confidence teacher outputs. |
| Self-supervised pretraining | Potentially useful but secondary | Use trajectories from all domains if the supervised baseline saturates or raw-set encoders need more representation learning. |

## 14. Recommended validation strategy

Use train-half-1/test-half-2 and the reverse direction as the primary internal checks. Tune only with contiguous time blocks inside the training half and remove a buffer at least as long as the model receptive field from both sides of every split. Never randomly split frames or allow overlapping windows across folds.

Report accuracy, macro-F1, binary ALIVE and DEAD F1, per-state F1, transition timing error at 0.5/1/2 s tolerances, temporal IoU, complete-segment F1 and possession-spell F1. Break results out by half, transition distance, movement tails, player coverage and ball visibility. WCSF cannot measure missing-ball transfer performance because that stratum is absent or negligible.

Meaningful independent validation is not possible from WCSF alone. A credible generalization claim needs another explicitly labeled match or manually verified challenge intervals covering slow live play, fast dead periods, ball-missing live play, ambiguous restarts and tracking degradation.

## 15. Modeling recommendations

1. Start with a tracking-only bidirectional TCN for ALIVE/DEAD using team speed summaries, count/coverage masks and optional de-mirrored geometry. Do not include phase or possession fields.
2. Define all windows in seconds and validate both half directions before any challenge pseudo-labeling.
3. Simulate competition conditions during training through player dropout, speed perturbation and ball masking; report performance separately with and without geometry.
4. Obtain a small independently annotated challenge validation set before treating teacher confidence as calibrated.
5. Add the H/A model only after the binary state model is credible; keep Game 3 weak anchors source-aware and lower weighted.

## 16. Open questions, assumptions and limitations

- The source does not document the exact semantic treatment of `PossessionFlag` during DEAD periods.
- The exact speed smoothing alignment and acceleration derivative convention are not documented.
- Row-level second-half conflicts cannot be semantically adjudicated without the source processing pipeline or video/events.
- The de-mirroring transform recovers one shared tactical pitch but not absolute stadium coordinates.
- One anonymous match cannot estimate competition-match generalization or between-match variance.
- Movement/phase agreement near boundaries establishes internal consistency, not verified event timing accuracy.

## 17. Supporting files and reproducibility

- `wcsf_analysis_summary.json`: machine-readable aggregate results, assumptions and input hashes.
- `wcsf_frame_anomalies.csv`: label conflicts, nonstandard player counts and other frame-level integrity anomalies only.
- `wcsf_state_segments.csv`: ALIVE/DEAD/UNKNOWN and H/A/D/UNKNOWN interval records; not a frame-level target file.
- `figures/`: the eight focused SVG figures linked above.

Run from the repository root:

```powershell
python data/open_data_groningen/analysis/analyze_wcsf_supervision.py
```

The script verifies input hashes after all outputs are written. It imports no competition feature code and writes nowhere outside this analysis directory.

