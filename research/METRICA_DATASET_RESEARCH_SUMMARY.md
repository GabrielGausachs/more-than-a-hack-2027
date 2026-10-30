# Metrica Tracking and Smart Tagging — Research Reference

**Project:** Metrica Sports, *More Than a Hack 2027*, Phase 1: possession reconstruction.  
**Prepared:** 30 October 2026.  
**Scope:** FC Barcelona–Athletic Club competition match, including tactical tracking, Smart Tagging, homographies, data quality and feature engineering.  
**Purpose:** Standalone exploration reference when rebuilding the coding repository. This is **not a feature archive, trained model, or verified possession ground truth**.

## Executive summary

The competition dataset is the FC Barcelona (home) versus Athletic Club (away) match: 25 Hz tactical ATD tracking, automatic Smart Tagging and homographies, with no possession labels permitted for model training. Tracking slots are unstable tracklets rather than stable player identities. The ball is missing in approximately 41% of scored frames, while player movement and reported player speeds remain broadly available. Home/away control may be supported by player–ball proximity where observed and team-specific tactical Smart Tags; live/dead status may be supported by collective activity and set-piece contexts. Neither source is verified possession ground truth.

This document retains the detailed source audits, measured statistics, feature-engineering decisions and research limitations for the competition match.

## A. Competition challenge contract

| Item | Verified / audited information |
|---|---|
| Submission labels | `H` = home possession while live; `A` = away possession while live; `D` = ball dead; `X` = abstain / unscored |
| Tactical timeline | 199,319 frames at 25 Hz; submit `frame,label` for zero-based frames `0..199318` |
| Scored first half | `11185..90651` inclusive = 79,467 frames |
| Scored second half | `110808..190197` inclusive = 79,390 frames |
| Scored total | 158,857 frames; remaining 40,462 unscored |
| Submission convention | Tactical ATD source `frame_id` is **1-based**, so `submission_frame=frame_id-1` |
| Smart Tagging | Seconds on tactical video timeline; convert to frame with documented rounding convention |
| Primary evaluation | Possession **spellF1**, one-to-one temporal-IoU matching at several thresholds (0.3–0.7); framewise macro-F1 secondary |

`D` denotes a genuine non-live period; `X` is not synonymous with `D`. The absence of a visible ball is **not** a dead-ball label. Do not treat loose balls as a fourth possession state: the competition convention retains the last controlling team until the opposing team acquires control, unless play is dead. All source-specific rules should be checked against the fresh clone's official challenge documentation before coding.

**Restrictions:** No training, model tuning or manual event annotation on the competition's public five-minute possession ground truth. The reference is useful for alignment/semantics only within the competition's permitted use. Do not manually annotate or correct challenge-match possession from video. External match supervision, rule-based challenge pseudo-labeling and permitted self-supervision are distinct, and any synthetic labels must be identified as such. Keep private match data outside public Git history.

## B. Tactical ATD: format, identity, quality

- FIFA/EPTS text format: `frame_id:player_1;...;player_n:ball`, with `x,y,speed` triples. Normalized x/y nominally `0..1`; **player speed is reported in metres per second**; missing triples are `NaN,NaN,NaN`. Pitch used for metric geometry: **105.3 × 68.0 m**.
- 199,319 contiguous raw records; 46 tracked slots (24 Barcelona, 22 Athletic); 45 ever observed. **Slots and ShirtNumber are not stable player IDs.** Tracks may split, switch, duplicate and overlap; occasionally more than 11 slots per side are simultaneously present. Preserve the full available set with position and speed masks rather than forcing 11 players or deriving velocities from consecutive slot positions.
- Tracked players per scored frame: mean **20.49**, median 22; 23,644 scored frames have <20 players, 78,597 exactly 22, 3,749 >22. Means: 10.20 home, 10.29 away. Full-video mean 18.98.
- 3,782,421 player-position observations; 3,412,124 finite player-speed observations (**90.21%** among observed positions). At least one valid player speed is available on **158,428 / 158,857 = 99.73%** of scored frames. The speed measurement itself is *not necessarily measurement-equivalent* to WCSF's one-second-smoothed speed.
- Ball observed **113,488/199,319 (56.94%)** across full tactical video and **94,214/158,857 (59.31%)** across scored frames; scored missing-ball frames **64,643**. Finite ball speed on only **9,909** full-video observations; don't rely on it. Scored first and second half visibility ≈59.35% / 59.27%.
- Missing-ball run counts within scored halves: **150** (69 H1, 81 H2). Median **10.72 seconds**, P90 **33.81 s**, maximum **221.48 s**; 19 runs ≥30 s and 4 ≥60 s. Long runs cannot be reconstructed with simple ball interpolation.
- Of visible ball positions, **4,964 (4.37%)** lie outside nominal pitch normalized [0,1]²; **43,911 (1.16%)** observed player positions also lie outside. Don't indiscriminately clip.
- Nearest observed player to visible ball: median **2.36 m**, P90 **9.45 m**, P99 **21.71 m**. Proximity is meaningful but not definitive ball control.

### Coordinate and orientation decisions

The raw competition x/y are normalized pitch coordinates, whereas WCSF uses team-relative metres. For the planned common model coordinate schema, WCSF can be de-mirrored to a shared home-centric physical pitch and then normalized. The **competition home-team attacking direction by half must be established from reliable metadata or a documented, label-independent check before rotation**; the previous discussion proposed rotating *both teams together* on any half where Barcelona attacks toward negative x: `(x,y) → (1-x,1-y)`. This was a **proposed preprocessing choice**, not a verified fact about the direction in the challenge files. Do not claim that a particular half must be rotated without verification. Never rotate home and away independently for the competition.

## C. Smart Tagging: counts, semantics, coverage

- 638 automatic, overlapping Smart Tag instances; 314 home team, 288 away team, 36 team `N/A`. Tag intervals are **tactical contexts**, not mutually exclusive live/dead or possession labels. Coverage anywhere on full video: **139,587/199,319 = 70.03%**; up to 12 simultaneous tags.
- Code counts: `PLAYERS IN THE BOX` 97, `BALL IN FINAL THIRD` 93, `BALL IN THE BOX` 72, `PROGRESSION` 66, `DEFENDING IN MIDDLE THIRD` 66, `SET PIECES` 59, `CREATING CHANCES` 30, `DEFENDING IN DEFENSIVE THIRD` 30, `LONG BALL` 29, `ATTACKING TRANSITION` 24, `DEFENSIVE TRANSITION` 24, `BUILD UP` 22, `DEFENDING IN ATTACKING THIRD` 22, `GOALS` 4.
- Set pieces: **26 throw-ins, 12 corners, 10 goal kicks, 5 free kicks, 4 kickoffs after goal, 2 half-start kickoffs**. **All 26 throw-ins have `Team=N/A`**, so throw-in tags alone cannot label H/A. Set-piece annotation windows often span ~35 seconds and can cover both preparation and resumed live play.
- Attacking and corresponding defending tactical tags are *paired views of the same phase*, not independent evidence. Count each phase once when creating soft labels. Goals tags are very broad (median ≈169 seconds), overlap other contexts, and must not be equated with an entire dead interval.
- Observed tracking during different tags varies: build-up/progression show ≈93% ball visibility and attacking transitions ≈92%; creating chances ≈57%, set pieces ≈68%, goals ≈12%. These figures reflect automated annotation/tracker behavior and should never be mistaken for true label precision.

### Completed feature engineering (in prior working repo)

**Stage 1 — `tracking_player_sets.npz`**, builder `feature_engineering/tracking_features.py`:

| Array | Intended shape | Meaning |
|---|---|---|
| `players` | `(199319,2,24,3)` | Per-slot normalized x, y, reported speed; padded slots and invalid numeric entries zeroed |
| `player_location_valid` | `(199319,2,24)` | Valid positions |
| `player_speed_valid` | `(199319,2,24)` | Valid provider speed |
| `ball` | `(199319,3)` | Normalized x/y and reported speed |
| `ball_location_valid`, `ball_speed_valid` | `(199319,)` each | Masks; zero numeric data alone is not absence |

Also preserved tracked counts, metadata (frame indices, half IDs, team IDs, fps); roster slots are **unordered sets**, not persistent ID sequences. Stage 1 was validated as deterministic at 25 Hz.

**Stage 2 — `derived_tracking_features.npz`**, builder `feature_engineering/derived_tracking_features.py`: `features` and `feature_valid`, each `(199319,17)`; zero numeric fill for invalid values **only when paired with validity masks**. Ordered names:

1. `nearest_home_dist`
2. `nearest_away_dist`
3. `nearest_dist_diff` (home minus away)
4. `home_players_within_5m`
5. `away_players_within_5m`
6–11. Home `mean_speed`, `median_speed`, `speed_std`, `tracked_player_count`, `valid_speed_count`, `speed_coverage`
12–17. Corresponding six Away measures.

Ball-based proximity features are available on ~59.2% of scored frames; per-team activity statistics remain available nearly all the time. The six team measures can be independently reconstructed per frame from masks and valid speeds; they **do not need player identity continuity**.

**Stage 3 — `smart_tag_features.npz`**, builder `feature_engineering/smart_tag_features.py`:

- `tactical_active (199319,2,4)` bool and `tactical_weight (199319,2,4)` float32; categories `BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, `CREATING CHANCES`. Matched attacking/defending source tags must not be double counted.
- Weight for interval `[start,end)` uses frame-center progress `u=(frame-start+0.5)/(end-start)` and Gaussian `exp[-0.5*((u-0.5)/0.22)^2]`; overlapping tags of same team/category use maximum.
- `context_active (199319,11)` bool, `context_since_start` and `context_until_end` int32: start kickoff, after-goal kickoff, throw-in, corner, goal kick, free kick, goals, long ball, ball in final third, ball in box, players in box. Nonactive timing zeros are only distinguishable using `context_active`. Team-agnostic and overlapping contexts allowed.
- Tactical assigned-team scored coverage **63,718 / 158,857 = 40.1%**. At least one Smart Tag context/tactical flag on **130,698 / 158,857 = 82.27%** scored frames. 31.50% have tags but missing ball; 9.19% have neither.

| Scored evidence regime | Frames | Percent |
|---|---:|---:|
| Ball + any Smart Tag | 80,657 | 50.77% |
| Ball, no Smart Tag | 13,557 | 8.53% |
| Smart Tag, no ball | 50,041 | 31.50% |
| Neither | 14,602 | 9.19% |

This is **data availability**, not verified possession coverage.

## D. Exploratory insights on missing ball, activity and stoppages

- Missing-ball intervals often overlap set-piece contexts; overlapping tag counts are **not mutually exclusive**. From the main EDA, 150 ball gaps overlap automatic goals 11 times, kickoffs 5, corners 23, throw-ins 34, free kicks 6.
- Team speed tends to fall during long missing-ball runs and rise around the ball's return. For long gaps (≥30 s), mean player speed was around 2.08 m/s one second before disappearance, 1.43 m/s five seconds into the gap, 0.88 m/s five seconds before reappearance, and 2.18 m/s five seconds after reappearance. This **does not establish dead-ball ground truth**.
- Coverage-controlled follow-up: where 21–22 players were tracked, mean speed was **2.205 m/s** on ball-visible frames vs **1.358 m/s** on ball-missing frames, difference **0.848 m/s** (95% block-bootstrap CI 0.724–0.974). The effect persisted in both scored halves and broadly in other sufficiently populated coverage strata. At equal interval weighting the effect was smaller (about 0.47 m/s for 21–22 players): long runs influence frame-weighted estimates.
- For missing-ball gaps with 21–22 players tracked, mean speed decreased with gap length (5–10 s: 1.814 m/s; 10–30 s: 1.299; 30–60 s: 1.146; ≥60 s: 0.921). Long low-motion gaps are likely enriched for stoppages, **not proof that every such gap is dead**.
- Set-piece windows contain differing numbers of ball disappearance/reappearance cycles, so `set-piece tag active = DEAD` is invalid. Ball recovery can occur before the actual restart. The investigation found some coordinate jumps and probable tracking artifacts across gaps; do not extrapolate a straight ball trajectory through them.
- Player speeds/observation counts must remain explicit measurement-quality channels; reduced player coverage explains some, but not all, differences in mean speed.

## E. Proposed model-ready preprocessing contract (design, not evidence of completed implementation)

- **Common frequency:** WCSF at native **10 Hz**; challenge 25 Hz resampled to a deterministic mask-preserving 10 Hz timeline. Never interpolate missing player observations into artificial valid observations. Save mapping back to 25 Hz; later interpolate **model probabilities** rather than inventing player tracks.
- **Per-frame unordered player set:** `(N,2,24,3)` x/y/ **provided speed**; `(N,2,24)` location-valid and speed-valid masks. The 24 slots are storage/padding only, not player IDs or aligned trajectories. Deep Sets or another permutation-invariant player encoder is appropriate.
- **Team activity:** `(N,2,6)`: mean, median and std of valid player speeds, tracked count, valid speed count, valid speed coverage, with separate feature-valid masks. Do not derive individual speed from consecutive Metrica slot positions or smooth individual reported speeds across unstable slots. Optional validity-aware temporal smoothing of **aggregated** team activity remains a separate experiment.
- **Coordinates:** WCSF provider coordinates in metres are mirrored by team; reconstruct shared field before normalization. Proposed common `(x,y)` normalized physical pitch with home attacking +X; apply whole-field 180° rotation on the competition half(s) only when the real attacking direction is verified without forbidden ground truth. Use `(x,y)→(1-x,1-y)` for normalized competition positions when appropriate.
- **Training labels:** only external WCSF provides `AliveFlag`; use a separate label-valid mask for its conflicted frames. Do not introduce competition labels or Smart Tags into the initial movement-only supervised classifier.
- **Splits:** WCSF H1 train/validation → H2 test and reverse; choose ~15% contiguous validation block within training half with configurable exclusion buffer (initially 10 s each side), at least as large as the eventual model temporal context; no random frame splits or normalization fit on held-out data.
- **Evaluation:** classification F1/precision/recall and calibration, transition timing and segment F1, then full H/A/D spellF1 once the team-control model is integrated. Half-held-out WCSF evaluation is **within-match**, not cross-match validation.

## F. Key unknowns and migration checklist

1. Re-fetch and inspect authoritative challenge specification and exact source CSV/XML schemas in the new checkout; old analysis filenames may not exist.
2. Preserve original data outside public Git. The previous working directory's NPZ archives, EDA scripts and supporting CSVs are **not automatically restored by these Markdown notes**.
3. Verify Barcelona's attacking direction separately for each scored half before any whole-pitch rotation. WCSF's home-centric orientation must not be blindly assumed for Metrica.
4. Inspect ball-missing and low-player-coverage regimes; do not equate absence of detection with a stoppage.
5. Treat automatic Smart Tags as **weak evidence** with provenance, not training/evaluation ground truth.
6. Do not use public competition possession labels for training/tuning. Never manually annotate the challenge match.
7. Any claimed generalization to the challenge remains unverified without independent permitted labeled evaluation data.

---

# Archived source reports (verbatim research context)

The sections below preserve the original competition-match analyses as standalone text. They are included for completeness, including methodology, raw tables, uncertainties, and old relative paths. **Embedded figures and referenced CSVs are not automatically included in this standalone Markdown; paths refer to the old analysis workspace.** Current code/feature status is not implied by an archived report.



---

## Archived source: `DATA_SUMMARY.md`

# Data Summary and Local Audit

**Project:** More than a Hack 2027 — Phase 1: Ball Possession  
**Match:** FC Barcelona vs Athletic Club, LaLiga EA Sports 2024–25, matchday 2  
**Audit date:** 2026-10-27  
**Reference timeline:** tactical recording, 25 frames per second

This document consolidates the data contract described by the repository with statistics measured
directly from the files currently available under `data/`. It is intended as a practical reference
for data loading, feature engineering, quality control, modeling, and submission generation.

## 1. Task and label contract

The goal is to produce exactly one label for every frame of the **tactical** recording.

| Label | Meaning |
|---|---|
| `H` | Home team, FC Barcelona, has possession while the ball is in play |
| `A` | Away team, Athletic Club, has possession while the ball is in play |
| `D` | Dead ball; the ball is not in play |
| `X` | No answer/abstention, or a frame without ground truth |

The required submission is a CSV with the exact header `frame,label` and exactly **199,319 data
rows**. Frames must be the integers `0` through `199318`, in order. The tactical video timeline is
the only submission timeline; the broadcast timeline is optional input and is not synchronized to
it.

Every scored ground-truth frame is one of `H`, `A`, or `D`. There is no contested-ball state. While
play is live, possession remains with the last controlling team through loose-ball situations until
the other team gains controlled possession. A dead-ball interval is a stoppage, not a possession.

## 2. Match timeline and scored frames

The tactical timeline contains **199,319 frames**, equivalent to approximately **132.88 minutes**
at 25 fps. The two scored half-blocks contain **158,857 frames**, approximately **105.90 minutes**.

| Period | First frame | Last frame | Frames | Scored? |
|---|---:|---:|---:|---|
| Before the match | 0 | 11,184 | 11,185 | No |
| First half | 11,185 | 90,651 | 79,467 | Yes |
| Halftime | 90,652 | 110,807 | 20,156 | No |
| Second half | 110,808 | 190,197 | 79,390 | Yes |
| After the final whistle | 190,198 | 199,318 | 9,121 | No |

There are **40,462 unscored frames**. `X` can safely be emitted for all of them.

The second scored block begins at frame `110808`, but the second-half kickoff occurs at frame
`113742`. The intervening **2,934 frames** are scored and are all dead-ball frames. The first scored
block begins at the first-half kickoff.

## 3. Local asset inventory

The current local bundle contains seven structured match-data files plus `data/README.md`, totaling
**376,656,707 bytes** (approximately 376.7 MB, or 359.2 MiB).

| Source | Asset | File size | Present? |
|---|---|---:|---|
| Tactical | ATD raw tracking | 195,714,285 bytes | Yes |
| Tactical | ATD metadata | 59,057 bytes | Yes |
| Tactical | Homography | 23,187,043 bytes | Yes |
| Tactical | Smart Tagging | 261,055 bytes | Yes |
| Tactical | MP4 video | — | **No** |
| Broadcast | ATD raw tracking | 141,966,465 bytes | Yes |
| Broadcast | ATD metadata | 64,956 bytes | Yes |
| Broadcast | Homography | 15,401,635 bytes | Yes |
| Broadcast | MP4 video | — | **No** |

The expected tactical and broadcast MP4 files were not present at audit time. Therefore, the
current bundle supports tracking-, tagging-, and geometry-based work, but not pixel-level computer
vision or visual inspection of tracking/homography alignment.

The match data is separately licensed and must not be redistributed or committed to the public
submission repository. The repository's `.gitignore` already excludes the contents of `data/`.

## 4. Timeline alignment

Each asset is indexed on the timeline of its own recording.

| Asset | Index convention | Conversion to tactical submission frame |
|---|---|---|
| Submission | Zero-based tactical video frame | Already aligned |
| Tactical ATD | `frame_id` from 1 to 199,319 | `video_frame = frame_id - 1` |
| Tactical Smart Tagging | Seconds from tactical video start | `video_frame = round(seconds * 25)` |
| Tactical homography | Zero-based `frame` | Already aligned |
| Broadcast assets | Broadcast's own timeline | No mapping is provided |

The per-half frame offsets in `games.py` describe alignment with the separate corrected-tracking
counter used to derive the ground truth. Participants do not need those offsets to align the
provided tactical ATD: its only required conversion is subtracting one from `frame_id`.

## 5. Automatic Tracking Data format

The ATD is a FIFA/EPTS export consisting of a metadata XML file and a raw text file. Each raw line
has this form:

```text
frame_id:player_1;player_2;...;player_n:ball
```

Each player or ball group contains:

```text
x,y,speed
```

- `x` and `y` are normalized pitch coordinates, nominally on a `0–1` scale.
- Player speed is in metres per second.
- A missing observation is represented as `NaN,NaN,NaN`.
- The real pitch dimensions used by `games.py` are **105.3 × 68.0 m**.
- Track slots identify tracklets, not people. Slot numbers and `ShirtNumber` fields must not be
  interpreted as real shirt numbers or stable player identities.

Slots can appear, disappear, switch identity, and duplicate a player. Possession features should
therefore be collective and geometric rather than dependent on persistent player identity.

## 6. Tactical ATD audit

### 6.1 Structure and continuity

| Statistic | Value |
|---|---:|
| Raw records | 199,319 |
| `frame_id` range | 1–199,319 |
| Contiguous | Yes |
| Malformed lines | 0 |
| Lines with the wrong slot count | 0 |
| Metadata track slots | 46 |
| Barcelona slots | 24 |
| Athletic Club slots | 22 |
| Slots observed at least once | 45 |

The team name in the metadata is written as `Atletic Club`; code should not silently assume an
accented or differently spelled version.

### 6.2 Tracked-player coverage

| Population | Mean | Median | P10 | P90 | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| All frames | 18.98 | 21 | 10 | 22 | 1 | 35 |
| Scored frames | 20.49 | 22 | 18 | 22 | 1 | 25 |
| Unscored frames | 13.05 | 12 | 5 | 21 | 1 | 35 |

Within the 158,857 scored frames:

- **23,644 frames (14.9%)** have fewer than 20 tracked players.
- **78,597 frames (49.5%)** have exactly 22 tracked players.
- **3,749 frames (2.4%)** have more than 22 tracked players, indicating duplicate tracks or other
  tracking noise.
- Mean observed counts are 10.20 Barcelona tracks and 10.29 Athletic tracks per frame.

Across the full tactical recording there are 3,782,421 observed player positions. Finite player
speed is available for 3,412,124 of them, giving **90.21% speed coverage**.

### 6.3 Ball coverage

| Population | Ball observed | Total frames | Coverage | Missing |
|---|---:|---:|---:|---:|
| Full tactical recording | 113,488 | 199,319 | 56.94% | 43.06% |
| Scored frames | 94,214 | 158,857 | 59.31% | 40.69% |
| Unscored frames | 19,274 | 40,462 | 47.63% | 52.37% |

Only **9,909** tactical ball observations contain a finite speed, approximately **8.7%** of frames
with a visible ball. Ball speed should therefore not be treated as a generally available feature.

Ball absence occurs in long contiguous intervals:

| Half | Missing-ball runs | Median run | 90th percentile | Longest run | Longest duration |
|---|---:|---:|---:|---:|---:|
| First half | 69 | 289 frames | 1,083 frames | 3,466 frames | 138.64 s |
| Second half | 81 | 245 frames | 639 frames | 5,537 frames | 221.48 s |

These gaps are much too long for unrestricted interpolation. Short-gap interpolation may be useful,
but longer gaps require team-shape, player-motion, tagging, and temporal-state evidence.

### 6.4 Coordinate and proximity quality

Of the 113,488 visible tactical ball observations, **4,964 (4.37%)** lie outside the nominal
normalized `[0,1] × [0,1]` pitch rectangle. Of 3,782,421 observed player positions, **43,911
(1.16%)** are outside that range.

Out-of-bounds positions should be inspected contextually rather than blindly clipped: they can
represent a legitimately out-of-play ball, people beyond the field boundary, projection error, or
tracking noise.

Using the measured 105.3 × 68.0 m pitch and the nearest observed player to each visible ball:

| Percentile | Nearest-player distance |
|---|---:|
| Minimum | 0.002 m |
| P10 | 0.50 m |
| Median | 2.36 m |
| P90 | 9.45 m |
| P99 | 21.71 m |
| Maximum | 86.90 m |

The heavy tail shows why nearest-player ownership is a useful signal but a weak standalone
possession rule. Aerial balls, occlusions, stale tracks, dead balls, and tracking mistakes can all
produce large distances.

## 7. Broadcast ATD audit

The broadcast source is optional and uses a separate, unsynchronized timeline with cuts and
replays.

| Statistic | Value |
|---|---:|
| Records | 164,488 |
| `frame_id` range | 1–164,488 |
| Contiguous | Yes |
| Track slots | 51 |
| Barcelona/Athletic slots | 26 / 25 |
| Ball-observed frames | 83,121 (50.53%) |
| Mean tracked players | 11.48 |
| Median tracked players | 14 |
| P90 tracked players | 21 |
| Longest missing-ball interval | 4,004 frames (160.16 s) |

The low player count reflects the broadcast camera's close-ups, restricted field of view, replays,
and non-pitch shots. Finite player speed covers **80.94%** of observed broadcast player positions;
only 5,590 visible-ball observations contain a finite speed.

Broadcast evidence cannot be copied into the tactical submission without first solving temporal
alignment. Any alignment procedure must also detect and exclude replays and non-live shots.

## 8. Homography audit

Each homography row contains a 3 × 3 matrix mapping normalized pitch coordinates to image pixels,
plus `reprojection_error` and `valid`. Matrix values are stored in row-major columns `h00` through
`h22`.

### Tactical homography

| Statistic | Value |
|---|---:|
| Rows | 199,319 |
| Frame range | 0–199,318 |
| Contiguous and zero-based | Yes |
| Valid matrices | 199,319 (100%) |
| Valid rows with missing matrix values | 0 |
| Median reprojection error | 0.239 |
| P90 reprojection error | 0.407 |
| P99 reprojection error | 4.808 |
| Maximum reprojection error | 5.476 |

The tactical matrix is available on every frame, but `reprojection_error` should still be used as a
soft confidence feature. A matrix can exist while being less reliable, particularly outside active
play.

### Broadcast homography

| Statistic | Value |
|---|---:|
| Rows | 164,488 |
| Frame range | 0–164,487 |
| Contiguous and zero-based | Yes |
| Valid matrices | 108,248 (65.81%) |
| Invalid matrices | 56,240 (34.19%) |
| Longest invalid interval | 2,719 frames (108.76 s) |
| Median valid-row reprojection error | 0.217 |
| P90 | 0.450 |
| P99 | 6.527 |
| Maximum | 303.824 |

The local file contains **123 invalid runs**, including one one-frame run, while `DATA.md` describes
122 stretches. Code should rely on the file's `valid` column instead of a hard-coded count.

Six rows marked valid, at frames `46073–46075` and `46077–46079`, have negative reprojection-error
values. A conventional residual should not be negative, so these values should be treated as a data
quality anomaly rather than high-confidence matrices.

## 9. Smart Tagging audit

Smart Tagging is an automatic SportsCode XML annotation of tactical phases and set pieces. Its
intervals use seconds from the start of the tactical recording. The file contains **638 contiguous
instance IDs**, from 1 through 638.

### 9.1 Team and half distribution

| Tag | Count |
|---|---:|
| Barcelona | 314 |
| Athletic Club | 288 |
| `N/A` | 36 |
| First half | 320 |
| Second half | 318 |

All 26 throw-ins have `Team = N/A`. Team tags must therefore be handled explicitly rather than
assuming that every restart identifies the restarting side.

### 9.2 Code distribution

| Code | Count | Code | Count |
|---|---:|---|---:|
| `PLAYERS IN THE BOX` | 97 | `CREATING CHANCES` | 30 |
| `BALL IN FINAL THIRD` | 93 | `DEFENDING IN DEFENSIVE THIRD` | 30 |
| `BALL IN THE BOX` | 72 | `LONG BALL` | 29 |
| `PROGRESSION` | 66 | `ATTACKING TRANSITION` | 24 |
| `DEFENDING IN MIDDLE THIRD` | 66 | `DEFENSIVE TRANSITION` | 24 |
| `SET PIECES` | 59 | `BUILD UP` | 22 |
| `DEFENDING IN ATTACKING THIRD` | 22 | `GOALS` | 4 |

### 9.3 Set-piece distribution

| Set-piece type | Count |
|---|---:|
| Throw-in | 26 |
| Corner kick | 12 |
| Goal kick | 10 |
| Free kick | 5 |
| Kickoff after goal | 4 |
| Half-start kickoff | 2 |

### 9.4 Duration and coverage

| Statistic | Value |
|---|---:|
| Earliest tag start | 442.76 s, approximately frame 11,069 |
| Latest tag end | 7,719.28 s, approximately frame 192,982 |
| Minimum duration | 5.00 s |
| Median duration | 22.82 s |
| Mean duration | 25.69 s |
| P90 duration | 35.10 s |
| Maximum duration | 250.72 s |
| Frames covered by at least one tag | 139,587 (70.03% of the video) |
| Frames covered by at least two tags | 95,742 |
| Maximum simultaneous tags | 12 |

The intervals are overlapping tactical concepts, not mutually exclusive states and not possession
segments. Smart Tagging should be treated as noisy evidence. Particularly useful signals include:

- Set-piece intervals as restart and dead-ball-boundary priors.
- Team-tagged build-up, progression, final-third, transition, and chance-creation intervals as soft
  possession priors.
- Attacking and defensive transition tags as candidate change-of-possession regions.

Copying a tag interval directly into the output would import false positives, false negatives, and
phase-boundary errors wholesale.

## 10. Where the long ball gaps occur

This section uses only the tactical ATD, Smart Tagging, and the published scored-half boundaries.
It does **not** inspect possession ground-truth labels. A missing-ball run is a contiguous sequence
of ATD frames whose ball group is `NaN,NaN,NaN`. Here, a **long gap** means at least 30 seconds
(750 frames).

The analysis is reproducible with:

```bash
python analysis/ball_gap_analysis.py
```

It writes the complete machine-readable results to `analysis/ball_gap_summary.json`, the long-gap
table to `analysis/ball_gap_long.csv`, and the following figure to
`figures/ball_gap_distribution.svg`.

![Ball visibility timeline and missing-run distribution](figures/ball_gap_distribution.svg)

### 10.1 Overall distribution

There are **150 separate missing-ball runs** inside the two scored blocks: 69 in the first half and
81 in the second. Together they account for 64,643 missing frames, or **2,585.72 seconds (43.10
minutes)**. Ball visibility is almost identical across the halves: 59.35% in the first and 59.27%
in the second.

| Minimum gap duration | Runs | Share of all runs | Share of all missing-ball time |
|---|---:|---:|---:|
| At least 1 second | 133 | 88.7% | 99.88% |
| At least 5 seconds | 132 | 88.0% | 99.70% |
| At least 10 seconds | 79 | 52.7% | 85.24% |
| At least 30 seconds | 19 | 12.7% | 43.83% |
| At least 60 seconds | 4 | 2.7% | 18.99% |

The non-cumulative duration bins are:

| Duration | Runs |
|---|---:|
| Less than 1 second | 17 |
| 1 to less than 5 seconds | 1 |
| 5 to less than 10 seconds | 53 |
| 10 to less than 30 seconds | 60 |
| 30 to less than 60 seconds | 15 |
| At least 60 seconds | 4 |

Run duration has a median of **10.72 s**, P75 of 21.06 s, P90 of 33.81 s, P95 of 50.80 s, and a
maximum of 221.48 s. This is not a pattern of predominantly momentary occlusions with only a few
outliers. More than half of all gaps last at least 10 seconds, and the 19 gaps of at least 30
seconds produce almost 44% of all missing-ball time.

### 10.2 Every gap of at least 30 seconds

“Half time” below means elapsed time from the start of the published scored block, not the stadium
clock. `Players min/median/mean/max` counts available player positions per frame. Speed coverage is
the percentage of those player-position observations that also have a finite speed. Tag coverage
is the fraction of gap frames covered by at least one automatic Smart Tagging interval.

| Half | Frames | Half time | Duration | Players min/median/mean/max | Speed coverage | Smart Tags: intervals / coverage | Overlapping set piece |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 29,030–29,780 | 11:54–12:24 | 30.04 s | 16 / 21 / 20.4 / 23 | 91.2% | 8 / 100% | Corner kick |
| 1 | 36,264–37,051 | 16:43–17:15 | 31.52 s | 15 / 18 / 18.3 / 21 | 92.6% | 10 / 100% | Corner kick |
| 1 | 46,678–48,030 | 23:40–24:34 | 54.12 s | 9 / 14 / 14.9 / 22 | 83.4% | 8 / 100% | Free kick |
| 1 | 48,640–49,712 | 24:58–25:41 | 42.92 s | 18 / 20 / 20.0 / 21 | 93.7% | 2 / 100% | Kickoff after goal |
| 1 | 58,081–61,546 | 31:16–33:34 | **138.64 s** | 4 / 11 / 12.1 / 23 | 74.6% | 2 / 100% | Corner kick |
| 1 | 70,196–71,519 | 39:20–40:13 | 52.96 s | 10 / 15 / 16.2 / 22 | 79.6% | 2 / 12.8% | — |
| 1 | 72,560–73,950 | 40:55–41:51 | 55.64 s | 10 / 15 / 15.6 / 21 | 81.1% | 7 / 100% | Free kick |
| 1 | 73,997–75,531 | 41:52–42:54 | **61.40 s** | 4 / 16 / 16.5 / 23 | 83.6% | 6 / 100% | Free kick; kickoff after goal |
| 1 | 80,051–81,174 | 45:55–46:40 | 44.96 s | 18 / 21 / 20.8 / 22 | 90.1% | 6 / 100% | Corner kick |
| 1 | 86,615–88,351 | 50:17–51:27 | **69.48 s** | 18 / 21 / 20.7 / 23 | 92.1% | 4 / 83.8% | — |
| 1 | 89,109–90,037 | 51:57–52:34 | 37.16 s | 2 / 4 / 4.0 / 8 | 63.3% | 0 / 0% | — |
| 2 | 129,693–130,471 | 12:35–13:07 | 31.16 s | 17 / 20 / 19.5 / 21 | 87.9% | 7 / 100% | Corner kick |
| 2 | 135,624–136,678 | 16:33–17:15 | 42.20 s | 17 / 21 / 20.4 / 23 | 89.0% | 14 / 100% | Corner kick |
| 2 | 146,038–146,873 | 23:29–24:03 | 33.44 s | 18 / 21 / 20.7 / 22 | 91.2% | 7 / 100% | Corner kick |
| 2 | 152,396–153,536 | 27:44–28:29 | 45.64 s | 21 / 22 / 21.9 / 23 | 94.1% | 7 / 36.4% | — |
| 2 | 154,923–155,898 | 29:25–30:04 | 39.04 s | 13 / 21 / 19.5 / 23 | 87.2% | 12 / 100% | Corner kick; free kick |
| 2 | 157,546–163,082 | 31:10–34:51 | **221.48 s** | 4 / 18 / 15.9 / 24 | 82.0% | 10 / 100% | Kickoff after goal; throw-in |
| 2 | 168,918–170,177 | 38:44–39:35 | 50.40 s | 19 / 21 / 20.9 / 22 | 97.2% | 1 / 36.0% | Throw-in |
| 2 | 175,715–176,992 | 43:16–44:07 | 51.12 s | 17 / 19 / 19.2 / 21 | 88.8% | 5 / 68.2% | — |

### 10.3 What remains usable during long gaps

- **Player positions never disappear completely.** Every one of the 19 long gaps has at least one
  player position on every frame. Ten gaps have a median of at least 20 tracked players.
- **Player speeds mostly remain usable.** Across all player observations in the long gaps, weighted
  speed coverage is **86.33%**. Per-gap coverage ranges from 63.33% to 97.22%.
- **Tracking quality is heterogeneous.** Some long gaps preserve nearly the full player set—for
  example frames 152,396–153,536 have a median of 22 players—while others coincide with broad
  tracking degradation. Frames 89,109–90,037 have only 2–8 players and a median of 4.
- **Smart Tagging is usually present.** Eighteen of 19 long gaps overlap at least one tag; 13 are
  covered by tags for their full duration. Fourteen overlap an automatically tagged set piece.
  These associations are informative priors, not proof that the entire gap is dead ball.
- **The two extreme gaps are not simple ball-only occlusions.** The 138.64-second first-half gap has
  a median of only 11 players and 74.6% speed coverage, showing substantial general tracking loss.
  The 221.48-second second-half gap is less degraded—a median of 18 players and 82.0% speed
  coverage—and spans ten automatic tactical intervals, so it contains multiple phases rather than
  one homogeneous event.

### 10.4 Relation to half boundaries

None of the 19 gaps of at least 30 seconds touches the first or last frame of a scored block. The
closest is frames 89,109–90,037, which ends **24.56 seconds before the end of the first-half scored
block** and also has severe player-tracking loss and no overlapping Smart Tagging interval. This
boundary-adjacent case looks different from the typical long gap and deserves a low-confidence or
special-quality state, but it must not be manually assigned a possession label.

The main conclusion is that long gaps occur throughout both halves, not only at kickoffs, halftime,
or the final whistle. A robust model therefore cannot solve missing-ball frames with boundary rules
or interpolation alone. It should distinguish at least:

1. Ball-only loss with strong player tracking and tactical context.
2. Joint ball/player degradation with partial geometric information.
3. Broad tracking collapse, where the previous temporal state and uncertainty handling dominate.

## 11. Public ground truth

`evaluation/gt/BAR-ATH_possession_gt_public.csv` is a full-length, 199,319-row label file, but only
the first five minutes of play contain scored labels. All other rows are `X` and ignored.

| Statistic | Value |
|---|---:|
| Covered frames | 11,185–18,684 |
| Scored frames | 7,500 |
| `H` frames | 3,454 |
| `A` frames | 2,577 |
| `D` frames | 1,469 |
| Possessions | 17 |
| Total segments, including dead ball | 21 |
| Mean possession duration | 14.191 s |

This window is for validating file shape, frame alignment, label meaning, and metric integration.
Competition rules prohibit using it for training, parameter tuning, threshold selection, or model
selection. With only 17 possessions, it is statistically too small to rank reasonable model
variants reliably in any case.

## 12. Evaluation implications

The ranking metric is **spellF1**, which treats every maximal run of `H` or `A` as one possession.
Predicted and true possessions are matched one-to-one by temporal intersection-over-union, and F1
is averaged over IoU thresholds 0.3, 0.4, 0.5, 0.6, and 0.7.

`macroF1` is also reported and is the first tie-break, but it scores frames independently. A model
can achieve reasonable frame accuracy while creating far too many short possessions. The published
nearest-player baseline illustrates this:

| Full-match baseline | macroF1 | spellF1 |
|---|---:|---:|
| Raw nearest player | 0.6532 | 0.0486 |
| Smoothed ±25 frames | 0.6752 | 0.1996 |
| Smoothed ±50 frames | 0.6724 | **0.2433** |
| Smoothed ±125 frames | 0.6356 | 0.2242 |

The target to beat is the ±50-frame baseline at **spellF1 0.2433**.

On scored frames, `X` counts as a miss for `macroF1`. Before spell evaluation, however, `X` is
forward-filled from the last non-`X` label within the same half, with back-filling at the beginning
of a half. Consequently:

- Abstaining during an uncertain patch inside a stable possession can avoid artificial splits.
- Abstaining across a true handover extends the previous possession and can badly misplace the
  transition.
- `X` should not be interpreted as a fourth state in the temporal model.

## 13. Recommended modeling interpretation

The data strongly favors a temporal segmentation system over independent frame classification.
A practical design should include:

1. **Separate evidence for live/dead status and team control.** Ball missingness is not equivalent
   to dead ball, and a visible ball does not guarantee live play.
2. **Team-control emissions.** Candidate features include nearest-player distance, several nearest
   players by team, local numerical superiority, team centroids and shapes, player and ball motion,
   pitch zone, and continuity from previous frames.
3. **A temporal state model.** An HMM, hidden semi-Markov model, constrained dynamic program, or
   change-point model can suppress flicker and impose plausible spell durations and transitions.
4. **Confidence-aware missing-data handling.** Interpolate only short gaps. Use collective player
   geometry, motion, tagging, and prior state during long ball gaps.
5. **Soft use of Smart Tagging.** Tags should modify probabilities around phases and restarts, not
   replace the possession state directly.
6. **Explicit boundary modeling.** Since spellF1 rewards possession intervals, the start and end of
   each possession are more important than isolated errors deep inside an otherwise correct spell.
7. **Tactical-first development.** Build a strong structured-data model before adding broadcast
   synchronization. Broadcast integration adds replay detection, shot classification, and temporal
   alignment as separate failure modes.

All rules, thresholds, and corrections must be algorithmic and reproducible. Frame-specific manual
corrections, manual labels on this match, or human-produced event data about this match are not
permitted.

## 14. Data-loading and validation checklist

- [ ] Use tactical frame indices for every submission row.
- [ ] Convert tactical ATD with `video_frame = frame_id - 1`.
- [ ] Confirm exactly 199,319 output rows and frames `0..199318`.
- [ ] Initialize all unscored frames to `X`.
- [ ] Treat track slots as temporary tracklets, not persistent players.
- [ ] Scale normalized coordinates by 105.3 × 68.0 before computing metric distances.
- [ ] Preserve missingness indicators as features rather than silently filling all gaps.
- [ ] Do not equate a missing ball with `D`.
- [ ] Check homography `valid` and reprojection error before projection.
- [ ] Handle negative broadcast reprojection errors defensively.
- [ ] Match Smart Tagging codes and tag values case-insensitively.
- [ ] Treat `Max Players in the box` as a string because it can contain `7+`.
- [ ] Handle `Team = N/A`, especially on throw-ins.
- [ ] Never map broadcast row numbers directly to tactical frames.
- [ ] Validate the exact final CSV with `evaluation/validate_submission.py`.
- [ ] Keep all restricted match assets out of Git and third-party hosted services.

## 15. Current environment note

During the audit, the official evaluation command could not start because the installed pandas
environment was missing the `pytz` and `python-dateutil` dependencies. This is an environment issue,
not a problem with the data files. The dependency environment should be repaired before beginning
the model-evaluation loop.

## 16. Authoritative repository references

- `README.md` — entry point and repository structure
- `CHALLENGE.md` — possession definition and task scope
- `DATA.md` — formats, timelines, and documented asset properties
- `EVALUATION.md` — metric behavior and public-window limitations
- `SUBMISSION.md` — exact output contract and deliverables
- `RULES.md` — permitted and prohibited methods
- `FAQ.md` — practical clarifications and common mistakes
- `games.py` — match constants, asset paths, and alignment helpers
- `evaluation/possession_metrics.py` — authoritative metric implementation

If narrative documentation and executable evaluation code ever disagree, the metric implementation
and official validation scripts are authoritative.



---

## Archived source: `EDA_REPORT.md`

# Exploratory Analysis of Ball Visibility, Player Activity, and Smart Tagging

**Scope:** tactical recording only  
**Frame rate:** 25 fps  
**Possession ground truth used:** no  
**Purpose:** exploratory research, not possession prediction or label construction

This report relates automatic tactical tracking to automatic Smart Tagging. Agreement between the
two sources is evidence of association only; it is not validation of either source.

## 1. Measurements and outputs

The analysis reads only:

- `data/tactical/BAR-ATH_tactical_atd.txt`
- `data/tactical/BAR-ATH_tactical_atd.xml`
- `data/tactical/BAR-ATH_tactical_smart_tagging.xml`
- the published tactical scored-half boundaries

It does not open public or private possession labels. Missing-ball intervals are maximal groups of
consecutive scored-block frames whose ATD ball observation is `NaN,NaN,NaN`.

Activity is calculated from finite player-speed observations. The descriptive thresholds of less
than 0.5 m/s and greater than 4 m/s are not optimized. Each missing interval is compared with the
five seconds immediately before and after it, clipped at half boundaries.

Primary outputs:

- `exploration_dashboard.html` — local complete timeline and interval selector
- `missing_ball_intervals.csv` — all 150 missing intervals with context features
- `missing_ball_tag_overlaps.csv` — one row per gap/tag overlap, including exact overlap duration
- `smart_tagging_category_summary.csv` — category-level duration, visibility, activity and position summaries
- `smart_tagging_overlap_matrix.csv` and `smart_tagging_interval_overlaps.csv`
- `set_piece_timing.csv` and `set_piece_aligned_curves.csv`
- `gap_transition_aligned_curves.csv`
- `player_activity_summary.csv` and `timeline_5s.csv`
- `frame_features.npz` — local frame-level feature cache used by the interval viewer

Static figures:

- [Complete timeline](figures/complete_timeline.svg)
- [Missing-ball transition alignment](figures/gap_transition_alignment.svg)
- [Set-piece boundary alignment](figures/set_piece_alignment.svg)
- [Smart Tagging overlap heatmap](figures/smart_tagging_overlap_heatmap.svg)
- [Speed distributions by tag category](figures/tag_speed_distributions.svg)

The reusable interval viewer can inspect any frame range:

```bash
python research/interval_diagnostic.py --gap-id 115
python research/interval_diagnostic.py --start-frame 157000 --end-frame 164000 --context-seconds 10
```

An example for the longest gap is at `interval_views/interval_gap_115.html`.

## 2. Missing-ball intervals

### 2.1 Frequency and duration

There are 150 missing-ball intervals within the scored blocks, totaling 64,643 frames or 43.10
minutes. The duration distribution established in the initial audit remains the main structural
finding: 79 intervals last at least 10 seconds, 19 last at least 30 seconds, and four last at least
60 seconds.

Activity changes systematically with gap duration:

| Gap duration | n | Mean tracked players | Mean speed inside | Mean speed before | Mean speed after | Mean share below 0.5 m/s |
|---|---:|---:|---:|---:|---:|---:|
| <5 s | 18 | 21.15 | 2.32 m/s | 2.16 | 2.26 | 9.98% |
| 5–10 s | 53 | 21.59 | 1.77 m/s | 2.19 | 1.83 | 12.46% |
| 10–30 s | 60 | 20.62 | 1.28 m/s | 1.93 | 1.60 | 18.88% |
| 30–60 s | 15 | 18.16 | 1.00 m/s | 2.17 | 2.24 | 31.77% |
| ≥60 s | 4 | 16.32 | 0.99 m/s | 1.61 | 1.26 | 35.16% |

Short gaps do not show a stoppage-like reduction in movement: the <5-second group is slightly
faster inside the gap than in its immediate context. The movement reduction becomes strong with
duration. Gaps lasting 30–60 seconds have less than half the mean player speed observed in the
five seconds before and after them.

This supports the hypothesis that many long gaps occur during low-activity situations, but it does
not establish that those frames are dead ball. Some reductions also coincide with falling
player-count coverage, and some long gaps span multiple automatic tactical phases.

### 2.2 Ball-visible versus ball-missing frames

Across all scored frames:

| State | Mean tracked players | Weighted mean speed | Median frame-median speed | P75 speed | P90 speed | Below 0.5 m/s | Above 4 m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ball visible | 21.39 | 2.172 m/s | 1.691 m/s | 3.00 | 4.25 | 7.58% | 12.41% |
| Ball missing | 19.17 | 1.216 m/s | 1.040 m/s | 1.50 | 2.25 | 23.97% | 2.71% |

Ball absence is strongly associated with lower activity and reduced tracking coverage. The
association is large enough to be potentially useful as a stoppage signal, but it is confounded:
the tracker may lose players and the ball together during non-pitch shots or difficult camera
conditions.

### 2.3 Disappearance and reappearance boundaries

The aligned curves show a consistent pattern over all 150 intervals:

- One second before disappearance, mean speed is about 2.06 m/s.
- At disappearance it is 1.96 m/s, falling to 1.54 m/s after five seconds.
- Immediately before reappearance it is about 1.41 m/s.
- At reappearance it rises to 1.52 m/s and reaches 1.96 m/s five seconds later.

For the 19 gaps of at least 30 seconds, the pattern is stronger:

- Speed is 2.08 m/s one second before disappearance and 1.43 m/s five seconds inside the gap.
- Five seconds before reappearance, speed is only 0.88 m/s and 42.1% of observed players are below
  0.5 m/s.
- At reappearance speed rises to 1.66 m/s, reaching 2.18 m/s five seconds later.

This supports the observation that ball detection can return shortly before or around renewed
collective activity. It does not prove the precise instant at which play resumes.

In 147 intervals with a visible ball immediately before and after the gap, the apparent pitch
displacement between those boundary observations has a median of 10.20 m, an interquartile range
of 4.65–22.48 m, and a maximum of 87.17 m. Large jumps can reflect real movement during the unseen
period, a restart at a new location, stale detections, or tracking error. Linear interpolation is
therefore unsafe for most substantial gaps.

### 2.4 Smart Tagging overlap

The 150 gaps create 601 individual overlaps with Smart Tagging intervals. Counts of gaps that
overlap selected automatic tags are:

| Automatic tag | Missing intervals overlapping it |
|---|---:|
| Goal | 11 |
| Kickoff | 5 |
| Corner | 23 |
| Throw-in | 34 |
| Free kick | 6 |

These counts are not mutually exclusive. A long missing interval can span a goal tag, a set piece,
and several tactical phases. Detailed overlap rows preserve each tag's category, team, start/end,
and exact overlap duration, so no interval is forced into a single event class.

Tag overlap becomes more common with gap duration: 83% of <5-second gaps, 82% of 10–30-second gaps,
93% of 30–60-second gaps, and all four ≥60-second gaps overlap at least one tag. Because both
systems are automatic and use related visual inputs, this is not independent corroboration.

## 3. Smart Tagging categories

### 3.1 Category summaries

| Category | Intervals | Median duration | Ball visible | Mean player speed |
|---|---:|---:|---:|---:|
| Attacking transition | 24 | 15.44 s | 91.80% | 3.157 m/s |
| Defensive transition | 24 | 15.44 s | 91.80% | 3.157 m/s |
| Build up | 22 | 20.52 s | 92.73% | 2.207 m/s |
| Progression | 66 | 17.64 s | 92.80% | 2.265 m/s |
| Long ball | 29 | 12.40 s | 87.14% | 2.497 m/s |
| Ball in final third | 93 | 27.00 s | 69.13% | 2.095 m/s |
| Ball in the box | 72 | 17.44 s | 70.72% | 2.271 m/s |
| Players in the box | 97 | 27.68 s | 60.75% | 1.934 m/s |
| Creating chances | 30 | 30.00 s | 57.26% | 1.828 m/s |
| Set pieces | 59 | 35.00 s | 67.93% | 1.954 m/s |
| Goals | 4 | 169.26 s | 11.66% | 1.099 m/s |

The category CSV contains the corresponding team assignments, N/A counts, duration percentiles,
tracking coverage, slow/fast proportions, team-specific speed means, ball-coordinate medians, and
team-centroid coordinates.

### 3.2 Paired categories are not independent

Four pairs have exactly identical temporal masks (Jaccard = 1.0):

- Attacking transition / defensive transition
- Build up / defending in attacking third
- Progression / defending in middle third
- Creating chances / defending in defensive third

The team assignments are reversed across each pair. These appear to be two perspectives on the
same automatically detected phase, not two independent detections. Treating both as separate
features would double-count the same timing evidence.

Other large overlaps reflect football hierarchy: ball-in-final-third overlaps players-in-the-box
with Jaccard 0.547 and ball-in-the-box with Jaccard 0.438. Set pieces also overlap final-third,
box, chance-creation and players-in-box tags; those categories are not mutually exclusive.

### 3.3 Transitions, build-up, and progression

Transition intervals have the highest observed activity: mean speed 3.16 m/s, 30.1% of player
speeds above 4 m/s, and ball visibility of 91.8%. Build-up and progression also have strong ball
visibility (92.7–92.8%), high player coverage, and mean speeds of 2.21–2.26 m/s.

These tags are potentially useful as high-confidence *live tactical activity* evidence. They must
not be copied into possession labels: the team tag is automatic, phase boundaries can be wrong,
and complementary defensive categories duplicate the same time spans.

### 3.4 Goal intervals

The four `GOALS` intervals are unusually broad: median duration 169.26 seconds and maximum 250.72
seconds. Within their union, ball visibility is only 11.66%, mean speed is 1.10 m/s, mean tracked
players is 16.21, and 57.75% of frames have fewer than 20 tracked players.

The goal tag therefore describes a broad post-goal context rather than a precise goal instant. It
can include celebration, review, player reorganization, restart preparation, the kickoff, and
tracking degradation. This strongly supports the observation that a long gap can contain multiple
distinct situations.

## 4. Set-piece timing

Set-piece timing is template-like rather than event-exact:

- All 26 throw-ins, 12 corners, 10 goal kicks and 5 free kicks last exactly 35 seconds.
- The four post-goal kickoffs and two half-start kickoffs last exactly 5 seconds.
- This fixed-duration structure is evidence that the intervals are contextual windows, not exact
  restart timestamps.

| Type | n | Ball visible during tag | Visible at tag start | Visible at tag end | Median first-visible offset | Mean speed |
|---|---:|---:|---:|---:|---:|---:|
| Corner | 12 | 59.14% | 0% | 83.33% | +6.76 s | 2.00 m/s |
| Free kick | 5 | 59.41% | 20% | 60% | +6.12 s | 1.84 m/s |
| Goal kick | 10 | 73.33% | 80% | 70% | 0.00 s | 1.94 m/s |
| Throw-in | 26 | 74.61% | 30.77% | 69.23% | +4.26 s | 1.99 m/s |
| Kickoff after goal | 4 | 27.00% | 0% | 75% | +3.28 s* | 0.74 m/s |
| Half-start kickoff | 2 | 64.00% | 50% | 100% | +1.80 s | 0.99 m/s |

\*Three of four post-goal kickoff windows contain a visible-ball observation; the fourth does not.

Corners provide the clearest preparation-to-restart pattern: the ball is missing at every tag
start, first appears a median 6.76 seconds later, and is visible at 83% of tag ends. Throw-ins and
free kicks are more variable. Goal-kick tags often begin with the ball already visible, so ball
recovery is not a universal restart marker.

The overall evidence indicates that most set-piece tags span restart preparation and subsequent
play, while kickoff tags are shorter and closer to the restart itself. Assigning an entire set-piece
interval to dead ball would be incorrect.

## 5. Team activity

| Team | Mean tracked players | Weighted mean speed | Median frame-median speed | P75 | P90 | Below 0.5 m/s | Above 4 m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Barcelona | 10.20 | 1.832 m/s | 1.402 | 2.50 | 3.75 | 13.36% | 8.91% |
| Athletic Club | 10.29 | 1.800 m/s | 1.370 | 2.25 | 3.75 | 14.01% | 8.68% |

The teams are extremely similar at match scale. Barcelona's weighted mean is only 0.032 m/s higher.
Any useful team-specific activity signal is therefore likely contextual and temporal rather than a
global team threshold.

The strongest activity relationship is with ball visibility, not team identity. Missing-ball
frames have lower speeds, more near-stationary players, fewer high-speed observations and lower
player-count coverage. Long-gap aligned curves also show that tracking coverage can decline while
activity declines, so speed and coverage should always be modeled jointly in future work.

## 6. Assessment of the starting hypotheses

### Supported

- **Ball tracking frequently disappears around stoppage-like contexts.** Long gaps strongly
  associate with low collective activity and set-piece/goal tags.
- **Ball detection can recover before collective activity fully resumes.** For ≥30-second gaps,
  the ball returns when mean speed is about 1.66 m/s; activity rises to 2.18 m/s five seconds later.
- **Long missing intervals contain multiple situations.** The longest gap overlaps ten tags and the
  goal intervals last several minutes.
- **Players often reduce movement during long gaps.** The relationship is monotonic across the main
  duration groups and especially large for 30–60-second gaps.
- **Set-piece timing is broad and type-dependent.** Fixed 35-second windows and recovery several
  seconds after the start show that tags are not exact restart markers.

### Qualified or contradicted

- **Not every ball gap resembles a stoppage.** Sub-five-second gaps retain high activity, and some
  short gaps occur during transition-like movement.
- **Missing ball alone is not a dead-ball indicator.** Many tactical phase tags overlap missing
  frames, and 30-second gaps can retain 20–22 players with substantial movement.
- **Set-piece tags do not uniformly start before ball recovery.** Most goal kicks already have the
  ball visible at tag start, unlike corners.
- **Smart Tagging categories are not independent.** Several attacking/defending pairs have exactly
  identical time spans.

## 7. Potentially useful future signals

These are research candidates, not implemented rules:

- Gap duration and time since last reliable ball observation.
- Collective speed level, its slope, and the proportion below 0.5 m/s.
- Reacceleration after ball reappearance rather than reappearance alone.
- Player-count and speed-observation coverage as explicit confidence variables.
- Set-piece type with type-specific temporal interpretation.
- High-activity transition/build-up/progression tags as evidence against a stoppage state.
- Ball boundary coordinates, out-of-bounds status, and the displacement between disappearance and
  reappearance, with uncertainty increasing sharply over long gaps.
- Agreement between multiple signals only after accounting for their shared automatic origin.

## 8. Uncertainty and limitations

- No video was available in the audited data directory, so timing could not be visually validated.
- ATD and Smart Tagging may share upstream detections; their errors are correlated.
- Player speed availability falls during some intervals, making apparent inactivity partly a
  measurement problem.
- Frame-level tag analyses count overlapping frames in multiple categories.
- Pitch-coordinate summaries are not direction-normalized between halves; they are descriptive,
  not attacking-direction comparisons.
- Boundary comparisons use five-second windows and the 0.5/4 m/s thresholds are descriptive.
- The analysis does not establish when the referee declares the ball live or dead.

## 9. Recommended investigations before modeling

1. Visually inspect a stratified sample when the tactical video is available: short/high-activity,
   long/full-tracking, long/degraded-tracking, and each set-piece type.
2. Test sensitivity to context windows of 2, 5, 10 and 20 seconds without using possession labels.
3. Separate camera/tracker failure from football inactivity using player count, homography error,
   centroid stability and spatial dispersion.
4. Analyze whether ball recovery followed by sustained acceleration is more stable than the first
   recovered frame.
5. Examine set-piece timing per instance, especially the outliers in throw-in and free-kick
   first-visible offsets.
6. Collapse perfectly paired Smart Tagging categories before any future feature engineering.
7. Investigate ball-coordinate plausibility at disappearance/reappearance and reject isolated
   one-frame recoveries before treating a gap as ended.
8. Repeat the analysis on other matches if accessible; all present findings come from one game.

Only after these checks should a temporal model or rule system be designed. This EDA intentionally
produces no possession predictions and no frame-specific H/A/D assignments.



---

## Archived source: `FOCUSED_UNCERTAINTY_REPORT.md`

# Focused Investigation of EDA Uncertainties

**Scope:** tactical tracking, Smart Tagging, and existing local diagnostic outputs only  
**Possession ground truth accessed:** no  
**Predictions or H/A/D labels produced:** no

This report tests the main unresolved explanations from `EDA_REPORT.md`. It is deliberately
narrower than the original EDA: tracking-coverage confounding, individual set-piece behavior,
paired-tag construction, and observation-quality artifacts.

## 1. Methods and uncertainty

Player speed is the ATD speed attached to each finite player observation. Frame-weighted estimates
weight every player-speed observation equally. Interval-weighted estimates first calculate one
mean per maximal ball-visible or ball-missing run and then weight each run equally.

Tracking coverage is stratified exactly as requested:

- 21–22 observed players
- 18–20
- 11–17
- fewer than 11

Frames with more than 22 tracks are excluded from the stratified comparison; there are 3,749 such
scored frames. They are likely to contain duplicate tracks and do not fit the requested strata.

Frame-weighted uncertainty uses a reproducible cluster bootstrap with 2,000 resamples and
non-overlapping 30-second temporal blocks. Interval-weighted uncertainty resamples whole runs.
These intervals address short-range temporal dependence better than treating frames as independent,
but they are not match-to-match confidence intervals: the dataset still contains only one match.

## 2. Does the speed reduction survive coverage control?

### 2.1 Combined halves: frame-weighted comparison

| Observed players | Ball visible: frames / speed observations | Visible speed (95% block CI) | Ball missing: frames / speed observations | Missing speed (95% block CI) | Difference, visible − missing (95% CI) |
|---|---:|---:|---:|---:|---:|
| 21–22 | 83,087 / 1,685,797 | 2.205 (2.131–2.282) | 36,150 / 723,808 | 1.358 (1.257–1.460) | **0.848 (0.724–0.974)** |
| 18–20 | 7,808 / 141,092 | 2.043 (1.801–2.291) | 13,447 / 232,260 | 0.982 (0.888–1.085) | **1.061 (0.811–1.310)** |
| 11–17 | 1,590 / 21,076 | 1.358 (1.159–1.586) | 7,860 / 93,605 | 0.930 (0.824–1.045) | **0.434 (0.220–0.658)** |
| <11 | 257 / 848 | 1.702 (1.445–2.160) | 4,909 / 17,715 | 0.746 (0.542–1.090) | 0.987 (0.498–1.547) |

The 21–22 stratum is the critical control. It retains almost the full player set, yet missing-ball
frames are slower by approximately 0.85 m/s. The confidence interval remains well above zero. The
association is therefore not merely an artifact of having fewer tracked players when the ball is
missing.

The <11 comparison is not reliable despite its positive interval: only 257 visible-ball frames,
848 speed observations and two temporal blocks are available. It should not be used as evidence.

### 2.2 Separate halves

| Half | Observed players | Visible − missing speed | 95% block-bootstrap CI |
|---:|---|---:|---:|
| 1 | 21–22 | 0.723 m/s | 0.519–0.913 |
| 2 | 21–22 | 0.952 m/s | 0.803–1.099 |
| 1 | 18–20 | 1.087 m/s | 0.742–1.445 |
| 2 | 18–20 | 1.013 m/s | 0.653–1.379 |
| 1 | 11–17 | 0.847 m/s | 0.474–1.448 |
| 2 | 11–17 | 0.279 m/s | 0.042–0.518 |

The speed reduction persists in both halves for the two best-supported coverage strata. The
second-half 11–17 result is much weaker and based on only seven visible-state temporal blocks.
There are no visible-ball frames with fewer than 11 players in the second half, so that comparison
is impossible.

### 2.3 Interval-weighted comparison

| Observed players | Visible runs: n / mean speed (95% interval CI) | Missing runs: n / mean speed (95% interval CI) |
|---|---:|---:|
| 21–22 | 135 / 2.064 (1.953–2.174) | 139 / 1.595 (1.469–1.725) |
| 18–20 | 61 / 1.966 (1.725–2.203) | 66 / 1.250 (1.110–1.422) |
| 11–17 | 14 / 1.739 (1.314–2.385) | 25 / 1.011 (0.873–1.186) |
| <11 | 2 / 1.802 (1.445–2.160) | 8 / 1.138 (0.682–1.721) |

Equal weighting of intervals reduces the 21–22 difference from 0.85 to 0.47 m/s, showing that long
low-activity intervals contribute substantially to the frame-weighted effect. Nevertheless, the
direction remains the same. The 18–20 difference also remains large. The two lower-coverage
comparisons are too sparse for strong inference.

### 2.4 Gap duration at matched coverage

Within the 21–22-player stratum, frame-weighted speed decreases with gap duration:

| Gap duration | Contributing intervals | Frames in stratum | Mean speed |
|---|---:|---:|---:|
| <5 s | 14 | 49 | 2.865 m/s |
| 5–10 s | 52 | 8,277 | 1.814 m/s |
| 10–30 s | 55 | 18,285 | 1.299 m/s |
| 30–60 s | 14 | 5,642 | 1.146 m/s |
| ≥60 s | 4 | 3,897 | 0.921 m/s |

The <5-second estimate contains only 49 frames and should not be overinterpreted. From 5 seconds
upward, however, the decline is monotonic despite holding player-count coverage at 21–22. This is
strong evidence that long ball gaps preferentially occur during genuinely lower measured activity,
not just broader tracker failure.

Coverage still matters. Within 30–60-second gaps, mean speed is 1.146 m/s with 21–22 players but
0.768 m/s with 18–20. At lower counts the samples fragment across few intervals and estimates are
unstable. Player count must remain an explicit confidence variable.

### Finding

**Verified:** the activity reduction associated with missing ball detections survives matching on
player-count coverage and appears in both halves.  
**Not verified:** that the reduction means dead ball. Slow live play, restart preparation, camera
conditions and shared tracking failure remain alternative explanations.

## 3. Individual set-piece visibility patterns

For every set-piece interval, the analysis decomposes ball visibility into maximal observed and
missing sequences. A sustained observed sequence is at least one second; this is a descriptive
reporting threshold, not an optimized decision rule. Transition activity compares the two seconds
before and after each disappearance or reappearance.

### 3.1 Sequence summaries

| Type | n intervals | Mean observed runs | Mean missing runs | Mean disappearances | Mean reappearances | Median longest missing run | Modal pattern share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Corner | 12 | 1.75 | 1.92 | 0.92 | 1.75 | 8.98 s | 41.7% |
| Throw-in | 26 | 1.38 | 1.38 | 0.69 | 1.08 | 6.58 s | 30.8% |
| Goal kick | 10 | 1.60 | 1.10 | 0.90 | 0.80 | 6.84 s | 30.0% |
| Free kick | 5 | 1.00 | 1.20 | 0.40 | 0.80 | 11.72 s | 60.0% |
| Kickoff after goal | 4 | 0.75 | 1.00 | 0.00 | 0.75 | 3.30 s | 75.0% |
| Half-start kickoff | 2 | 1.00 | 0.50 | 0.00 | 0.50 | 1.80 s | 50.0% |

The modal pattern is defined by disappearance count, reappearance count, visibility at tag start
and visibility at tag end. Corners, throw-ins and goal kicks have modal shares of only 30–42%, so
their visibility behavior is not internally consistent enough for one deterministic rule.

Corners are particularly multi-phase: 21 reappearances occur within 12 intervals. Throw-ins contain
up to three disappearances and three reappearances in one 35-second tag; goal kicks contain up to
two of each. This directly confirms that a set-piece interval can contain preparation, recovery,
continued play and renewed loss of the ball track.

One post-goal kickoff contains no observed ball at all. The other three contain a sustained
observation. The kickoff pattern looks more consistent than other types, but n=4 is far too small
for a general claim.

### 3.2 Isolated versus sustained observations

Across the complete scored blocks there are **no one-frame observed-ball runs**. There are only
three observed runs shorter than one second; the shortest lasts 0.64 seconds. Thus the gap structure
is not being fragmented by single-frame ball detections.

Within set-piece windows, one throw-in has a one-frame observed fragment and two local observed
fragments are shorter than one second (one corner and one throw-in). None is globally isolated:
the apparent fragment is produced by clipping a longer observed run at the tag boundary. This is an
important distinction—tag-window truncation must not be confused with detector flicker.

### 3.3 Activity around reappearance

| Type | Reappearance events | Mean speed before | Mean speed after | Mean change |
|---|---:|---:|---:|---:|
| Corner | 21 | 1.627 | 1.992 | +0.366 m/s |
| Throw-in | 28 | 1.384 | 1.707 | +0.323 m/s |
| Goal kick | 8 | 1.752 | 2.117 | +0.366 m/s |
| Free kick | 4 | 1.181 | 1.659 | +0.479 m/s |
| Kickoff after goal | 3 | 0.512 | 0.968 | +0.456 m/s |
| Half-start kickoff | 1 | 0.374 | 1.250 | +0.876 m/s |

Every category has a positive mean change around reappearance, supporting an association between
ball recovery and collective acceleration. Individual events remain variable: some reappearances
have negative speed changes, and the kickoff samples are tiny. Reappearance is therefore a useful
temporal cue but cannot be equated with the referee's restart or live play.

Detailed per-instance sequences are in `set_piece_sequences.csv`; every transition and its local
activity change is in `set_piece_transition_activity.csv`.

### Finding

**Verified:** set-piece intervals generally contain multiple observation states, and most categories
are internally heterogeneous.  
**Supported but not verified:** ball reappearance often accompanies acceleration.  
**Weakened:** the hypothesis that one set-piece tag has one standard missing-to-visible pattern.

## 4. Exact verification of paired Smart Tagging categories

All intervals were matched by exact integer start and end frame. Results:

| Pair | Exact pairs | Opposite Barcelona/Athletic assignments | Paired N/A | Missing counterpart | Same-team contradiction |
|---|---:|---:|---:|---:|---:|
| Attacking transition / defensive transition | 24 | 24 | 0 | 0 | 0 |
| Build up / defending in attacking third | 22 | 22 | 0 | 0 | 0 |
| Progression / defending in middle third | 66 | 66 | 0 | 0 | 0 |
| Creating chances / defending in defensive third | 30 | 28 | 2 | 0 | 0 |

Across 142 exact pairs, 140 assign the two opposing teams and two assign `N/A` to both sides. No
interval is missing a counterpart and none assigns the same named team to both perspectives.

### Finding

**Verified:** these pairs encode one underlying temporal phase from complementary team perspectives.
They are structurally coupled, not independent evidence. A later feature system should represent
each pair once, preserving the attacking/defending team relationship rather than counting both
tags as separate confirmations.

## 5. Observation-quality checks

### 5.1 Observed-ball runs

The scored blocks contain 149 observed-ball runs:

- Median duration: 13.80 seconds
- Interquartile range: 5.48–30.32 seconds
- Minimum: 0.64 seconds
- Maximum: 146.68 seconds
- One-frame runs: 0
- Runs shorter than one second: 3

The previous definition of a missing interval is therefore not materially distorted by isolated
single-frame recoveries.

### 5.2 Consecutive-frame coordinate jumps

There are 94,065 consecutive observed-frame pairs. Using descriptive thresholds:

- 119 steps are at least 2 m in one frame (implied speed ≥50 m/s).
- 13 are at least 3 m (≥75 m/s).
- 7 are at least 4 m (≥100 m/s).
- None reaches 5 m.
- Maximum: 4.384 m in one frame, or approximately 109.6 m/s.

These 119 steps occur in only nine consecutive clusters, rather than as 119 isolated teleports.
Player count is unchanged on 115 of them, and 70 occur with 21–22 players tracked on both frames.
Only two involve an out-of-bounds ball coordinate. That pattern points to a ball-specific smooth
trajectory/interpolation artifact rather than wholesale player-tracking collapse.

Six clusters totaling 75 of the 119 steps occur between frames 111,245 and 112,038, before the
documented second-half kickoff at frame 113,742. The largest cluster occurs at frames
179,905–179,922 and contains successive steps above 4 m. Speeds above 100 m/s are not plausible
football movement, so at least those extreme clusters should be treated as measurement artifacts.
The 2 m threshold alone is not a validity rule: some high-speed ball motion may be genuine.

### 5.3 Continuity around gaps

| Measurement | n | Median | P25–P75 | Maximum |
|---|---:|---:|---:|---:|
| Last observed step before disappearance | 149 | 0.063 m | 0.023–0.158 | 1.134 m |
| First observed step after reappearance | 148 | 0.466 m | 0.212–0.684 | 4.384 m |
| Last-before to first-after gap displacement | 147 | 10.198 m | 4.651–22.485 | 87.175 m |

The first post-recovery step is substantially larger than the last pre-loss step. That asymmetry
suggests that recovery can begin with a less stable coordinate track. It also reinforces the need
for a sustained-observation check before trusting the recovered ball location.

Player count changes from immediately before a gap to the recovery frame have median zero, mean
−0.42, and range −6 to +4. Some gaps clearly include coverage changes, but the typical boundary
does not. Combined with the coverage-stratified results, player-count loss explains part—but not
all—of the activity curve.

### Finding

**Verified:** there are no isolated one-frame recoveries in the scored blocks, but there are several
smooth, physically implausible high-speed coordinate clusters.  
**Supported:** recovered coordinates are initially less continuous than coordinates immediately
before loss.  
**Uncertain:** moderate 2–3 m steps cannot be separated into genuine kicked-ball motion and tracker
artifacts without video or an independent ball source.

## 6. Practical implications

### Dead-ball research

- Missing ball plus low activity remains informative even after controlling for player coverage.
- Neither signal is sufficient alone: short gaps can be high activity and long gaps can retain
  nearly complete tracking.
- A future detector should distinguish ball absence, collective deceleration, low player coverage,
  set-piece context and sustained post-recovery acceleration.
- Reappearance should be treated as a candidate boundary with a stabilization period, not as the
  live-ball instant.
- Set-piece logic must be type-specific and probabilistic. Corners and throw-ins in particular can
  contain several visibility cycles.

### Team-possession research

- Player activity indicates phase intensity, not which team controls the ball.
- Paired Smart Tags must be collapsed to one phase representation; otherwise the same automatic
  evidence is double-counted.
- The team assignments in paired tags are internally consistent, but this does not validate them
  against true possession.
- Large gap-boundary displacement and early recovery instability weaken any inference based solely
  on the last and first observed ball positions.
- Missing-ball intervals should carry explicit confidence and coverage features rather than being
  filled by deterministic team rules.

No H/A/D labels or possession decisions are generated by this analysis.

## 7. Statistical limitations and confounders

- This is one match; bootstrap intervals quantify within-match temporal variability only.
- Thirty-second blocks reduce but do not eliminate temporal dependence.
- Visible and missing periods differ in football context, so the coverage-controlled association is
  not a causal effect of ball visibility.
- Interval weighting changes the estimand and gives a 0.04-second run the same weight as a long run.
- Low-coverage visible samples are extremely sparse, especially in the second half.
- Speed is itself an automatic tracking output and may degrade with image conditions.
- Smart Tagging and ATD may share upstream information, so agreement is correlated evidence.
- Set-piece transition windows are clipped to tag boundaries; local sequence fragments can be
  boundary artifacts.
- Physically implausible jump thresholds are descriptive and were not tuned against labels.

## 8. Remaining unanswered questions

1. Do visually confirmed stoppages show the same coverage-controlled activity profile?
2. How long after ball recovery does the coordinate become spatially stable?
3. Are moderate high-speed clusters genuine long passes/shots or smooth tracker interpolation?
4. Does the activity/recovery pattern generalize to other matches and camera systems?
5. Can homography error or image-based camera-state indicators separate tracker failure from
   football inactivity?
6. Which point inside each broad set-piece window best corresponds to the observable restart?
7. How should simultaneous, structurally paired tags be represented without double-counting?
8. Can player-shape stability add information beyond count and speed during long ball gaps?

The next justified step is targeted visual validation of stratified examples—not possession-model
implementation or parameter tuning on competition ground truth.



---

## Archived source: `UNTAGGED_FRAMES_REPORT.md`

# Smart Tag Coverage Gaps During Game Time

**Reference:** tactical video at 25 fps  
**Game-time windows:** first half `00:07:26.000–00:59:22.000`; second half `01:15:49.000–02:06:40.000`  
**Possession ground truth used:** no

An untagged run is a maximal consecutive group of frames with no active Smart Tag. Runs
are calculated independently within each half and never joined across halftime. Smart Tag
overlap is collapsed to a Boolean union, so a frame with several simultaneous tags counts
once as tagged.

## Coverage

| Measure | Frames | Percentage of game-time frames |
|---|---:|---:|
| Game-time frames | 154,175 | 100.00% |
| At least one Smart Tag | 130,668 | 84.75% |
| No Smart Tag | 23,507 | 15.25% |

### Coverage by half

| Half | Game-time frames | Tagged frames | Untagged frames | Untagged share | Untagged runs | Median run |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77,900 | 66,509 | 11,391 | 14.62% | 34 | 259 frames (10.36 s) |
| 2 | 76,275 | 64,159 | 12,116 | 15.88% | 35 | 222 frames (8.88 s) |

## Longest untagged run

The longest run is **run 25 in half 1**, containing **1,867 frames (74.68 seconds)**.

- Video interval: **`00:46:54.600–00:48:09.280`**
- Frame-exact 25 fps timecode: **`00:46:54:15–00:48:09:07`**
- Frame interval: **`[70365, 72232)`**
- Nearest previous tag(s): #40 PLAYERS IN THE BOX · Atletic Club [00:46:36.240–00:46:54.600]
- Nearest following tag(s): #41 PLAYERS IN THE BOX · Atletic Club [00:48:09.280–00:49:32.960]

The interval is half-open: the end timestamp is the first frame after the untagged run.

## Run-length findings

The 23,507 frames form **69 contiguous untagged runs**. Median run length is **232 frames (9.28 s)**; maximum is **1,867 frames (74.68 s)**.

### Cumulative thresholds

| Condition | Runs | Frames in those runs | Share of all untagged frames |
|---|---:|---:|---:|
| Exactly one frame | 0 | 0 | 0.00% |
| More than 4 frames | 68 | 23,503 | 99.98% |
| More than 8 frames | 67 | 23,496 | 99.95% |
| More than 12 frames | 67 | 23,496 | 99.95% |
| More than 25 frames | 64 | 23,441 | 99.72% |

### Mutually exclusive bins

| Run length | Runs | Frames |
|---|---:|---:|
| 1 frames | 0 | 0 |
| 2–4 frames | 1 | 4 |
| 5–8 frames | 1 | 7 |
| 9–12 frames | 0 | 0 |
| 13–25 frames | 3 | 55 |
| >25 frames | 64 | 23,441 |

There are no isolated one-frame coverage holes. Sixty-four of 69 runs exceed 25 frames
and contain 99.72% of all untagged frames. Missing Smart Tag coverage therefore occurs in
substantial continuous blocks, not frame-level flicker.

## Nearest tag categories around gaps

Several tags can share the nearest boundary because complementary attacking/defending tags
have identical intervals. Counts below count tag instances, not untagged runs.
All 69 runs have at least one immediately preceding tag. Sixty-seven have an immediately
following tag; runs 34 and 69 end at the supplied half cutoffs, so no within-half following
tag can be reported for them. Twenty-six runs have multiple preceding tags and 28 have
multiple following tags because of simultaneous paired categories.

| Previous tag category | Boundary occurrences | Next tag category | Boundary occurrences |
|---|---:|---|---:|
| SET PIECES | 17 | DEFENDING IN MIDDLE THIRD | 21 |
| PLAYERS IN THE BOX | 13 | PROGRESSION | 21 |
| DEFENDING IN MIDDLE THIRD | 12 | SET PIECES | 13 |
| PROGRESSION | 12 | LONG BALL | 12 |
| LONG BALL | 10 | BALL IN FINAL THIRD | 7 |
| CREATING CHANCES | 6 | PLAYERS IN THE BOX | 5 |
| DEFENDING IN DEFENSIVE THIRD | 6 | BUILD UP | 4 |
| ATTACKING TRANSITION | 5 | DEFENDING IN ATTACKING THIRD | 4 |
| DEFENSIVE TRANSITION | 5 | ATTACKING TRANSITION | 2 |
| BALL IN FINAL THIRD | 3 | DEFENSIVE TRANSITION | 2 |
| BUILD UP | 3 | BALL IN THE BOX | 2 |
| DEFENDING IN ATTACKING THIRD | 3 | CREATING CHANCES | 1 |
|  |  | DEFENDING IN DEFENSIVE THIRD | 1 |

## All untagged runs and neighboring tags

`Previous` contains every tag sharing the latest end before the run. `Next` contains every
tag sharing the earliest start after it. Searches are restricted to the same half.

| Run | Half | Frames `[start,end)` | Video interval | Duration | Previous nearest tag(s) | Next nearest tag(s) |
|---:|---:|---:|---|---:|---|---|
| 1 | 1 | `12851–13263` | `00:08:34.040–00:08:50.520` | 16.48 s | #349 ATTACKING TRANSITION · Atletic Club [00:08:22.960–00:08:34.040]<br>#573 DEFENSIVE TRANSITION · Barcelona [00:08:22.960–00:08:34.040] | #526 DEFENDING IN MIDDLE THIRD · Atletic Club [00:08:50.520–00:09:12.160]<br>#302 PROGRESSION · Barcelona [00:08:50.520–00:09:12.160] |
| 2 | 1 | `14849–15398` | `00:09:53.960–00:10:15.920` | 21.96 s | #548 DEFENDING IN MIDDLE THIRD · Atletic Club [00:09:40.800–00:09:53.960]<br>#324 PROGRESSION · Barcelona [00:09:40.800–00:09:53.960] | #52 PLAYERS IN THE BOX · Barcelona [00:10:15.920–00:10:38.600] |
| 3 | 1 | `15965–16843` | `00:10:38.600–00:11:13.720` | 35.12 s | #52 PLAYERS IN THE BOX · Barcelona [00:10:15.920–00:10:38.600] | #534 DEFENDING IN MIDDLE THIRD · Barcelona [00:11:13.720–00:11:34.040]<br>#310 PROGRESSION · Atletic Club [00:11:13.720–00:11:34.040] |
| 4 | 1 | `19195–19535` | `00:12:47.800–00:13:01.400` | 13.60 s | #454 SET PIECES · Barcelona · Goal Kick [00:12:12.800–00:12:47.800] | #442 SET PIECES · N/A · Throw-in [00:13:01.400–00:13:36.400] |
| 5 | 1 | `20410–20417` | `00:13:36.400–00:13:36.680` | 0.28 s | #442 SET PIECES · N/A · Throw-in [00:13:01.400–00:13:36.400] | #545 DEFENDING IN MIDDLE THIRD · Atletic Club [00:13:36.680–00:14:08.440]<br>#321 PROGRESSION · Barcelona [00:13:36.680–00:14:08.440] |
| 6 | 1 | `21211–21576` | `00:14:08.440–00:14:23.040` | 14.60 s | #545 DEFENDING IN MIDDLE THIRD · Atletic Club [00:13:36.680–00:14:08.440]<br>#321 PROGRESSION · Barcelona [00:13:36.680–00:14:08.440] | #448 SET PIECES · N/A · Throw-in [00:14:23.040–00:14:58.040] |
| 7 | 1 | `27040–27236` | `00:18:01.600–00:18:09.440` | 7.84 s | #482 CREATING CHANCES · Barcelona [00:17:31.600–00:18:01.600]<br>#624 DEFENDING IN DEFENSIVE THIRD · Atletic Club [00:17:31.600–00:18:01.600] | #389 LONG BALL · Atletic Club [00:18:09.440–00:18:17.520] |
| 8 | 1 | `27438–27794` | `00:18:17.520–00:18:31.760` | 14.24 s | #389 LONG BALL · Atletic Club [00:18:09.440–00:18:17.520] | #437 SET PIECES · N/A · Throw-in [00:18:31.760–00:19:06.760] |
| 9 | 1 | `30681–30962` | `00:20:27.240–00:20:38.480` | 11.24 s | #395 LONG BALL · Atletic Club [00:20:14.040–00:20:27.240] | #544 DEFENDING IN MIDDLE THIRD · Atletic Club [00:20:38.480–00:20:53.760]<br>#320 PROGRESSION · Barcelona [00:20:38.480–00:20:53.760] |
| 10 | 1 | `31344–31512` | `00:20:53.760–00:21:00.480` | 6.72 s | #544 DEFENDING IN MIDDLE THIRD · Atletic Club [00:20:38.480–00:20:53.760]<br>#320 PROGRESSION · Barcelona [00:20:38.480–00:20:53.760] | #533 DEFENDING IN MIDDLE THIRD · Atletic Club [00:21:00.480–00:21:26.960]<br>#309 PROGRESSION · Barcelona [00:21:00.480–00:21:26.960] |
| 11 | 1 | `32174–32431` | `00:21:26.960–00:21:37.240` | 10.28 s | #533 DEFENDING IN MIDDLE THIRD · Atletic Club [00:21:00.480–00:21:26.960]<br>#309 PROGRESSION · Barcelona [00:21:00.480–00:21:26.960] | #412 LONG BALL · Barcelona [00:21:37.240–00:21:54.200] |
| 12 | 1 | `33224–33807` | `00:22:08.960–00:22:32.280` | 23.32 s | #347 ATTACKING TRANSITION · Barcelona [00:21:53.400–00:22:08.960]<br>#571 DEFENSIVE TRANSITION · Atletic Club [00:21:53.400–00:22:08.960] | #156 BALL IN FINAL THIRD · Barcelona [00:22:32.280–00:22:58.000] |
| 13 | 1 | `38424–38450` | `00:25:36.960–00:25:38.000` | 1.04 s | #410 LONG BALL · Barcelona [00:25:18.400–00:25:36.960] | #504 DEFENDING IN MIDDLE THIRD · Atletic Club [00:25:38.000–00:26:06.000]<br>#280 PROGRESSION · Barcelona [00:25:38.000–00:26:06.000] |
| 14 | 1 | `49756–49814` | `00:33:10.240–00:33:12.560` | 2.32 s | #2 SET PIECES · N/A · Kick Off (After Goal) [00:33:05.240–00:33:10.240] | #540 DEFENDING IN MIDDLE THIRD · Barcelona [00:33:12.560–00:33:38.360]<br>#316 PROGRESSION · Atletic Club [00:33:12.560–00:33:38.360] |
| 15 | 1 | `50847–51000` | `00:33:53.880–00:34:00.000` | 6.12 s | #34 PLAYERS IN THE BOX · Atletic Club [00:33:16.480–00:33:53.880] | #453 SET PIECES · Barcelona · Goal Kick [00:34:00.000–00:34:35.000] |
| 16 | 1 | `53554–54114` | `00:35:42.160–00:36:04.560` | 22.40 s | #492 CREATING CHANCES · Barcelona [00:35:12.160–00:35:42.160]<br>#634 DEFENDING IN DEFENSIVE THIRD · Atletic Club [00:35:12.160–00:35:42.160] | #391 LONG BALL · Atletic Club [00:36:04.560–00:36:14.880] |
| 17 | 1 | `54372–54481` | `00:36:14.880–00:36:19.240` | 4.36 s | #391 LONG BALL · Atletic Club [00:36:04.560–00:36:14.880] | #354 ATTACKING TRANSITION · Barcelona [00:36:19.240–00:36:31.520]<br>#578 DEFENSIVE TRANSITION · Atletic Club [00:36:19.240–00:36:31.520] |
| 18 | 1 | `54788–54887` | `00:36:31.520–00:36:35.480` | 3.96 s | #354 ATTACKING TRANSITION · Barcelona [00:36:19.240–00:36:31.520]<br>#578 DEFENSIVE TRANSITION · Atletic Club [00:36:19.240–00:36:31.520] | #516 DEFENDING IN MIDDLE THIRD · Atletic Club [00:36:35.480–00:37:02.440]<br>#292 PROGRESSION · Barcelona [00:36:35.480–00:37:02.440] |
| 19 | 1 | `55862–56014` | `00:37:14.480–00:37:20.560` | 6.08 s | #65 PLAYERS IN THE BOX · Barcelona [00:36:49.200–00:37:14.480] | #560 DEFENDING IN MIDDLE THIRD · Barcelona [00:37:20.560–00:37:38.240]<br>#336 PROGRESSION · Atletic Club [00:37:20.560–00:37:38.240] |
| 20 | 1 | `62137–62411` | `00:41:25.480–00:41:36.440` | 10.96 s | #3 SET PIECES · N/A · Kick Off (After Goal) [00:41:20.480–00:41:25.480] | #376 BUILD UP · Barcelona [00:41:36.440–00:41:54.600]<br>#600 DEFENDING IN ATTACKING THIRD · Atletic Club [00:41:36.440–00:41:54.600] |
| 21 | 1 | `63590–63822` | `00:42:23.600–00:42:32.880` | 9.28 s | #562 DEFENDING IN MIDDLE THIRD · Atletic Club [00:42:09.240–00:42:23.600]<br>#338 PROGRESSION · Barcelona [00:42:09.240–00:42:23.600] | #527 DEFENDING IN MIDDLE THIRD · Barcelona [00:42:32.880–00:42:47.800]<br>#303 PROGRESSION · Atletic Club [00:42:32.880–00:42:47.800] |
| 22 | 1 | `64195–64667` | `00:42:47.800–00:43:06.680` | 18.88 s | #527 DEFENDING IN MIDDLE THIRD · Barcelona [00:42:32.880–00:42:47.800]<br>#303 PROGRESSION · Atletic Club [00:42:32.880–00:42:47.800] | #535 DEFENDING IN MIDDLE THIRD · Barcelona [00:43:06.680–00:43:18.560]<br>#311 PROGRESSION · Atletic Club [00:43:06.680–00:43:18.560] |
| 23 | 1 | `66433–66522` | `00:44:17.320–00:44:20.880` | 3.56 s | #432 SET PIECES · N/A · Throw-in [00:43:42.320–00:44:17.320] | #511 DEFENDING IN MIDDLE THIRD · Atletic Club [00:44:20.880–00:44:42.560]<br>#287 PROGRESSION · Barcelona [00:44:20.880–00:44:42.560] |
| 24 | 1 | `67627–68185` | `00:45:05.080–00:45:27.400` | 22.32 s | #117 BALL IN FINAL THIRD · Atletic Club [00:44:25.560–00:45:05.080] | #557 DEFENDING IN MIDDLE THIRD · Atletic Club [00:45:27.400–00:45:39.080]<br>#333 PROGRESSION · Barcelona [00:45:27.400–00:45:39.080] |
| 25 | 1 | `70365–72232` | `00:46:54.600–00:48:09.280` | 74.68 s | #40 PLAYERS IN THE BOX · Atletic Club [00:46:36.240–00:46:54.600] | #41 PLAYERS IN THE BOX · Atletic Club [00:48:09.280–00:49:32.960] |
| 26 | 1 | `75582–75597` | `00:50:23.280–00:50:23.880` | 0.60 s | #4 SET PIECES · N/A · Kick Off (After Goal) [00:50:18.280–00:50:23.280] | #539 DEFENDING IN MIDDLE THIRD · Atletic Club [00:50:23.880–00:50:41.360]<br>#315 PROGRESSION · Barcelona [00:50:23.880–00:50:41.360] |
| 27 | 1 | `76034–76072` | `00:50:41.360–00:50:42.880` | 1.52 s | #539 DEFENDING IN MIDDLE THIRD · Atletic Club [00:50:23.880–00:50:41.360]<br>#315 PROGRESSION · Barcelona [00:50:23.880–00:50:41.360] | #166 BALL IN FINAL THIRD · Barcelona [00:50:42.880–00:51:08.080] |
| 28 | 1 | `77029–77259` | `00:51:21.160–00:51:30.360` | 9.20 s | #392 LONG BALL · Atletic Club [00:51:12.080–00:51:21.160] | #70 PLAYERS IN THE BOX · Barcelona [00:51:30.360–00:51:48.400] |
| 29 | 1 | `78050–78794` | `00:52:02.000–00:52:31.760` | 29.76 s | #71 PLAYERS IN THE BOX · Barcelona [00:51:44.000–00:52:02.000] | #213 BALL IN THE BOX · Atletic Club [00:52:31.760–00:52:49.120] |
| 30 | 1 | `79425–79686` | `00:52:57.000–00:53:07.440` | 10.44 s | #42 PLAYERS IN THE BOX · Atletic Club [00:52:31.880–00:52:57.000] | #552 DEFENDING IN MIDDLE THIRD · Atletic Club [00:53:07.440–00:53:20.040]<br>#328 PROGRESSION · Barcelona [00:53:07.440–00:53:20.040] |
| 31 | 1 | `82600–82748` | `00:55:04.000–00:55:09.920` | 5.92 s | #467 CREATING CHANCES · Barcelona [00:54:34.000–00:55:04.000]<br>#609 DEFENDING IN DEFENSIVE THIRD · Atletic Club [00:54:34.000–00:55:04.000] | #121 BALL IN FINAL THIRD · Atletic Club [00:55:09.920–00:55:40.200] |
| 32 | 1 | `84685–84847` | `00:56:27.400–00:56:33.880` | 6.48 s | #74 PLAYERS IN THE BOX · Barcelona [00:55:30.840–00:56:27.400] | #452 SET PIECES · Atletic Club · Goal Kick [00:56:33.880–00:57:08.880] |
| 33 | 1 | `88070–88457` | `00:58:42.800–00:58:58.280` | 15.48 s | #76 PLAYERS IN THE BOX · Barcelona [00:57:59.200–00:58:42.800] | #385 LONG BALL · Atletic Club [00:58:58.280–00:59:07.400] |
| 34 | 1 | `88685–89050` | `00:59:07.400–00:59:22.000` | 14.60 s | #385 LONG BALL · Atletic Club [00:58:58.280–00:59:07.400] | None within half |
| 35 | 2 | `113785–113891` | `01:15:51.400–01:15:55.640` | 4.24 s | #5 SET PIECES · N/A · Kick Off (Start) [01:15:46.400–01:15:51.400] | #371 BUILD UP · Barcelona [01:15:55.640–01:16:18.320]<br>#595 DEFENDING IN ATTACKING THIRD · Atletic Club [01:15:55.640–01:16:18.320] |
| 36 | 2 | `115727–116071` | `01:17:09.080–01:17:22.840` | 13.76 s | #426 SET PIECES · N/A · Throw-in [01:16:34.080–01:17:09.080] | #531 DEFENDING IN MIDDLE THIRD · Atletic Club [01:17:22.840–01:17:46.000]<br>#307 PROGRESSION · Barcelona [01:17:22.840–01:17:46.000] |
| 37 | 2 | `118754–120152` | `01:19:10.160–01:20:06.080` | 55.92 s | #123 BALL IN FINAL THIRD · Barcelona [01:18:39.160–01:19:10.160] | #398 LONG BALL · Barcelona [01:20:06.080–01:20:17.840] |
| 38 | 2 | `120446–120624` | `01:20:17.840–01:20:24.960` | 7.12 s | #398 LONG BALL · Barcelona [01:20:06.080–01:20:17.840] | #443 SET PIECES · N/A · Throw-in [01:20:24.960–01:20:59.960] |
| 39 | 2 | `122741–123010` | `01:21:49.640–01:22:00.400` | 10.76 s | #484 CREATING CHANCES · Barcelona [01:21:19.640–01:21:49.640]<br>#626 DEFENDING IN DEFENSIVE THIRD · Atletic Club [01:21:19.640–01:21:49.640] | #393 LONG BALL · Barcelona [01:22:00.400–01:22:10.800] |
| 40 | 2 | `123528–123604` | `01:22:21.120–01:22:24.160` | 3.04 s | #87 PLAYERS IN THE BOX · Atletic Club [01:22:02.160–01:22:21.120] | #431 SET PIECES · N/A · Throw-in [01:22:24.160–01:22:59.160] |
| 41 | 2 | `124941–125053` | `01:23:17.640–01:23:22.120` | 4.48 s | #381 BUILD UP · Atletic Club [01:23:01.680–01:23:17.640]<br>#605 DEFENDING IN ATTACKING THIRD · Barcelona [01:23:01.680–01:23:17.640] | #555 DEFENDING IN MIDDLE THIRD · Barcelona [01:23:22.120–01:23:36.840]<br>#331 PROGRESSION · Atletic Club [01:23:22.120–01:23:36.840] |
| 42 | 2 | `126238–126384` | `01:24:09.520–01:24:15.360` | 5.84 s | #510 DEFENDING IN MIDDLE THIRD · Atletic Club [01:23:56.440–01:24:09.520]<br>#286 PROGRESSION · Barcelona [01:23:56.440–01:24:09.520] | #549 DEFENDING IN MIDDLE THIRD · Atletic Club [01:24:15.360–01:24:37.880]<br>#325 PROGRESSION · Barcelona [01:24:15.360–01:24:37.880] |
| 43 | 2 | `126947–127252` | `01:24:37.880–01:24:50.080` | 12.20 s | #549 DEFENDING IN MIDDLE THIRD · Atletic Club [01:24:15.360–01:24:37.880]<br>#325 PROGRESSION · Barcelona [01:24:15.360–01:24:37.880] | #433 SET PIECES · N/A · Throw-in [01:24:50.080–01:25:25.080] |
| 44 | 2 | `128317–128539` | `01:25:32.680–01:25:41.560` | 8.88 s | #497 DEFENDING IN MIDDLE THIRD · Atletic Club [01:25:05.720–01:25:32.680]<br>#273 PROGRESSION · Barcelona [01:25:05.720–01:25:32.680] | #126 BALL IN FINAL THIRD · Barcelona [01:25:41.560–01:26:07.920] |
| 45 | 2 | `131196–131353` | `01:27:27.840–01:27:34.120` | 6.28 s | #414 SET PIECES · Barcelona · Corner Kick [01:26:52.840–01:27:27.840] | #388 LONG BALL · Atletic Club [01:27:34.120–01:27:46.520] |
| 46 | 2 | `132064–132084` | `01:28:02.560–01:28:03.360` | 0.80 s | #561 DEFENDING IN MIDDLE THIRD · Atletic Club [01:27:46.320–01:28:02.560]<br>#337 PROGRESSION · Barcelona [01:27:46.320–01:28:02.560] | #538 DEFENDING IN MIDDLE THIRD · Atletic Club [01:28:03.360–01:28:14.240]<br>#314 PROGRESSION · Barcelona [01:28:03.360–01:28:14.240] |
| 47 | 2 | `132945–133222` | `01:28:37.800–01:28:48.880` | 11.08 s | #352 ATTACKING TRANSITION · Barcelona [01:28:15.840–01:28:37.800]<br>#576 DEFENSIVE TRANSITION · Atletic Club [01:28:15.840–01:28:37.800] | #355 ATTACKING TRANSITION · Atletic Club [01:28:48.880–01:28:59.360]<br>#579 DEFENSIVE TRANSITION · Barcelona [01:28:48.880–01:28:59.360] |
| 48 | 2 | `134273–134388` | `01:29:30.920–01:29:35.520` | 4.60 s | #485 CREATING CHANCES · Barcelona [01:29:00.920–01:29:30.920]<br>#627 DEFENDING IN DEFENSIVE THIRD · Atletic Club [01:29:00.920–01:29:30.920] | #394 LONG BALL · Barcelona [01:29:35.520–01:29:48.160] |
| 49 | 2 | `134704–135245` | `01:29:48.160–01:30:09.800` | 21.64 s | #394 LONG BALL · Barcelona [01:29:35.520–01:29:48.160] | #481 CREATING CHANCES · Atletic Club [01:30:09.800–01:30:39.800]<br>#623 DEFENDING IN DEFENSIVE THIRD · Barcelona [01:30:09.800–01:30:39.800] |
| 50 | 2 | `138938–139866` | `01:32:37.520–01:33:14.640` | 37.12 s | #420 SET PIECES · Barcelona · Corner Kick [01:32:02.520–01:32:37.520] | #15 PLAYERS IN THE BOX · Barcelona [01:33:14.640–01:33:34.720] |
| 51 | 2 | `140839–140843` | `01:33:53.560–01:33:53.720` | 0.16 s | #457 SET PIECES · Atletic Club · Goal Kick [01:33:18.560–01:33:53.560] | #134 BALL IN FINAL THIRD · Barcelona [01:33:53.720–01:34:50.960] |
| 52 | 2 | `142928–142976` | `01:35:17.120–01:35:19.040` | 1.92 s | #495 CREATING CHANCES · Barcelona [01:34:47.120–01:35:17.120]<br>#637 DEFENDING IN DEFENSIVE THIRD · Atletic Club [01:34:47.120–01:35:17.120] | #401 LONG BALL · Atletic Club [01:35:19.040–01:35:37.120] |
| 53 | 2 | `143428–143502` | `01:35:37.120–01:35:40.080` | 2.96 s | #401 LONG BALL · Atletic Club [01:35:19.040–01:35:37.120] | #515 DEFENDING IN MIDDLE THIRD · Barcelona [01:35:40.080–01:35:57.840]<br>#291 PROGRESSION · Atletic Club [01:35:40.080–01:35:57.840] |
| 54 | 2 | `152737–153463` | `01:41:49.480–01:42:18.520` | 29.04 s | #21 PLAYERS IN THE BOX · Barcelona [01:41:08.000–01:41:49.480] | #390 LONG BALL · Atletic Club [01:42:18.520–01:42:33.440] |
| 55 | 2 | `153836–153912` | `01:42:33.440–01:42:36.480` | 3.04 s | #390 LONG BALL · Atletic Club [01:42:18.520–01:42:33.440] | #383 BUILD UP · Barcelona [01:42:36.480–01:43:00.560]<br>#607 DEFENDING IN ATTACKING THIRD · Atletic Club [01:42:36.480–01:43:00.560] |
| 56 | 2 | `163125–163349` | `01:48:45.000–01:48:53.960` | 8.96 s | #6 SET PIECES · N/A · Kick Off (After Goal) [01:48:40.000–01:48:45.000] | #188 BALL IN FINAL THIRD · Atletic Club [01:48:53.960–01:49:18.800] |
| 57 | 2 | `165731–165859` | `01:50:29.240–01:50:34.360` | 5.12 s | #430 SET PIECES · N/A · Throw-in [01:49:54.240–01:50:29.240] | #436 SET PIECES · N/A · Throw-in [01:50:34.360–01:51:09.360] |
| 58 | 2 | `167324–167646` | `01:51:32.960–01:51:45.840` | 12.88 s | #530 DEFENDING IN MIDDLE THIRD · Barcelona [01:51:19.600–01:51:32.960]<br>#306 PROGRESSION · Atletic Club [01:51:19.600–01:51:32.960] | #409 LONG BALL · Atletic Club [01:51:45.840–01:52:08.040] |
| 59 | 2 | `168745–169724` | `01:52:29.800–01:53:08.960` | 39.16 s | #94 PLAYERS IN THE BOX · Atletic Club [01:51:53.560–01:52:29.800] | #444 SET PIECES · N/A · Throw-in [01:53:08.960–01:53:43.960] |
| 60 | 2 | `172312–172718` | `01:54:52.480–01:55:08.720` | 16.24 s | #96 PLAYERS IN THE BOX · Atletic Club [01:54:29.840–01:54:52.480] | #192 BALL IN FINAL THIRD · Atletic Club [01:55:08.720–01:55:44.840] |
| 61 | 2 | `174331–174362` | `01:56:13.240–01:56:14.480` | 1.24 s | #366 BUILD UP · Barcelona [01:55:54.040–01:56:13.240]<br>#590 DEFENDING IN ATTACKING THIRD · Atletic Club [01:55:54.040–01:56:13.240] | #517 DEFENDING IN MIDDLE THIRD · Barcelona [01:56:14.480–01:56:40.560]<br>#293 PROGRESSION · Atletic Club [01:56:14.480–01:56:40.560] |
| 62 | 2 | `175299–176121` | `01:56:51.960–01:57:24.840` | 32.88 s | #356 ATTACKING TRANSITION · Barcelona [01:56:37.560–01:56:51.960]<br>#580 DEFENSIVE TRANSITION · Atletic Club [01:56:37.560–01:56:51.960] | #25 PLAYERS IN THE BOX · Barcelona [01:57:24.840–01:58:42.360] |
| 63 | 2 | `178059–178141` | `01:58:42.360–01:58:45.640` | 3.28 s | #25 PLAYERS IN THE BOX · Barcelona [01:57:24.840–01:58:42.360] | #380 BUILD UP · Atletic Club [01:58:45.640–01:59:04.240]<br>#604 DEFENDING IN ATTACKING THIRD · Barcelona [01:58:45.640–01:59:04.240] |
| 64 | 2 | `180951–182413` | `02:00:38.040–02:01:36.520` | 58.48 s | #196 BALL IN FINAL THIRD · Atletic Club [02:00:04.640–02:00:38.040] | #456 SET PIECES · Atletic Club · Goal Kick [02:01:36.520–02:02:11.520] |
| 65 | 2 | `183737–184199` | `02:02:29.480–02:02:47.960` | 18.48 s | #446 SET PIECES · N/A · Throw-in [02:01:54.480–02:02:29.480] | #268 BALL IN THE BOX · Atletic Club [02:02:47.960–02:03:13.960] |
| 66 | 2 | `186316–186616` | `02:04:12.640–02:04:24.640` | 12.00 s | #451 SET PIECES · N/A · Throw-in [02:03:37.640–02:04:12.640] | #399 LONG BALL · Atletic Club [02:04:24.640–02:04:38.200] |
| 67 | 2 | `187474–187494` | `02:04:58.960–02:04:59.760` | 0.80 s | #372 BUILD UP · Barcelona [02:04:47.240–02:04:58.960]<br>#596 DEFENDING IN ATTACKING THIRD · Atletic Club [02:04:47.240–02:04:58.960] | #445 SET PIECES · N/A · Throw-in [02:04:59.760–02:05:34.760] |
| 68 | 2 | `188369–188432` | `02:05:34.760–02:05:37.280` | 2.52 s | #445 SET PIECES · N/A · Throw-in [02:04:59.760–02:05:34.760] | #449 SET PIECES · N/A · Throw-in [02:05:37.280–02:06:12.280] |
| 69 | 2 | `189307–190000` | `02:06:12.280–02:06:40.000` | 27.72 s | #449 SET PIECES · N/A · Throw-in [02:05:37.280–02:06:12.280] | None within half |

## Interpretation

- Smart Tagging covers **84.75%** of the supplied game-time windows, substantially more
  than the 70.03% calculated over the entire video.
- Untagged frames are organized into long blocks. They are not predominantly isolated
  annotation holes.
- A coverage gap only means that none of the exported automatic tactical categories is
  active. It does not mean dead ball, live play, contested possession, or tracking failure.
- The neighboring tags provide visual-inspection context but are not validated event or
  possession boundaries.
- Complementary paired categories often appear together at a boundary and must not be
  interpreted as independent confirmations.
- A SET PIECES interval is immediately before 17 runs and immediately after 13 runs.
  This is boundary context only: it does not establish that the gap itself is a stoppage.

The machine-readable equivalent is `untagged_runs_with_neighboring_tags.csv`. Restricted
derived information remains below gitignored `data/` and must not be uploaded publicly.



---

## Archived source: `UNTAGGED_SCORED_FRAMES_REPORT.md`

﻿# Smart Tag Coverage Gaps in the Repository Scored Blocks

**Scope:** repository-defined scored frames only  
**Reference:** tactical recording at 25 fps  
**Possession ground truth used:** no  
**Tracking feature source:** `data\eda_outputs\frame_features.npz`

The canonical scored blocks are inclusive frame ranges: first half `11,185-90,651` and
second half `110,808-190,197`. Together they contain 158,857 frames. An untagged run is
a maximal consecutive run inside one scored block with no active Smart Tag. Smart Tag
overlaps are collapsed to a Boolean union.

The second scored block begins 2,934 frames (117.36 seconds) before the documented
second-half kickoff at frame 113,742. Boundary-touching runs are therefore reported
explicitly and must not be interpreted as ordinary in-play gaps.

## Smart Tag coverage

| Measure | Frames | Share of scored frames |
|---|---:|---:|
| Scored frames | 158,857 | 100.00% |
| At least one Smart Tag | 130,698 | 82.27% |
| No Smart Tag | 28,159 | 17.73% |

### Coverage by scored half

| Half | Scored frames | Tagged | Untagged | Untagged share | Runs | Median run | Longest run |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 79,467 | 66,474 | 12,993 | 16.35% | 34 | 259 (10.36 s) | 1,967 (78.68 s) |
| 2 | 79,390 | 64,224 | 15,166 | 19.10% | 36 | 223 (8.92 s) | 2,852 (114.08 s) |

The 28,159 untagged frames form **70 runs**. The median is
**244 frames (9.78 s)** and the longest is
**2,852 frames (114.08 s)**.

## Longest untagged run

- Run: **35**, half 2
- Video interval: **`01:13:52.320-01:15:46.400`**
- Frames: **`[110808, 113660)`**
- Ball observed: **2,288/2,852 frames (80.22%)**
- Mean tracked players: **18.56**
- Weighted mean player speed: **1.283 m/s**
- Previous tag(s): none within scored half
- Next tag(s): #5 SET PIECES / N/A / Kick Off (Start) [01:15:46.400-01:15:51.400]

The end time is exclusive: it is the first frame after the untagged run.

## Smart Tag activity versus ball visibility

Counts are frame counts. Percentages in parentheses are within each Smart Tag row.

| Smart Tag state | Ball observed | Ball missing | Total |
|---|---:|---:|---:|
| Active | 80,657 (61.71%) | 50,041 (38.29%) | 130,698 |
| No Smart Tag | 13,557 (48.14%) | 14,602 (51.86%) | 28,159 |

Ball missingness is **51.86%** among untagged frames and
**38.29%** among tagged frames: a **13.57
percentage-point difference**, risk ratio **1.354**, odds ratio
**1.736**, and phi association **0.105**. These are descriptive frame-level
associations; temporal autocorrelation makes ordinary independent-frame significance tests
inappropriate.

From the reverse conditional view, **22.59%** of all
ball-missing scored frames are untagged, versus **14.39%**
of ball-observed scored frames.

## Observation regimes

`21-22 players` is used as a descriptive near-complete coverage band. Speed coverage is
the fraction of observed player positions that also have a finite speed.

| Regime | Frames | Mean players | 21-22 players | <11 players | 0 players | Speed coverage | Mean speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tag active / ball observed | 80,657 | 21.52 | 90.70% | 0.00% | 0.00% | 93.50% | 2.268 m/s |
| Tag active / ball missing | 50,041 | 19.27 | 56.95% | 6.98% | 0.00% | 89.81% | 1.266 m/s |
| No tag / ball observed | 13,557 | 20.65 | 73.25% | 1.90% | 0.00% | 91.77% | 1.566 m/s |
| No tag / ball missing | 14,602 | 18.80 | 52.41% | 9.71% | 0.00% | 90.95% | 1.042 m/s |

## Run-length distribution

| Condition | Runs | Frames in those runs | Share of untagged frames |
|---|---:|---:|---:|
| Exactly 1 frame | 0 | 0 | 0.00% |
| More than 4 frames | 69 | 28,155 | 99.99% |
| More than 8 frames | 68 | 28,148 | 99.96% |
| More than 12 frames | 68 | 28,148 | 99.96% |
| More than 25 frames | 65 | 28,093 | 99.77% |

## Neighboring Smart Tag categories

Counts are tag instances at the immediately preceding or following boundary. Paired
categories can contribute two instances to one boundary.

| Previous category | Occurrences | Following category | Occurrences |
|---|---:|---|---:|
| SET PIECES | 17 | DEFENDING IN MIDDLE THIRD | 21 |
| PLAYERS IN THE BOX | 13 | PROGRESSION | 21 |
| DEFENDING IN MIDDLE THIRD | 12 | SET PIECES | 14 |
| PROGRESSION | 12 | LONG BALL | 12 |
| LONG BALL | 10 | BALL IN FINAL THIRD | 7 |
| CREATING CHANCES | 6 | PLAYERS IN THE BOX | 5 |
| DEFENDING IN DEFENSIVE THIRD | 6 | BUILD UP | 4 |
| ATTACKING TRANSITION | 5 | DEFENDING IN ATTACKING THIRD | 4 |
| DEFENSIVE TRANSITION | 5 | ATTACKING TRANSITION | 2 |
| BALL IN FINAL THIRD | 3 | DEFENSIVE TRANSITION | 2 |
| BUILD UP | 3 | BALL IN THE BOX | 2 |
| DEFENDING IN ATTACKING THIRD | 3 | CREATING CHANCES | 1 |
|  |  | DEFENDING IN DEFENSIVE THIRD | 1 |

## Every untagged run

Detailed player-count bands, ball counts, speed availability, speed percentiles, slow/fast
fractions, boundary flags, and full neighboring-tag metadata are in
`untagged_runs_scored_with_tracking.csv`.

| Run | Half | Video interval | Frames | Ball observed | Mean players | 21-22 players | Mean speed | Previous tag(s) | Next tag(s) |
|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 1 | `00:08:34.040-00:08:50.520` | 412 | 68.93% | 21.00 | 100.00% | 1.906 m/s | #349 ATTACKING TRANSITION / Atletic Club [00:08:22.960-00:08:34.040]<br>#573 DEFENSIVE TRANSITION / Barcelona [00:08:22.960-00:08:34.040] | #526 DEFENDING IN MIDDLE THIRD / Atletic Club [00:08:50.520-00:09:12.160]<br>#302 PROGRESSION / Barcelona [00:08:50.520-00:09:12.160] |
| 2 | 1 | `00:09:53.960-00:10:15.920` | 549 | 100.00% | 22.00 | 100.00% | 2.377 m/s | #548 DEFENDING IN MIDDLE THIRD / Atletic Club [00:09:40.800-00:09:53.960]<br>#324 PROGRESSION / Barcelona [00:09:40.800-00:09:53.960] | #52 PLAYERS IN THE BOX / Barcelona [00:10:15.920-00:10:38.600] |
| 3 | 1 | `00:10:38.600-00:11:13.720` | 878 | 84.97% | 21.92 | 100.00% | 1.644 m/s | #52 PLAYERS IN THE BOX / Barcelona [00:10:15.920-00:10:38.600] | #534 DEFENDING IN MIDDLE THIRD / Barcelona [00:11:13.720-00:11:34.040]<br>#310 PROGRESSION / Atletic Club [00:11:13.720-00:11:34.040] |
| 4 | 1 | `00:12:47.800-00:13:01.400` | 340 | 0.00% | 21.15 | 77.94% | 1.171 m/s | #454 SET PIECES / Barcelona / Goal Kick [00:12:12.800-00:12:47.800] | #442 SET PIECES / N/A / Throw-in [00:13:01.400-00:13:36.400] |
| 5 | 1 | `00:13:36.400-00:13:36.680` | 7 | 100.00% | 22.00 | 100.00% | 1.292 m/s | #442 SET PIECES / N/A / Throw-in [00:13:01.400-00:13:36.400] | #545 DEFENDING IN MIDDLE THIRD / Atletic Club [00:13:36.680-00:14:08.440]<br>#321 PROGRESSION / Barcelona [00:13:36.680-00:14:08.440] |
| 6 | 1 | `00:14:08.440-00:14:23.040` | 365 | 100.00% | 21.93 | 100.00% | 2.590 m/s | #545 DEFENDING IN MIDDLE THIRD / Atletic Club [00:13:36.680-00:14:08.440]<br>#321 PROGRESSION / Barcelona [00:13:36.680-00:14:08.440] | #448 SET PIECES / N/A / Throw-in [00:14:23.040-00:14:58.040] |
| 7 | 1 | `00:18:01.600-00:18:09.440` | 196 | 100.00% | 21.78 | 100.00% | 1.106 m/s | #482 CREATING CHANCES / Barcelona [00:17:31.600-00:18:01.600]<br>#624 DEFENDING IN DEFENSIVE THIRD / Atletic Club [00:17:31.600-00:18:01.600] | #389 LONG BALL / Atletic Club [00:18:09.440-00:18:17.520] |
| 8 | 1 | `00:18:17.520-00:18:31.760` | 356 | 56.74% | 22.54 | 45.51% | 1.787 m/s | #389 LONG BALL / Atletic Club [00:18:09.440-00:18:17.520] | #437 SET PIECES / N/A / Throw-in [00:18:31.760-00:19:06.760] |
| 9 | 1 | `00:20:27.240-00:20:38.480` | 281 | 100.00% | 22.00 | 100.00% | 1.564 m/s | #395 LONG BALL / Atletic Club [00:20:14.040-00:20:27.240] | #544 DEFENDING IN MIDDLE THIRD / Atletic Club [00:20:38.480-00:20:53.760]<br>#320 PROGRESSION / Barcelona [00:20:38.480-00:20:53.760] |
| 10 | 1 | `00:20:53.760-00:21:00.480` | 168 | 100.00% | 22.00 | 100.00% | 1.485 m/s | #544 DEFENDING IN MIDDLE THIRD / Atletic Club [00:20:38.480-00:20:53.760]<br>#320 PROGRESSION / Barcelona [00:20:38.480-00:20:53.760] | #533 DEFENDING IN MIDDLE THIRD / Atletic Club [00:21:00.480-00:21:26.960]<br>#309 PROGRESSION / Barcelona [00:21:00.480-00:21:26.960] |
| 11 | 1 | `00:21:26.960-00:21:37.240` | 257 | 100.00% | 22.00 | 100.00% | 1.892 m/s | #533 DEFENDING IN MIDDLE THIRD / Atletic Club [00:21:00.480-00:21:26.960]<br>#309 PROGRESSION / Barcelona [00:21:00.480-00:21:26.960] | #412 LONG BALL / Barcelona [00:21:37.240-00:21:54.200] |
| 12 | 1 | `00:22:08.960-00:22:32.280` | 583 | 40.99% | 21.77 | 96.23% | 0.958 m/s | #347 ATTACKING TRANSITION / Barcelona [00:21:53.400-00:22:08.960]<br>#571 DEFENSIVE TRANSITION / Atletic Club [00:21:53.400-00:22:08.960] | #156 BALL IN FINAL THIRD / Barcelona [00:22:32.280-00:22:58.000] |
| 13 | 1 | `00:25:36.960-00:25:38.000` | 26 | 100.00% | 21.00 | 100.00% | 1.388 m/s | #410 LONG BALL / Barcelona [00:25:18.400-00:25:36.960] | #504 DEFENDING IN MIDDLE THIRD / Atletic Club [00:25:38.000-00:26:06.000]<br>#280 PROGRESSION / Barcelona [00:25:38.000-00:26:06.000] |
| 14 | 1 | `00:33:10.240-00:33:12.560` | 58 | 100.00% | 20.64 | 63.79% | 2.067 m/s | #2 SET PIECES / N/A / Kick Off (After Goal) [00:33:05.240-00:33:10.240] | #540 DEFENDING IN MIDDLE THIRD / Barcelona [00:33:12.560-00:33:38.360]<br>#316 PROGRESSION / Atletic Club [00:33:12.560-00:33:38.360] |
| 15 | 1 | `00:33:53.880-00:34:00.000` | 153 | 2.61% | 22.00 | 100.00% | 0.556 m/s | #34 PLAYERS IN THE BOX / Atletic Club [00:33:16.480-00:33:53.880] | #453 SET PIECES / Barcelona / Goal Kick [00:34:00.000-00:34:35.000] |
| 16 | 1 | `00:35:42.160-00:36:04.560` | 560 | 100.00% | 21.79 | 100.00% | 1.164 m/s | #492 CREATING CHANCES / Barcelona [00:35:12.160-00:35:42.160]<br>#634 DEFENDING IN DEFENSIVE THIRD / Atletic Club [00:35:12.160-00:35:42.160] | #391 LONG BALL / Atletic Club [00:36:04.560-00:36:14.880] |
| 17 | 1 | `00:36:14.880-00:36:19.240` | 109 | 100.00% | 22.00 | 100.00% | 3.091 m/s | #391 LONG BALL / Atletic Club [00:36:04.560-00:36:14.880] | #354 ATTACKING TRANSITION / Barcelona [00:36:19.240-00:36:31.520]<br>#578 DEFENSIVE TRANSITION / Atletic Club [00:36:19.240-00:36:31.520] |
| 18 | 1 | `00:36:31.520-00:36:35.480` | 99 | 100.00% | 22.00 | 100.00% | 2.342 m/s | #354 ATTACKING TRANSITION / Barcelona [00:36:19.240-00:36:31.520]<br>#578 DEFENSIVE TRANSITION / Atletic Club [00:36:19.240-00:36:31.520] | #516 DEFENDING IN MIDDLE THIRD / Atletic Club [00:36:35.480-00:37:02.440]<br>#292 PROGRESSION / Barcelona [00:36:35.480-00:37:02.440] |
| 19 | 1 | `00:37:14.480-00:37:20.560` | 152 | 100.00% | 22.00 | 100.00% | 1.620 m/s | #65 PLAYERS IN THE BOX / Barcelona [00:36:49.200-00:37:14.480] | #560 DEFENDING IN MIDDLE THIRD / Barcelona [00:37:20.560-00:37:38.240]<br>#336 PROGRESSION / Atletic Club [00:37:20.560-00:37:38.240] |
| 20 | 1 | `00:41:25.480-00:41:36.440` | 274 | 28.10% | 20.87 | 70.44% | 0.795 m/s | #3 SET PIECES / N/A / Kick Off (After Goal) [00:41:20.480-00:41:25.480] | #376 BUILD UP / Barcelona [00:41:36.440-00:41:54.600]<br>#600 DEFENDING IN ATTACKING THIRD / Atletic Club [00:41:36.440-00:41:54.600] |
| 21 | 1 | `00:42:23.600-00:42:32.880` | 232 | 37.93% | 22.00 | 100.00% | 1.919 m/s | #562 DEFENDING IN MIDDLE THIRD / Atletic Club [00:42:09.240-00:42:23.600]<br>#338 PROGRESSION / Barcelona [00:42:09.240-00:42:23.600] | #527 DEFENDING IN MIDDLE THIRD / Barcelona [00:42:32.880-00:42:47.800]<br>#303 PROGRESSION / Atletic Club [00:42:32.880-00:42:47.800] |
| 22 | 1 | `00:42:47.800-00:43:06.680` | 472 | 69.92% | 21.98 | 100.00% | 1.927 m/s | #527 DEFENDING IN MIDDLE THIRD / Barcelona [00:42:32.880-00:42:47.800]<br>#303 PROGRESSION / Atletic Club [00:42:32.880-00:42:47.800] | #535 DEFENDING IN MIDDLE THIRD / Barcelona [00:43:06.680-00:43:18.560]<br>#311 PROGRESSION / Atletic Club [00:43:06.680-00:43:18.560] |
| 23 | 1 | `00:44:17.320-00:44:20.880` | 89 | 100.00% | 22.00 | 100.00% | 2.773 m/s | #432 SET PIECES / N/A / Throw-in [00:43:42.320-00:44:17.320] | #511 DEFENDING IN MIDDLE THIRD / Atletic Club [00:44:20.880-00:44:42.560]<br>#287 PROGRESSION / Barcelona [00:44:20.880-00:44:42.560] |
| 24 | 1 | `00:45:05.080-00:45:27.400` | 558 | 100.00% | 21.54 | 90.32% | 0.821 m/s | #117 BALL IN FINAL THIRD / Atletic Club [00:44:25.560-00:45:05.080] | #557 DEFENDING IN MIDDLE THIRD / Atletic Club [00:45:27.400-00:45:39.080]<br>#333 PROGRESSION / Barcelona [00:45:27.400-00:45:39.080] |
| 25 | 1 | `00:46:54.600-00:48:09.280` | 1,867 | 5.84% | 16.56 | 9.43% | 0.744 m/s | #40 PLAYERS IN THE BOX / Atletic Club [00:46:36.240-00:46:54.600] | #41 PLAYERS IN THE BOX / Atletic Club [00:48:09.280-00:49:32.960] |
| 26 | 1 | `00:50:23.280-00:50:23.880` | 15 | 100.00% | 22.00 | 100.00% | 1.635 m/s | #4 SET PIECES / N/A / Kick Off (After Goal) [00:50:18.280-00:50:23.280] | #539 DEFENDING IN MIDDLE THIRD / Atletic Club [00:50:23.880-00:50:41.360]<br>#315 PROGRESSION / Barcelona [00:50:23.880-00:50:41.360] |
| 27 | 1 | `00:50:41.360-00:50:42.880` | 38 | 100.00% | 21.34 | 100.00% | 2.546 m/s | #539 DEFENDING IN MIDDLE THIRD / Atletic Club [00:50:23.880-00:50:41.360]<br>#315 PROGRESSION / Barcelona [00:50:23.880-00:50:41.360] | #166 BALL IN FINAL THIRD / Barcelona [00:50:42.880-00:51:08.080] |
| 28 | 1 | `00:51:21.160-00:51:30.360` | 230 | 100.00% | 21.85 | 100.00% | 1.570 m/s | #392 LONG BALL / Atletic Club [00:51:12.080-00:51:21.160] | #70 PLAYERS IN THE BOX / Barcelona [00:51:30.360-00:51:48.400] |
| 29 | 1 | `00:52:02.000-00:52:31.760` | 744 | 24.73% | 21.76 | 87.90% | 1.159 m/s | #71 PLAYERS IN THE BOX / Barcelona [00:51:44.000-00:52:02.000] | #213 BALL IN THE BOX / Atletic Club [00:52:31.760-00:52:49.120] |
| 30 | 1 | `00:52:57.000-00:53:07.440` | 261 | 65.52% | 22.04 | 96.17% | 1.066 m/s | #42 PLAYERS IN THE BOX / Atletic Club [00:52:31.880-00:52:57.000] | #552 DEFENDING IN MIDDLE THIRD / Atletic Club [00:53:07.440-00:53:20.040]<br>#328 PROGRESSION / Barcelona [00:53:07.440-00:53:20.040] |
| 31 | 1 | `00:55:04.000-00:55:09.920` | 148 | 0.00% | 22.00 | 100.00% | 1.506 m/s | #467 CREATING CHANCES / Barcelona [00:54:34.000-00:55:04.000]<br>#609 DEFENDING IN DEFENSIVE THIRD / Atletic Club [00:54:34.000-00:55:04.000] | #121 BALL IN FINAL THIRD / Atletic Club [00:55:09.920-00:55:40.200] |
| 32 | 1 | `00:56:27.400-00:56:33.880` | 162 | 0.62% | 21.00 | 100.00% | 1.310 m/s | #74 PLAYERS IN THE BOX / Barcelona [00:55:30.840-00:56:27.400] | #452 SET PIECES / Atletic Club / Goal Kick [00:56:33.880-00:57:08.880] |
| 33 | 1 | `00:58:42.800-00:58:58.280` | 387 | 27.13% | 20.87 | 83.98% | 0.795 m/s | #76 PLAYERS IN THE BOX / Barcelona [00:57:59.200-00:58:42.800] | #385 LONG BALL / Atletic Club [00:58:58.280-00:59:07.400] |
| 34 | 1 | `00:59:07.400-01:00:26.080` | 1,967 | 17.03% | 6.16 | 5.85% | 1.344 m/s | #385 LONG BALL / Atletic Club [00:58:58.280-00:59:07.400] | None within scored half |
| 35 | 2 | `01:13:52.320-01:15:46.400` | 2,852 | 80.22% | 18.56 | 21.56% | 1.283 m/s | None within scored half | #5 SET PIECES / N/A / Kick Off (Start) [01:15:46.400-01:15:51.400] |
| 36 | 2 | `01:15:51.400-01:15:55.640` | 106 | 100.00% | 22.00 | 100.00% | 2.238 m/s | #5 SET PIECES / N/A / Kick Off (Start) [01:15:46.400-01:15:51.400] | #371 BUILD UP / Barcelona [01:15:55.640-01:16:18.320]<br>#595 DEFENDING IN ATTACKING THIRD / Atletic Club [01:15:55.640-01:16:18.320] |
| 37 | 2 | `01:17:09.080-01:17:22.840` | 344 | 25.29% | 19.46 | 0.00% | 1.135 m/s | #426 SET PIECES / N/A / Throw-in [01:16:34.080-01:17:09.080] | #531 DEFENDING IN MIDDLE THIRD / Atletic Club [01:17:22.840-01:17:46.000]<br>#307 PROGRESSION / Barcelona [01:17:22.840-01:17:46.000] |
| 38 | 2 | `01:19:10.160-01:20:06.080` | 1,398 | 44.42% | 20.97 | 87.70% | 1.281 m/s | #123 BALL IN FINAL THIRD / Barcelona [01:18:39.160-01:19:10.160] | #398 LONG BALL / Barcelona [01:20:06.080-01:20:17.840] |
| 39 | 2 | `01:20:17.840-01:20:24.960` | 178 | 10.67% | 21.00 | 100.00% | 1.849 m/s | #398 LONG BALL / Barcelona [01:20:06.080-01:20:17.840] | #443 SET PIECES / N/A / Throw-in [01:20:24.960-01:20:59.960] |
| 40 | 2 | `01:21:49.640-01:22:00.400` | 269 | 0.00% | 22.00 | 100.00% | 1.217 m/s | #484 CREATING CHANCES / Barcelona [01:21:19.640-01:21:49.640]<br>#626 DEFENDING IN DEFENSIVE THIRD / Atletic Club [01:21:19.640-01:21:49.640] | #393 LONG BALL / Barcelona [01:22:00.400-01:22:10.800] |
| 41 | 2 | `01:22:21.120-01:22:24.160` | 76 | 0.00% | 21.84 | 100.00% | 1.195 m/s | #87 PLAYERS IN THE BOX / Atletic Club [01:22:02.160-01:22:21.120] | #431 SET PIECES / N/A / Throw-in [01:22:24.160-01:22:59.160] |
| 42 | 2 | `01:23:17.640-01:23:22.120` | 112 | 47.32% | 21.71 | 100.00% | 1.412 m/s | #381 BUILD UP / Atletic Club [01:23:01.680-01:23:17.640]<br>#605 DEFENDING IN ATTACKING THIRD / Barcelona [01:23:01.680-01:23:17.640] | #555 DEFENDING IN MIDDLE THIRD / Barcelona [01:23:22.120-01:23:36.840]<br>#331 PROGRESSION / Atletic Club [01:23:22.120-01:23:36.840] |
| 43 | 2 | `01:24:09.520-01:24:15.360` | 146 | 100.00% | 22.00 | 100.00% | 1.774 m/s | #510 DEFENDING IN MIDDLE THIRD / Atletic Club [01:23:56.440-01:24:09.520]<br>#286 PROGRESSION / Barcelona [01:23:56.440-01:24:09.520] | #549 DEFENDING IN MIDDLE THIRD / Atletic Club [01:24:15.360-01:24:37.880]<br>#325 PROGRESSION / Barcelona [01:24:15.360-01:24:37.880] |
| 44 | 2 | `01:24:37.880-01:24:50.080` | 305 | 0.00% | 22.00 | 100.00% | 0.902 m/s | #549 DEFENDING IN MIDDLE THIRD / Atletic Club [01:24:15.360-01:24:37.880]<br>#325 PROGRESSION / Barcelona [01:24:15.360-01:24:37.880] | #433 SET PIECES / N/A / Throw-in [01:24:50.080-01:25:25.080] |
| 45 | 2 | `01:25:32.680-01:25:41.560` | 222 | 100.00% | 22.00 | 100.00% | 2.135 m/s | #497 DEFENDING IN MIDDLE THIRD / Atletic Club [01:25:05.720-01:25:32.680]<br>#273 PROGRESSION / Barcelona [01:25:05.720-01:25:32.680] | #126 BALL IN FINAL THIRD / Barcelona [01:25:41.560-01:26:07.920] |
| 46 | 2 | `01:27:27.840-01:27:34.120` | 157 | 100.00% | 21.34 | 100.00% | 1.539 m/s | #414 SET PIECES / Barcelona / Corner Kick [01:26:52.840-01:27:27.840] | #388 LONG BALL / Atletic Club [01:27:34.120-01:27:46.520] |
| 47 | 2 | `01:28:02.560-01:28:03.360` | 20 | 100.00% | 21.00 | 100.00% | 1.444 m/s | #561 DEFENDING IN MIDDLE THIRD / Atletic Club [01:27:46.320-01:28:02.560]<br>#337 PROGRESSION / Barcelona [01:27:46.320-01:28:02.560] | #538 DEFENDING IN MIDDLE THIRD / Atletic Club [01:28:03.360-01:28:14.240]<br>#314 PROGRESSION / Barcelona [01:28:03.360-01:28:14.240] |
| 48 | 2 | `01:28:37.800-01:28:48.880` | 277 | 100.00% | 20.80 | 79.78% | 2.227 m/s | #352 ATTACKING TRANSITION / Barcelona [01:28:15.840-01:28:37.800]<br>#576 DEFENSIVE TRANSITION / Atletic Club [01:28:15.840-01:28:37.800] | #355 ATTACKING TRANSITION / Atletic Club [01:28:48.880-01:28:59.360]<br>#579 DEFENSIVE TRANSITION / Barcelona [01:28:48.880-01:28:59.360] |
| 49 | 2 | `01:29:30.920-01:29:35.520` | 115 | 0.00% | 21.00 | 100.00% | 1.265 m/s | #485 CREATING CHANCES / Barcelona [01:29:00.920-01:29:30.920]<br>#627 DEFENDING IN DEFENSIVE THIRD / Atletic Club [01:29:00.920-01:29:30.920] | #394 LONG BALL / Barcelona [01:29:35.520-01:29:48.160] |
| 50 | 2 | `01:29:48.160-01:30:09.800` | 541 | 100.00% | 21.90 | 100.00% | 1.999 m/s | #394 LONG BALL / Barcelona [01:29:35.520-01:29:48.160] | #481 CREATING CHANCES / Atletic Club [01:30:09.800-01:30:39.800]<br>#623 DEFENDING IN DEFENSIVE THIRD / Barcelona [01:30:09.800-01:30:39.800] |
| 51 | 2 | `01:32:37.520-01:33:14.640` | 928 | 14.98% | 22.08 | 57.65% | 0.871 m/s | #420 SET PIECES / Barcelona / Corner Kick [01:32:02.520-01:32:37.520] | #15 PLAYERS IN THE BOX / Barcelona [01:33:14.640-01:33:34.720] |
| 52 | 2 | `01:33:53.560-01:33:53.720` | 4 | 100.00% | 22.00 | 100.00% | 3.109 m/s | #457 SET PIECES / Atletic Club / Goal Kick [01:33:18.560-01:33:53.560] | #134 BALL IN FINAL THIRD / Barcelona [01:33:53.720-01:34:50.960] |
| 53 | 2 | `01:35:17.120-01:35:19.040` | 48 | 0.00% | 21.69 | 100.00% | 1.190 m/s | #495 CREATING CHANCES / Barcelona [01:34:47.120-01:35:17.120]<br>#637 DEFENDING IN DEFENSIVE THIRD / Atletic Club [01:34:47.120-01:35:17.120] | #401 LONG BALL / Atletic Club [01:35:19.040-01:35:37.120] |
| 54 | 2 | `01:35:37.120-01:35:40.080` | 74 | 100.00% | 22.00 | 100.00% | 1.924 m/s | #401 LONG BALL / Atletic Club [01:35:19.040-01:35:37.120] | #515 DEFENDING IN MIDDLE THIRD / Barcelona [01:35:40.080-01:35:57.840]<br>#291 PROGRESSION / Atletic Club [01:35:40.080-01:35:57.840] |
| 55 | 2 | `01:41:49.480-01:42:18.520` | 726 | 0.00% | 21.73 | 74.24% | 0.775 m/s | #21 PLAYERS IN THE BOX / Barcelona [01:41:08.000-01:41:49.480] | #390 LONG BALL / Atletic Club [01:42:18.520-01:42:33.440] |
| 56 | 2 | `01:42:33.440-01:42:36.480` | 76 | 100.00% | 21.00 | 100.00% | 1.420 m/s | #390 LONG BALL / Atletic Club [01:42:18.520-01:42:33.440] | #383 BUILD UP / Barcelona [01:42:36.480-01:43:00.560]<br>#607 DEFENDING IN ATTACKING THIRD / Atletic Club [01:42:36.480-01:43:00.560] |
| 57 | 2 | `01:48:45.000-01:48:53.960` | 224 | 32.59% | 22.00 | 100.00% | 2.082 m/s | #6 SET PIECES / N/A / Kick Off (After Goal) [01:48:40.000-01:48:45.000] | #188 BALL IN FINAL THIRD / Atletic Club [01:48:53.960-01:49:18.800] |
| 58 | 2 | `01:50:29.240-01:50:34.360` | 128 | 0.00% | 22.00 | 100.00% | 1.036 m/s | #430 SET PIECES / N/A / Throw-in [01:49:54.240-01:50:29.240] | #436 SET PIECES / N/A / Throw-in [01:50:34.360-01:51:09.360] |
| 59 | 2 | `01:51:32.960-01:51:45.840` | 322 | 100.00% | 21.57 | 100.00% | 1.926 m/s | #530 DEFENDING IN MIDDLE THIRD / Barcelona [01:51:19.600-01:51:32.960]<br>#306 PROGRESSION / Atletic Club [01:51:19.600-01:51:32.960] | #409 LONG BALL / Atletic Club [01:51:45.840-01:52:08.040] |
| 60 | 2 | `01:52:29.800-01:53:08.960` | 979 | 17.67% | 21.11 | 87.33% | 1.165 m/s | #94 PLAYERS IN THE BOX / Atletic Club [01:51:53.560-01:52:29.800] | #444 SET PIECES / N/A / Throw-in [01:53:08.960-01:53:43.960] |
| 61 | 2 | `01:54:52.480-01:55:08.720` | 406 | 0.00% | 19.19 | 9.36% | 0.804 m/s | #96 PLAYERS IN THE BOX / Atletic Club [01:54:29.840-01:54:52.480] | #192 BALL IN FINAL THIRD / Atletic Club [01:55:08.720-01:55:44.840] |
| 62 | 2 | `01:56:13.240-01:56:14.480` | 31 | 100.00% | 21.65 | 100.00% | 1.455 m/s | #366 BUILD UP / Barcelona [01:55:54.040-01:56:13.240]<br>#590 DEFENDING IN ATTACKING THIRD / Atletic Club [01:55:54.040-01:56:13.240] | #517 DEFENDING IN MIDDLE THIRD / Barcelona [01:56:14.480-01:56:40.560]<br>#293 PROGRESSION / Atletic Club [01:56:14.480-01:56:40.560] |
| 63 | 2 | `01:56:51.960-01:57:24.840` | 822 | 27.01% | 19.80 | 22.63% | 0.960 m/s | #356 ATTACKING TRANSITION / Barcelona [01:56:37.560-01:56:51.960]<br>#580 DEFENSIVE TRANSITION / Atletic Club [01:56:37.560-01:56:51.960] | #25 PLAYERS IN THE BOX / Barcelona [01:57:24.840-01:58:42.360] |
| 64 | 2 | `01:58:42.360-01:58:45.640` | 82 | 0.00% | 22.00 | 100.00% | 1.469 m/s | #25 PLAYERS IN THE BOX / Barcelona [01:57:24.840-01:58:42.360] | #380 BUILD UP / Atletic Club [01:58:45.640-01:59:04.240]<br>#604 DEFENDING IN ATTACKING THIRD / Barcelona [01:58:45.640-01:59:04.240] |
| 65 | 2 | `02:00:38.040-02:01:36.520` | 1,462 | 37.89% | 22.74 | 30.85% | 0.730 m/s | #196 BALL IN FINAL THIRD / Atletic Club [02:00:04.640-02:00:38.040] | #456 SET PIECES / Atletic Club / Goal Kick [02:01:36.520-02:02:11.520] |
| 66 | 2 | `02:02:29.480-02:02:47.960` | 462 | 35.71% | 21.98 | 100.00% | 1.013 m/s | #446 SET PIECES / N/A / Throw-in [02:01:54.480-02:02:29.480] | #268 BALL IN THE BOX / Atletic Club [02:02:47.960-02:03:13.960] |
| 67 | 2 | `02:04:12.640-02:04:24.640` | 300 | 100.00% | 21.86 | 100.00% | 1.765 m/s | #451 SET PIECES / N/A / Throw-in [02:03:37.640-02:04:12.640] | #399 LONG BALL / Atletic Club [02:04:24.640-02:04:38.200] |
| 68 | 2 | `02:04:58.960-02:04:59.760` | 20 | 0.00% | 21.30 | 80.00% | 2.135 m/s | #372 BUILD UP / Barcelona [02:04:47.240-02:04:58.960]<br>#596 DEFENDING IN ATTACKING THIRD / Atletic Club [02:04:47.240-02:04:58.960] | #445 SET PIECES / N/A / Throw-in [02:04:59.760-02:05:34.760] |
| 69 | 2 | `02:05:34.760-02:05:37.280` | 63 | 100.00% | 22.00 | 100.00% | 2.034 m/s | #445 SET PIECES / N/A / Throw-in [02:04:59.760-02:05:34.760] | #449 SET PIECES / N/A / Throw-in [02:05:37.280-02:06:12.280] |
| 70 | 2 | `02:06:12.280-02:06:47.920` | 891 | 21.55% | 16.18 | 0.00% | 0.912 m/s | #449 SET PIECES / N/A / Throw-in [02:05:37.280-02:06:12.280] | None within scored half |

## Findings and limitations

- Untagged frames with a visible ball number **13,557**.
  Their mean player count is **20.65**, with
  **9,931 frames (73.25%)**
  in the 21-22 player band. This is the
  regime where ball/player geometry can potentially provide information absent from tags.
- Untagged frames with a missing ball number **14,602**.
  Their mean player count is **18.80**, with
  **1,418 frames (9.71%)**
  below 11 tracked players. Frames where
  both sources are weak require more temporal context and explicit uncertainty handling.
- Smart Tags and tracking are automatic systems. Their agreement is not evidence that
  either one identifies the true football state.
- Missing tags do not imply dead ball, live play, or unknown possession. Low player speed
  is descriptive and is not a ground-truth stoppage label.
- Frame counts are highly autocorrelated. Treat the matrix as an exposure summary, not as
  158,857 independent statistical trials.
- One run starts exactly at a scored-block boundary and two end exactly at a scored-block
  boundary. The start-boundary run has no preceding within-half tag; the two end-boundary
  runs have no following within-half tag.
- The tactical video timeline is used directly. Smart Tag endpoints are rounded from
  seconds to 25 fps frames; ATD frame IDs are converted to zero-based frames.

Restricted derived tables remain below gitignored `data/` and must not be published.



---
