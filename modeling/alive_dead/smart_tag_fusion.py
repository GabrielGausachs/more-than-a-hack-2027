"""Prepare and one-time evaluate the conservative Metrica Smart Tag candidate.

Preparation is intentionally label-free::

    python -m modeling.alive_dead.smart_tag_fusion prepare

Evaluation is a separate operation over an already frozen run::

    python -m modeling.alive_dead.smart_tag_fusion evaluate --run-dir RUN --ground-truth GT.csv
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from evaluation import possession_metrics as pm
from feature_engineering.smart_tags import (
    AWAY_TEAM,
    HOME_TEAM,
    audit_smart_tags,
    build_smart_tag_evidence,
    load_smart_tags,
)

from .metrica_had import (
    _comparison_row,
    _load_ground_truth,
    _run_official_scorer,
    _sha256,
    _validate_model_metadata,
    _write_comparison_csv,
    average_categorical_probabilities,
    decode_probabilities_by_half,
    infer_metrica_checkpoint,
    load_comparison_config,
    load_global_decoder,
    reconstruct_source_labels,
    reconstruct_source_probabilities,
    write_submission,
)
from .evaluate import load_metrica_features, select_device, validate_transfer_configs
from .postprocess_had import Decoder
from .runs import fold_checkpoint_path, load_run_metadata


SCHEMA_VERSION = "metrica_smart_tag_fusion_run_v1"
CONFIG_SCHEMA_VERSION = "metrica_smart_tag_fusion_v1"
DEFAULT_CONFIG = Path("configs/alive_dead/metrica_smart_tag_fusion.json")
CONSERVATIVE_POLICY = {
    "home_team": "Barcelona",
    "away_team": "Atletic Club",
    "ownership_codes": [
        "BUILD UP",
        "PROGRESSION",
        "ATTACKING TRANSITION",
        "DEFENDING IN ATTACKING THIRD",
        "DEFENDING IN MIDDLE THIRD",
        "DEFENSIVE TRANSITION",
    ],
    "ownership_taper_seconds": 2.0,
    "ownership_ball_missing_only": True,
    "ownership_odds_multiplier": 1.5,
    "kickoff_types": ["Kick Off (Start)", "Kick Off (After Goal)"],
    "kickoff_odds_multiplier": 1.5,
    "kickoff_live_seconds": 2.0,
    "decoder": {"method": "viterbi", "transition_penalty": 1.0},
}
ALL_FRAME_POLICY = {**CONSERVATIVE_POLICY, "ownership_ball_missing_only": False}
SUPPORTED_POLICIES = (CONSERVATIVE_POLICY, ALL_FRAME_POLICY)
VARIANTS = (
    "baseline_raw",
    "baseline_decoded",
    "smart_tag_raw",
    "smart_tag_decoded",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_fusion_config(path: str | Path) -> dict[str, Any]:
    """Load the one allowed, predeclared conservative candidate."""
    raw = _read_json(path)
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported Smart Tag fusion config schema")
    if raw.get("policy") not in SUPPORTED_POLICIES:
        raise ValueError("Smart Tag fusion policy is not one of the two registered candidates")
    required = ("model_config", "model_name", "features", "smart_tags", "output_root")
    missing = [name for name in required if not raw.get(name)]
    if missing:
        raise ValueError(f"Smart Tag fusion config is missing: {missing}")
    return raw


def fuse_probabilities(
    probabilities: np.ndarray,
    ball_location_valid: np.ndarray,
    home_evidence: np.ndarray,
    away_evidence: np.ndarray,
    kickoff_dead: np.ndarray,
    kickoff_live: np.ndarray,
    *,
    ownership_odds_multiplier: float = 1.5,
    kickoff_odds_multiplier: float = 1.5,
    ownership_ball_missing_only: bool = True,
) -> np.ndarray:
    """Apply separated ownership and alive/dead evidence in hierarchical log-odds space."""
    values = np.asarray(probabilities, dtype=np.float64)
    ball_valid = np.asarray(ball_location_valid, dtype=bool)
    evidence = [np.asarray(item, dtype=np.float64) for item in (
        home_evidence, away_evidence, kickoff_dead, kickoff_live
    )]
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Probabilities must have shape (N, 3)")
    if any(item.shape != (len(values),) for item in [ball_valid, *evidence]):
        raise ValueError("Fusion probabilities, masks and evidence are not aligned")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Fusion probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Fusion probability rows must sum to one")
    if any(not np.isfinite(item).all() or np.any((item < 0.0) | (item > 1.0)) for item in evidence):
        raise ValueError("Smart Tag evidence must be finite and lie in [0, 1]")
    if ownership_odds_multiplier <= 1.0 or kickoff_odds_multiplier <= 1.0:
        raise ValueError("Fusion odds multipliers must be greater than one")

    ownership_gate = (~ball_valid).astype(np.float64) if ownership_ball_missing_only else 1.0
    owner_delta = ownership_gate * (evidence[0] - evidence[1])
    alive_delta = evidence[3] - evidence[2]
    active = (owner_delta != 0.0) | (alive_delta != 0.0)
    # Preserve every uninfluenced row bit-for-bit.  Besides being easier to audit, this prevents
    # a logit round trip from manufacturing tiny changes where Smart Tags supplied no evidence.
    fused = np.asarray(probabilities, dtype=np.float32).copy()
    if not np.any(active):
        return fused

    eps = 1e-7
    selected = values[active]
    alive = np.clip(selected[:, pm.H] + selected[:, pm.A], eps, 1.0 - eps)
    home_given_alive = np.clip(selected[:, pm.H] / alive, eps, 1.0 - eps)
    owner_logit = np.log(home_given_alive / (1.0 - home_given_alive))
    owner_logit += math.log(ownership_odds_multiplier) * owner_delta[active]
    home_given_alive = 1.0 / (1.0 + np.exp(-owner_logit))

    alive_logit = np.log(alive / (1.0 - alive))
    alive_logit += math.log(kickoff_odds_multiplier) * alive_delta[active]
    fused_alive = 1.0 / (1.0 + np.exp(-alive_logit))
    fused[active] = np.column_stack(
        (fused_alive * home_given_alive, fused_alive * (1.0 - home_given_alive), 1.0 - fused_alive)
    ).astype(np.float32)
    if not np.isfinite(fused).all() or not np.allclose(fused.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Smart Tag fusion produced invalid probabilities")
    return fused


def _sample_evidence(evidence_25hz: Mapping[str, np.ndarray], source_frames: np.ndarray) -> dict[str, np.ndarray]:
    frames = np.asarray(source_frames, dtype=np.int64)
    if frames.ndim != 1 or np.any(frames < 0):
        raise ValueError("Source frame indices must be a non-negative vector")
    output = {}
    for name, values in evidence_25hz.items():
        array = np.asarray(values)
        if np.any(frames >= len(array)):
            raise ValueError("Source frame index lies outside Smart Tag evidence")
        output[name] = array[frames]
    return output


def _source_mask(mask_10hz: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    reverse = np.asarray(reverse, dtype=np.int64)
    mask = np.asarray(mask_10hz, dtype=bool)
    output = np.zeros(len(reverse), dtype=bool)
    valid = reverse >= 0
    output[valid] = mask[reverse[valid]]
    return output


def _label_free_diagnostics(
    baseline: np.ndarray,
    fused: np.ndarray,
    evidence_10hz: Mapping[str, np.ndarray],
    ball_valid: np.ndarray,
    baseline_labels: Mapping[str, np.ndarray],
    fused_labels: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    ownership = np.maximum(evidence_10hz["home_ownership"], evidence_10hz["away_ownership"])
    conflict = np.minimum(evidence_10hz["home_ownership"], evidence_10hz["away_ownership"])
    kickoff = np.maximum(evidence_10hz["kickoff_dead"], evidence_10hz["kickoff_live"])
    changed = np.max(np.abs(fused - baseline), axis=1) > 1e-8
    l1 = np.abs(fused - baseline).sum(axis=1)
    return {
        "n_samples": int(len(baseline)),
        "ball_missing_samples": int(np.count_nonzero(~ball_valid)),
        "ownership_evidence_samples": int(np.count_nonzero(ownership > 0.0)),
        "ownership_evidence_while_ball_missing": int(np.count_nonzero((ownership > 0.0) & ~ball_valid)),
        "opposing_evidence_conflict_samples": int(np.count_nonzero(conflict > 0.0)),
        "kickoff_evidence_samples": int(np.count_nonzero(kickoff > 0.0)),
        "probability_rows_changed": int(np.count_nonzero(changed)),
        "mean_probability_l1": float(l1.mean()),
        "mean_probability_l1_changed": float(l1[changed].mean()) if np.any(changed) else 0.0,
        "max_probability_l1": float(l1.max(initial=0.0)),
        "raw_label_changes": int(np.count_nonzero(baseline_labels["raw"] != fused_labels["raw"])),
        "decoded_label_changes": int(
            np.count_nonzero(baseline_labels["decoded"] != fused_labels["decoded"])
        ),
    }


def prepare_run(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> Path:
    """Generate and freeze the candidate without loading any ground truth."""
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_fusion_config(config_path)
    policy = config["policy"]
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    feature_path = Path(config["features"])
    xml_path = Path(config["smart_tags"])
    model_config_path = Path(config["model_config"])
    specs, weights = load_comparison_config(model_config_path)
    if len(specs) != 1 or specs[0].name != config["model_name"]:
        raise ValueError("Fusion requires exactly the configured mask-augmented no-transition model")
    spec = specs[0]
    if dict(spec.expected_decoder) != policy["decoder"]:
        raise ValueError("Configured model does not use the frozen Viterbi penalty 1.0")

    features = load_metrica_features(feature_path)
    if features["team_order"].tolist() != ["home", "away"] or features["team_names"].tolist() != [
        HOME_TEAM, AWAY_TEAM
    ]:
        raise ValueError("Metrica feature archive team order does not match Barcelona H / Atletic Club A")
    n_source = int(features["original_frame_count"])
    fps = float(features["original_source_fps"])
    events = load_smart_tags(xml_path, fps=fps, n_frames=n_source)
    audit = audit_smart_tags(events, n_frames=n_source, fps=fps)
    evidence_25hz = build_smart_tag_evidence(
        events,
        n_frames=n_source,
        fps=fps,
        ownership_taper_seconds=policy["ownership_taper_seconds"],
        kickoff_live_seconds=policy["kickoff_live_seconds"],
    )
    evidence_10hz = _sample_evidence(evidence_25hz, features["source_frame_index"])

    metadata = load_run_metadata(spec.run_dir, "had")
    _validate_model_metadata(metadata, spec)
    decoder, decoder_report = load_global_decoder(spec.decoder_selection, spec.expected_decoder)
    if decoder.to_dict() != policy["decoder"]:
        raise ValueError("Loaded decoder is not the predeclared global Viterbi decoder")
    device = select_device(device_name)
    fold_probabilities: dict[int, np.ndarray] = {}
    configs = {}
    checkpoints = {}
    checkpoint_paths = {fold: fold_checkpoint_path(spec.run_dir, fold) for fold in (1, 2)}
    for fold in (1, 2):
        print(f"Inferring frozen fold {fold}/2 ...", flush=True)
        values, checkpoint_config, checkpoint = infer_metrica_checkpoint(
            checkpoint_paths[fold], fold, metadata, features, device
        )
        fold_probabilities[fold] = values
        configs[fold] = checkpoint_config
        checkpoints[fold] = checkpoint
    validate_transfer_configs(configs[1], configs[2])
    baseline = average_categorical_probabilities(fold_probabilities, weights)
    fused = fuse_probabilities(
        baseline,
        features["ball_location_valid"],
        evidence_10hz["home_ownership"],
        evidence_10hz["away_ownership"],
        evidence_10hz["kickoff_dead"],
        evidence_10hz["kickoff_live"],
        ownership_odds_multiplier=policy["ownership_odds_multiplier"],
        kickoff_odds_multiplier=policy["kickoff_odds_multiplier"],
        ownership_ball_missing_only=policy["ownership_ball_missing_only"],
    )

    baseline_labels_10 = {
        "raw": decode_probabilities_by_half(baseline, features["half_id"], Decoder("raw")),
        "decoded": decode_probabilities_by_half(baseline, features["half_id"], decoder),
    }
    fused_labels_10 = {
        "raw": decode_probabilities_by_half(fused, features["half_id"], Decoder("raw")),
        "decoded": decode_probabilities_by_half(fused, features["half_id"], decoder),
    }
    labels_25: dict[str, np.ndarray] = {}
    for prefix, variants in (("baseline", baseline_labels_10), ("smart_tag", fused_labels_10)):
        for variant, labels in variants.items():
            labels_25[f"{prefix}_{variant}"], valid = reconstruct_source_labels(
                labels, features["source_to_sample_index"]
            )
            if not np.array_equal(valid, features["source_scored"]):
                raise ValueError("Prediction coverage differs from the feature archive scored timeline")

    baseline_source, source_valid = reconstruct_source_probabilities(
        baseline, features["source_to_sample_index"]
    )
    fused_source, fused_valid = reconstruct_source_probabilities(
        fused, features["source_to_sample_index"]
    )
    if not np.array_equal(source_valid, fused_valid):
        raise ValueError("Baseline and fused source coverage disagree")

    output_dir.mkdir(parents=True)
    submissions_dir = output_dir / "submissions"
    predictions_dir = output_dir / "predictions"
    reports_dir = output_dir / "reports"
    for directory in (submissions_dir, predictions_dir, reports_dir):
        directory.mkdir()
    submission_paths = {
        name: write_submission(submissions_dir / f"{name}.csv", labels_25[name]) for name in VARIANTS
    }

    prediction_path = predictions_dir / "smart_tag_fusion.npz"
    np.savez_compressed(
        prediction_path,
        fold1_probabilities_10hz=fold_probabilities[1],
        fold2_probabilities_10hz=fold_probabilities[2],
        baseline_probabilities_10hz=baseline,
        fused_probabilities_10hz=fused,
        baseline_probabilities_25hz=baseline_source,
        fused_probabilities_25hz=fused_source,
        baseline_raw_labels_10hz=baseline_labels_10["raw"],
        baseline_decoded_labels_10hz=baseline_labels_10["decoded"],
        smart_tag_raw_labels_10hz=fused_labels_10["raw"],
        smart_tag_decoded_labels_10hz=fused_labels_10["decoded"],
        baseline_raw_labels_25hz=labels_25["baseline_raw"],
        baseline_decoded_labels_25hz=labels_25["baseline_decoded"],
        smart_tag_raw_labels_25hz=labels_25["smart_tag_raw"],
        smart_tag_decoded_labels_25hz=labels_25["smart_tag_decoded"],
        home_ownership_evidence_10hz=evidence_10hz["home_ownership"],
        away_ownership_evidence_10hz=evidence_10hz["away_ownership"],
        kickoff_dead_evidence_10hz=evidence_10hz["kickoff_dead"],
        kickoff_live_evidence_10hz=evidence_10hz["kickoff_live"],
        ball_location_valid_10hz=features["ball_location_valid"],
        ball_location_valid_25hz=_source_mask(
            features["ball_location_valid"], features["source_to_sample_index"]
        ),
        kickoff_boundary_25hz=evidence_25hz["kickoff_boundary"],
        prediction_valid_25hz=source_valid,
        sample_index=features["sample_index"],
        half_id=features["half_id"],
        source_frame_index=features["source_frame_index"],
        source_to_sample_index=features["source_to_sample_index"],
    )
    diagnostics = _label_free_diagnostics(
        baseline,
        fused,
        evidence_10hz,
        features["ball_location_valid"],
        baseline_labels_10,
        fused_labels_10,
    )
    audit["scored_timeline"] = {
        "frames": int(np.count_nonzero(features["source_scored"])),
        "frames_with_any_tag": int(
            np.count_nonzero(features["source_scored"] & (_coverage_from_events(events, n_source) > 0))
        ),
    }
    audit_path = reports_dir / "smart_tag_audit.json"
    diagnostics_path = reports_dir / "label_free_diagnostics.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    artifact_paths = {
        "predictions": prediction_path,
        "audit": audit_path,
        "diagnostics": diagnostics_path,
        **{f"submission_{name}": path for name, path in submission_paths.items()},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_label_free",
        "created_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "ground_truth_accessed": False,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "model_config": str(model_config_path),
        "model_config_sha256": _sha256(model_config_path),
        "feature_archive": str(feature_path),
        "feature_archive_sha256": _sha256(feature_path),
        "smart_tag_xml": str(xml_path),
        "smart_tag_xml_sha256": _sha256(xml_path),
        "model": {
            "name": spec.name,
            "run_dir": str(spec.run_dir),
            "run_name": metadata["run_name"],
            "ensemble_weights": {f"fold{fold}": weights[fold] for fold in (1, 2)},
            "decoder_selection": str(spec.decoder_selection),
            "decoder_selection_sha256": _sha256(spec.decoder_selection),
            "decoder_schema": decoder_report["schema_version"],
            "decoder": decoder.to_dict(),
            "checkpoints": {
                f"fold{fold}": {
                    "path": str(checkpoint_paths[fold]),
                    "sha256": _sha256(checkpoint_paths[fold]),
                    "epoch": int(checkpoints[fold]["epoch"]),
                }
                for fold in (1, 2)
            },
        },
        "policy": policy,
        "experiment_context": (
            "label_free_preregistered_candidate"
            if policy["ownership_ball_missing_only"]
            else "posthoc_all_frame_candidate_after_public_result"
        ),
        "timeline": {
            "model_fps": float(features["source_fps"]),
            "source_fps": fps,
            "n_model_samples": int(len(features["sample_index"])),
            "n_source_frames": n_source,
            "scored_frame_intervals": features["scored_frame_intervals"].tolist(),
        },
        "artifacts": {
            name: {"path": str(path.relative_to(output_dir)), "sha256": _sha256(path)}
            for name, path in artifact_paths.items()
        },
        "warning": "Candidate was frozen without labels. Do not change its policy after evaluation.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen label-free Smart Tag candidate: {output_dir}")
    print(
        f"Changed {diagnostics['decoded_label_changes']} decoded 10 Hz labels and "
        f"{diagnostics['raw_label_changes']} raw labels."
    )
    return output_dir


def _coverage_from_events(events: tuple[Any, ...], n_frames: int) -> np.ndarray:
    difference = np.zeros(n_frames + 1, dtype=np.int32)
    for event in events:
        difference[event.start_frame] += 1
        difference[event.end_frame] -= 1
    return np.cumsum(difference[:-1])


def _verify_frozen_run(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "frozen_label_free":
        raise ValueError("Run is not a frozen label-free Smart Tag candidate")
    if manifest.get("ground_truth_accessed") is not False or manifest.get("policy") not in SUPPORTED_POLICIES:
        raise ValueError("Frozen-run provenance or policy is invalid")
    for name, record in manifest.get("artifacts", {}).items():
        path = run_dir / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen artifact failed hash verification: {name}")
    if set(name.removeprefix("submission_") for name in manifest["artifacts"] if name.startswith("submission_")) != set(VARIANTS):
        raise ValueError("Frozen run does not contain exactly the four expected submissions")
    return manifest


def _masked_metrics(gt: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool) & (gt != pm.X)
    cm = pm.scored_confusion(pm.confusion(gt[selected], prediction[selected]))
    strict, conditional = pm.team_accuracy(cm)
    return {
        "n_frames": int(np.count_nonzero(selected)),
        "accuracy": pm.accuracy(cm),
        "macro_f1": pm.macro_f1(cm),
        "alive_dead_f1": pm.alive_dead_f1(cm),
        "team_acc_gt_alive": strict,
        "team_acc_both_alive": conditional,
        "confusion": cm.tolist(),
        "per_class": {
            name: dict(zip(("precision", "recall", "f1"), pm.per_class_prf(cm, index)))
            for index, name in enumerate(pm.SCORED)
        },
    }


def evaluate_frozen_run(
    run_dir: str | Path,
    ground_truth_path: str | Path,
    *,
    bootstrap_n: int = 1000,
) -> Path:
    """Evaluate the already-frozen candidate once; never recompute fusion here."""
    run_dir = Path(run_dir)
    ground_truth_path = Path(ground_truth_path)
    manifest = _verify_frozen_run(run_dir)
    output_dir = run_dir / "evaluation"
    complete_path = output_dir / "evaluation_manifest.json"
    if complete_path.exists():
        raise FileExistsError("This frozen candidate already has a completed evaluation")
    n_source = int(manifest["timeline"]["n_source_frames"])
    gt = _load_ground_truth(ground_truth_path, n_source)
    prediction_record = manifest["artifacts"]["predictions"]
    with np.load(run_dir / prediction_record["path"], allow_pickle=False) as archive:
        ball_valid = archive["ball_location_valid_25hz"].astype(bool)
        kickoff_boundary = archive["kickoff_boundary_25hz"].astype(bool)

    output_dir.mkdir()
    reports_dir = output_dir / "reports"
    reports_dir.mkdir()
    direct_reports: dict[str, dict[str, Any]] = {}
    rows = []
    diagnostics: dict[str, Any] = {}
    for name in VARIANTS:
        path = run_dir / manifest["artifacts"][f"submission_{name}"]["path"]
        frame = pm.read_label_csv(path)
        problem = pm.validate_submission(frame, n_source)
        if problem is not None:
            raise ValueError(f"Frozen submission {name} is invalid: {problem}")
        labels = pm.label_codes(frame)
        metrics = pm.evaluate(gt, labels, bootstrap_n=bootstrap_n)
        from .metrica_had import _json_safe_metrics

        safe = _json_safe_metrics(metrics)
        direct_reports[name] = safe
        rows.append(_comparison_row(name, safe))
        diagnostics[name] = {
            "ball_visible": _masked_metrics(gt, labels, ball_valid),
            "ball_missing": _masked_metrics(gt, labels, ~ball_valid),
            "kickoff_boundary": _masked_metrics(gt, labels, kickoff_boundary),
        }
        (reports_dir / f"{name}.json").write_text(json.dumps(safe, indent=2), encoding="utf-8")

    rows.sort(key=lambda row: (-row["spell_f1_mean"], -row["macro_f1"], row["submission"]))
    _write_comparison_csv(reports_dir / "comparison.csv", rows)
    (reports_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (reports_dir / "stratified_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    official_dir = output_dir / "official"
    _run_official_scorer(
        run_dir / "submissions", official_dir, ground_truth_path, bootstrap_n, direct_reports
    )

    by_name = {row["submission"]: row for row in rows}
    base = by_name["baseline_decoded"]
    fused = by_name["smart_tag_decoded"]
    primary = {
        "baseline": "baseline_decoded",
        "candidate": "smart_tag_decoded",
        "spell_f1_mean_delta": fused["spell_f1_mean"] - base["spell_f1_mean"],
        "macro_f1_delta": fused["macro_f1"] - base["macro_f1"],
        "accuracy_delta": fused["accuracy"] - base["accuracy"],
        "alive_dead_f1_delta": fused["alive_dead_f1"] - base["alive_dead_f1"],
        "team_acc_gt_alive_delta": fused["team_acc_gt_alive"] - base["team_acc_gt_alive"],
        "fragmentation_delta": fused["fragmentation"] - base["fragmentation"],
    }
    (reports_dir / "primary_comparison.json").write_text(
        json.dumps(primary, indent=2), encoding="utf-8"
    )
    evaluation_manifest = {
        "schema_version": "metrica_smart_tag_fusion_evaluation_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "frozen_manifest_sha256": _sha256(run_dir / "manifest.json"),
        "ground_truth": str(ground_truth_path),
        "ground_truth_sha256": _sha256(ground_truth_path),
        "bootstrap_draws": bootstrap_n,
        "n_public_frames": int(np.count_nonzero(gt != pm.X)),
        "official_scorer_verified": True,
        "experiment_context": manifest.get("experiment_context", "label_free_preregistered_candidate"),
        "public_window_warning": (
            "Post-hoc diagnostic after observing the first candidate; it is not unbiased evidence."
            if not manifest["policy"]["ownership_ball_missing_only"]
            else "One-time diagnostic over 7,500 frames and 17 possessions; do not tune the fusion policy."
        ),
        "primary_comparison": primary,
    }
    complete_path.write_text(json.dumps(evaluation_manifest, indent=2), encoding="utf-8")
    print("One-time public-window comparison (17 possessions; diagnostics only):")
    for row in rows:
        print(
            f"  {row['submission']:24s} spellF1={row['spell_f1_mean']:.4f} "
            f"macroF1={row['macro_f1']:.4f} possessions={row['n_pred_spells']}"
        )
    print(
        f"Primary decoded delta: spellF1={primary['spell_f1_mean_delta']:+.4f}, "
        f"macroF1={primary['macro_f1_delta']:+.4f}"
    )
    print(f"Saved: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="create the frozen label-free candidate")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate", help="evaluate one already-frozen candidate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    if args.command == "prepare":
        config = load_fusion_config(args.config)
        output = args.output_dir or Path(config["output_root"]) / datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S"
        )
        prepare_run(args.config, output, device_name=args.device)
    else:
        if args.bootstrap < 0:
            parser.error("--bootstrap must be non-negative")
        evaluate_frozen_run(args.run_dir, args.ground_truth, bootstrap_n=args.bootstrap)


if __name__ == "__main__":
    main()
