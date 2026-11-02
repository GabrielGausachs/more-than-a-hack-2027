"""Diagnostic timelines for temporal classification predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


BINARY_COLORS = {0: "#b8b8b8", 1: "#2ca25f"}
HAD_COLORS = {0: "#2166ac", 1: "#d95f02", 2: "#b8b8b8"}


def _group_runs_by_half(
    runs: Sequence[object], allowed_labels: Sequence[int] = (0, 1)
) -> tuple[tuple[int, tuple[object, ...]], ...]:
    """Validate and chronologically group prediction runs by half."""
    if not runs:
        raise ValueError("At least one prediction run is required")
    allowed = tuple(allowed_labels)
    if not allowed:
        raise ValueError("At least one timeline class is required")

    grouped: dict[int, list[object]] = {}
    for run in runs:
        if int(run.half_id) not in (1, 2):
            raise ValueError("Prediction run half_id must be 1 or 2")
        labels = np.asarray(run.labels)
        predictions = np.asarray(run.predictions)
        if run.start < 0 or run.end <= run.start or run.length != len(labels):
            raise ValueError("Prediction run has invalid boundaries")
        if labels.ndim != 1 or predictions.shape != labels.shape:
            raise ValueError("Prediction labels and predictions must be aligned 1D arrays")
        if not np.isin(labels, allowed).all() or not np.isin(predictions, allowed).all():
            label = "binary" if allowed == (0, 1) else "declared"
            raise ValueError(f"Prediction timelines require {label} class values")
        grouped.setdefault(int(run.half_id), []).append(run)

    result = []
    for half_id in sorted(grouped):
        half_runs = tuple(sorted(grouped[half_id], key=lambda run: run.start))
        if any(left.end > right.start for left, right in zip(half_runs, half_runs[1:])):
            raise ValueError(f"Prediction runs overlap in half {half_id}")
        result.append((half_id, half_runs))
    return tuple(result)


def _segments(values: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return tuple(
        (int(start), int(end), int(values[start]))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    )


def _validate_aligned_run_sets(
    run_sets: Mapping[str, Sequence[object]], allowed_labels: Sequence[int]
) -> tuple[tuple[int, tuple[object, ...]], ...]:
    if not run_sets:
        raise ValueError("At least one prediction set is required")
    grouped_sets = {
        name: _group_runs_by_half(runs, allowed_labels) for name, runs in run_sets.items()
    }
    reference = next(iter(grouped_sets.values()))
    reference_flat = [run for _, runs in reference for run in runs]
    for name, grouped in grouped_sets.items():
        runs = [run for _, half_runs in grouped for run in half_runs]
        if len(runs) != len(reference_flat):
            raise ValueError(f"Prediction set {name!r} has different temporal runs")
        for reference_run, run in zip(reference_flat, runs):
            if (
                (run.start, run.end, run.half_id)
                != (reference_run.start, reference_run.end, reference_run.half_id)
                or not np.array_equal(run.labels, reference_run.labels)
            ):
                raise ValueError(f"Prediction set {name!r} is not timeline-aligned")
    return reference


def save_prediction_comparison_timeline(
    run_sets: Mapping[str, Sequence[object]],
    source_fps: float,
    output_path: str | Path,
    title: str,
    *,
    class_names: Sequence[str],
    class_colors: Mapping[int, str],
) -> Path:
    """Save ground truth and one or more aligned prediction bands per half."""
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be a positive finite value")
    if not class_names or set(class_colors) != set(range(len(class_names))):
        raise ValueError("Class names and colors must define the same zero-based classes")
    grouped = _validate_aligned_run_sets(run_sets, tuple(range(len(class_names))))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Timeline plotting requires Matplotlib; install requirements-starter.txt"
        ) from exc

    set_names = tuple(run_sets)
    grouped_lookup = {
        name: dict(_group_runs_by_half(runs, tuple(range(len(class_names)))))
        for name, runs in run_sets.items()
    }
    n_bands = 1 + len(set_names)
    figure, axes = plt.subplots(
        len(grouped),
        1,
        figsize=(16, (1.1 + 0.65 * n_bands) * len(grouped) + 0.8),
        squeeze=False,
    )
    for axis, (half_id, ground_truth_runs) in zip(axes[:, 0], grouped):
        origin = ground_truth_runs[0].start
        bands = [("Ground truth", ground_truth_runs, "labels")]
        bands.extend(
            (name, grouped_lookup[name][half_id], "predictions") for name in set_names
        )
        ticks, tick_labels = [], []
        for band_index, (band_name, runs, attribute) in enumerate(bands):
            y = n_bands - band_index - 0.85
            ticks.append(y + 0.35)
            tick_labels.append(band_name)
            for run in runs:
                for start, end, label in _segments(np.asarray(getattr(run, attribute))):
                    start_minutes = (run.start - origin + start) / source_fps / 60.0
                    width_minutes = (end - start) / source_fps / 60.0
                    axis.broken_barh(
                        [(start_minutes, width_minutes)],
                        (y, 0.7),
                        facecolors=class_colors[label],
                        edgecolors="white",
                        linewidth=0.25,
                    )

        end_minutes = (
            max(run.end for run in ground_truth_runs) - origin
        ) / source_fps / 60.0
        axis.set_xlim(0.0, end_minutes)
        axis.set_ylim(0.0, float(n_bands))
        axis.set_yticks(ticks, labels=tick_labels)
        axis.set_xlabel("Elapsed time within half (minutes)")
        axis.set_title("First half" if half_id == 1 else "Second half")
        axis.grid(axis="x", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)

    figure.suptitle(title)
    figure.legend(
        handles=tuple(
            Patch(facecolor=class_colors[index], label=name)
            for index, name in enumerate(class_names)
        ),
        loc="upper center",
        ncol=len(class_names),
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_prediction_timeline(
    runs: Sequence[object],
    source_fps: float,
    output_path: str | Path,
    title: str,
    *,
    class_names: Sequence[str] = ("DEAD", "ALIVE"),
    class_colors: Mapping[int, str] | None = None,
    prediction_label: str = "Prediction",
) -> Path:
    """Save one ground-truth/prediction timeline, binary by default."""
    colors = BINARY_COLORS if class_colors is None else class_colors
    return save_prediction_comparison_timeline(
        {prediction_label: runs},
        source_fps,
        output_path,
        title,
        class_names=class_names,
        class_colors=colors,
    )
