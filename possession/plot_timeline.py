"""Plot prediction and available ground truth on the same video timeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from games import GAMES

EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))
import possession_metrics as metrics  # noqa: E402
from run_evaluation import find_gt  # noqa: E402

COLORS = {"H": "#2774AE", "A": "#D1495B", "D": "#8D99AE", "X": "#FFFFFF"}


def _draw_strip(axis, codes: np.ndarray, start: int, stop: int, y: float) -> None:
    labels = metrics.LABELS
    for segment_start, segment_stop, code in metrics.segments(codes, [(start, stop)]):
        axis.broken_barh(
            [(segment_start, segment_stop - segment_start)], (y, 0.75),
            facecolors=COLORS[labels[int(code)]], edgecolors="none")


def plot_timeline(gt: str, submission: Path) -> Path:
    """Save one aligned H/A/D strip pair for every available ground-truth run."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    gt_path = Path(find_gt(gt))
    gt_table = metrics.read_label_csv(gt_path)
    prediction_table = metrics.read_label_csv(submission)
    reason = metrics.validate_submission(gt_table, len(gt_table))
    if reason:
        raise ValueError("invalid ground truth: {}".format(reason))
    reason = metrics.validate_submission(prediction_table, len(gt_table))
    if reason:
        raise ValueError("invalid submission: {}".format(reason))

    gt_codes = metrics.label_codes(gt_table)
    prediction_codes = metrics.label_codes(prediction_table)
    runs = metrics.valid_runs(gt_codes)
    if not runs:
        raise ValueError("ground truth contains no graded frames")

    figure, axes = plt.subplots(
        len(runs), 1, figsize=(14, max(2.6, 2.4 * len(runs))), squeeze=False)
    fps = float(GAMES["BAR-ATH"]["fps"])
    for axis, (start, stop) in zip(axes[:, 0], runs):
        _draw_strip(axis, gt_codes, start, stop, 1.05)
        _draw_strip(axis, prediction_codes, start, stop, 0.05)
        axis.set_xlim(start, stop)
        axis.set_ylim(0, 2)
        axis.set_yticks([0.425, 1.425], labels=["Prediction", "Ground truth"])
        ticks = axis.get_xticks()
        axis.set_xticks(ticks, ["{:.1f}".format(value / fps / 60.0) for value in ticks])
        axis.set_xlabel("Tactical-video time (minutes)")
        axis.set_title("Frames {}-{}".format(start, stop - 1))
        axis.grid(axis="x", alpha=0.2)

    figure.suptitle("Possession timeline - {} ground-truth diagnostic window".format(gt))
    figure.legend(
        handles=[Patch(facecolor=COLORS[label], label=label) for label in ("H", "A", "D")],
        loc="upper right", ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output = submission.parent / "timeline_comparison.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("Wrote {}".format(output))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default="public")
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args(argv)
    plot_timeline(args.gt, args.submission)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
