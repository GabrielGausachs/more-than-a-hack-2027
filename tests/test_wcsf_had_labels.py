import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_engineering.wcsf_had_labels import (
    A,
    D,
    H,
    UNKNOWN,
    build_had_labels,
    save_had_labels,
    validate_had_labels,
)


HEADER = ["timestamp_ms", "group_id", "player_id", "AliveFlag", "PossessionFlag"]


def write_votes(path, frames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for timestamp, rows in frames:
            for player_id, (alive, possession) in enumerate(rows):
                writer.writerow(
                    {
                        "timestamp_ms": timestamp,
                        "group_id": 1 if player_id % 2 == 0 else 2,
                        "player_id": player_id,
                        "AliveFlag": alive,
                        "PossessionFlag": possession,
                    }
                )


def write_timeline(path, timestamps):
    np.savez_compressed(
        path,
        timestamp_ms=np.asarray(timestamps, dtype=np.int64),
        half_id=np.ones(len(timestamps), dtype=np.int8),
        sample_index=np.arange(len(timestamps), dtype=np.int64),
    )


class WcsfHadLabelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.csv_path = root / "votes.csv"
        self.feature_path = root / "features.npz"

    def tearDown(self):
        self.temporary.cleanup()

    def test_strict_rules_and_dead_precedence(self):
        frames = [
            (0, [(1, 1), (1, 1)]),
            (100, [(1, 2), (1, 2)]),
            (200, [(0, 1), (0, 2)]),
            (300, [(1, 1), (1, 2)]),
            (400, [(1, 1), (0, 1)]),
        ]
        write_votes(self.csv_path, frames)
        write_timeline(self.feature_path, [0, 100, 200, 300, 400])
        result = build_had_labels([self.csv_path], self.feature_path)
        np.testing.assert_array_equal(result["labels"], [H, A, D, UNKNOWN, UNKNOWN])
        np.testing.assert_array_equal(
            result["label_valid"], [True, True, True, False, False]
        )
        self.assertTrue(result["possession_conflict"][2])
        self.assertTrue(result["possession_conflict"][3])
        self.assertTrue(result["alive_conflict"][4])

    def test_duplicate_and_invalid_possession_are_rejected(self):
        write_votes(self.csv_path, [(0, [(1, 3)])])
        write_timeline(self.feature_path, [0])
        with self.assertRaisesRegex(ValueError, "PossessionFlag must be 1 or 2"):
            build_had_labels([self.csv_path], self.feature_path)

    def test_round_trip_and_validation(self):
        write_votes(self.csv_path, [(0, [(1, 1)]), (100, [(0, 2)])])
        write_timeline(self.feature_path, [0, 100])
        result = build_had_labels([self.csv_path], self.feature_path)
        output = Path(self.temporary.name) / "had.npz"
        save_had_labels(result, output, self.feature_path)
        with np.load(output, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["labels"], [H, D])
            np.testing.assert_array_equal(archive["label_valid"], [True, True])
        result["labels"][0] = D
        with self.assertRaisesRegex(ValueError, "strict H/A/D"):
            validate_had_labels(result, self.feature_path)


class FullWcsfHadAcceptanceTests(unittest.TestCase):
    def test_full_dataset_contract(self):
        root = Path(__file__).resolve().parents[1]
        data = root / "data/open_data_groningen"
        paths = [data / "positions_FIRST_HALF.csv", data / "positions_SECOND_HALF.csv"]
        features = data / "processed/wcsf_player_team_features.npz"
        if not all(path.exists() for path in [*paths, features]):
            self.skipTest("Local gated WCSF data or generated features are unavailable")
        result = build_had_labels(paths, features)
        values, counts = np.unique(result["labels"], return_counts=True)
        self.assertEqual(
            dict(zip(values.tolist(), counts.tolist())),
            {UNKNOWN: 110, H: 18249, A: 12239, D: 26776},
        )
        self.assertEqual(np.count_nonzero(result["alive_conflict"]), 39)
        self.assertEqual(np.count_nonzero(result["possession_conflict"]), 90)
        self.assertEqual(
            np.count_nonzero(
                result["label_valid"]
                & result["possession_conflict"]
                & (result["labels"] == D)
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
