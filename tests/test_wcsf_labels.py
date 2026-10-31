import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_engineering.wcsf_labels import (
    build_alive_dead_labels,
    save_labels,
    validate_labels,
)


HEADER = ["timestamp_ms", "group_id", "player_id", "AliveFlag"]


def write_votes(path, timestamp_votes):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for timestamp_ms, votes in timestamp_votes:
            for player_id, alive_flag in enumerate(votes):
                writer.writerow(
                    {
                        "timestamp_ms": timestamp_ms,
                        "group_id": 1 if player_id % 2 == 0 else 2,
                        "player_id": player_id,
                        "AliveFlag": alive_flag,
                    }
                )


def write_feature_timeline(path, timestamps, half_id=1):
    np.savez_compressed(
        path,
        timestamp_ms=np.asarray(timestamps, dtype=np.int64),
        half_id=np.full(len(timestamps), half_id, dtype=np.int8),
        sample_index=np.arange(len(timestamps), dtype=np.int64),
    )


class WcsfLabelTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.csv_path = base / "half.csv"
        self.feature_path = base / "features.npz"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_unanimous_and_conflicting_votes(self):
        timestamp_votes = [
            (0, [0, 0, 0, 0]),
            (100, [1, 1, 1, 1]),
            (200, [1, 1, 1, 1, 0]),
            (300, [0, 0, 0, 0, 1]),
        ]
        write_votes(self.csv_path, timestamp_votes)
        write_feature_timeline(self.feature_path, [0, 100, 200, 300])
        result = build_alive_dead_labels([self.csv_path], self.feature_path)

        np.testing.assert_array_equal(result["labels"], [0, 1, -1, -1])
        np.testing.assert_array_equal(result["label_valid"], [True, True, False, False])
        np.testing.assert_array_equal(result["alive_conflict"], [False, False, True, True])
        np.testing.assert_array_equal(result["alive_vote_count"], [0, 4, 4, 1])
        np.testing.assert_array_equal(result["dead_vote_count"], [4, 0, 1, 4])
        np.testing.assert_allclose(result["alive_vote_fraction"], [0, 1, 0.8, 0.2])
        self.assertEqual(result["labels"].dtype, np.int8)
        self.assertEqual(str(result["aggregation_policy"]), "strict_unanimity")

    def test_all_non_unanimous_votes_are_invalid(self):
        for votes in ([1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0]):
            with self.subTest(votes=votes):
                write_votes(self.csv_path, [(0, votes)])
                write_feature_timeline(self.feature_path, [0])
                result = build_alive_dead_labels([self.csv_path], self.feature_path)
                self.assertEqual(result["labels"][0], -1)
                self.assertFalse(result["label_valid"][0])

    def test_missing_and_non_binary_flags_are_rejected(self):
        for value in ("", "2", "nan", "ALIVE"):
            with self.subTest(value=value):
                write_votes(self.csv_path, [(0, [value])])
                write_feature_timeline(self.feature_path, [0])
                with self.assertRaisesRegex(ValueError, "AliveFlag must be 0 or 1"):
                    build_alive_dead_labels([self.csv_path], self.feature_path)

    def test_duplicate_player_vote_is_rejected(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            writer.writeheader()
            row = {"timestamp_ms": 0, "group_id": 1, "player_id": 7, "AliveFlag": 1}
            writer.writerow(row)
            writer.writerow(row)
        write_feature_timeline(self.feature_path, [0])
        with self.assertRaisesRegex(ValueError, "Duplicate AliveFlag vote"):
            build_alive_dead_labels([self.csv_path], self.feature_path)

    def test_feature_timeline_mismatch_is_rejected(self):
        write_votes(self.csv_path, [(0, [1]), (100, [0])])
        write_feature_timeline(self.feature_path, [0, 200])
        with self.assertRaisesRegex(ValueError, "does not align exactly"):
            build_alive_dead_labels([self.csv_path], self.feature_path)

    def test_validation_rejects_invalid_value(self):
        write_votes(self.csv_path, [(0, [1])])
        write_feature_timeline(self.feature_path, [0])
        result = build_alive_dead_labels([self.csv_path], self.feature_path)
        result["labels"][0] = 3
        with self.assertRaisesRegex(ValueError, "values -1"):
            validate_labels(result, self.feature_path)

    def test_archive_round_trip(self):
        write_votes(self.csv_path, [(0, [1]), (100, [0])])
        write_feature_timeline(self.feature_path, [0, 100])
        result = build_alive_dead_labels([self.csv_path], self.feature_path)
        output = Path(self.temp_dir.name) / "labels.npz"
        save_labels(result, output, self.feature_path)
        with np.load(output, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["labels"], [1, 0])
            np.testing.assert_array_equal(archive["label_valid"], [True, True])


class FullWcsfLabelAcceptanceTests(unittest.TestCase):
    def test_full_dataset_contract(self):
        root = Path(__file__).resolve().parents[1]
        data_dir = root / "data" / "open_data_groningen"
        paths = [data_dir / "positions_FIRST_HALF.csv", data_dir / "positions_SECOND_HALF.csv"]
        feature_path = data_dir / "processed" / "wcsf_player_team_features.npz"
        if not all(path.exists() for path in [*paths, feature_path]):
            self.skipTest("Local gated WCSF data or generated features are unavailable")

        result = build_alive_dead_labels(paths, feature_path)
        labels, counts = np.unique(result["labels"], return_counts=True)
        self.assertEqual(dict(zip(labels.tolist(), counts.tolist())), {-1: 39, 0: 26776, 1: 30559})
        self.assertEqual(len(result["labels"]), 57374)
        self.assertEqual(
            np.count_nonzero(
                (result["alive_vote_count"] > 0) & (result["dead_vote_count"] > 0)
            ),
            39,
        )
        self.assertEqual(np.count_nonzero(~result["label_valid"]), 39)
        self.assertTrue(np.all(result["labels"][~result["label_valid"]] == -1))


if __name__ == "__main__":
    unittest.main()
