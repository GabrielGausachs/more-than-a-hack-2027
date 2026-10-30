import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_engineering.wcsf_features import (
    TEAM_FEATURE_NAMES,
    build_features,
    build_player_features,
    build_team_features,
    load_wcsf_positions,
    save_features,
    validate_features,
)


HEADER = ["timestamp_ms", "group_id", "player_id", "x", "y", "speed"]


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


class WcsfFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "half.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_orientation_normalization_masks_and_dense_sorting(self):
        write_rows(
            self.path,
            [
                {"timestamp_ms": 0, "group_id": 1, "player_id": 20, "x": 52.5, "y": 34, "speed": ""},
                {"timestamp_ms": 0, "group_id": 1, "player_id": 10, "x": -54.6, "y": -35, "speed": 2},
                {"timestamp_ms": 0, "group_id": 2, "player_id": 30, "x": -52.5, "y": -34, "speed": 4},
                {"timestamp_ms": 100, "group_id": 1, "player_id": 10, "x": "", "y": 1, "speed": 3},
                {"timestamp_ms": 100, "group_id": 2, "player_id": 30, "x": 0, "y": 0, "speed": 1},
            ],
        )
        result = build_player_features(load_wcsf_positions([self.path]))

        # Home is sorted by player_id; valid out-of-bounds coordinates are not clipped.
        self.assertLess(result["players"][0, 0, 0, 0], 0.0)
        np.testing.assert_allclose(result["players"][0, 0, 1, :2], [1.0, 1.0])
        # Away is rotated before normalization and therefore also maps to (1, 1).
        np.testing.assert_allclose(result["players"][0, 1, 0, :2], [1.0, 1.0])
        self.assertFalse(result["player_speed_valid"][0, 0, 1])
        self.assertEqual(result["players"][0, 0, 1, 2], 0.0)
        # Speed remains independently valid when the location is missing.
        self.assertFalse(result["player_location_valid"][1, 0, 0])
        self.assertTrue(result["player_speed_valid"][1, 0, 0])
        self.assertEqual(result["players"][1, 0, 0, 2], 3.0)
        self.assertFalse(result["player_location_valid"][0, 0, 2:].any())
        self.assertTrue(np.all(result["players"][0, 0, 2:] == 0.0))

    def test_team_statistics_and_validity(self):
        players = np.zeros((2, 2, 24, 3), dtype=np.float32)
        location_valid = np.zeros((2, 2, 24), dtype=bool)
        speed_valid = np.zeros_like(location_valid)
        location_valid[0, 0, :3] = True
        speed_valid[0, 0, :3] = True
        players[0, 0, :3, 2] = [1, 2, 6]

        features, valid = build_team_features(players, location_valid, speed_valid)
        expected = [3.0, 2.0, np.std([1, 2, 6], ddof=0), 3.0, 3.0, 1.0]
        np.testing.assert_allclose(features[0, 0], expected)
        self.assertTrue(valid[0, 0].all())
        self.assertTrue(valid[1, 1, 3])
        self.assertTrue(valid[1, 1, 4])
        self.assertFalse(valid[1, 1, [0, 1, 2, 5]].any())
        self.assertTrue(np.all(features[1, 1] == 0.0))
        self.assertEqual(TEAM_FEATURE_NAMES.tolist()[2], "speed_std")

    def test_duplicate_player_is_rejected(self):
        row = {"timestamp_ms": 0, "group_id": 1, "player_id": 10, "x": 0, "y": 0, "speed": 1}
        write_rows(self.path, [row, row])
        with self.assertRaisesRegex(ValueError, "Duplicate player observation"):
            build_player_features(load_wcsf_positions([self.path]))

    def test_more_than_24_players_is_rejected(self):
        rows = [
            {"timestamp_ms": 0, "group_id": 1, "player_id": player, "x": 0, "y": 0, "speed": 1}
            for player in range(25)
        ]
        write_rows(self.path, rows)
        with self.assertRaisesRegex(ValueError, "More than 24"):
            build_player_features(load_wcsf_positions([self.path]))

    def test_archive_round_trip(self):
        write_rows(
            self.path,
            [
                {"timestamp_ms": 0, "group_id": 1, "player_id": 1, "x": 0, "y": 0, "speed": 1},
                {"timestamp_ms": 0, "group_id": 2, "player_id": 2, "x": 0, "y": 0, "speed": 2},
                {"timestamp_ms": 100, "group_id": 1, "player_id": 1, "x": 1, "y": 1, "speed": 3},
                {"timestamp_ms": 100, "group_id": 2, "player_id": 2, "x": -1, "y": -1, "speed": 4},
            ],
        )
        result = build_features([self.path])
        validate_features(result)
        output = Path(self.temp_dir.name) / "features.npz"
        save_features(result, output)
        with np.load(output) as archive:
            self.assertEqual(archive["players"].shape, (2, 2, 24, 3))
            self.assertEqual(archive["schema_version"].item(), "1.0")


class FullWcsfAcceptanceTests(unittest.TestCase):
    def test_full_dataset_contract(self):
        data_dir = Path(__file__).resolve().parents[1] / "data" / "open_data_groningen"
        paths = [data_dir / "positions_FIRST_HALF.csv", data_dir / "positions_SECOND_HALF.csv"]
        if not all(path.exists() for path in paths):
            self.skipTest("Local gated WCSF data is not available")

        result = build_features(paths)
        self.assertEqual(result["players"].shape, (57374, 2, 24, 3))
        self.assertEqual(result["team_features"].shape, (57374, 2, 6))
        self.assertEqual(np.sum(result["half_id"] == 1), 28014)
        self.assertEqual(np.sum(result["half_id"] == 2), 29360)
        self.assertTrue(np.isfinite(result["players"]).all())
        self.assertTrue(np.isfinite(result["team_features"]).all())


if __name__ == "__main__":
    unittest.main()
