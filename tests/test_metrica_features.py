"""Tests for WCSF-compatible Metrica player and team features."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_engineering.metrica_features import (
    build_resampling_index,
    load_metrica_metadata,
    load_selected_player_features,
    validate_metrica_features,
)
from feature_engineering.wcsf_features import build_team_features


def metadata_xml(path: Path) -> None:
    players = []
    for slot in range(46):
        team_id = "home" if slot < 24 else "away"
        players.append(f'<Player id="{slot + 1}" teamId="{team_id}"/>')
    path.write_text(
        """<main><Metadata><GlobalConfig><FrameRate>25.0</FrameRate></GlobalConfig>
        <Sessions><Session><FieldSize><Width>105</Width><Height>68</Height></FieldSize>
        </Session></Sessions><Teams>
        <Team id="home"><Name>Barcelona</Name><ProviderTeamsParameters>
        <ProviderParameter><Name>attack_direction_first_half</Name>
        <Value>left_to_right</Value></ProviderParameter></ProviderTeamsParameters></Team>
        <Team id="away"><Name>Atletic Club</Name></Team></Teams><Players>"""
        + "".join(players)
        + "</Players></Metadata></main>",
        encoding="utf-8",
    )


def atd_group(x="NaN", y="NaN", speed="NaN") -> str:
    return f"{x},{y},{speed}"


class ResamplingTests(unittest.TestCase):
    def test_nearest_frame_resampling_prefers_earlier_ties(self):
        result = build_resampling_index(30, [(10, 20), (22, 29)])
        np.testing.assert_array_equal(result["source_frame_index"][:5], [10, 12, 15, 17, 20])
        np.testing.assert_array_equal(result["timestamp_ms"][:5], [400, 500, 600, 700, 800])
        self.assertEqual(result["half_id"].tolist(), [1] * 5 + [2] * 3)
        self.assertTrue(np.all(result["source_to_sample_index"][:10] == -1))
        self.assertTrue(np.all(result["source_to_sample_index"][21:22] == -1))
        self.assertTrue(np.all(result["source_to_sample_index"][10:21] >= 0))


class PlayerFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.xml_path = self.root / "tracking.xml"
        self.raw_path = self.root / "tracking.txt"
        metadata_xml(self.xml_path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_metadata_team_mapping(self):
        metadata = load_metrica_metadata(self.xml_path)
        self.assertEqual(metadata["team_names"].tolist(), ["Barcelona", "Atletic Club"])
        self.assertEqual(np.count_nonzero(metadata["slot_team_index"] == 0), 24)
        self.assertEqual(np.count_nonzero(metadata["slot_team_index"] == 1), 22)
        self.assertEqual(metadata["slot_within_team"][-1], 21)

    def test_slot_preservation_masks_rotation_and_padding(self):
        metadata = load_metrica_metadata(self.xml_path)
        first = [atd_group() for _ in range(46)]
        second = [atd_group() for _ in range(46)]
        first[0] = atd_group("1.2", "-0.1", "NaN")
        first[24] = atd_group("0.2", "0.3", "4")
        second[0] = atd_group("0.2", "0.3", "5")
        second[1] = atd_group("NaN", "0.4", "3")
        second[24] = atd_group("1.2", "-0.1", "NaN")
        self.raw_path.write_text(
            f"1:{';'.join(first)}:1.2,-0.1,7\n"
            f"2:{';'.join(second)}:0.2,0.3,NaN\n",
            encoding="utf-8",
        )
        alignment = {
            "source_frame_index": np.array([0, 1], dtype=np.int64),
            "half_id": np.array([1, 2], dtype=np.int8),
        }
        result = load_selected_player_features(
            self.raw_path, metadata, alignment, n_source_frames=2
        )

        np.testing.assert_allclose(result["players"][0, 0, 0, :2], [1.2, -0.1])
        np.testing.assert_allclose(result["players"][0, 1, 0], [0.2, 0.3, 4.0])
        np.testing.assert_allclose(result["players"][1, 0, 0], [0.8, 0.7, 5.0])
        np.testing.assert_allclose(result["players"][1, 1, 0, :2], [-0.2, 1.1])
        self.assertTrue(result["player_location_valid"][0, 0, 0])
        self.assertFalse(result["player_speed_valid"][0, 0, 0])
        self.assertFalse(result["player_location_valid"][1, 0, 1])
        self.assertTrue(result["player_speed_valid"][1, 0, 1])
        self.assertEqual(result["players"][1, 0, 1, 2], 3.0)
        self.assertFalse(result["player_location_valid"][:, 1, 22:].any())
        self.assertTrue(np.all(result["players"][:, 1, 22:] == 0.0))
        np.testing.assert_allclose(result["ball"][0], [1.2, -0.1, 7.0])
        np.testing.assert_allclose(result["ball"][1, :2], [0.8, 0.7])
        self.assertTrue(result["ball_location_valid"].all())
        self.assertEqual(result["ball_speed_valid"].tolist(), [True, False])

        team_features, team_valid = build_team_features(
            result["players"],
            result["player_location_valid"],
            result["player_speed_valid"],
        )
        self.assertEqual(team_features[0, 1, 0], 4.0)
        self.assertEqual(team_features[1, 0, 4], 2.0)
        self.assertTrue(team_valid[0, 1].all())


class FullArchiveTests(unittest.TestCase):
    def test_generated_archive_contract(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "data/metrica/processed/metrica_player_team_ball_features.npz"
        if not path.exists():
            self.skipTest("Generated Metrica feature archive is unavailable")
        with np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        validate_metrica_features(result)
        self.assertEqual(result["players"].shape, (63543, 2, 24, 3))
        self.assertEqual(result["team_features"].shape, (63543, 2, 6))
        self.assertEqual(result["ball"].shape, (63543, 3))
        self.assertEqual(np.count_nonzero(result["half_id"] == 1), 31787)
        self.assertEqual(np.count_nonzero(result["half_id"] == 2), 31756)
        self.assertEqual(result["players"].dtype, np.float32)
        self.assertEqual(result["player_location_valid"].dtype, np.bool_)
        self.assertEqual(result["team_names"].tolist(), ["Barcelona", "Atletic Club"])
        self.assertTrue(np.isfinite(result["players"]).all())
        self.assertTrue(np.isfinite(result["team_features"]).all())

        wcsf_path = (
            root
            / "data/open_data_groningen/processed/wcsf_player_team_ball_features.npz"
        )
        if wcsf_path.exists():
            with np.load(wcsf_path, allow_pickle=False) as wcsf:
                for name in (
                    "player_feature_names",
                    "ball_feature_names",
                    "team_feature_names",
                    "team_order",
                    "source_fps",
                    "pitch_length_m",
                    "pitch_width_m",
                    "schema_version",
                ):
                    np.testing.assert_array_equal(result[name], wcsf[name])


if __name__ == "__main__":
    unittest.main()
