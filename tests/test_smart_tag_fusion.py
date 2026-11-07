"""Synthetic tests for conservative Smart Tag parsing and probability fusion."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_engineering.smart_tags import (
    audit_smart_tags,
    build_smart_tag_evidence,
    load_smart_tags,
)
from modeling.alive_dead.smart_tag_fusion import fuse_probabilities


XML = """<?xml version="1.0"?>
<file><ALL_INSTANCES>
  <instance><start>1</start><end>5</end><code>BUILD UP</code>
    <label><group>Team</group><text>Barcelona</text></label>
    <label><group>Half</group><text>1st Half</text></label>
  </instance>
  <instance><start>1</start><end>5</end><code>DEFENDING IN ATTACKING THIRD</code>
    <label><group>Team</group><text>Atletic Club</text></label>
    <label><group>Half</group><text>1st Half</text></label>
  </instance>
  <instance><start>6</start><end>8</end><code>PROGRESSION</code>
    <label><group>Team</group><text>Atletic Club</text></label>
    <label><group>Half</group><text>1st Half</text></label>
  </instance>
  <instance><start>9</start><end>10</end><code>SET PIECES</code>
    <label><group>Team</group><text>N/A</text></label>
    <label><group>Half</group><text>1st Half</text></label>
    <label><group>Type</group><text>Kick Off (Start)</text></label>
  </instance>
</ALL_INSTANCES></file>"""


class SmartTagFusionTests(unittest.TestCase):
    def _events(self):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "tags.xml"
        path.write_text(XML, encoding="utf-8")
        return temporary, load_smart_tags(path, fps=10.0, n_frames=120)

    def test_parser_audit_and_half_open_intervals(self):
        temporary, events = self._events()
        self.addCleanup(temporary.cleanup)
        self.assertEqual((events[0].start_frame, events[0].end_frame), (10, 50))
        audit = audit_smart_tags(events, n_frames=120, fps=10.0)
        self.assertEqual(audit["n_events"], 4)
        self.assertEqual(audit["team_counts"], {"Atletic Club": 2, "Barcelona": 1, "N/A": 1})
        self.assertEqual(audit["coverage"]["frames_two_or_more"], 40)

    def test_complementary_tags_collapse_and_conflicts_cancel(self):
        temporary, events = self._events()
        self.addCleanup(temporary.cleanup)
        evidence = build_smart_tag_evidence(
            events, n_frames=120, fps=10.0, ownership_taper_seconds=1.0, kickoff_live_seconds=2.0
        )
        # BUILD UP and its complementary defensive tag both mean home possession; max keeps 1.0.
        self.assertAlmostEqual(float(evidence["home_ownership"][30]), 1.0)
        self.assertEqual(float(evidence["away_ownership"][30]), 0.0)
        # Add equal opposing arrays directly: hierarchical fusion must cancel their net effect.
        probabilities = np.tile(np.array([[0.4, 0.3, 0.3]], dtype=np.float32), (3, 1))
        equal = np.ones(3, dtype=np.float32)
        zeros = np.zeros(3, dtype=np.float32)
        result = fuse_probabilities(probabilities, np.zeros(3, bool), equal, equal, zeros, zeros)
        np.testing.assert_allclose(result, probabilities, atol=1e-6)

    def test_ownership_only_preserves_alive_probability_and_requires_missing_ball(self):
        probabilities = np.array([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]], dtype=np.float32)
        home = np.ones(2, dtype=np.float32)
        zeros = np.zeros(2, dtype=np.float32)
        result = fuse_probabilities(
            probabilities, np.array([False, True]), home, zeros, zeros, zeros
        )
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        np.testing.assert_allclose(result[:, :2].sum(axis=1), probabilities[:, :2].sum(axis=1))
        self.assertGreater(result[0, 0], probabilities[0, 0])
        np.testing.assert_allclose(result[1], probabilities[1], atol=1e-6)
        self.assertTrue(np.array_equal(result[1], probabilities[1]))

    def test_zero_evidence_is_bit_identical(self):
        probabilities = np.array([[0.12345678, 0.2345679, 0.64197534]], dtype=np.float32)
        zeros = np.zeros(1, dtype=np.float32)
        result = fuse_probabilities(probabilities, np.zeros(1, bool), zeros, zeros, zeros, zeros)
        self.assertTrue(np.array_equal(result, probabilities))

    def test_all_frame_policy_applies_ownership_with_visible_ball(self):
        probabilities = np.array([[0.3, 0.3, 0.4]], dtype=np.float32)
        ones = np.ones(1, dtype=np.float32)
        zeros = np.zeros(1, dtype=np.float32)
        result = fuse_probabilities(
            probabilities,
            np.ones(1, bool),
            ones,
            zeros,
            zeros,
            zeros,
            ownership_ball_missing_only=False,
        )
        self.assertGreater(result[0, 0], probabilities[0, 0])
        np.testing.assert_allclose(result[0, :2].sum(), probabilities[0, :2].sum())

    def test_kickoff_changes_alive_dead_but_not_conditional_owner(self):
        probabilities = np.array([[0.4, 0.2, 0.4], [0.4, 0.2, 0.4]], dtype=np.float32)
        zeros = np.zeros(2, dtype=np.float32)
        result = fuse_probabilities(
            probabilities,
            np.ones(2, bool),
            zeros,
            zeros,
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        )
        self.assertGreater(result[0, 2], probabilities[0, 2])
        self.assertLess(result[1, 2], probabilities[1, 2])
        np.testing.assert_allclose(result[:, 0] / result[:, 1], [2.0, 2.0], atol=1e-6)

    def test_kickoff_evidence_uses_interval_then_two_second_decay(self):
        temporary, events = self._events()
        self.addCleanup(temporary.cleanup)
        evidence = build_smart_tag_evidence(
            events, n_frames=120, fps=10.0, ownership_taper_seconds=1.0, kickoff_live_seconds=2.0
        )
        self.assertEqual(float(evidence["kickoff_dead"][90]), 1.0)
        self.assertEqual(float(evidence["kickoff_dead"][99]), 1.0)
        self.assertEqual(float(evidence["kickoff_dead"][100]), 0.0)
        self.assertGreater(float(evidence["kickoff_live"][100]), 0.9)
        self.assertAlmostEqual(float(evidence["kickoff_live"][119]), 0.025, places=6)


if __name__ == "__main__":
    unittest.main()
