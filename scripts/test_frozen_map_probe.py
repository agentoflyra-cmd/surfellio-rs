"""Geometry checks for the offline perturbation probe."""

import unittest

import numpy as np

from scripts.frozen_map_probe import changed_scan


class PerturbationTests(unittest.TestCase):
    def test_clean_input_is_unchanged(self):
        points = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 1.0]])
        changed, count = changed_scan(points, np.eye(3), np.zeros(3), np.eye(3),
                                      np.zeros(3), 270, 41, "clean")
        np.testing.assert_array_equal(changed, points)
        self.assertEqual(count, 0)

    def test_occluder_is_reproducible_and_keeps_sensor_rays(self):
        angle = np.linspace(-0.08, 0.08, 100)
        points_l = np.column_stack((5 * np.cos(angle), 5 * np.sin(angle), np.zeros(100)))
        rotation_wi = np.eye(3)
        position_wi = np.array([2.0, -1.0, 0.5])
        rotation_il = np.eye(3)
        translation_il = np.array([0.1, 0.2, 0.3])
        points_w = points_l + translation_il + position_wi
        first, modified = changed_scan(points_w, rotation_wi, position_wi,
                                       rotation_il, translation_il, 270, 41, "moving_occluder")
        second, repeated = changed_scan(points_w, rotation_wi, position_wi,
                                         rotation_il, translation_il, 270, 41, "moving_occluder")
        np.testing.assert_array_equal(first, second)
        self.assertEqual(modified, repeated)
        self.assertEqual(modified, len(points_w))
        local = first - translation_il - position_wi
        np.testing.assert_allclose(local / np.linalg.norm(local, axis=1)[:, None],
                                   points_l / np.linalg.norm(points_l, axis=1)[:, None], atol=1e-12)
        self.assertTrue(np.all(np.linalg.norm(local, axis=1) < 3.0))


if __name__ == "__main__":
    unittest.main()
