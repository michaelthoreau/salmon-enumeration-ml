import unittest

from infer_video_detr import _FuzzyCrossingEstimate
from simulate_fuzzy_count import run_no_error_simulation


class FuzzyCountTest(unittest.TestCase):
    def test_constant_flux_is_independent_of_sampling_stride(self):
        for stride in (1, 3, 10):
            estimate = _FuzzyCrossingEstimate(100, 30, 5)
            for frame in range(0, 31, stride):
                result = estimate.update(frame, 20, [[0, 10]])
            self.assertAlmostEqual(result['down'], 2)
            self.assertEqual(result['up'], 0)

    def test_opposing_flows_do_not_cancel_total(self):
        estimate = _FuzzyCrossingEstimate(100, 30, 5)
        estimate.update(100, 20, [[0, 10], [0, -10]])
        result = estimate.update(130, 20, [[0, 10], [0, -10]])
        self.assertAlmostEqual(result['up'], 1)
        self.assertAlmostEqual(result['down'], 1)
        self.assertAlmostEqual(result['total'], 2)
        self.assertEqual(result['net'], 0)

    def test_sideways_or_missing_motion_has_zero_flux(self):
        estimate = _FuzzyCrossingEstimate(100, 30, 5)
        estimate.update(0, 20, [[100, 0]])
        result = estimate.update(30, 20, [])
        self.assertEqual(result['total'], 0)

    def test_count_is_smoothed_and_time_uses_source_indices(self):
        estimate = _FuzzyCrossingEstimate(100, 30, 2)
        estimate.update(60, 10, [[0, 10]])
        result = estimate.update(90, 30, [[0, 10]])
        self.assertEqual(result['moving_average_count'], 20)
        self.assertAlmostEqual(result['down'], 1.5)

    def test_perfect_detection_baseline(self):
        summary, rows = run_no_error_simulation(
            fish_count=10,
            duration_s=60.0,
            fps=30.0,
            height=100.0,
            line_position=0.5,
            speed_px_s=5.0,
            arrival_interval_s=1.0,
            ma_window=30,
        )

        self.assertEqual(summary['actual_up'], 10)
        self.assertEqual(summary['actual_down'], 0)
        self.assertAlmostEqual(summary['fuzzy_up'], 10.0, places=2)
        self.assertEqual(summary['fuzzy_down'], 0.0)
        self.assertEqual(rows[-1]['time_s'], 59.96666666666667)


if __name__ == '__main__':
    unittest.main()