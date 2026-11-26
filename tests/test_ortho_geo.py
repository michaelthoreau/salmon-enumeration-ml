import math
import unittest

import torch

from eval_ortho_local import (AUGS, augment, lonlat_to_merc, merc_res, merc_to_lonlat, merge_points,
                              tile_origin_merc, unaugment_points)
from predict_modal import _latlon_to_tile, _pixel_to_latlon

LNG, LAT = -158.5464975, 59.2148506


class OrthoGeoTest(unittest.TestCase):
    def test_point_round_trips_through_tile_pixel(self):
        for zoom in (18, 21, 22, 23):
            tx, ty = _latlon_to_tile(zoom, LAT, LNG)
            ox, oy = tile_origin_merc(zoom, tx, ty)
            x, y = lonlat_to_merc(LNG, LAT)
            col, row = (x - ox) / merc_res(zoom), (oy - y) / merc_res(zoom)
            self.assertTrue(0 <= col < 256 and 0 <= row < 256)
            lng, lat = _pixel_to_latlon(col, row, zoom, tx, ty)
            self.assertAlmostEqual(lng, LNG, places=9)
            self.assertAlmostEqual(lat, LAT, places=9)

    def test_tile_origin_matches_pixel_to_latlon_corner(self):
        tx, ty = _latlon_to_tile(22, LAT, LNG)
        lng, lat = merc_to_lonlat(*tile_origin_merc(22, tx, ty))
        expected_lng, expected_lat = _pixel_to_latlon(0, 0, 22, tx, ty)
        self.assertAlmostEqual(lng, expected_lng, places=9)
        self.assertAlmostEqual(lat, expected_lat, places=9)

    def test_zoom_22_ground_resolution_at_site(self):
        self.assertAlmostEqual(merc_res(22) * math.cos(math.radians(LAT)), 0.0191, delta=2e-4)


class TtaMappingTest(unittest.TestCase):
    def test_unaugment_returns_points_to_original_position(self):
        n, row, col = 8, 2, 5
        img = torch.zeros(1, n, n)
        img[0, row, col] = 1
        expected = torch.tensor([[(col + 0.5) / n, (row + 0.5) / n]])
        for aug in range(len(AUGS)):
            r, c = (augment(img, aug)[0] == 1).nonzero()[0].tolist()
            predicted = torch.tensor([[(c + 0.5) / n, (r + 0.5) / n]])
            self.assertTrue(torch.allclose(unaugment_points(predicted, aug), expected), f"aug {AUGS[aug]}")

    def test_single_view_merge_is_distance_nms(self):
        lat0 = LAT
        metre = 1 / 111320
        raw = [(LNG, LAT, 0.9, 0), (LNG, LAT + 0.2 * metre, 0.5, 0), (LNG, LAT + 2 * metre, 0.4, 0)]
        merged = merge_points(raw, 1, 0.5, lat0)
        self.assertEqual(sorted(round(p[2], 2) for p in merged), [0.4, 0.9])

    def test_multi_view_merge_scores_by_vote(self):
        raw = [(LNG, LAT, 0.8, 0), (LNG, LAT, 0.6, 1), (LNG, LAT, 0.7, 1)]
        [(_, _, score)] = merge_points(raw, 4, 0.5, LAT)
        self.assertAlmostEqual(score, (0.8 + 0.7) / 4)


if __name__ == "__main__":
    unittest.main()
