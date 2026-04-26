from __future__ import annotations

import unittest

from powergrid.board_layout import load_board_layout


class BoardLayoutTests(unittest.TestCase):
    def test_germany_layout_uses_reference_image_size(self) -> None:
        layout = load_board_layout("germany")
        self.assertEqual(layout["board_art"]["natural_size"]["width"], 1424)
        self.assertEqual(layout["board_art"]["natural_size"]["height"], 2000)

    def test_germany_city_anchors_are_populated(self) -> None:
        layout = load_board_layout("germany")
        missing = []
        for city_id, payload in layout["cities"].items():
            anchor = payload["anchor"]
            if anchor["x"] is None or anchor["y"] is None:
                missing.append(city_id)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
