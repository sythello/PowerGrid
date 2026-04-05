from __future__ import annotations

import unittest

from powergrid.rules_data import (
    DataValidationError,
    load_map,
    load_power_plants,
    load_rule_tables,
    validate_static_data,
)


class StaticDataTests(unittest.TestCase):
    def test_load_rule_tables(self) -> None:
        rules = load_rule_tables()
        self.assertEqual(rules.starting_money, 50)
        self.assertEqual(rules.houses_per_player, 22)
        self.assertEqual(sorted(rules.player_count_rules), [3, 4, 5, 6])
        self.assertEqual(rules.player_count_rules[3]["step_2_cities"], 7)
        self.assertEqual(rules.player_count_rules[6]["end_game_cities"], 14)

    def test_load_power_plants(self) -> None:
        plants = load_power_plants()
        self.assertEqual(len(plants), 42)
        self.assertEqual(plants[0].price, 3)
        self.assertEqual(plants[-1].price, 50)
        self.assertTrue(any(plant.is_hybrid for plant in plants))
        self.assertTrue(any(plant.is_ecological for plant in plants))

    def test_load_germany_map(self) -> None:
        game_map = load_map("germany")
        self.assertEqual(game_map.name, "Germany")
        self.assertEqual(len(game_map.regions), 6)
        self.assertGreaterEqual(len(game_map.cities), 42)
        self.assertTrue(any(city.id == "mannheim" for city in game_map.cities))
        self.assertTrue(any(connection.cost == 0 for connection in game_map.connections))

    def test_load_usa_map(self) -> None:
        game_map = load_map("usa")
        self.assertEqual(game_map.name, "USA")
        self.assertEqual(len(game_map.regions), 6)
        self.assertGreaterEqual(len(game_map.cities), 42)
        self.assertTrue(any(city.id == "portland" for city in game_map.cities))
        self.assertTrue(
            any(
                "seattle" in {connection.city_1, connection.city_2}
                for connection in game_map.connections
            )
        )

    def test_load_test_map(self) -> None:
        game_map = load_map("test")
        self.assertEqual(game_map.name, "Test Map")
        self.assertEqual(len(game_map.regions), 3)
        self.assertEqual(len(game_map.cities), 3)
        self.assertEqual(len(game_map.connections), 2)
        self.assertEqual(game_map.special_rules, ())
        self.assertEqual(
            game_map.region_adjacency,
            {
                "alpha": ("beta",),
                "beta": ("alpha", "gamma"),
                "gamma": ("beta",),
            },
        )

    def test_validate_static_data(self) -> None:
        report = validate_static_data()
        self.assertEqual(report.maps_loaded, ("germany", "usa"))
        self.assertEqual(report.power_plant_count, 42)

    def test_validate_static_data_raises_cleanly(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_static_data(data_root="tests/fixtures/does-not-exist")


if __name__ == "__main__":
    unittest.main()
