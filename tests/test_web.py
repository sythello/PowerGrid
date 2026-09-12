from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from powergrid.model import ModelValidationError
from powergrid.session import GameSession
from powergrid.web.server import PowerGridWebController, STATIC_ROOT, make_server


class PowerGridWebControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = PowerGridWebController()

    def test_new_game_snapshot_contains_browser_ready_layout(self) -> None:
        payload = self.controller.new_game(
            {
                "map_id": "germany",
                "seed": 7,
                "players": [
                    {"name": "Alice", "controller": "human"},
                    {"name": "Bob", "controller": "human"},
                    {"name": "Carol", "controller": "human"},
                ],
            }
        )

        self.assertTrue(payload["has_game"])
        self.assertEqual(payload["state"]["game_map"]["id"], "germany")
        self.assertNotIn("rules", payload["state"])
        self.assertIn("deck_count", payload["state"])
        self.assertEqual(payload["layout"]["size"], {"width": 1424, "height": 2000})
        self.assertEqual(len(payload["layout"]["cities"]), 42)
        self.assertEqual(
            payload["layout"]["house_slot_offsets"],
            [{"x": 0, "y": -22}, {"x": -22, "y": 13}, {"x": 22, "y": 13}],
        )
        self.assertIsNotNone(payload["request"])

    def test_germany_edge_city_anchors_match_the_board_art(self) -> None:
        layout = self.controller._layout_payload("germany")

        self.assertAlmostEqual(layout["cities"]["duisburg"]["x"], 0.05478)
        self.assertAlmostEqual(layout["cities"]["duisburg"]["y"], 0.42600)
        self.assertAlmostEqual(layout["cities"]["regensburg"]["x"], 0.68400)
        self.assertAlmostEqual(layout["cities"]["regensburg"]["y"], 0.80050)

    def test_city_layout_save_requires_confirmation_and_complete_coordinates(self) -> None:
        snapshot = self.controller.new_game(
            {
                "map_id": "test",
                "seed": 7,
                "players": [
                    {"name": "One", "controller": "human"},
                    {"name": "Two", "controller": "human"},
                    {"name": "Three", "controller": "human"},
                ],
            }
        )
        cities = {
            city["id"]: {
                "x": snapshot["layout"]["cities"][city["id"]]["x"],
                "y": snapshot["layout"]["cities"][city["id"]]["y"],
            }
            for city in snapshot["state"]["game_map"]["cities"]
        }

        with self.assertRaisesRegex(ModelValidationError, "explicit confirmation"):
            self.controller.save_city_layout(
                {"map_id": "test", "confirmed": False, "cities": cities}
            )
        with self.assertRaisesRegex(ModelValidationError, "coordinate set mismatch"):
            self.controller.save_city_layout(
                {"map_id": "test", "confirmed": True, "cities": {}}
            )
        cities["amber_falls"]["x"] = 1.01
        with self.assertRaisesRegex(ModelValidationError, "between 0 and 1"):
            self.controller.save_city_layout(
                {"map_id": "test", "confirmed": True, "cities": cities}
            )

    def test_city_layout_save_atomically_updates_current_map_section(self) -> None:
        source = STATIC_ROOT / "data" / "web_layouts.json"
        with TemporaryDirectory() as directory:
            target = Path(directory) / "web_layouts.json"
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            controller = PowerGridWebController(web_layouts_path=target)
            snapshot = controller.new_game(
                {
                    "map_id": "germany",
                    "seed": 7,
                    "players": [
                        {"name": "One", "controller": "human"},
                        {"name": "Two", "controller": "human"},
                        {"name": "Three", "controller": "human"},
                    ],
                }
            )
            cities = {
                city["id"]: {
                    "x": snapshot["layout"]["cities"][city["id"]]["x"],
                    "y": snapshot["layout"]["cities"][city["id"]]["y"],
                }
                for city in snapshot["state"]["game_map"]["cities"]
            }
            cities["duisburg"]["x"] = 0.06123449

            result = controller.save_city_layout(
                {"map_id": "germany", "confirmed": True, "cities": cities}
            )
            persisted = json.loads(target.read_text(encoding="utf-8"))

            self.assertTrue(result["saved"])
            self.assertEqual(result["city_count"], 42)
            self.assertEqual(result["config_file"], "web_layouts.json")
            self.assertEqual(persisted["germany"]["cities"]["duisburg"]["x"], 0.061234)
            self.assertEqual(len(persisted["germany"]["cities"]), 42)
            self.assertIn("house_slot_offsets", persisted["germany"])
            self.assertIn("seattle", persisted["usa"]["cities"])
            self.assertEqual(result["layout"]["cities"]["duisburg"]["x"], 0.061234)

    def test_auction_intent_advances_to_next_human_request(self) -> None:
        snapshot = self.controller.new_game(
            {
                "map_id": "test",
                "seed": 4,
                "players": [
                    {"name": "One", "controller": "human"},
                    {"name": "Two", "controller": "human"},
                    {"name": "Three", "controller": "human"},
                ],
            }
        )
        request = snapshot["request"]
        action = next(action for action in request["legal_actions"] if action["action_type"] == "auction_start")
        result = self.controller.submit_intent(
            {
                "intent_type": "auction_start",
                "player_id": request["player_id"],
                "payload": {
                    "plant_price": action["payload"]["plant_price"],
                    "bid": action["payload"]["min_bid"],
                },
            }
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["request"]["decision_type"], "auction_bid")
        self.assertEqual(
            result["state"]["auction_state"]["active_plant_price"],
            action["payload"]["plant_price"],
        )

    def test_build_quote_uses_rules_engine(self) -> None:
        self.controller._session = GameSession.from_scenario("build_test")
        snapshot = self.controller.snapshot_payload()
        request = snapshot["request"]
        action = next(action for action in request["legal_actions"] if action["action_type"] == "build_city")

        quote = self.controller.quote_build(
            {
                "player_id": request["player_id"],
                "city_ids": [action["payload"]["city_id"]],
            }
        )

        self.assertTrue(quote["valid"])
        self.assertEqual(quote["cost"], action["payload"]["total_cost"])

    def test_every_power_plant_definition_has_an_image(self) -> None:
        definitions = json.loads(
            (Path(__file__).parents[1] / "src/powergrid/data/rules/power_plants.json").read_text(
                encoding="utf-8"
            )
        )
        missing = [
            plant["price"]
            for plant in definitions
            if not (STATIC_ROOT / "assets" / "plants" / f"{plant['price']}.png").is_file()
        ]
        self.assertEqual(missing, [])


class PowerGridWebHttpTests(unittest.TestCase):
    def test_server_serves_shell_and_metadata(self) -> None:
        try:
            server = make_server("127.0.0.1", 0)
        except PermissionError:
            self.skipTest("local socket binding is disabled in this sandbox")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(f"{base}/", timeout=3) as response:
                body = response.read().decode("utf-8")
            self.assertIn("PowerGrid · 电力调度台", body)

            request = Request(
                f"{base}/api/game",
                data=json.dumps(
                    {
                        "map_id": "germany",
                        "seed": 7,
                        "players": [
                            {"name": "One", "controller": "human"},
                            {"name": "Two", "controller": "ai_deterministic"},
                            {"name": "Three", "controller": "ai_deterministic"},
                        ],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read())
            self.assertTrue(payload["has_game"])
            self.assertEqual(payload["state"]["players"][0]["name"], "One")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
