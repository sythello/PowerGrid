from __future__ import annotations

import json
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from powergrid.model import ModelValidationError
from powergrid.session import GameSession
from powergrid.web.server import PowerGridWebController, STATIC_ROOT, make_server


class PowerGridWebControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = PowerGridWebController()

    def test_metadata_uses_versioned_nn_rl_v1_as_the_default_ai(self) -> None:
        metadata = self.controller.metadata()
        controller_ids = [controller["id"] for controller in metadata["controllers"]]

        self.assertEqual(controller_ids, ["human", "ai_nn_rl_v1", "ai_deterministic"])
        self.assertNotIn("ai_heuristics", controller_ids)
        self.assertEqual(metadata["defaults"]["ai_controller"], "ai_nn_rl_v1")
        self.assertEqual(
            metadata["defaults"]["controllers"],
            ["human", "ai_nn_rl_v1", "ai_nn_rl_v1"],
        )
        nn_option = metadata["controllers"][1]
        self.assertEqual(nn_option["name"], "NN RL V1 (v 2.01)")
        self.assertEqual(nn_option["supported_maps"], ["germany"])
        self.assertEqual(nn_option["supported_player_counts"], [3])

    def test_new_game_maps_nn_rl_v1_to_the_latest_release_controller(self) -> None:
        payload = self.controller.new_game(
            {
                "map_id": "germany",
                "seed": 7,
                "players": [
                    {"name": "Alice", "controller": "human"},
                    {"name": "Bob", "controller": "ai_nn_rl_v1"},
                    {"name": "Carol", "controller": "ai_nn_rl_v1"},
                ],
            }
        )

        controllers = {
            player["name"]: player["controller"]
            for player in payload["state"]["players"]
        }
        self.assertEqual(controllers["Bob"], "ai_nn_rl_based_v1")
        self.assertEqual(controllers["Carol"], "ai_nn_rl_based_v1")

    def test_legacy_nn_rl_v2_web_id_remains_compatible(self) -> None:
        payload = self.controller.new_game(
            {
                "map_id": "germany",
                "seed": 7,
                "players": [
                    {"name": "Alice", "controller": "human"},
                    {"name": "Bob", "controller": "ai_nn_rl_v2"},
                    {"name": "Carol", "controller": "ai_nn_rl_v2"},
                ],
            }
        )

        controllers = {
            player["name"]: player["controller"]
            for player in payload["state"]["players"]
        }
        self.assertEqual(controllers["Bob"], "ai_nn_rl_based_v1")
        self.assertEqual(controllers["Carol"], "ai_nn_rl_based_v1")

    def test_nn_rl_v1_rejects_unsupported_map_or_player_count(self) -> None:
        with self.assertRaisesRegex(ModelValidationError, "3-player Germany"):
            self.controller.new_game(
                {
                    "map_id": "usa",
                    "seed": 7,
                    "players": [
                        {"name": "Alice", "controller": "human"},
                        {"name": "Bob", "controller": "ai_nn_rl_v1"},
                        {"name": "Carol", "controller": "ai_nn_rl_v1"},
                    ],
                }
            )

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

    def test_snapshot_exposes_global_parameters_and_only_the_deck_top_back(self) -> None:
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

        parameters = payload["global_parameters"]
        self.assertEqual(parameters["current_step"], 1)
        self.assertEqual(parameters["player_count"], 3)
        self.assertEqual(parameters["step_2_cities"], 7)
        self.assertEqual(parameters["end_game_cities"], 17)
        self.assertEqual(parameters["payment_schedule"]["0"], 10)
        self.assertEqual(parameters["payment_schedule"]["20"], 150)
        self.assertEqual(
            parameters["resource_refill"]["step_2"],
            {"coal": 5, "oil": 3, "garbage": 2, "uranium": 1},
        )
        expected_back = self.controller._session.snapshot().state.power_plant_draw_stack[0].deck_back
        self.assertEqual(payload["state"]["deck_top_back"], expected_back)
        self.assertNotIn("power_plant_draw_stack", payload["state"])
        self.assertNotIn("rules", payload["state"])

    def test_pending_step_3_card_reports_its_physical_socket_back(self) -> None:
        self.controller.new_game(
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
        session = self.controller._session
        session._state = replace(
            session.snapshot().state,
            power_plant_draw_stack=(),
            step_3_card_pending=True,
        )

        payload = self.controller.snapshot_payload()

        self.assertEqual(payload["state"]["deck_count"], 0)
        self.assertEqual(payload["state"]["deck_top_back"], "socket")

    def test_germany_saved_city_overrides_are_applied(self) -> None:
        layout = self.controller._layout_payload("germany")
        saved = json.loads(
            (STATIC_ROOT / "data" / "web_layouts.json").read_text(encoding="utf-8")
        )["germany"]["cities"]

        self.assertEqual(layout["cities"]["duisburg"]["x"], saved["duisburg"]["x"])
        self.assertEqual(layout["cities"]["duisburg"]["y"], saved["duisburg"]["y"])
        self.assertEqual(layout["cities"]["regensburg"]["x"], saved["regensburg"]["x"])
        self.assertEqual(layout["cities"]["regensburg"]["y"], saved["regensburg"]["y"])

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

    def test_web_controller_accepts_multiple_build_batches_before_finish(self) -> None:
        self.controller._session = GameSession.from_scenario("build_test")
        snapshot = self.controller.snapshot_payload()
        player_id = snapshot["request"]["player_id"]

        for _batch in range(2):
            action = next(
                action
                for action in snapshot["request"]["legal_actions"]
                if action["action_type"] == "build_city"
            )
            snapshot = self.controller.submit_intent(
                {
                    "intent_type": "commit_build",
                    "player_id": player_id,
                    "payload": {"city_ids": [action["payload"]["city_id"]]},
                }
            )
            self.assertNotIn("error", snapshot)
            self.assertEqual(snapshot["request"]["player_id"], player_id)
            self.assertEqual(snapshot["request"]["decision_type"], "build_houses")

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

    def test_owned_power_plants_expose_hover_card_previews(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="plant-preview-tooltip"', html)
        self.assertIn('data-owned-plant-preview="/assets/plants/${plant.price}.png"', script)
        self.assertIn('aria-describedby="plant-preview-tooltip"', script)
        self.assertIn("function positionPlantPreview(target)", script)

    def test_web_build_submission_resets_the_batch_for_continued_building(self) -> None:
        script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            'submitCurrentIntent("commit_build", { city_ids: cities }, { forceReset: true })',
            script,
        )
        self.assertIn("可以继续选择城市，或结束建设", script)

    def test_bureaucracy_defaults_to_descending_greedy_plant_selection(self) -> None:
        script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("ui.runs = greedyRunDefaults(player)", script)
        self.assertIn("plants.filter((plant) => plant.resource_cost === 0)", script)
        self.assertIn("const targetCities = player.network_city_ids.length", script)
        self.assertIn("if (selectedOutput >= targetCities) continue", script)
        self.assertIn(".sort((left, right) => right.price - left.price)", script)
        self.assertIn("selected = remaining.coal + remaining.oil >= plant.resource_cost", script)
        self.assertIn("selected = remaining[resource] >= plant.resource_cost", script)
        self.assertIn("if (selected) selectedOutput += plant.output_cities", script)

    def test_global_parameters_render_city_thresholds(self) -> None:
        script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("进入 STEP 2", script)
        self.assertIn("parameters.step_2_cities", script)
        self.assertIn("触发游戏结束", script)
        self.assertIn("parameters.end_game_cities", script)


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
