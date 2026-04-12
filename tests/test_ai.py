from __future__ import annotations

import unittest

from powergrid.ai import BaseAiController, build_ai_controller, DeterministicAiController, register_ai_controller
from powergrid.model import GameConfig, ModelValidationError, SeatConfig, advance_phase, initialize_game
from powergrid.session import GameSession
from powergrid.session_types import HumanSeat


class AiFrameworkTests(unittest.TestCase):
    def test_build_ai_controller_returns_registered_base_ai_controller(self) -> None:
        controller = build_ai_controller("ai")

        self.assertIsInstance(controller, BaseAiController)
        self.assertIsInstance(controller, DeterministicAiController)

    def test_register_ai_controller_requires_base_ai_subclass(self) -> None:
        with self.assertRaises(ModelValidationError):
            register_ai_controller("bad-ai", HumanSeat)  # type: ignore[arg-type]

    def test_game_session_requires_base_ai_controller_for_ai_seats(self) -> None:
        config = GameConfig(
            map_id="germany",
            players=(
                SeatConfig("p1", "Player 1", controller="ai"),
                SeatConfig("p2", "Player 2", controller="human"),
                SeatConfig("p3", "Player 3", controller="human"),
            ),
            seed=7,
        )
        state = advance_phase(initialize_game(config, controllers=None))
        seat_agents = {
            "p1": HumanSeat(),
            "p2": HumanSeat(),
            "p3": HumanSeat(),
        }

        with self.assertRaises(ModelValidationError):
            GameSession(state, seat_agents)

    def test_deterministic_ai_auction_strategy_chooses_cheapest_opening_bid(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)
        snapshot = session.snapshot()
        assert snapshot.active_request is not None
        controller = DeterministicAiController()

        intent = controller.choose_intent(snapshot.active_request, snapshot)

        self.assertEqual(intent.intent_type, "auction_start")
        self.assertEqual(intent.payload["plant_price"], 6)
        self.assertEqual(intent.payload["bid"], 1)

    def test_deterministic_ai_build_strategy_picks_cheapest_legal_city(self) -> None:
        session = GameSession.from_scenario("build_test", seed=7)
        snapshot = session.snapshot()
        assert snapshot.active_request is not None
        controller = DeterministicAiController()

        intent = controller.choose_intent(snapshot.active_request, snapshot)
        legal_actions = [
            action
            for action in snapshot.active_request.legal_actions
            if action.action_type == "build_city"
        ]
        expected = min(
            legal_actions,
            key=lambda action: (
                int(action.payload["total_cost"]),
                str(action.payload["city_id"]),
            ),
        )

        self.assertEqual(intent.intent_type, "commit_build")
        self.assertEqual(intent.payload["city_ids"], [expected.payload["city_id"]])


if __name__ == "__main__":
    unittest.main()
