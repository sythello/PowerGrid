from __future__ import annotations

from dataclasses import replace
import unittest

from powergrid.ai import (
    BaseAiController,
    build_ai_controller,
    DeterministicAiController,
    register_ai_controller,
    StrategicAiController,
)
from powergrid.model import (
    GameConfig,
    ModelValidationError,
    PowerPlantCard,
    ResourceStorage,
    SeatConfig,
    advance_phase,
    add_power_plant_to_player,
    create_initial_state,
    initialize_game,
    make_default_seat_configs,
    set_player_resource_totals,
)
from powergrid.rules_data import load_power_plants
from powergrid.session import GameSession, GuiIntent, default_game_config
from powergrid.session_types import HumanSeat


class AiFrameworkTests(unittest.TestCase):
    def test_build_ai_controller_returns_registered_base_ai_controller(self) -> None:
        controller = build_ai_controller("ai_heuristics")

        self.assertIsInstance(controller, BaseAiController)
        self.assertIsInstance(controller, StrategicAiController)

    def test_generic_ai_alias_builds_deterministic_ai_controller(self) -> None:
        controller = build_ai_controller("ai")

        self.assertIsInstance(controller, BaseAiController)
        self.assertIsInstance(controller, DeterministicAiController)

    def test_deterministic_ai_remains_available_via_fallback_alias(self) -> None:
        controller = build_ai_controller("ai_deterministic")

        self.assertIsInstance(controller, BaseAiController)
        self.assertIsInstance(controller, DeterministicAiController)

    def test_register_ai_controller_requires_base_ai_subclass(self) -> None:
        with self.assertRaises(ModelValidationError):
            register_ai_controller("bad-ai", HumanSeat)  # type: ignore[arg-type]

    def test_game_session_requires_base_ai_controller_for_ai_seats(self) -> None:
        config = GameConfig(
            map_id="germany",
            players=(
                SeatConfig("p1", "Player 1", controller="ai_heuristics"),
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


class StrategicAiBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = StrategicAiController()
        self.plant_definitions = {definition.price: definition for definition in load_power_plants()}

    def test_strategic_ai_opening_prefers_stronger_plant_over_cheapest_visible_one(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)
        snapshot = session.snapshot()
        assert snapshot.active_request is not None

        intent = self.controller.choose_intent(snapshot.active_request, snapshot)

        self.assertEqual(intent.intent_type, "auction_start")
        self.assertEqual(intent.payload["plant_price"], 10)
        self.assertNotEqual(intent.payload["plant_price"], 6)

    def test_strategic_ai_first_round_does_not_pass(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)
        snapshot = session.snapshot()
        assert snapshot.active_request is not None

        intent = self.controller.choose_intent(snapshot.active_request, snapshot)

        self.assertNotEqual(intent.intent_type, "auction_pass")

    def test_strategic_ai_passes_on_overpriced_bid(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)
        snapshot = session.submit_intent(GuiIntent.auction_start("p3", plant_price=10, bid=40))
        assert snapshot.active_request is not None

        intent = self.controller.choose_intent(snapshot.active_request, snapshot)

        self.assertEqual(intent.intent_type, "auction_pass")

    def test_strategic_ai_discards_forward_looking_portfolio_not_lowest_number(self) -> None:
        base = GameSession.from_scenario("opening", seed=7).snapshot().state
        state = add_power_plant_to_player(base, "p3", 10)
        state = add_power_plant_to_player(state, "p3", 11)
        state = add_power_plant_to_player(state, "p3", 13)
        state = add_power_plant_to_player(state, "p3", 15)
        session = GameSession(state, {player.player_id: self.controller for player in state.players})
        snapshot = session.snapshot()
        assert snapshot.active_request is not None

        intent = self.controller.choose_intent(snapshot.active_request, snapshot)

        self.assertEqual(intent.intent_type, "discard_power_plant")
        self.assertNotEqual(intent.payload["plant_price"], 10)

    def test_strategic_ai_hybrid_discard_preserves_future_flexibility(self) -> None:
        state = GameSession.from_scenario("resource", seed=7).snapshot().state
        state = replace(
            state,
            players=tuple(
                replace(player, resource_storage=ResourceStorage())
                if player.player_id == "p1"
                else player
                for player in state.players
            ),
        )
        state = set_player_resource_totals(state, "p1", {"coal": 6, "oil": 4})
        session = GameSession(state, {player.player_id: self.controller for player in state.players})
        snapshot = session.snapshot()
        assert snapshot.active_request is not None

        intent = self.controller.choose_intent(snapshot.active_request, snapshot)

        self.assertEqual(intent.intent_type, "discard_hybrid_resources")
        self.assertEqual(intent.payload, {"coal": 2, "oil": 0})

    def test_strategic_ai_resource_strategy_buys_useful_fuel(self) -> None:
        base = create_initial_state(
            GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
        )
        allowed_city_ids = [
            city.id for city in base.game_map.cities if city.region in base.selected_regions
        ]
        player_order = ("p1", "p2", "p3")
        players = []
        for player in base.players:
            if player.player_id == "p3":
                players.append(
                    replace(
                        player,
                        elektro=40,
                        network_city_ids=tuple(allowed_city_ids[:2]),
                        houses_in_supply=20,
                        power_plants=(
                            PowerPlantCard.from_definition(self.plant_definitions[10]),
                        ),
                        turn_order_position=3,
                    )
                )
            else:
                players.append(
                    replace(
                        player,
                        elektro=40,
                        turn_order_position=player_order.index(player.player_id) + 1,
                    )
                )
        state = replace(
            base,
            players=tuple(players),
            player_order=player_order,
            phase="buy_resources",
            round_number=2,
            step=1,
            auction_state=None,
            pending_decision=None,
        )
        session = GameSession(state, {player.player_id: self.controller for player in state.players})
        snapshot = session.snapshot()
        assert snapshot.active_request is not None

        intent = self.controller.choose_intent(snapshot.active_request, snapshot)

        self.assertEqual(snapshot.active_request.player_id, "p3")
        self.assertEqual(intent.intent_type, "buy_resource")
        self.assertEqual(intent.payload["resource"], "coal")
        self.assertGreaterEqual(intent.payload["amount"], 1)

    def test_strategic_ai_build_strategy_can_commit_multi_city_plan(self) -> None:
        base = create_initial_state(
            GameConfig(
                map_id="test",
                players=make_default_seat_configs(3),
                seed=7,
                selected_regions=("alpha", "beta", "gamma"),
            )
        )
        players = tuple(
            replace(
                player,
                elektro=80,
                houses_in_supply=22,
                network_city_ids=(),
            )
            for player in base.players
        )
        state = replace(
            base,
            players=players,
            phase="build_houses",
            round_number=2,
            step=1,
            auction_state=None,
            pending_decision=None,
        )
        session = GameSession(state, {player.player_id: self.controller for player in state.players})
        snapshot = session.snapshot()
        assert snapshot.active_request is not None

        intent = self.controller.choose_intent(snapshot.active_request, snapshot)

        self.assertEqual(intent.intent_type, "commit_build")
        self.assertGreaterEqual(len(intent.payload["city_ids"]), 2)

    def test_strategic_ai_final_round_bureaucracy_maximizes_power(self) -> None:
        session = GameSession.from_scenario("endgame", seed=7)
        snapshot = session.snapshot()
        assert snapshot.active_request is not None

        intent = self.controller.choose_intent(snapshot.active_request, snapshot)

        self.assertEqual(intent.intent_type, "run_plants")
        self.assertEqual(
            [plan["plant_price"] for plan in intent.payload["plans"]],
            [18, 22],
        )

    def test_ai_vs_ai_smoke_completes_on_representative_matrix(self) -> None:
        for map_id, player_count in (("germany", 3), ("usa", 4)):
            with self.subTest(map_id=map_id, player_count=player_count):
                snapshot = _play_full_ai_game(map_id=map_id, player_count=player_count, seed=7)
                self.assertIsNotNone(snapshot.winner_result)
                assert snapshot.winner_result is not None
                self.assertTrue(snapshot.winner_result.winner_ids)


def _play_full_ai_game(*, map_id: str, player_count: int, seed: int):
    session = GameSession.new_game(
        default_game_config(
            player_count=player_count,
            ai_players=player_count,
            map_id=map_id,
            seed=seed,
        )
    )
    turn_count = 0
    while True:
        request = session.current_request()
        if request is None:
            snapshot = session.snapshot()
            if snapshot.winner_result is not None:
                return snapshot
            session._state = advance_phase(session._state)
            session._sync_phase_cursor(force_reset=True)
            continue
        snapshot = session.snapshot()
        intent = session._seat_agents[request.player_id].choose_intent(request, snapshot)
        if not session._apply_and_log(intent, auto_generated=True):
            raise AssertionError(session.snapshot().event_log[-1].message)
        turn_count += 1
        if turn_count > 1800:
            raise AssertionError(
                f"AI game did not finish for map={map_id}, players={player_count}, seed={seed}"
            )


if __name__ == "__main__":
    unittest.main()
