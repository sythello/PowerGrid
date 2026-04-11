from __future__ import annotations

from dataclasses import replace
import unittest

from powergrid.model import (
    Action,
    add_power_plant_to_player,
    DecisionRequest,
    GameConfig,
    GameState,
    PlantRunPlan,
    PowerPlantCard,
    ResourceStorage,
    SeatConfig,
    set_player_resource_totals,
)
from powergrid.rules_data import load_power_plants
from powergrid.session import (
    default_game_config,
    DeterministicAiSeat,
    GameSession,
    GuiIntent,
    HumanSeat,
)


def _human_seats(state: GameState) -> dict[str, HumanSeat]:
    return {player.player_id: HumanSeat() for player in state.players}


def _player(state: GameState, player_id: str):
    return next(player for player in state.players if player.player_id == player_id)


class GameSessionTests(unittest.TestCase):
    def test_from_scenario_exposes_opening_request(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)

        snapshot = session.snapshot()

        self.assertEqual(snapshot.state.phase, "auction")
        self.assertIsNotNone(snapshot.active_request)
        assert snapshot.active_request is not None
        self.assertEqual(snapshot.active_request.player_id, "p3")
        self.assertEqual(snapshot.active_request.decision_type, "auction_start")

    def test_new_game_uses_configured_seat_types(self) -> None:
        config = GameConfig(
            map_id="germany",
            players=(
                SeatConfig("p1", "Player 1", controller="ai"),
                SeatConfig("p2", "Player 2", controller="human"),
                SeatConfig("p3", "Player 3", controller="human"),
            ),
            seed=7,
        )

        session = GameSession.new_game(config)
        snapshot = session.advance_until_blocked()

        self.assertEqual(snapshot.state.phase, "auction")
        self.assertIsNotNone(snapshot.active_request)
        assert snapshot.active_request is not None
        self.assertEqual(snapshot.active_request.player_id, "p3")

    def test_submit_intent_updates_auction_state(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)

        snapshot = session.submit_intent(GuiIntent.auction_start("p3", plant_price=6, bid=1))

        self.assertEqual(snapshot.state.phase, "auction")
        self.assertIsNotNone(snapshot.state.auction_state)
        assert snapshot.state.auction_state is not None
        self.assertTrue(snapshot.state.auction_state.has_active_auction)
        self.assertEqual(snapshot.state.auction_state.active_plant_price, 6)
        self.assertEqual(snapshot.state.auction_state.current_bid, 1)
        self.assertIsNotNone(snapshot.active_request)
        assert snapshot.active_request is not None
        self.assertEqual(snapshot.active_request.player_id, "p1")
        self.assertEqual(snapshot.active_request.decision_type, "auction_bid")

    def test_invalid_intent_keeps_state_unchanged_and_logs_error(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)
        before = session.snapshot().state.to_dict()

        snapshot = session.submit_intent(GuiIntent.auction_pass("p3"))

        self.assertEqual(snapshot.state.to_dict(), before)
        self.assertTrue(snapshot.event_log)
        self.assertEqual(snapshot.event_log[-1].level, "error")
        self.assertIn("must buy a power plant", snapshot.event_log[-1].message)

    def test_ai_auto_advance_stops_at_next_human_boundary(self) -> None:
        session = GameSession.from_scenario(
            "opening",
            seed=7,
            seat_agents={
                "p1": DeterministicAiSeat(),
                "p2": HumanSeat(),
                "p3": DeterministicAiSeat(),
            },
        )

        snapshot = session.advance_until_blocked()

        self.assertIsNotNone(snapshot.active_request)
        assert snapshot.active_request is not None
        self.assertEqual(snapshot.state.phase, "auction")
        self.assertEqual(snapshot.active_request.player_id, "p2")
        self.assertEqual(snapshot.active_request.decision_type, "auction_bid")
        self.assertGreaterEqual(len(snapshot.event_log), 2)

    def test_later_round_auction_pass_advances_to_next_chooser(self) -> None:
        session = GameSession.from_scenario("auction_step3", seed=7)
        first_request = session.snapshot().active_request
        assert first_request is not None

        snapshot = session.submit_intent(GuiIntent.auction_pass(first_request.player_id))

        self.assertEqual(snapshot.state.phase, "auction")
        self.assertIsNotNone(snapshot.active_request)
        assert snapshot.active_request is not None
        self.assertNotEqual(snapshot.active_request.player_id, first_request.player_id)

    def test_pending_discard_power_plant_request_can_be_resolved(self) -> None:
        base = GameSession.from_scenario("opening", seed=7).snapshot().state
        state = add_power_plant_to_player(base, "p3", 5)
        state = add_power_plant_to_player(state, "p3", 10)
        state = add_power_plant_to_player(state, "p3", 11)
        state = add_power_plant_to_player(state, "p3", 13)
        session = GameSession(state, _human_seats(state))

        snapshot = session.snapshot()
        self.assertIsNotNone(snapshot.active_request)
        assert snapshot.active_request is not None
        self.assertEqual(snapshot.active_request.decision_type, "discard_power_plant")

        snapshot = session.submit_intent(GuiIntent.discard_plant("p3", 5))

        self.assertIsNone(snapshot.state.pending_decision)
        self.assertEqual(tuple(plant.price for plant in _player(snapshot.state, "p3").power_plants), (10, 11, 13))

    def test_pending_hybrid_discard_request_can_be_resolved(self) -> None:
        state = GameSession.from_scenario("resource", seed=7).snapshot().state
        player = replace(_player(state, "p1"), resource_storage=ResourceStorage())
        state = replace(
            state,
            players=tuple(player if existing.player_id == "p1" else existing for existing in state.players),
        )
        state = set_player_resource_totals(state, "p1", {"coal": 6, "oil": 4})
        session = GameSession(state, _human_seats(state))

        snapshot = session.snapshot()
        self.assertIsNotNone(snapshot.active_request)
        assert snapshot.active_request is not None
        self.assertEqual(snapshot.active_request.decision_type, "discard_hybrid_resources")

        snapshot = session.submit_intent(
            GuiIntent.discard_hybrid_resources("p1", coal=1, oil=1)
        )

        self.assertIsNone(snapshot.state.pending_decision)
        self.assertEqual(
            _player(snapshot.state, "p1").resource_storage,
            ResourceStorage(coal=4, hybrid_coal=1, hybrid_oil=3),
        )

    def test_resource_phase_buy_and_done_advance_turn_order(self) -> None:
        session = GameSession.from_scenario("resource", seed=7)
        first_snapshot = session.snapshot()
        assert first_snapshot.active_request is not None
        active_player_id = first_snapshot.active_request.player_id
        buy_action = next(
            action
            for action in first_snapshot.active_request.legal_actions
            if action.action_type == "buy_resource"
        )
        before_money = _player(first_snapshot.state, active_player_id).elektro

        snapshot = session.submit_intent(
            GuiIntent.buy_resource(
                active_player_id,
                resource=str(buy_action.payload["resource"]),
                amount=1,
            )
        )
        mid_money = _player(snapshot.state, active_player_id).elektro
        snapshot = session.submit_intent(GuiIntent.finish_buying(active_player_id))

        self.assertLess(mid_money, before_money)
        self.assertIsNotNone(snapshot.active_request)
        assert snapshot.active_request is not None
        self.assertNotEqual(snapshot.active_request.player_id, active_player_id)

    def test_build_phase_quote_is_non_mutating_and_commit_build_updates_state(self) -> None:
        session = GameSession.from_scenario("build_test", seed=7)
        first_snapshot = session.snapshot()
        assert first_snapshot.active_request is not None
        active_player_id = first_snapshot.active_request.player_id
        build_action = next(
            action
            for action in first_snapshot.active_request.legal_actions
            if action.action_type == "build_city"
        )
        target_city = str(build_action.payload["city_id"])
        before_state = first_snapshot.state.to_dict()

        quoted = session.submit_intent(GuiIntent.quote_build(active_player_id, [target_city]))
        built = session.submit_intent(GuiIntent.commit_build(active_player_id, [target_city]))

        self.assertEqual(quoted.state.to_dict(), before_state)
        self.assertTrue(any("Quote for" in event.message for event in quoted.event_log))
        self.assertIn(target_city, _player(built.state, active_player_id).network_city_ids)

    def test_bureaucracy_skip_all_records_summary_and_advances_round(self) -> None:
        session = GameSession.from_scenario("step2", seed=7)

        for _ in range(3):
            snapshot = session.snapshot()
            assert snapshot.active_request is not None
            session.submit_intent(GuiIntent.skip_bureaucracy(snapshot.active_request.player_id))

        snapshot = session.snapshot()

        self.assertIsNotNone(snapshot.last_round_summary)
        assert snapshot.last_round_summary is not None
        self.assertTrue(snapshot.last_round_summary.triggered_step_2)
        self.assertEqual(snapshot.state.round_number, 3)
        self.assertEqual(snapshot.state.phase, "auction")

    def test_endgame_bureaucracy_can_finish_the_game(self) -> None:
        session = GameSession.from_scenario("endgame", seed=7)
        snapshot = session.snapshot()
        assert snapshot.active_request is not None
        snapshot = session.submit_intent(GuiIntent.skip_bureaucracy(snapshot.active_request.player_id))
        assert snapshot.active_request is not None
        snapshot = session.submit_intent(
            GuiIntent.run_plants(snapshot.active_request.player_id, [PlantRunPlan(13, {})])
        )
        assert snapshot.active_request is not None
        snapshot = session.submit_intent(
            GuiIntent.run_plants(
                snapshot.active_request.player_id,
                [
                    PlantRunPlan(20, {"coal": 3}),
                    PlantRunPlan(23, {"uranium": 1}),
                ],
            )
        )

        self.assertIsNotNone(snapshot.winner_result)
        assert snapshot.winner_result is not None
        self.assertEqual(snapshot.winner_result.winner_ids, ("p2",))
        self.assertEqual(snapshot.state.phase, "bureaucracy")

    def test_mixed_seat_auto_advance_works_for_player_counts_three_to_six(self) -> None:
        for player_count in range(3, 7):
            config = default_game_config(player_count=player_count, ai_players=player_count - 1, seed=7)
            config = replace(
                config,
                players=tuple(
                    replace(seat, controller="human" if index == player_count - 1 else "ai")
                    for index, seat in enumerate(config.players)
                ),
            )
            session = GameSession.new_game(config)
            snapshot = session.advance_until_blocked()

            self.assertIsNotNone(snapshot.active_request)
            assert snapshot.active_request is not None
            self.assertEqual(snapshot.active_request.player_id, f"p{player_count}")


if __name__ == "__main__":
    unittest.main()
