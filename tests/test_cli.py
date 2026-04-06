from __future__ import annotations

import unittest

from powergrid.cli import GameRunResult, PhaseTraceEntry, ScriptedController, run_game
from powergrid.scenarios import build_game_scenario


def _phase_signature(result: GameRunResult) -> tuple[tuple[int, str, int], ...]:
    return tuple(
        (entry.round_number, entry.phase, entry.step)
        for entry in result.phase_history
    )


def _player(result: GameRunResult, player_id: str):
    return next(player for player in result.final_state.players if player.player_id == player_id)


class CLIGameLoopTests(unittest.TestCase):
    def test_run_game_progresses_through_opening_round(self) -> None:
        state = build_game_scenario("opening", seed=7)
        controllers = {
            "p1": ScriptedController(
                player_id="p1",
                commands=["pass", "start 7 7", "buy oil 3", "done", "done", "run 7"],
            ),
            "p2": ScriptedController(
                player_id="p2",
                commands=["pass", "pass", "start 10 10", "buy coal 2", "done", "done", "run 10"],
            ),
            "p3": ScriptedController(
                player_id="p3",
                commands=["start 6 1", "buy garbage 1", "done", "done", "run 6"],
            ),
        }

        result = run_game(
            state,
            controllers,
            output_fn=None,
            render_state=False,
            stop_condition=lambda current_state: (
                current_state.phase == "determine_order" and current_state.round_number == 2
            ),
        )

        self.assertFalse(result.quit_requested)
        self.assertIsNone(result.winner_result)
        self.assertEqual(result.final_state.phase, "determine_order")
        self.assertEqual(result.final_state.round_number, 2)
        self.assertEqual(
            _phase_signature(result),
            (
                (1, "auction", 1),
                (1, "buy_resources", 1),
                (1, "build_houses", 1),
                (1, "bureaucracy", 1),
                (2, "determine_order", 1),
            ),
        )
        self.assertEqual(len(result.round_summaries), 1)

    def test_run_game_applies_step_2_transition_during_bureaucracy(self) -> None:
        state = build_game_scenario("step2", seed=7)
        controllers = {
            "p1": ScriptedController(player_id="p1", commands=["skip"]),
            "p2": ScriptedController(player_id="p2", commands=["skip"]),
            "p3": ScriptedController(player_id="p3", commands=["skip"]),
        }

        result = run_game(
            state,
            controllers,
            output_fn=None,
            render_state=False,
            stop_condition=lambda current_state: (
                current_state.phase == "determine_order" and current_state.round_number == 3
            ),
        )

        self.assertEqual(result.final_state.phase, "determine_order")
        self.assertEqual(result.final_state.step, 2)
        self.assertEqual(
            _phase_signature(result),
            (
                (2, "bureaucracy", 1),
                (3, "determine_order", 2),
            ),
        )
        self.assertEqual(len(result.round_summaries), 1)
        self.assertTrue(result.round_summaries[0].triggered_step_2)
        self.assertFalse(result.round_summaries[0].triggered_step_3)

    def test_run_game_handles_auction_time_step_3_transition(self) -> None:
        state = build_game_scenario("auction_step3", seed=7)
        controllers = {
            "p1": ScriptedController(
                player_id="p1",
                commands=["pass", "pass", "done", "done", "skip"],
            ),
            "p2": ScriptedController(
                player_id="p2",
                commands=["pass", "pass", "done", "done", "skip"],
            ),
            "p3": ScriptedController(
                player_id="p3",
                commands=["start 6 1", "done", "done", "skip"],
            ),
        }

        result = run_game(
            state,
            controllers,
            output_fn=None,
            render_state=False,
            stop_condition=lambda current_state: (
                current_state.phase == "determine_order" and current_state.round_number == 3
            ),
        )

        self.assertEqual(result.final_state.phase, "determine_order")
        self.assertEqual(result.final_state.step, 3)
        self.assertEqual(
            _phase_signature(result),
            (
                (2, "auction", 2),
                (2, "buy_resources", 3),
                (2, "build_houses", 3),
                (2, "bureaucracy", 3),
                (3, "determine_order", 3),
            ),
        )
        self.assertEqual(len(result.round_summaries), 1)
        self.assertFalse(result.round_summaries[0].triggered_step_2)
        self.assertFalse(result.round_summaries[0].triggered_step_3)

    def test_run_game_can_resolve_end_game(self) -> None:
        state = build_game_scenario("endgame", seed=7)
        controllers = {
            "p1": ScriptedController(player_id="p1", commands=["run 13"]),
            "p2": ScriptedController(player_id="p2", commands=["run 20 23"]),
            "p3": ScriptedController(player_id="p3", commands=["skip"]),
        }

        result = run_game(
            state,
            controllers,
            output_fn=None,
            render_state=False,
        )

        self.assertFalse(result.quit_requested)
        self.assertIsNotNone(result.winner_result)
        self.assertEqual(result.final_state.phase, "bureaucracy")
        self.assertEqual(result.final_state.step, 2)
        self.assertEqual(len(result.round_summaries), 1)
        self.assertTrue(result.round_summaries[0].game_end_triggered)
        self.assertEqual(result.winner_result.winner_ids, ("p2",))

    def test_run_game_debug_add_plant_can_trigger_discard_prompt(self) -> None:
        state = build_game_scenario("opening", seed=7)
        controllers = {
            "p1": ScriptedController(player_id="p1", commands=[]),
            "p2": ScriptedController(player_id="p2", commands=[]),
            "p3": ScriptedController(
                player_id="p3",
                commands=["add-plant 5", "add-plant 10", "add-plant 11", "add-plant 13", "discard 5", "quit"],
            ),
        }

        result = run_game(
            state,
            controllers,
            output_fn=None,
            render_state=False,
            allow_debug_commands=True,
        )

        self.assertTrue(result.quit_requested)
        self.assertEqual(tuple(plant.price for plant in _player(result, "p3").power_plants), (10, 11, 13))
        self.assertIsNone(result.final_state.pending_decision)

    def test_run_game_debug_add_and_clear_city_preserve_house_supply(self) -> None:
        state = build_game_scenario("opening", seed=7)
        controllers = {
            "p1": ScriptedController(player_id="p1", commands=[]),
            "p2": ScriptedController(player_id="p2", commands=[]),
            "p3": ScriptedController(
                player_id="p3",
                commands=["add-city berlin", "clear-city berlin", "quit"],
            ),
        }

        result = run_game(
            state,
            controllers,
            output_fn=None,
            render_state=False,
            allow_debug_commands=True,
        )

        self.assertTrue(result.quit_requested)
        self.assertEqual(_player(result, "p3").houses_in_supply, 22)
        self.assertEqual(_player(result, "p3").network_city_ids, ())


if __name__ == "__main__":
    unittest.main()
