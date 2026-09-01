from __future__ import annotations

import unittest

from powergrid.ai import (
    EfficiencyDeterministicAiController,
    ExpansionDeterministicAiController,
    ReserveDeterministicAiController,
    build_ai_controller,
)
from powergrid.ai.nn_rank_value.dataset import DEFAULT_BEHAVIOR_CONTROLLERS
from powergrid.model import GameConfig, SeatConfig
from powergrid.session import GameSession


PROFILE_NAMES = (
    "ai_deterministic_efficiency",
    "ai_deterministic_expansion",
    "ai_deterministic_reserve",
)


class ProfiledDeterministicAiTests(unittest.TestCase):
    def test_profile_controllers_are_registered(self) -> None:
        self.assertIsInstance(
            build_ai_controller("ai_deterministic_efficiency"),
            EfficiencyDeterministicAiController,
        )
        self.assertIsInstance(
            build_ai_controller("ai_deterministic_expansion"),
            ExpansionDeterministicAiController,
        )
        self.assertIsInstance(
            build_ai_controller("ai_deterministic_reserve"),
            ReserveDeterministicAiController,
        )
        self.assertEqual(
            DEFAULT_BEHAVIOR_CONTROLLERS,
            ("ai_deterministic", *PROFILE_NAMES),
        )

    def test_every_profile_differs_from_baseline_and_each_other(self) -> None:
        snapshots = [
            GameSession.from_scenario(name, seed=7).snapshot()
            for name in ("opening", "resource", "build_test", "endgame")
        ]

        def signature(controller_name: str):
            controller = build_ai_controller(controller_name)
            return tuple(
                controller.choose_intent(snapshot.active_request, snapshot).to_dict()
                for snapshot in snapshots
                if snapshot.active_request is not None
            )

        baseline = signature("ai_deterministic")
        profile_signatures = [signature(name) for name in PROFILE_NAMES]
        self.assertTrue(all(value != baseline for value in profile_signatures))
        self.assertEqual(len({repr(value) for value in profile_signatures}), 3)

    def test_profile_intents_are_accepted_in_all_supported_scenarios(self) -> None:
        for controller_name in PROFILE_NAMES:
            for scenario_name in ("opening", "resource", "build_test", "endgame"):
                with self.subTest(controller=controller_name, scenario=scenario_name):
                    session = GameSession.from_scenario(scenario_name, seed=7)
                    snapshot = session.snapshot()
                    assert snapshot.active_request is not None
                    intent = build_ai_controller(controller_name).choose_intent(
                        snapshot.active_request,
                        snapshot,
                    )
                    result = session.submit_intent(intent, auto_advance=False)
                    self.assertNotEqual(result.event_log[-1].level, "error")

    def test_mixed_profile_game_completes_without_intent_errors(self) -> None:
        controllers = PROFILE_NAMES
        config = GameConfig(
            map_id="germany",
            players=tuple(
                SeatConfig(
                    f"p{index + 1}",
                    f"Player {index + 1}",
                    controller=controller,
                )
                for index, controller in enumerate(controllers)
            ),
            seed=701,
        )
        session = GameSession.new_game(config)

        snapshot = session.advance_until_blocked()

        self.assertIsNotNone(snapshot.winner_result)
        self.assertFalse(
            [entry for entry in session.game_log_entries() if entry.event_type == "intent_error"]
        )


if __name__ == "__main__":
    unittest.main()
