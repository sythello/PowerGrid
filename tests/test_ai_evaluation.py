from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from tempfile import TemporaryDirectory
import unittest

from powergrid.ai import (
    AiEvaluationBucketConfig,
    AiEvaluationStanding,
    build_default_evaluation_lineups,
    derive_final_standings,
    evaluate_ai_bucket,
    select_evaluation_regions,
)
from powergrid.ai.evaluation import (
    _compute_pairwise_game_updates,
    _controller_pair_actual_score,
    _pair_k_factor,
)
from powergrid.model import (
    GameConfig,
    advance_phase,
    create_initial_state,
    legal_region_sets,
    make_default_seat_configs,
    resolve_winner,
)
from powergrid.tools.evaluate_ai_ratings import main as evaluate_ai_ratings_main


class AiEvaluationTests(unittest.TestCase):
    def test_default_lineups_for_three_player_germany_only_use_mixed_compositions(self) -> None:
        lineups = build_default_evaluation_lineups(
            ("ai_deterministic", "ai_heuristics"),
            3,
            map_id="germany",
        )

        self.assertEqual(
            lineups,
            (
                ("ai_deterministic", "ai_deterministic", "ai_heuristics"),
                ("ai_deterministic", "ai_heuristics", "ai_heuristics"),
            ),
        )

    def test_controller_pair_actual_score_averages_duplicate_seats(self) -> None:
        standings = (
            AiEvaluationStanding("p1", "ai_deterministic", 1, 8, 70, 15),
            AiEvaluationStanding("p2", "ai_heuristics", 2, 6, 60, 14),
            AiEvaluationStanding("p3", "ai_deterministic", 3, 4, 50, 12),
        )

        actual_score, wins, draws, losses = _controller_pair_actual_score(
            standings,
            "ai_deterministic",
            "ai_heuristics",
        )

        self.assertEqual((wins, draws, losses), (1, 0, 1))
        self.assertAlmostEqual(actual_score, 0.5)

    def test_same_controller_pairings_are_ignored(self) -> None:
        standings = (
            AiEvaluationStanding("p1", "ai_deterministic", 1, 8, 70, 15),
            AiEvaluationStanding("p2", "ai_deterministic", 2, 6, 60, 14),
            AiEvaluationStanding("p3", "ai_deterministic", 3, 4, 50, 12),
        )

        updates = _compute_pairwise_game_updates(
            standings,
            {"ai_deterministic": 1500.0},
            base_k=24.0,
        )

        self.assertEqual(updates, ())

    def test_pairwise_draw_uses_equal_core_signature(self) -> None:
        standings = (
            AiEvaluationStanding("p1", "ai_deterministic", 1, 6, 60, 10),
            AiEvaluationStanding("p2", "ai_heuristics", 1, 6, 60, 10),
            AiEvaluationStanding("p3", "ai_other", 3, 2, 20, 8),
        )

        actual_score, wins, draws, losses = _controller_pair_actual_score(
            standings,
            "ai_deterministic",
            "ai_heuristics",
        )

        self.assertEqual((wins, draws, losses), (0, 1, 0))
        self.assertAlmostEqual(actual_score, 0.5)

    def test_pair_k_scales_by_distinct_controller_count(self) -> None:
        self.assertEqual(_pair_k_factor(24.0, 2), 24.0)
        self.assertEqual(_pair_k_factor(24.0, 3), 12.0)

    def test_random_region_selection_is_legal_reproducible_and_varies(self) -> None:
        config = AiEvaluationBucketConfig(region_sampling_seed=17)
        legal_sets = set(legal_region_sets("germany", 3))
        first_schedule = tuple(select_evaluation_regions(config, seed) for seed in range(100))
        repeated_schedule = tuple(select_evaluation_regions(config, seed) for seed in range(100))

        self.assertEqual(first_schedule, repeated_schedule)
        self.assertTrue(set(first_schedule).issubset(legal_sets))
        self.assertGreater(len(set(first_schedule)), 1)

    def test_explicit_regions_disable_random_sampling(self) -> None:
        selected_regions = ("black", "blue", "magenta")
        config = AiEvaluationBucketConfig(
            selected_regions=selected_regions,
            region_sampling_seed=99,
        )

        self.assertEqual(config.region_selection_mode, "fixed_explicit")
        self.assertTrue(
            all(
                select_evaluation_regions(config, seed) == selected_regions
                for seed in range(20)
            )
        )

    def test_derive_final_standings_matches_winner_tiebreak_rules(self) -> None:
        base_state = advance_phase(
            create_initial_state(
                GameConfig(
                    map_id="germany",
                    players=make_default_seat_configs(3),
                    seed=7,
                )
            )
        )
        allowed_city_ids = [
            city.id for city in base_state.game_map.cities if city.region in base_state.selected_regions
        ]

        money_tiebreak_state = replace(
            base_state,
            players=tuple(
                replace(player, elektro=70 if player.player_id == "p2" else 60 if player.player_id == "p1" else 10)
                for player in base_state.players
            ),
        )
        winner_result = resolve_winner(money_tiebreak_state, {"p1": 6, "p2": 6, "p3": 1})
        standings = derive_final_standings(money_tiebreak_state, winner_result)
        self.assertEqual([standing.player_id for standing in standings], ["p2", "p1", "p3"])

        connected_tiebreak_state = replace(
            money_tiebreak_state,
            players=tuple(
                replace(
                    player,
                    elektro=60 if player.player_id in {"p1", "p2"} else player.elektro,
                    network_city_ids=(
                        tuple(allowed_city_ids[:2]) if player.player_id == "p1"
                        else (allowed_city_ids[0],) if player.player_id == "p2"
                        else player.network_city_ids
                    ),
                )
                for player in money_tiebreak_state.players
            ),
        )
        winner_result = resolve_winner(connected_tiebreak_state, {"p1": 6, "p2": 6, "p3": 1})
        standings = derive_final_standings(connected_tiebreak_state, winner_result)
        self.assertEqual([standing.player_id for standing in standings], ["p1", "p2", "p3"])

    def test_evaluate_ai_bucket_returns_report_for_germany_three_players(self) -> None:
        report = evaluate_ai_bucket(
            AiEvaluationBucketConfig(
                games_per_lineup=1,
                seed_start=1,
            )
        )

        self.assertEqual(report.config.map_id, "germany")
        self.assertEqual(report.config.player_count, 3)
        self.assertEqual(report.config.region_selection_mode, "random_all_legal")
        self.assertEqual(report.resolved_selected_regions, ())
        self.assertEqual(len(report.sampled_region_sets), 1)
        self.assertEqual(len(report.scheduled_lineups), 2)
        self.assertEqual(len(report.game_summaries), 2)
        self.assertEqual(
            report.game_summaries[0].selected_regions,
            report.game_summaries[1].selected_regions,
        )
        self.assertIn(report.game_summaries[0].selected_regions, report.legal_region_sets)
        self.assertEqual(
            {summary.controller_name for summary in report.controller_summaries},
            {"ai_deterministic", "ai_heuristics"},
        )
        self.assertEqual(len(report.controller_pair_summaries), 1)
        self.assertTrue(all(game_summary.winner_ids for game_summary in report.game_summaries))

    def test_cli_tool_writes_json_and_prints_summary(self) -> None:
        with TemporaryDirectory() as tempdir:
            output_path = f"{tempdir}/ratings.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = evaluate_ai_ratings_main(
                    [
                        "--games-per-lineup",
                        "1",
                        "--seed-start",
                        "1",
                        "--output",
                        output_path,
                    ]
                )

            self.assertEqual(result, 0)
            with open(output_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["bucket"]["map_id"], "germany")
            self.assertEqual(payload["bucket"]["player_count"], 3)
            self.assertEqual(payload["bucket"]["region_selection_mode"], "random_all_legal")
            self.assertEqual(len(payload["bucket"]["region_game_counts"]), 1)
            self.assertTrue(all(game["selected_regions"] for game in payload["games"]))
            self.assertEqual(len(payload["schedule"]["lineups"]), 2)
            self.assertIn("Leaderboard:", stdout.getvalue())
            self.assertIn("Wrote report to", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
