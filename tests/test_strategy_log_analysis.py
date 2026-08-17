from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from powergrid.tools.analyze_strategy_logs import analyze_strategy_logs, main as analyze_strategy_logs_main


class StrategyLogAnalysisTests(unittest.TestCase):
    def test_analyzer_classifies_own_score_miss_before_relative_opponent_miss(self) -> None:
        with TemporaryDirectory() as tempdir:
            strategy_dir = Path(tempdir)
            _write_strategy_log(
                strategy_dir / "germany_3p_seed7.json",
                projected_scoreboard={
                    "own_score": 100.0,
                    "opponent_scores": [
                        _opponent_score("p2", 80.0),
                        _opponent_score("p3", 60.0),
                    ],
                    "max_opponent_score": 80.0,
                    "average_opponent_score": 70.0,
                    "opponent_adjustment": -66.0,
                    "relative_score": 34.0,
                },
                actual_scoreboard={
                    "own_score": 180.0,
                    "opponent_scores": [
                        _opponent_score("p2", 82.0),
                        _opponent_score("p3", 60.0),
                    ],
                    "max_opponent_score": 82.0,
                    "average_opponent_score": 71.0,
                    "opponent_adjustment": -67.5,
                    "relative_score": 112.5,
                },
            )

            report = analyze_strategy_logs(strategy_dir, player_id="p1")

        self.assertEqual(report["decision_pair_count"], 1)
        top = report["top_relative_discrepancies"][0]
        self.assertEqual(top["primary_driver"], "own_score_miss")
        self.assertEqual(top["own_signed_delta"], 80.0)
        self.assertEqual(top["projection_horizon"], "immediate_state")
        self.assertEqual(report["primary_driver_counts"], {"own_score_miss": 1})

    def test_cli_writes_strategy_analysis_report(self) -> None:
        with TemporaryDirectory() as tempdir:
            strategy_dir = Path(tempdir) / "strategy"
            strategy_dir.mkdir()
            output_path = Path(tempdir) / "report.json"
            _write_strategy_log(
                strategy_dir / "germany_3p_seed8.json",
                projected_scoreboard={
                    "own_score": 100.0,
                    "opponent_scores": [_opponent_score("p2", 80.0)],
                    "max_opponent_score": 80.0,
                    "average_opponent_score": 80.0,
                    "opponent_adjustment": -68.0,
                    "relative_score": 32.0,
                },
                actual_scoreboard={
                    "own_score": 101.0,
                    "opponent_scores": [_opponent_score("p2", 140.0)],
                    "max_opponent_score": 140.0,
                    "average_opponent_score": 140.0,
                    "opponent_adjustment": -119.0,
                    "relative_score": -18.0,
                },
            )

            result = analyze_strategy_logs_main(
                [
                    "--strategy-dir",
                    str(strategy_dir),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(result, 0)
            with output_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(
            payload["top_relative_discrepancies"][0]["primary_driver"],
            "opponent_score_or_relative_miss",
        )

    def test_analyzer_classifies_zero_delta_as_no_material_miss(self) -> None:
        with TemporaryDirectory() as tempdir:
            strategy_dir = Path(tempdir)
            scoreboard = {
                "own_score": 100.0,
                "opponent_scores": [_opponent_score("p2", 80.0)],
                "max_opponent_score": 80.0,
                "average_opponent_score": 80.0,
                "opponent_adjustment": -68.0,
                "relative_score": 32.0,
            }
            _write_strategy_log(
                strategy_dir / "germany_3p_seed9.json",
                projected_scoreboard=scoreboard,
                actual_scoreboard=scoreboard,
            )

            report = analyze_strategy_logs(strategy_dir, player_id="p1")

        self.assertEqual(report["primary_driver_counts"], {"no_material_miss": 1})

    def test_analyzer_keeps_cross_horizon_auction_predictions_out_of_actionable_top(self) -> None:
        with TemporaryDirectory() as tempdir:
            strategy_dir = Path(tempdir)
            payload = {
                "format_version": 1,
                "strategy_log": [
                    _entry(
                        index=1,
                        current_scoreboard=_scoreboard(own=100.0, relative=50.0),
                        projected_scoreboard=_scoreboard(own=500.0, relative=450.0),
                        phase="auction",
                        intent_type="auction_pass",
                        projected_kind="auction_fallback_purchase_at_min_bid",
                        projection_horizon="auction_fallback",
                    ),
                    _entry(
                        index=3,
                        current_scoreboard=_scoreboard(own=120.0, relative=70.0),
                        projected_scoreboard=_scoreboard(own=140.0, relative=90.0),
                        phase="buy_resources",
                        intent_type="finish_buying",
                    ),
                    _entry(
                        index=5,
                        current_scoreboard=_scoreboard(own=190.0, relative=140.0),
                        projected_scoreboard=_scoreboard(own=190.0, relative=140.0),
                        phase="build_houses",
                        intent_type="finish_building",
                    ),
                ],
            }
            (strategy_dir / "germany_3p_seed10.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            report = analyze_strategy_logs(strategy_dir, player_id="p1")

        self.assertEqual(report["decision_pair_count"], 2)
        self.assertEqual(report["actionable_decision_pair_count"], 1)
        self.assertEqual(report["top_relative_discrepancies"][0]["entry_index"], 3)
        self.assertEqual(report["top_relative_discrepancies_all"][0]["entry_index"], 1)
        horizon_counts = {
            item["projection_horizon"]: item["count"]
            for item in report["summary_by_projection_horizon"]
        }
        self.assertEqual(horizon_counts["auction_fallback"], 1)


def _write_strategy_log(
    path: Path,
    *,
    projected_scoreboard: dict[str, object],
    actual_scoreboard: dict[str, object],
) -> None:
    payload = {
        "format_version": 1,
        "strategy_log": [
            _entry(
                index=1,
                current_scoreboard=projected_scoreboard,
                projected_scoreboard=projected_scoreboard,
                phase="buy_resources",
                intent_type="finish_buying",
            ),
            _entry(
                index=3,
                current_scoreboard=actual_scoreboard,
                projected_scoreboard=actual_scoreboard,
                phase="build_houses",
                intent_type="finish_building",
            ),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _entry(
    *,
    index: int,
    current_scoreboard: dict[str, object],
    projected_scoreboard: dict[str, object],
    phase: str,
    intent_type: str,
    projected_kind: str = "current_state_after_pass",
    projection_horizon: str = "immediate_state",
) -> dict[str, object]:
    return {
        "index": index,
        "source": "ai",
        "event_type": "ai_state",
        "level": "debug",
        "message": "Heuristic AI selected an intent.",
        "player_id": "p1",
        "phase": phase,
        "round_number": 1,
        "step": 1,
        "payload": {
            "label": "heuristic_decision",
            "state": {
                "schema_version": 4,
                "decision_type": phase,
                "intent_type": intent_type,
                "intent_payload": {},
                "current_evaluation": {
                    "player_id": "p1",
                    "scoreboard": current_scoreboard,
                },
                "selected_action": {
                    "projected_kind": projected_kind,
                    "projection_horizon": projection_horizon,
                    "score_terms": {"selection_rule": "test"},
                    "projected_evaluation": {
                        "player_id": "p1",
                        "scoreboard": projected_scoreboard,
                    },
                },
            },
        },
    }


def _scoreboard(own: float, relative: float) -> dict[str, object]:
    return {
        "own_score": own,
        "opponent_scores": [_opponent_score("p2", 80.0)],
        "max_opponent_score": 80.0,
        "average_opponent_score": 80.0,
        "opponent_adjustment": relative - own,
        "relative_score": relative,
    }


def _opponent_score(player_id: str, score: float) -> dict[str, object]:
    return {
        "player_id": player_id,
        "current_score": score,
        "refuel_projected_score": score,
        "score_for_relative": score,
        "threat_applied": False,
    }


if __name__ == "__main__":
    unittest.main()
