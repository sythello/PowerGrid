from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any


OWN_SCORE_MISS_ABS_THRESHOLD = 50.0
OWN_SCORE_MISS_SHARE_THRESHOLD = 0.4
MATERIAL_DELTA_EPSILON = 1e-6
NON_ACTIONABLE_PROJECTION_HORIZONS = {
    "auction_fallback",
    "post_auction_economy",
    "terminal_if_won",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze heuristic AI strategy logs for score prediction discrepancies."
    )
    parser.add_argument("--strategy-dir", required=True, help="Directory containing strategy log JSON files.")
    parser.add_argument("--game-dir", help="Directory containing full game log JSON files.")
    parser.add_argument("--player-id", default="p1", help="Heuristic player id to analyze.")
    parser.add_argument(
        "--output",
        help="Optional JSON report path. Defaults to {strategy-dir}/discrepancy_report.json.",
    )
    parser.add_argument(
        "--own-threshold",
        type=float,
        default=OWN_SCORE_MISS_ABS_THRESHOLD,
        help="Absolute own-score delta threshold for primary own-score miss classification.",
    )
    args = parser.parse_args(argv)

    strategy_dir = Path(args.strategy_dir)
    game_dir = Path(args.game_dir) if args.game_dir else None
    output_path = Path(args.output) if args.output else strategy_dir / "discrepancy_report.json"
    report = analyze_strategy_logs(
        strategy_dir,
        game_dir=game_dir,
        player_id=args.player_id,
        own_threshold=args.own_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Strategy Log Analysis")
    print(f"Strategy dir: {strategy_dir}")
    print(f"Player: {args.player_id}")
    print(f"Decision pairs: {report['decision_pair_count']}")
    print(f"Heuristic wins: {report['heuristic_wins']} / {report['game_count']}")
    if report["top_relative_discrepancies"]:
        top = report["top_relative_discrepancies"][0]
        print(
            "Largest relative miss: "
            f"seed={top['seed']} entry={top['entry_index']} "
            f"phase={top['phase']} intent={top['intent_type']} "
            f"delta={top['relative_signed_delta']:.4f} "
            f"primary={top['primary_driver']}"
        )
    print(f"Wrote report to {output_path}")
    return 0


def analyze_strategy_logs(
    strategy_dir: Path,
    *,
    game_dir: Path | None = None,
    player_id: str = "p1",
    own_threshold: float = OWN_SCORE_MISS_ABS_THRESHOLD,
) -> dict[str, object]:
    strategy_paths = sorted(strategy_dir.glob("*.json"))
    records = []
    schema_versions = Counter()
    for path in strategy_paths:
        if path.name == "discrepancy_report.json":
            continue
        seed = _seed_from_path(path)
        payload = _read_json(path)
        entries = _heuristic_entries(payload, player_id)
        for entry in entries:
            schema_versions[int(entry["payload"]["state"].get("schema_version", 0))] += 1
        for index, entry in enumerate(entries[:-1]):
            records.append(_build_prediction_record(seed, entry, entries[index + 1], own_threshold=own_threshold))

    records.sort(key=lambda record: record["relative_absolute_delta"], reverse=True)
    actionable_records = [
        record for record in records if _is_actionable_for_immediate_tuning(record)
    ]
    own_score_misses = [
        record
        for record in actionable_records
        if record["primary_driver"] == "own_score_miss"
    ]
    opponent_relative_misses = [
        record
        for record in actionable_records
        if record["primary_driver"] != "own_score_miss"
    ]

    games = _summarize_games(game_dir, player_id=player_id) if game_dir is not None else []
    heuristic_wins = sum(1 for game in games if game["heuristic_won"])

    return {
        "format_version": 2,
        "analysis_method": {
            "relative_delta": (
                "next current_evaluation.scoreboard.relative_score minus "
                "selected_action.projected_evaluation.scoreboard.relative_score"
            ),
            "own_first_rule": (
                "classify as own_score_miss when abs(own_delta) >= own_threshold "
                "and abs(own_delta) >= own_share_threshold * abs(relative_delta)"
            ),
            "own_threshold": own_threshold,
            "own_share_threshold": OWN_SCORE_MISS_SHARE_THRESHOLD,
            "material_delta_epsilon": MATERIAL_DELTA_EPSILON,
            "actionable_filter": (
                "top discrepancy lists exclude projection horizons that intentionally "
                "look beyond the next heuristic decision"
            ),
            "non_actionable_projection_horizons": sorted(NON_ACTIONABLE_PROJECTION_HORIZONS),
        },
        "strategy_dir": str(strategy_dir),
        "game_dir": str(game_dir) if game_dir is not None else None,
        "player_id": player_id,
        "game_count": len(games),
        "heuristic_wins": heuristic_wins,
        "win_rate": round(heuristic_wins / len(games), 4) if games else None,
        "games": games,
        "decision_pair_count": len(records),
        "actionable_decision_pair_count": len(actionable_records),
        "schema_versions": {str(version): count for version, count in sorted(schema_versions.items())},
        "primary_driver_counts": dict(Counter(record["primary_driver"] for record in records)),
        "actionable_primary_driver_counts": dict(
            Counter(record["primary_driver"] for record in actionable_records)
        ),
        "top_relative_discrepancies": actionable_records[:20],
        "top_relative_discrepancies_all": records[:20],
        "top_own_score_misses": sorted(
            own_score_misses,
            key=lambda record: record["own_absolute_delta"],
            reverse=True,
        )[:20],
        "top_opponent_relative_misses": opponent_relative_misses[:20],
        "summary_by_phase_intent": _summarize_by_phase_intent(records),
        "summary_by_projection_horizon": _summarize_by_projection_horizon(records),
    }


def _is_actionable_for_immediate_tuning(record: dict[str, object]) -> bool:
    horizon = str(record.get("projection_horizon") or "unknown")
    return horizon not in NON_ACTIONABLE_PROJECTION_HORIZONS


def _heuristic_entries(payload: dict[str, Any], player_id: str) -> list[dict[str, Any]]:
    entries = []
    for entry in payload.get("strategy_log", []):
        state = entry.get("payload", {}).get("state", {})
        current = state.get("current_evaluation", {})
        if (
            entry.get("source") == "ai"
            and entry.get("payload", {}).get("label") == "heuristic_decision"
            and current.get("player_id") == player_id
            and "scoreboard" in current
        ):
            entries.append(entry)
    return entries


def _build_prediction_record(
    seed: int | None,
    entry: dict[str, Any],
    next_entry: dict[str, Any],
    *,
    own_threshold: float,
) -> dict[str, object]:
    state = entry["payload"]["state"]
    next_state = next_entry["payload"]["state"]
    projected = state["selected_action"]["projected_evaluation"]
    actual = next_state["current_evaluation"]
    projected_board = projected["scoreboard"]
    actual_board = actual["scoreboard"]

    relative_delta = float(actual_board["relative_score"]) - float(projected_board["relative_score"])
    own_delta = float(actual_board["own_score"]) - float(projected_board["own_score"])
    opponent_adjustment_delta = (
        float(actual_board["opponent_adjustment"])
        - float(projected_board["opponent_adjustment"])
    )
    opponent_deltas = _opponent_score_deltas(
        projected_board.get("opponent_scores", []),
        actual_board.get("opponent_scores", []),
    )
    primary_driver = _classify_primary_driver(
        relative_delta,
        own_delta,
        opponent_adjustment_delta,
        own_threshold=own_threshold,
    )

    return {
        "seed": seed,
        "entry_index": entry["index"],
        "round_number": entry["round_number"],
        "step": entry["step"],
        "phase": entry["phase"],
        "decision_type": state["decision_type"],
        "intent_type": state["intent_type"],
        "intent_payload": state["intent_payload"],
        "selection_rule": state["selected_action"].get("score_terms", {}).get("selection_rule"),
        "projected_kind": state["selected_action"].get("projected_kind")
        or state["selected_action"].get("score_terms", {}).get("projected_kind"),
        "projection_horizon": state["selected_action"].get("projection_horizon"),
        "next_entry_index": next_entry["index"],
        "next_round_number": next_entry["round_number"],
        "next_step": next_entry["step"],
        "next_phase": next_entry["phase"],
        "next_decision_type": next_state["decision_type"],
        "projected_scores": projected_board,
        "actual_scores": actual_board,
        "relative_signed_delta": round(relative_delta, 4),
        "relative_absolute_delta": round(abs(relative_delta), 4),
        "own_signed_delta": round(own_delta, 4),
        "own_absolute_delta": round(abs(own_delta), 4),
        "opponent_adjustment_signed_delta": round(opponent_adjustment_delta, 4),
        "opponent_adjustment_absolute_delta": round(abs(opponent_adjustment_delta), 4),
        "opponent_score_deltas": opponent_deltas,
        "largest_opponent_score_delta": _largest_opponent_delta(opponent_deltas),
        "primary_driver": primary_driver,
    }


def _classify_primary_driver(
    relative_delta: float,
    own_delta: float,
    opponent_adjustment_delta: float,
    *,
    own_threshold: float,
) -> str:
    own_abs = abs(own_delta)
    relative_abs = abs(relative_delta)
    if (
        relative_abs < MATERIAL_DELTA_EPSILON
        and own_abs < MATERIAL_DELTA_EPSILON
        and abs(opponent_adjustment_delta) < MATERIAL_DELTA_EPSILON
    ):
        return "no_material_miss"
    if own_abs >= own_threshold and own_abs >= relative_abs * OWN_SCORE_MISS_SHARE_THRESHOLD:
        return "own_score_miss"
    if abs(opponent_adjustment_delta) >= relative_abs * 0.4:
        return "opponent_score_or_relative_miss"
    return "mixed_or_small_relative_miss"


def _opponent_score_deltas(
    projected_scores: list[dict[str, Any]],
    actual_scores: list[dict[str, Any]],
) -> list[dict[str, object]]:
    actual_by_id = {score["player_id"]: score for score in actual_scores}
    deltas = []
    for projected in projected_scores:
        player_id = projected["player_id"]
        actual = actual_by_id.get(player_id)
        if actual is None:
            continue
        projected_score = float(projected["score_for_relative"])
        actual_score = float(actual["score_for_relative"])
        deltas.append(
            {
                "player_id": player_id,
                "projected_score_for_relative": round(projected_score, 4),
                "actual_score_for_relative": round(actual_score, 4),
                "signed_delta": round(actual_score - projected_score, 4),
                "absolute_delta": round(abs(actual_score - projected_score), 4),
                "projected_threat_applied": bool(projected.get("threat_applied")),
                "actual_threat_applied": bool(actual.get("threat_applied")),
            }
        )
    return sorted(deltas, key=lambda item: item["absolute_delta"], reverse=True)


def _largest_opponent_delta(deltas: list[dict[str, object]]) -> dict[str, object] | None:
    return deltas[0] if deltas else None


def _summarize_by_phase_intent(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["phase"]), str(record["intent_type"]))].append(record)
    summary = []
    for (phase, intent_type), values in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        relative_values = [float(value["relative_signed_delta"]) for value in values]
        own_values = [float(value["own_signed_delta"]) for value in values]
        summary.append(
            {
                "phase": phase,
                "intent_type": intent_type,
                "count": len(values),
                "average_relative_signed_delta": round(mean(relative_values), 4),
                "max_relative_absolute_delta": round(max(abs(value) for value in relative_values), 4),
                "average_own_signed_delta": round(mean(own_values), 4),
                "max_own_absolute_delta": round(max(abs(value) for value in own_values), 4),
                "primary_driver_counts": dict(Counter(value["primary_driver"] for value in values)),
            }
        )
    return summary


def _summarize_by_projection_horizon(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("projection_horizon") or "unknown")].append(record)
    summary = []
    for horizon, values in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        relative_values = [float(value["relative_signed_delta"]) for value in values]
        own_values = [float(value["own_signed_delta"]) for value in values]
        summary.append(
            {
                "projection_horizon": horizon,
                "count": len(values),
                "average_relative_signed_delta": round(mean(relative_values), 4),
                "max_relative_absolute_delta": round(max(abs(value) for value in relative_values), 4),
                "average_own_signed_delta": round(mean(own_values), 4),
                "max_own_absolute_delta": round(max(abs(value) for value in own_values), 4),
                "primary_driver_counts": dict(Counter(value["primary_driver"] for value in values)),
            }
        )
    return summary


def _summarize_games(game_dir: Path, *, player_id: str) -> list[dict[str, object]]:
    games = []
    for game_path in sorted(game_dir.glob("*.json")):
        payload = _read_json(game_path)
        winner = payload.get("winner_result") or {}
        players = {player["player_id"]: player["controller"] for player in payload["config"]["players"]}
        powered = winner.get("powered_cities", {})
        money = winner.get("money", {})
        connected = winner.get("connected_cities", {})
        winner_ids = list(winner.get("winner_ids", []))
        games.append(
            {
                "seed": _seed_from_path(game_path),
                "winner_ids": winner_ids,
                "heuristic_won": player_id in winner_ids,
                "standings": [
                    {
                        "player_id": pid,
                        "controller": players[pid],
                        "powered": powered.get(pid),
                        "money": money.get(pid),
                        "connected": connected.get(pid),
                    }
                    for pid in sorted(players)
                ],
            }
        )
    return games


def _seed_from_path(path: Path) -> int | None:
    marker = "seed"
    stem = path.stem
    if marker not in stem:
        return None
    suffix = stem.rsplit(marker, 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
