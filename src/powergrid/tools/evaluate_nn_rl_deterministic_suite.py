from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from powergrid.ai import build_ai_controller, derive_final_standings
from powergrid.ai.nn_rl_based.controller import NnRlBasedAiController
from powergrid.model import GameConfig, ModelValidationError, SeatConfig, legal_region_sets
from powergrid.session import GameSession


RL_CONTROLLER = "ai_nn_rl_based_v1"
DEFAULT_OPPONENTS = (
    "ai_deterministic",
    "ai_deterministic_efficiency",
    "ai_deterministic_expansion",
    "ai_deterministic_reserve",
)


def evaluate_deterministic_suite(
    checkpoint: str | Path,
    *,
    opponents: tuple[str, ...] = DEFAULT_OPPONENTS,
    games_per_lineup: int,
    seed_start: int,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 8301,
    paired_rollout_report: str | Path | None = None,
    paired_checkpoint_label: str = "d010",
    progress_callback: Callable[[str, int, int, float], None] | None = None,
) -> dict[str, Any]:
    if games_per_lineup <= 0 or bootstrap_samples <= 0:
        raise ValueError("games_per_lineup and bootstrap_samples must be positive")
    if not opponents or len(opponents) != len(set(opponents)):
        raise ValueError("opponents must be non-empty and unique")
    if RL_CONTROLLER in opponents:
        raise ValueError("opponents may not contain the RL controller")
    checkpoint_path = Path(checkpoint)
    region_sets = legal_region_sets("germany", 3)
    started = time.perf_counter()
    opponent_results: dict[str, Any] = {}
    all_games: dict[str, list[dict[str, Any]]] = {}

    for opponent_index, opponent in enumerate(opponents):
        rl_controller = NnRlBasedAiController(checkpoint_path=checkpoint_path)
        opponent_controller = build_ai_controller(opponent)
        lineups = (
            (RL_CONTROLLER, RL_CONTROLLER, opponent),
            (RL_CONTROLLER, opponent, opponent),
        )
        games: list[dict[str, Any]] = []
        game_index = 0
        for lineup_index, lineup in enumerate(lineups):
            for offset in range(games_per_lineup):
                game_index += 1
                seed = seed_start + offset
                selected_regions = region_sets[offset % len(region_sets)]
                games.append(
                    _run_game(
                        checkpoint_controller=rl_controller,
                        opponent_controller=opponent_controller,
                        opponent_name=opponent,
                        lineup=lineup,
                        lineup_index=lineup_index,
                        game_index=game_index,
                        seed=seed,
                        selected_regions=selected_regions,
                    )
                )
                if progress_callback is not None:
                    progress_callback(
                        opponent,
                        game_index,
                        len(lineups) * games_per_lineup,
                        time.perf_counter() - started,
                    )
        opponent_results[opponent] = _summarize_opponent(
            games,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + opponent_index,
        )
        all_games[opponent] = games

    paired = _load_paired_result(
        paired_rollout_report, checkpoint_label=paired_checkpoint_label
    )
    end_to_end_all_above_half = all(
        result["pairwise_score"] > 0.5 for result in opponent_results.values()
    )
    paired_above_half = bool(paired is not None and paired["paired_score"] > 0.5)
    return {
        "format_name": "powergrid.nn_rl_deterministic_suite",
        "format_version": 1,
        "configuration": {
            "checkpoint": str(checkpoint_path),
            "opponents": list(opponents),
            "games_per_lineup": games_per_lineup,
            "games_per_opponent": 2 * games_per_lineup,
            "seed_start": seed_start,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "region_schedule": "game offset cycles all legal Germany/3p region sets",
        },
        "paired_rollout": paired,
        "opponents": opponent_results,
        "eligibility": {
            "criterion": (
                "paired rollout score > 0.50 and every end-to-end opponent "
                "pairwise score > 0.50"
            ),
            "paired_rollout_score_above_0_50": paired_above_half,
            "all_end_to_end_scores_above_0_50": end_to_end_all_above_half,
            "retain_as_current_best": paired_above_half
            and end_to_end_all_above_half,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "games": all_games,
    }


def _run_game(
    *,
    checkpoint_controller: NnRlBasedAiController,
    opponent_controller: Any,
    opponent_name: str,
    lineup: tuple[str, ...],
    lineup_index: int,
    game_index: int,
    seed: int,
    selected_regions: tuple[str, ...],
) -> dict[str, Any]:
    config = GameConfig(
        map_id="germany",
        players=tuple(
            SeatConfig(
                player_id=f"p{index + 1}",
                name=f"Player {index + 1}",
                controller=controller,
            )
            for index, controller in enumerate(lineup)
        ),
        seed=seed,
        selected_regions=selected_regions,
    )
    agents = {
        f"p{index + 1}": (
            checkpoint_controller
            if controller == RL_CONTROLLER
            else opponent_controller
        )
        for index, controller in enumerate(lineup)
    }
    session = GameSession.new_game(config, seat_agents=agents)
    snapshot = session.advance_until_blocked()
    if snapshot.winner_result is None:
        message = (
            snapshot.event_log[-1].message
            if snapshot.event_log
            else "game ended without a winner"
        )
        raise ModelValidationError(
            f"suite game failed for opponent={opponent_name} lineup={lineup}: {message}"
        )
    standings = derive_final_standings(snapshot.state, snapshot.winner_result)
    score = wins = draws = losses = 0
    rl_standings = [
        standing for standing in standings if standing.controller_name == RL_CONTROLLER
    ]
    opponent_standings = [
        standing for standing in standings if standing.controller_name == opponent_name
    ]
    for rl_standing in rl_standings:
        for opponent_standing in opponent_standings:
            rl_signature = _standing_signature(rl_standing)
            opponent_signature = _standing_signature(opponent_standing)
            if rl_signature == opponent_signature:
                draws += 1
                score += 0.5
            elif rl_standing.place < opponent_standing.place:
                wins += 1
                score += 1.0
            else:
                losses += 1
    comparisons = wins + draws + losses
    if comparisons <= 0:
        raise ModelValidationError("suite game contains no cross-controller comparison")
    return {
        "game_index": game_index,
        "lineup_index": lineup_index,
        "seed": seed,
        "selected_regions": list(selected_regions),
        "lineup": list(lineup),
        "score": score,
        "comparisons": comparisons,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "standings": [standing.to_dict() for standing in standings],
    }


def _standing_signature(standing: Any) -> tuple[int, int, int]:
    return (
        int(standing.powered_cities),
        int(standing.money),
        int(standing.connected_cities),
    )


def _summarize_opponent(
    games: list[dict[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    if not games:
        raise ValueError("opponent summary requires completed games")
    score_total = sum(float(game["score"]) for game in games)
    comparisons = sum(int(game["comparisons"]) for game in games)
    wins = sum(int(game["wins"]) for game in games)
    draws = sum(int(game["draws"]) for game in games)
    losses = sum(int(game["losses"]) for game in games)
    rl_places: list[int] = []
    opponent_places: list[int] = []
    region_values: dict[str, list[tuple[float, int]]] = defaultdict(list)
    opponent_name = ""
    for game in games:
        region_key = ",".join(game["selected_regions"])
        region_values[region_key].append(
            (float(game["score"]), int(game["comparisons"]))
        )
        for standing in game["standings"]:
            if standing["controller_name"] == RL_CONTROLLER:
                rl_places.append(int(standing["place"]))
            else:
                opponent_name = str(standing["controller_name"])
                opponent_places.append(int(standing["place"]))
    interval = _game_bootstrap_score_interval(
        games, samples=bootstrap_samples, seed=bootstrap_seed
    )
    return {
        "opponent": opponent_name,
        "games_completed": len(games),
        "seat_pair_comparisons": comparisons,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "pairwise_score": score_total / comparisons,
        "pairwise_score_95_ci": interval,
        "rl_average_finish": float(np.mean(rl_places)),
        "opponent_average_finish": float(np.mean(opponent_places)),
        "score_above_0_50": score_total / comparisons > 0.5,
        "region_game_counts": {
            key: len(values) for key, values in sorted(region_values.items())
        },
        "region_pairwise_scores": {
            key: sum(item[0] for item in values)
            / sum(item[1] for item in values)
            for key, values in sorted(region_values.items())
        },
    }


def _game_bootstrap_score_interval(
    games: list[dict[str, Any]], *, samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    scores = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        indices = rng.integers(0, len(games), size=len(games))
        selected = [games[int(index)] for index in indices]
        scores[sample_index] = sum(float(game["score"]) for game in selected) / sum(
            int(game["comparisons"]) for game in selected
        )
    return [float(value) for value in np.quantile(scores, [0.025, 0.975])]


def _load_paired_result(
    path: str | Path | None, *, checkpoint_label: str
) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("format_name") != "powergrid.nn_rl_paired_terminal_rollouts":
        raise ValueError("paired rollout report format is incompatible")
    try:
        result = report["checkpoints"][checkpoint_label]
    except KeyError as exc:
        raise ValueError(
            f"paired rollout report has no checkpoint {checkpoint_label!r}"
        ) from exc
    return {
        "report": str(path),
        "checkpoint_label": checkpoint_label,
        "deviations": int(result["deviations"]),
        "paired_score": float(result["paired_score"]),
        "paired_score_95_ci": list(result["paired_score_95_ci"]),
        "mean_advantage": float(result["mean_advantage_on_deviations"]),
        "mean_advantage_95_ci": list(result["mean_advantage_95_ci"]),
        "statistically_positive": bool(result["statistically_positive"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare one RL checkpoint with every deterministic profile."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--games-per-lineup", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=80001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=8301)
    parser.add_argument("--paired-rollout-report")
    parser.add_argument("--paired-checkpoint-label", default="d010")
    parser.add_argument(
        "--output",
        default="artifacts/validation/ai_nn_rl_based_v1_deterministic_suite.json",
    )
    args = parser.parse_args(argv)

    def progress(opponent: str, completed: int, total: int, elapsed: float) -> None:
        if completed == total or completed % 20 == 0:
            print(
                f"Progress: opponent={opponent} games={completed}/{total} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    report = evaluate_deterministic_suite(
        args.checkpoint,
        opponents=tuple(args.opponents),
        games_per_lineup=args.games_per_lineup,
        seed_start=args.seed_start,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        paired_rollout_report=args.paired_rollout_report,
        paired_checkpoint_label=args.paired_checkpoint_label,
        progress_callback=progress,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("NN RL deterministic-suite evaluation")
    for opponent, result in report["opponents"].items():
        print(
            f"{opponent}: score={result['pairwise_score']:.4f} "
            f"ci={result['pairwise_score_95_ci']} "
            f"avg_finish={result['rl_average_finish']:.4f}/"
            f"{result['opponent_average_finish']:.4f} "
            f"pass={result['score_above_0_50']}"
        )
    print(f"Eligibility: {report['eligibility']}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
