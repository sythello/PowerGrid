from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from powergrid.ai import derive_final_standings
from powergrid.ai.nn_rank_value.dataset import sha256_file
from powergrid.ai.nn_rl_based.controller import (
    CONTROLLER_NAME,
    NnRlBasedAiController,
)
from powergrid.model import GameConfig, ModelValidationError, SeatConfig, legal_region_sets
from powergrid.session import GameSession


CANDIDATE = "candidate"
INCUMBENT = "incumbent"
DUEL_LINEUPS = (
    (CANDIDATE, CANDIDATE, INCUMBENT),
    (CANDIDATE, INCUMBENT, CANDIDATE),
    (INCUMBENT, CANDIDATE, CANDIDATE),
    (CANDIDATE, INCUMBENT, INCUMBENT),
    (INCUMBENT, CANDIDATE, INCUMBENT),
    (INCUMBENT, INCUMBENT, CANDIDATE),
)


def evaluate_checkpoint_duel(
    candidate_checkpoint: str | Path,
    incumbent_checkpoint: str | Path,
    *,
    seeds: int,
    seed_start: int,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 9301,
    progress_callback: Callable[[int, int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Compare two RL checkpoints over both compositions and all seat placements."""

    if seeds <= 0 or bootstrap_samples <= 0:
        raise ValueError("seeds and bootstrap_samples must be positive")
    if seed_start < 0 or bootstrap_seed < 0:
        raise ValueError("seed_start and bootstrap_seed may not be negative")
    candidate_path = Path(candidate_checkpoint)
    incumbent_path = Path(incumbent_checkpoint)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate checkpoint does not exist: {candidate_path}")
    if not incumbent_path.is_file():
        raise FileNotFoundError(f"incumbent checkpoint does not exist: {incumbent_path}")

    candidate_controller = NnRlBasedAiController(candidate_path)
    incumbent_controller = NnRlBasedAiController(incumbent_path)
    controllers = {
        CANDIDATE: candidate_controller,
        INCUMBENT: incumbent_controller,
    }
    region_sets = legal_region_sets("germany", 3)
    total_games = seeds * len(DUEL_LINEUPS)
    games: list[dict[str, Any]] = []
    started = time.perf_counter()

    for seed_offset in range(seeds):
        seed = seed_start + seed_offset
        selected_regions = region_sets[seed_offset % len(region_sets)]
        for lineup_index, lineup in enumerate(DUEL_LINEUPS):
            games.append(
                _run_duel_game(
                    controllers=controllers,
                    lineup=lineup,
                    lineup_index=lineup_index,
                    game_index=len(games) + 1,
                    seed=seed,
                    selected_regions=selected_regions,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    len(games),
                    total_games,
                    seed,
                    time.perf_counter() - started,
                )

    overall = _summarize_duel_games(
        games,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    compositions = {
        name: _summarize_duel_games(
            [game for game in games if game["composition"] == name],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index + 1,
        )
        for index, name in enumerate(("candidate_majority", "incumbent_majority"))
    }
    lineups = {
        str(index): {
            "roles": list(lineup),
            **_summarize_duel_games(
                [game for game in games if game["lineup_index"] == index],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 10 + index,
            ),
        }
        for index, lineup in enumerate(DUEL_LINEUPS)
    }
    candidate_above_half = overall["pairwise_score"] > 0.5
    confidence_lower_above_half = overall["pairwise_score_95_ci"][0] > 0.5
    both_compositions_noninferior = all(
        result["pairwise_score"] >= 0.5 for result in compositions.values()
    )
    return {
        "format_name": "powergrid.nn_rl_checkpoint_duel",
        "format_version": 1,
        "method": {
            "roles": [CANDIDATE, INCUMBENT],
            "lineups_per_seed": len(DUEL_LINEUPS),
            "lineup_schedule": "both 2v1 compositions and every seat placement",
            "score": "candidate wins + 0.5 * draws over cross-checkpoint seat pairs",
            "tie": "equal powered cities, money, and connected cities",
            "confidence_interval": (
                "seed-cluster bootstrap; every sampled seed keeps all six lineups"
            ),
        },
        "configuration": {
            "candidate_checkpoint": str(candidate_path),
            "candidate_checkpoint_sha256": sha256_file(candidate_path),
            "incumbent_checkpoint": str(incumbent_path),
            "incumbent_checkpoint_sha256": sha256_file(incumbent_path),
            "map": "germany",
            "players": 3,
            "seeds": seeds,
            "seed_start": seed_start,
            "games_per_seed": len(DUEL_LINEUPS),
            "games": total_games,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "region_schedule": "seed offset cycles all legal Germany/3p region sets",
        },
        "overall": overall,
        "compositions": compositions,
        "lineups": lineups,
        "eligibility": {
            "criterion": (
                "overall candidate score > 0.50, seed-cluster 95% CI lower bound "
                "> 0.50, and both composition point scores >= 0.50"
            ),
            "candidate_score_above_0_50": candidate_above_half,
            "candidate_95_ci_lower_above_0_50": confidence_lower_above_half,
            "both_compositions_score_at_least_0_50": both_compositions_noninferior,
            "promote_candidate": (
                candidate_above_half
                and confidence_lower_above_half
                and both_compositions_noninferior
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "games": games,
    }


def _run_duel_game(
    *,
    controllers: dict[str, NnRlBasedAiController],
    lineup: tuple[str, str, str],
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
                controller=CONTROLLER_NAME,
            )
            for index in range(3)
        ),
        seed=seed,
        selected_regions=selected_regions,
    )
    roles_by_player = {
        f"p{index + 1}": role for index, role in enumerate(lineup)
    }
    agents = {
        player_id: controllers[role]
        for player_id, role in roles_by_player.items()
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
            f"checkpoint duel game failed for seed={seed} lineup={lineup}: {message}"
        )
    standings = derive_final_standings(snapshot.state, snapshot.winner_result)
    candidate_standings = [
        standing
        for standing in standings
        if roles_by_player[standing.player_id] == CANDIDATE
    ]
    incumbent_standings = [
        standing
        for standing in standings
        if roles_by_player[standing.player_id] == INCUMBENT
    ]
    score = wins = draws = losses = 0
    for candidate in candidate_standings:
        for incumbent in incumbent_standings:
            if _standing_signature(candidate) == _standing_signature(incumbent):
                draws += 1
                score += 0.5
            elif candidate.place < incumbent.place:
                wins += 1
                score += 1.0
            else:
                losses += 1
    comparisons = wins + draws + losses
    if comparisons <= 0:
        raise ModelValidationError("checkpoint duel game has no cross-checkpoint pairs")
    standing_payloads = []
    for standing in standings:
        payload = standing.to_dict()
        payload["duel_role"] = roles_by_player[standing.player_id]
        standing_payloads.append(payload)
    return {
        "game_index": game_index,
        "lineup_index": lineup_index,
        "seed": seed,
        "selected_regions": list(selected_regions),
        "lineup": list(lineup),
        "composition": (
            "candidate_majority"
            if lineup.count(CANDIDATE) == 2
            else "incumbent_majority"
        ),
        "score": score,
        "comparisons": comparisons,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "standings": standing_payloads,
    }


def _standing_signature(standing: Any) -> tuple[int, int, int]:
    return (
        int(standing.powered_cities),
        int(standing.money),
        int(standing.connected_cities),
    )


def _summarize_duel_games(
    games: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not games:
        raise ValueError("checkpoint duel summary requires completed games")
    score_total = sum(float(game["score"]) for game in games)
    comparisons = sum(int(game["comparisons"]) for game in games)
    wins = sum(int(game["wins"]) for game in games)
    draws = sum(int(game["draws"]) for game in games)
    losses = sum(int(game["losses"]) for game in games)
    candidate_places: list[int] = []
    incumbent_places: list[int] = []
    region_values: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for game in games:
        region_key = ",".join(game["selected_regions"])
        region_values[region_key].append(
            (float(game["score"]), int(game["comparisons"]))
        )
        for standing in game["standings"]:
            destination = (
                candidate_places
                if standing["duel_role"] == CANDIDATE
                else incumbent_places
            )
            destination.append(int(standing["place"]))
    interval = _seed_cluster_bootstrap_score_interval(
        games,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    score = score_total / comparisons
    return {
        "seeds": len({int(game["seed"]) for game in games}),
        "games_completed": len(games),
        "seat_pair_comparisons": comparisons,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "pairwise_score": score,
        "pairwise_score_95_ci": interval,
        "candidate_average_finish": float(np.mean(candidate_places)),
        "incumbent_average_finish": float(np.mean(incumbent_places)),
        "score_above_0_50": score > 0.5,
        "statistically_superior": interval[0] > 0.5,
        "region_game_counts": {
            key: len(values) for key, values in sorted(region_values.items())
        },
        "region_pairwise_scores": {
            key: sum(item[0] for item in values)
            / sum(item[1] for item in values)
            for key, values in sorted(region_values.items())
        },
    }


def _seed_cluster_bootstrap_score_interval(
    games: list[dict[str, Any]], *, samples: int, seed: int
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        by_seed[int(game["seed"])].append(game)
    clusters = [by_seed[key] for key in sorted(by_seed)]
    rng = np.random.default_rng(seed)
    scores = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        indices = rng.integers(0, len(clusters), size=len(clusters))
        selected = [
            game
            for cluster_index in indices
            for game in clusters[int(cluster_index)]
        ]
        scores[sample_index] = sum(float(game["score"]) for game in selected) / sum(
            int(game["comparisons"]) for game in selected
        )
    return [float(value) for value in np.quantile(scores, [0.025, 0.975])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate and incumbent ai_nn_rl_based_v1 checkpoints with "
            "both compositions and all seat placements."
        )
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--seed-start", type=int, default=400001)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=9301)
    parser.add_argument(
        "--output",
        default="artifacts/validation/ai_nn_rl_checkpoint_duel.json",
    )
    args = parser.parse_args(argv)

    def progress(completed: int, total: int, seed: int, elapsed: float) -> None:
        if completed == total or completed % 20 == 0:
            print(
                f"Progress: games={completed}/{total} seed={seed} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    report = evaluate_checkpoint_duel(
        args.candidate,
        args.incumbent,
        seeds=args.seeds,
        seed_start=args.seed_start,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        progress_callback=progress,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    overall = report["overall"]
    print("NN RL checkpoint duel")
    print(f"Candidate: {args.candidate}")
    print(f"Incumbent: {args.incumbent}")
    print(
        f"Score={overall['pairwise_score']:.4f} "
        f"CI={overall['pairwise_score_95_ci']} "
        f"W/D/L={overall['wins']}/{overall['draws']}/{overall['losses']}"
    )
    for name, result in report["compositions"].items():
        print(
            f"{name}: score={result['pairwise_score']:.4f} "
            f"CI={result['pairwise_score_95_ci']}"
        )
    print(f"Eligibility: {report['eligibility']}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
