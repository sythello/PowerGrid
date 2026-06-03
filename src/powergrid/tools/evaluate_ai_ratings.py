from __future__ import annotations

import argparse
import json
from pathlib import Path

from powergrid.ai import AiEvaluationBucketConfig, evaluate_ai_bucket


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Power Grid AI controllers with Elo ratings.")
    parser.add_argument("--map", dest="map_id", default="germany")
    parser.add_argument("--players", dest="player_count", type=int, default=3)
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["ai_deterministic", "ai_heuristics"],
        help="Controller ids to evaluate. V1 expects exactly two controllers.",
    )
    parser.add_argument("--games-per-lineup", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--k-factor", type=float, default=24.0)
    parser.add_argument(
        "--output",
        help="Optional JSON output path. Defaults to artifacts/ai_ratings/{map}_{players}p.json",
    )
    args = parser.parse_args(argv)

    config = AiEvaluationBucketConfig(
        map_id=args.map_id,
        player_count=args.player_count,
        controller_names=tuple(args.controllers),
        games_per_lineup=args.games_per_lineup,
        seed_start=args.seed_start,
        initial_rating=args.initial_rating,
        k_factor=args.k_factor,
    )
    report = evaluate_ai_bucket(config)
    output_path = Path(args.output) if args.output else Path("artifacts") / "ai_ratings" / f"{args.map_id}_{args.player_count}p.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("AI Elo Evaluation")
    print(
        f"Bucket: map={report.config.map_id} players={report.config.player_count} "
        f"regions={','.join(report.resolved_selected_regions)}"
    )
    print(
        f"Games: {len(report.game_summaries)} "
        f"({report.config.games_per_lineup} per lineup across {len(report.scheduled_lineups)} lineups)"
    )
    print("Leaderboard:")
    for index, summary in enumerate(report.controller_summaries, start=1):
        print(
            f"{index}. {summary.controller_name} "
            f"rating={summary.rating:.2f} "
            f"games={summary.games} "
            f"seats={summary.seat_appearances} "
            f"wins={summary.controller_wins} "
            f"avg_finish={summary.average_finish:.3f}"
        )
    print(f"Wrote report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
