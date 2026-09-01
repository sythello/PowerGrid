from __future__ import annotations

import argparse

from powergrid.ai import (
    AiEvaluationBucketConfig,
    build_ai_controller,
    evaluate_ai_bucket,
)
from powergrid.session import GameSession


BASELINE = "ai_deterministic"
PROFILE_CONTROLLERS = (
    "ai_deterministic_efficiency",
    "ai_deterministic_expansion",
    "ai_deterministic_reserve",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate behavior diversity and held-out strength of deterministic profiles."
    )
    parser.add_argument("--games-per-lineup", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=9001)
    parser.add_argument("--min-pair-score", type=float, default=0.35)
    parser.add_argument("--max-pair-score", type=float, default=0.65)
    parser.add_argument("--max-average-finish-gap", type=float, default=0.4)
    parser.add_argument("--max-rating-gap", type=float, default=110.0)
    args = parser.parse_args()

    _validate_behavior_diversity()
    print("Profiled deterministic behavior diversity: PASS")
    for controller_name in PROFILE_CONTROLLERS:
        report = evaluate_ai_bucket(
            AiEvaluationBucketConfig(
                controller_names=(BASELINE, controller_name),
                games_per_lineup=args.games_per_lineup,
                seed_start=args.seed_start,
            )
        )
        summaries = {
            summary.controller_name: summary for summary in report.controller_summaries
        }
        baseline = summaries[BASELINE]
        profile = summaries[controller_name]
        pair = report.controller_pair_summaries[0]
        baseline_pair_score = pair.actual_score_total_for_a / pair.games
        if pair.controller_a != BASELINE:
            baseline_pair_score = 1.0 - baseline_pair_score
        profile_pair_score = 1.0 - baseline_pair_score
        finish_gap = abs(profile.average_finish - baseline.average_finish)
        rating_gap = abs(profile.rating - baseline.rating)
        total_games = len(report.game_summaries)
        winner_share = profile.controller_wins / total_games
        assert args.min_pair_score <= profile_pair_score <= args.max_pair_score, (
            f"{controller_name} pair score {profile_pair_score:.3f} outside calibrated range"
        )
        assert args.min_pair_score <= winner_share <= args.max_pair_score, (
            f"{controller_name} winner share {winner_share:.3f} outside calibrated range"
        )
        assert finish_gap <= args.max_average_finish_gap, (
            f"{controller_name} average-finish gap {finish_gap:.3f} is too large"
        )
        assert rating_gap <= args.max_rating_gap, (
            f"{controller_name} rating gap {rating_gap:.2f} is too large"
        )
        print(f"{controller_name}: PASS")
        print(
            f"  games={total_games} profile_wins={profile.controller_wins} "
            f"winner_share={winner_share:.3f} pair_score={profile_pair_score:.3f}"
        )
        print(
            f"  avg_finish={profile.average_finish:.3f} "
            f"baseline_avg_finish={baseline.average_finish:.3f} "
            f"rating_gap={rating_gap:.2f}"
        )


def _validate_behavior_diversity() -> None:
    scenarios = tuple(
        GameSession.from_scenario(name, seed=7).snapshot()
        for name in ("opening", "resource", "build_test", "endgame")
    )
    baseline = build_ai_controller(BASELINE)
    baseline_intents = []
    for snapshot in scenarios:
        assert snapshot.active_request is not None
        baseline_intents.append(
            baseline.choose_intent(snapshot.active_request, snapshot).to_dict()
        )
    profile_signatures = set()
    for controller_name in PROFILE_CONTROLLERS:
        controller = build_ai_controller(controller_name)
        intents = []
        for snapshot in scenarios:
            assert snapshot.active_request is not None
            intents.append(controller.choose_intent(snapshot.active_request, snapshot).to_dict())
        assert intents != baseline_intents, f"{controller_name} duplicates baseline behavior"
        signature = repr(intents)
        assert signature not in profile_signatures, f"{controller_name} duplicates another profile"
        profile_signatures.add(signature)


if __name__ == "__main__":
    main()
