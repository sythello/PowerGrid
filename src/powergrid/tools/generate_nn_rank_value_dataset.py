from __future__ import annotations

import argparse

from powergrid.ai.nn_rank_value.dataset import (
    DEFAULT_BEHAVIOR_CONTROLLERS,
    DatasetGenerationProgress,
    generate_rank_value_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate terminal-rank training data for ai_nn_rank_value_v1."
    )
    parser.add_argument("--output", required=True, help="Output Parquet dataset directory.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--map", dest="map_id", default="germany")
    parser.add_argument("--players", dest="player_count", type=int, choices=range(3, 7), default=3)
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=list(DEFAULT_BEHAVIOR_CONTROLLERS),
        help="Behavior controllers rotated across seats and games.",
    )
    regions_group = parser.add_mutually_exclusive_group()
    regions_group.add_argument(
        "--regions",
        help="Optional comma-separated selected region ids (single region set).",
    )
    regions_group.add_argument(
        "--region-set",
        action="append",
        default=[],
        help="Repeatable comma-separated region set; games rotate across supplied sets.",
    )
    parser.add_argument(
        "--counterfactual-every",
        type=int,
        default=0,
        help="At every Nth decision, finish candidate rollouts for direct action labels; 0 disables.",
    )
    parser.add_argument("--counterfactual-max-candidates", type=int, default=8)
    parser.add_argument("--rollout-controller", default="ai_deterministic")
    parser.add_argument("--max-actions-per-game", type=int, default=5000)
    parser.add_argument("--target-shard-size-mib", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N completed games; 0 disables progress output.",
    )
    args = parser.parse_args(argv)
    selected_regions = tuple(
        value.strip() for value in (args.regions or "").split(",") if value.strip()
    )
    region_sets = tuple(
        tuple(value.strip() for value in raw.split(",") if value.strip())
        for raw in args.region_set
    )

    def report_progress(progress: DatasetGenerationProgress) -> None:
        if not args.progress_every:
            return
        if (
            progress.games_completed % args.progress_every
            and progress.games_completed != progress.games_total
        ):
            return
        rate = progress.games_completed / max(progress.elapsed_seconds, 1e-9)
        size_mib = progress.parquet_bytes / (1024 * 1024)
        print(
            f"Progress: {progress.games_completed}/{progress.games_total} games "
            f"rows={progress.rows} parquet={size_mib:.1f} MiB "
            f"rate={rate:.2f} games/s",
            flush=True,
        )

    summary = generate_rank_value_dataset(
        args.output,
        games=args.games,
        seed_start=args.seed_start,
        map_id=args.map_id,
        player_count=args.player_count,
        behavior_controllers=tuple(args.controllers),
        selected_regions=selected_regions,
        region_sets=region_sets,
        counterfactual_every=args.counterfactual_every,
        counterfactual_max_candidates=args.counterfactual_max_candidates,
        rollout_controller=args.rollout_controller,
        max_actions_per_game=args.max_actions_per_game,
        target_shard_size_bytes=args.target_shard_size_mib * 1024 * 1024,
        split_fractions=(
            args.train_fraction,
            args.validation_fraction,
            args.test_fraction,
        ),
        split_seed=args.split_seed,
        workers=args.workers,
        progress_callback=report_progress,
    )
    print("NN Rank-Value Dataset Generated")
    print(f"Output: {summary.output_path}")
    print(f"Manifest: {summary.metadata_path}")
    print(f"Games: {summary.games}")
    print(f"Behavior samples: {summary.behavior_samples}")
    print(f"Counterfactual samples: {summary.counterfactual_samples}")
    print(f"Feature dimensions: state={summary.state_dim} action={summary.action_dim}")
    print(
        f"Parquet: shards={summary.shards} "
        f"size={summary.parquet_bytes / (1024 * 1024):.1f} MiB"
    )
    for split in ("train", "validation", "test"):
        print(
            f"{split}: games={summary.split_games[split]} "
            f"rows={summary.split_rows[split]}"
        )
    print(f"Example JSONL games: {len(summary.example_jsonl_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
