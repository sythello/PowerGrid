from __future__ import annotations

import argparse

from powergrid.ai.nn_rl_based.dataset import RlDatasetProgress, generate_rl_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate decision-grouped Policy/vector-Q data for ai_nn_rl_based_v1."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--map", dest="map_id", default="germany")
    parser.add_argument("--players", dest="player_count", type=int, default=3)
    parser.add_argument("--behavior-controller", default="ai_deterministic")
    parser.add_argument("--continuation-controller", default="ai_deterministic")
    regions_group = parser.add_mutually_exclusive_group()
    regions_group.add_argument(
        "--regions", help="Optional comma-separated Germany region ids (one fixed set)."
    )
    regions_group.add_argument(
        "--region-set",
        action="append",
        default=[],
        help=(
            "Repeatable comma-separated legal region set. Without either region option, "
            "games cycle by seed through every legal set."
        ),
    )
    parser.add_argument("--target-checkpoint")
    parser.add_argument("--search-fraction", type=float, default=0.0)
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument(
        "--adaptive-depth-2",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-search-nodes", type=int, default=512)
    parser.add_argument("--max-boundary-actions", type=int, default=128)
    parser.add_argument(
        "--leaf-policy", choices=("deterministic", "checkpoint"), default="deterministic"
    )
    parser.add_argument("--search-policy-mix", type=float, default=0.5)
    parser.add_argument("--search-temperature", type=float, default=0.25)
    parser.add_argument("--max-actions-per-game", type=int, default=5000)
    parser.add_argument("--target-shard-size-mib", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args(argv)
    regions = tuple(
        item.strip() for item in (args.regions or "").split(",") if item.strip()
    )
    region_sets = tuple(
        tuple(item.strip() for item in raw.split(",") if item.strip())
        for raw in args.region_set
    )

    def progress(value: RlDatasetProgress) -> None:
        if not args.progress_every:
            return
        if value.games_completed % args.progress_every and value.games_completed != value.games_total:
            return
        print(
            f"Progress: {value.games_completed}/{value.games_total} games "
            f"decisions={value.decisions} searched={value.searched_decisions} "
            f"nodes={value.search_nodes} parquet={value.parquet_bytes / (1024 * 1024):.1f} MiB "
            f"elapsed={value.elapsed_seconds:.1f}s",
            flush=True,
        )

    summary = generate_rl_dataset(
        args.output,
        games=args.games,
        seed_start=args.seed_start,
        map_id=args.map_id,
        player_count=args.player_count,
        behavior_controller=args.behavior_controller,
        continuation_controller=args.continuation_controller,
        selected_regions=regions,
        region_sets=region_sets,
        target_checkpoint=args.target_checkpoint,
        search_fraction=args.search_fraction,
        search_depth=args.search_depth,
        adaptive_depth_2=args.adaptive_depth_2,
        max_search_nodes=args.max_search_nodes,
        max_boundary_actions=args.max_boundary_actions,
        leaf_policy=args.leaf_policy,
        search_policy_mix=args.search_policy_mix,
        search_temperature=args.search_temperature,
        max_actions_per_game=args.max_actions_per_game,
        target_shard_size_bytes=args.target_shard_size_mib * 1024 * 1024,
        split_fractions=(
            args.train_fraction,
            args.validation_fraction,
            args.test_fraction,
        ),
        split_seed=args.split_seed,
        workers=args.workers,
        progress_callback=progress,
    )
    print("NN RL Dataset Generated")
    print(f"Output: {summary.output_path}")
    print(f"Manifest: {summary.manifest_path}")
    print(
        f"Games={summary.games} decisions={summary.decisions} "
        f"searched={summary.searched_decisions} nodes={summary.search_nodes} "
        f"depth2={summary.depth_2_completed}"
    )
    print(
        f"Features: state={summary.state_dim} action={summary.action_dim}; "
        f"Parquet: shards={summary.shards} size={summary.parquet_bytes / (1024 * 1024):.1f} MiB"
    )
    print(f"Elapsed: {summary.elapsed_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
