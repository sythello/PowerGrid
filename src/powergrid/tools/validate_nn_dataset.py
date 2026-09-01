from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from powergrid.ai.nn_rank_value.dataset import (
    DATASET_SPLITS,
    generate_rank_value_dataset,
    iter_parquet_batches,
    load_dataset_metadata,
    load_dataset_records,
    verify_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate terminal-label dataset generation.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    temporary = None
    output = args.output
    if output is None:
        temporary = tempfile.TemporaryDirectory(prefix="powergrid-nn-dataset-")
        output = Path(temporary.name) / "validation_dataset"
    summary = generate_rank_value_dataset(
        output,
        games=3,
        seed_start=211,
        behavior_controllers=("ai_deterministic",),
        counterfactual_every=250,
        counterfactual_max_candidates=2,
        target_shard_size_bytes=64 * 1024,
        split_fractions=(0.5, 0.25, 0.25),
        split_seed=7,
    )
    records = load_dataset_records(output)
    metadata = load_dataset_metadata(output)
    assert len(records) == summary.behavior_samples + summary.counterfactual_samples
    assert metadata["state_dim"] == summary.state_dim
    assert metadata["action_dim"] == summary.action_dim
    assert all(len(row["state_features"]) == summary.state_dim for row in records)
    assert all(len(row["action_features"]) == summary.action_dim for row in records)
    assert all(row["final_place"] in (1, 2, 3) for row in records)
    assert all(row["rank_value"] in (-1.0, 0.0, 1.0) for row in records)
    assert all(row["is_winner"] in (0, 1) for row in records)
    assert summary.counterfactual_samples > 0
    assert len(summary.example_jsonl_paths) == 3
    assert verify_dataset_manifest(output) == {
        "shards": summary.shards,
        "examples": 3,
    }
    split_game_ids: dict[str, set[str]] = {}
    for split in DATASET_SPLITS:
        split_game_ids[split] = {
            str(game_id)
            for batch in iter_parquet_batches(
                output,
                split,
                batch_size=1024,
                columns=("game_id",),
            )
            for game_id in batch.column(0).to_pylist()
        }
    assert not (split_game_ids["train"] & split_game_ids["validation"])
    assert not (split_game_ids["train"] & split_game_ids["test"])
    assert not (split_game_ids["validation"] & split_game_ids["test"])
    assert sum(len(values) for values in split_game_ids.values()) == summary.games

    print("Dataset validation: PASS")
    print(f"  behavior samples: {summary.behavior_samples}")
    print(f"  counterfactual rollout samples: {summary.counterfactual_samples}")
    print(f"  dimensions: state={summary.state_dim}, action={summary.action_dim}")
    print(f"  parquet shards/checksums: {summary.shards} PASS")
    print("  three whole-game JSONL examples: PASS")
    print("  deterministic game-exclusive train/validation/test split: PASS")
    print("  terminal labels and manifest schema: PASS")
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
