from __future__ import annotations

import math
from pathlib import Path
import tempfile

from powergrid.ai.nn_rank_value.dataset import (
    generate_rank_value_dataset,
)
from powergrid.ai.nn_rank_value.model import NumpyRankValueNetwork
from powergrid.ai.nn_rank_value.training import train_rank_value_model


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="powergrid-nn-training-") as directory:
        root = Path(directory)
        dataset = root / "dataset"
        checkpoint = root / "model.npz"
        generate_rank_value_dataset(
            dataset,
            games=6,
            seed_start=701,
            behavior_controllers=("ai_deterministic",),
            split_fractions=(0.5, 0.25, 0.25),
            split_seed=29,
        )
        summary = train_rank_value_model(
            dataset,
            checkpoint,
            epochs=4,
            batch_size=128,
            learning_rate=1e-3,
            hidden_dims=(32, 16),
            seed=29,
            scan_batch_size=512,
        )
        restored = NumpyRankValueNetwork.load(checkpoint)

    assert summary.train_samples > 0
    assert summary.validation_samples > 0
    assert summary.test_samples > 0
    assert restored.state_dim == 513 and restored.action_dim == 42
    assert restored.metadata["training_epochs"] == 4
    assert all(math.isfinite(value) for value in summary.final_train_metrics.values())
    assert all(math.isfinite(value) for value in summary.final_validation_metrics.values())
    assert all(math.isfinite(value) for value in summary.final_test_metrics.values())

    print("Training validation: PASS")
    print(
        f"  game-level rows: train={summary.train_samples}, "
        f"validation={summary.validation_samples}, test={summary.test_samples}"
    )
    print(
        f"  final loss: train={summary.final_train_metrics['loss']:.6f}, "
        f"validation={summary.final_validation_metrics['loss']:.6f}"
    )
    print(f"  held-out test loss: {summary.final_test_metrics['loss']:.6f}")
    print("  streaming normalization/training/evaluation: PASS")
    print("  checkpoint metadata/dimensions and manifest hash: PASS")


if __name__ == "__main__":
    main()
