from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Callable, Iterator

import numpy as np

from .dataset import iter_parquet_batches, load_dataset_metadata
from .model import NumpyRankValueNetwork


TRAINING_COLUMNS = (
    "state_features",
    "action_features",
    "is_winner",
    "rank_value",
)


@dataclass(frozen=True)
class TrainingSummary:
    checkpoint_path: Path
    train_samples: int
    validation_samples: int
    test_samples: int
    epochs: int
    final_train_metrics: dict[str, float]
    final_validation_metrics: dict[str, float]
    final_test_metrics: dict[str, float]
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingProgress:
    stage: str
    epoch: int
    epochs: int
    samples: int
    elapsed_seconds: float


def train_rank_value_model(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dims: tuple[int, int] = (128, 64),
    seed: int = 0,
    scan_batch_size: int = 8192,
    progress_callback: Callable[[TrainingProgress], None] | None = None,
) -> TrainingSummary:
    """Train from pre-split Parquet shards without materializing the dataset in memory."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if scan_batch_size <= 0:
        raise ValueError("scan_batch_size must be positive")
    started = time.perf_counter()
    dataset_root = Path(dataset_path)
    metadata = load_dataset_metadata(dataset_root)
    split_samples = {
        name: int(metadata["splits"][name]["rows"])
        for name in ("train", "validation", "test")
    }
    if split_samples["train"] <= 0:
        raise ValueError("training dataset split contains no samples")
    model = NumpyRankValueNetwork(
        int(metadata["state_dim"]),
        int(metadata["action_dim"]),
        hidden_dims=hidden_dims,
        seed=seed,
        state_feature_names=tuple(metadata["state_feature_names"]),
        action_feature_names=tuple(metadata["action_feature_names"]),
    )
    mean, scale = _stream_normalization(
        dataset_root,
        state_dim=model.state_dim,
        action_dim=model.action_dim,
        batch_size=scan_batch_size,
    )
    model.input_mean = mean
    model.input_scale = scale
    if progress_callback is not None:
        progress_callback(
            TrainingProgress(
                stage="normalization",
                epoch=0,
                epochs=epochs,
                samples=split_samples["train"],
                elapsed_seconds=time.perf_counter() - started,
            )
        )

    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        trained_samples = 0
        for arrays in _iter_array_batches(
            dataset_root,
            "train",
            state_dim=model.state_dim,
            action_dim=model.action_dim,
            batch_size=batch_size,
            shuffle_seed=seed + epoch,
        ):
            order = rng.permutation(len(arrays["wins"]))
            model.train_batch(
                arrays["states"][order],
                arrays["actions"][order],
                arrays["wins"][order],
                arrays["ranks"][order],
                learning_rate=learning_rate,
            )
            trained_samples += len(order)
        if progress_callback is not None:
            progress_callback(
                TrainingProgress(
                    stage="epoch",
                    epoch=epoch + 1,
                    epochs=epochs,
                    samples=trained_samples,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )

    final_train_metrics = _evaluate_split(
        model, dataset_root, "train", batch_size=scan_batch_size
    )
    if progress_callback is not None:
        progress_callback(
            TrainingProgress(
                stage="evaluate_train",
                epoch=epochs,
                epochs=epochs,
                samples=split_samples["train"],
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    final_validation_metrics = _evaluate_split(
        model, dataset_root, "validation", batch_size=scan_batch_size
    )
    if progress_callback is not None:
        progress_callback(
            TrainingProgress(
                stage="evaluate_validation",
                epoch=epochs,
                epochs=epochs,
                samples=split_samples["validation"],
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    final_test_metrics = _evaluate_split(
        model, dataset_root, "test", batch_size=scan_batch_size
    )
    elapsed_seconds = time.perf_counter() - started
    if progress_callback is not None:
        progress_callback(
            TrainingProgress(
                stage="evaluate_test",
                epoch=epochs,
                epochs=epochs,
                samples=split_samples["test"],
                elapsed_seconds=elapsed_seconds,
            )
        )
    manifest_path = dataset_root / "manifest.json"
    output = model.save(
        checkpoint_path,
        metadata={
            "model_name": "ai_nn_rank_value_v1",
            "training_dataset": str(dataset_root),
            "training_dataset_manifest_sha256": _sha256_file(manifest_path),
            "training_epochs": epochs,
            "training_seed": seed,
            "train_samples": split_samples["train"],
            "validation_samples": split_samples["validation"],
            "test_samples": split_samples["test"],
            "final_train_metrics": final_train_metrics,
            "final_validation_metrics": final_validation_metrics,
            "final_test_metrics": final_test_metrics,
            "training_elapsed_seconds": elapsed_seconds,
            "label_definition": {
                "win": "1 iff the acting player is in winner_ids",
                "rank_value": "(player_count + 1 - 2 * final_place) / (player_count - 1)",
            },
        },
    )
    return TrainingSummary(
        checkpoint_path=output,
        train_samples=split_samples["train"],
        validation_samples=split_samples["validation"],
        test_samples=split_samples["test"],
        epochs=epochs,
        final_train_metrics=final_train_metrics,
        final_validation_metrics=final_validation_metrics,
        final_test_metrics=final_test_metrics,
        elapsed_seconds=elapsed_seconds,
    )


def _stream_normalization(
    dataset_path: Path,
    *,
    state_dim: int,
    action_dim: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = state_dim + action_dim
    sums = np.zeros(feature_count, dtype=np.float64)
    square_sums = np.zeros(feature_count, dtype=np.float64)
    rows = 0
    for arrays in _iter_array_batches(
        dataset_path,
        "train",
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=batch_size,
    ):
        combined = np.concatenate([arrays["states"], arrays["actions"]], axis=1)
        sums += combined.sum(axis=0, dtype=np.float64)
        square_sums += np.square(combined, dtype=np.float64).sum(axis=0, dtype=np.float64)
        rows += len(combined)
    if rows <= 0:
        raise ValueError("training dataset split contains no samples")
    mean64 = sums / rows
    variance = np.maximum((square_sums / rows) - np.square(mean64), 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return mean64.astype(np.float32), scale.astype(np.float32)


def _evaluate_split(
    model: NumpyRankValueNetwork,
    dataset_path: Path,
    split: str,
    *,
    batch_size: int,
) -> dict[str, float]:
    totals = {
        "win_loss": 0.0,
        "rank_mse": 0.0,
        "rank_mae": 0.0,
        "win_correct": 0.0,
    }
    rows = 0
    eps = 1e-7
    for arrays in _iter_array_batches(
        dataset_path,
        split,
        state_dim=model.state_dim,
        action_dim=model.action_dim,
        batch_size=batch_size,
    ):
        predictions = model.predict(arrays["states"], arrays["actions"])
        wins = arrays["wins"]
        ranks = arrays["ranks"]
        count = len(wins)
        totals["win_loss"] += float(
            -np.sum(
                wins * np.log(predictions.win_probability + eps)
                + (1.0 - wins) * np.log(1.0 - predictions.win_probability + eps),
                dtype=np.float64,
            )
        )
        rank_error = predictions.rank_value - ranks
        totals["rank_mse"] += float(np.square(rank_error).sum(dtype=np.float64))
        totals["rank_mae"] += float(np.abs(rank_error).sum(dtype=np.float64))
        totals["win_correct"] += float(
            np.sum((predictions.win_probability >= 0.5) == (wins >= 0.5))
        )
        rows += count
    if rows == 0:
        return {}
    win_loss = totals["win_loss"] / rows
    rank_mse = totals["rank_mse"] / rows
    return {
        "loss": win_loss + rank_mse,
        "win_loss": win_loss,
        "rank_mse": rank_mse,
        "win_accuracy": totals["win_correct"] / rows,
        "rank_mae": totals["rank_mae"] / rows,
    }


def _iter_array_batches(
    dataset_path: Path,
    split: str,
    *,
    state_dim: int,
    action_dim: int,
    batch_size: int,
    shuffle_seed: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    for batch in iter_parquet_batches(
        dataset_path,
        split,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        columns=TRAINING_COLUMNS,
    ):
        states = _fixed_size_list_to_numpy(batch.column(0), state_dim)
        actions = _fixed_size_list_to_numpy(batch.column(1), action_dim)
        wins = np.asarray(batch.column(2).to_numpy(zero_copy_only=False), dtype=np.float32)
        ranks = np.asarray(batch.column(3).to_numpy(zero_copy_only=False), dtype=np.float32)
        yield {"states": states, "actions": actions, "wins": wins, "ranks": ranks}


def _fixed_size_list_to_numpy(array: Any, width: int) -> np.ndarray:
    values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    result = values.reshape(len(array), width)
    if result.shape != (len(array), width):
        raise ValueError("dataset feature dimension does not match manifest")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["TrainingProgress", "TrainingSummary", "train_rank_value_model"]
