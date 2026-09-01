from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Iterator

import numpy as np

from ..nn_rank_value.dataset import sha256_file
from .dataset import iter_rl_parquet_batches, load_rl_dataset_metadata
from .model import NumpyRlPolicyQNetwork


TRAINING_COLUMNS = (
    "state_features",
    "candidate_action_features",
    "teacher_action_index",
    "terminal_rank_values",
    "player_mask",
    "has_search_targets",
    "search_q_values",
)
TRAINING_SAMPLING_MODES = ("all", "balanced_search")


@dataclass(frozen=True)
class RlTrainingProgress:
    stage: str
    epoch: int
    epochs: int
    decisions: int
    elapsed_seconds: float


@dataclass(frozen=True)
class RlTrainingSummary:
    checkpoint_path: Path
    epochs: int
    train_decisions: int
    validation_decisions: int
    test_decisions: int
    final_train_metrics: dict[str, float]
    final_validation_metrics: dict[str, float]
    final_test_metrics: dict[str, float]
    elapsed_seconds: float


def train_rl_model(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    init_checkpoint: str | Path | None = None,
    epochs: int = 20,
    batch_decisions: int = 128,
    learning_rate: float = 1e-3,
    hidden_dims: tuple[int, int, int] = (128, 64, 64),
    seed: int = 0,
    policy_weight: float = 1.0,
    q_mc_weight: float = 1.0,
    q_search_weight: float = 1.0,
    policy_target_mode: str = "legacy_soft_mix",
    search_policy_mix: float = 0.5,
    search_temperature: float = 0.25,
    improved_action_weight: float = 0.75,
    min_search_advantage: float = 0.0,
    training_sampling: str = "all",
    progress_callback: Callable[[RlTrainingProgress], None] | None = None,
) -> RlTrainingSummary:
    if epochs <= 0 or batch_decisions <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_decisions, and learning_rate must be positive")
    if any(weight < 0 for weight in (policy_weight, q_mc_weight, q_search_weight)):
        raise ValueError("loss weights may not be negative")
    if training_sampling not in TRAINING_SAMPLING_MODES:
        raise ValueError(
            "training_sampling must be one of " + ", ".join(TRAINING_SAMPLING_MODES)
        )
    if training_sampling == "balanced_search":
        if init_checkpoint is None:
            raise ValueError("balanced_search requires an initial Stage-0 checkpoint")
        if batch_decisions % 2:
            raise ValueError("balanced_search requires an even batch_decisions value")
    started = time.perf_counter()
    root = Path(dataset_path)
    metadata = load_rl_dataset_metadata(root)
    split_counts = {
        split: int(metadata["splits"][split]["rows"])
        for split in ("train", "validation", "test")
    }
    if split_counts["train"] <= 0:
        raise ValueError("RL training split contains no decisions")
    train_searched, train_non_search = _count_search_rows(root, "train")
    if training_sampling == "balanced_search":
        if train_searched <= 0:
            raise ValueError("balanced_search requires searched training decisions")
        if train_non_search < train_searched:
            raise ValueError(
                "balanced_search requires at least as many non-search as searched decisions"
            )
    if init_checkpoint is None:
        model = NumpyRlPolicyQNetwork(
            int(metadata["state_dim"]),
            int(metadata["action_dim"]),
            hidden_dims=hidden_dims,
            seed=seed,
            state_feature_names=tuple(metadata["state_feature_names"]),
            action_feature_names=tuple(metadata["action_feature_names"]),
        )
        state_mean, state_scale, action_mean, action_scale = _stream_normalization(
            root, batch_decisions=batch_decisions
        )
        model.state_mean = state_mean
        model.state_scale = state_scale
        model.action_mean = action_mean
        model.action_scale = action_scale
    else:
        model = NumpyRlPolicyQNetwork.load(init_checkpoint)
        _validate_checkpoint_schema(model, metadata)
    if progress_callback is not None:
        progress_callback(
            RlTrainingProgress(
                stage="normalization",
                epoch=0,
                epochs=epochs,
                decisions=split_counts["train"],
                elapsed_seconds=time.perf_counter() - started,
            )
        )

    epoch_sampling_counts: list[dict[str, int]] = []
    for epoch in range(epochs):
        trained = 0
        trained_searched = 0
        trained_accepted = 0
        for arrays in _iter_array_batches(
            root,
            "train",
            batch_decisions=batch_decisions,
            shuffle_seed=seed + epoch,
            training_sampling=training_sampling,
            searched_count=train_searched,
            non_search_count=train_non_search,
        ):
            result = model.train_batch(
                arrays["states"],
                arrays["actions"],
                arrays["offsets"],
                arrays["teacher"],
                arrays["terminal"],
                arrays["player_masks"],
                arrays["searched"],
                arrays["search_q"],
                learning_rate=learning_rate,
                policy_weight=policy_weight,
                q_mc_weight=q_mc_weight,
                q_search_weight=q_search_weight,
                policy_target_mode=policy_target_mode,
                search_policy_mix=search_policy_mix,
                search_temperature=search_temperature,
                improved_action_weight=improved_action_weight,
                min_search_advantage=min_search_advantage,
            )
            trained += len(arrays["states"])
            trained_searched += int(np.sum(arrays["searched"]))
            trained_accepted += int(result["accepted_improvement_decisions"])
        epoch_sampling_counts.append(
            {
                "epoch": epoch + 1,
                "decisions": trained,
                "searched": trained_searched,
                "non_search": trained - trained_searched,
                "accepted_improvements": trained_accepted,
            }
        )
        if progress_callback is not None:
            progress_callback(
                RlTrainingProgress(
                    stage="epoch",
                    epoch=epoch + 1,
                    epochs=epochs,
                    decisions=trained,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )

    metrics = {}
    for split in ("train", "validation", "test"):
        metrics[split] = _evaluate_split(
            model,
            root,
            split,
            batch_decisions=batch_decisions,
            policy_target_mode=policy_target_mode,
            search_policy_mix=search_policy_mix,
            search_temperature=search_temperature,
            improved_action_weight=improved_action_weight,
            min_search_advantage=min_search_advantage,
        )
        if progress_callback is not None:
            progress_callback(
                RlTrainingProgress(
                    stage=f"evaluate_{split}",
                    epoch=epochs,
                    epochs=epochs,
                    decisions=split_counts[split],
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
    elapsed = time.perf_counter() - started
    output = model.save(
        checkpoint_path,
        metadata={
            "supported_map": "germany",
            "supported_player_count": 3,
            "training_dataset": str(root),
            "training_dataset_manifest_sha256": sha256_file(root / "manifest.json"),
            "target_checkpoint_sha256": metadata["generation"].get(
                "target_checkpoint_sha256", ""
            ),
            "training_iteration": 1 if init_checkpoint is not None else 0,
            "training_seed": seed,
            "training_epochs": epochs,
            "training_batch_decisions": batch_decisions,
            "learning_rate": learning_rate,
            "loss_weights": {
                "policy": policy_weight,
                "q_mc": q_mc_weight,
                "q_search": q_search_weight,
            },
            "search_policy_mix": search_policy_mix,
            "search_temperature": search_temperature,
            "policy_target_mode": policy_target_mode,
            "improved_action_weight": improved_action_weight,
            "min_search_advantage": min_search_advantage,
            "training_sampling": training_sampling,
            "training_sampling_source_counts": {
                "searched": train_searched,
                "non_search": train_non_search,
            },
            "training_sampling_epoch_counts": epoch_sampling_counts,
            "source_search_configuration": {
                key: metadata["generation"].get(key)
                for key in (
                    "search_fraction",
                    "search_depth",
                    "adaptive_depth_2",
                    "max_search_nodes",
                    "max_boundary_actions",
                    "leaf_policy",
                    "continuation_controller",
                    "hidden_state_sampling",
                )
            },
            "label_definition": {
                "rank": "(player_count + 1 - 2 * final_place) / (player_count - 1)",
                "q_mc": "terminal rank vector for the behavior action",
                "q_search": "full-action frozen-Q semantic-search target",
                "discount": 1.0,
            },
            "split_decisions": split_counts,
            "final_train_metrics": metrics["train"],
            "final_validation_metrics": metrics["validation"],
            "final_test_metrics": metrics["test"],
            "training_elapsed_seconds": elapsed,
        },
    )
    return RlTrainingSummary(
        checkpoint_path=output,
        epochs=epochs,
        train_decisions=split_counts["train"],
        validation_decisions=split_counts["validation"],
        test_decisions=split_counts["test"],
        final_train_metrics=metrics["train"],
        final_validation_metrics=metrics["validation"],
        final_test_metrics=metrics["test"],
        elapsed_seconds=elapsed,
    )


def _iter_array_batches(
    root: Path,
    split: str,
    *,
    batch_decisions: int,
    shuffle_seed: int | None = None,
    training_sampling: str = "all",
    searched_count: int | None = None,
    non_search_count: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    if training_sampling not in TRAINING_SAMPLING_MODES:
        raise ValueError("unsupported training sampling mode")
    if training_sampling == "balanced_search":
        if split != "train":
            raise ValueError("balanced_search may only be used for the training split")
        if batch_decisions % 2:
            raise ValueError("balanced_search requires an even batch_decisions value")
        if searched_count is None or non_search_count is None:
            searched_count, non_search_count = _count_search_rows(root, split)
        yield from _iter_balanced_search_batches(
            root,
            split,
            batch_decisions=batch_decisions,
            shuffle_seed=shuffle_seed,
            searched_count=searched_count,
            non_search_count=non_search_count,
        )
        return
    for batch in iter_rl_parquet_batches(
        root,
        split,
        batch_size=batch_decisions,
        shuffle_seed=shuffle_seed,
        columns=TRAINING_COLUMNS,
    ):
        rows = batch.to_pylist()
        if not rows:
            continue
        yield _rows_to_arrays(rows)


def _iter_balanced_search_batches(
    root: Path,
    split: str,
    *,
    batch_decisions: int,
    shuffle_seed: int | None,
    searched_count: int,
    non_search_count: int,
) -> Iterator[dict[str, np.ndarray]]:
    if searched_count <= 0 or non_search_count < searched_count:
        raise ValueError("balanced_search source counts are invalid")
    half_batch = batch_decisions // 2
    seed = int(shuffle_seed or 0)
    rng = np.random.default_rng(seed)
    offset = float(rng.random())
    next_anchor = int(np.floor(offset * non_search_count / searched_count))
    anchor_number = 0
    non_search_index = 0
    search_buffer: list[dict[str, Any]] = []
    anchor_buffer: list[dict[str, Any]] = []

    def emit(count: int) -> dict[str, np.ndarray]:
        selected = search_buffer[:count] + anchor_buffer[:count]
        del search_buffer[:count]
        del anchor_buffer[:count]
        order = rng.permutation(len(selected))
        return _rows_to_arrays([selected[int(index)] for index in order])

    source_batch_size = max(batch_decisions * 4, 512)
    for batch in iter_rl_parquet_batches(
        root,
        split,
        batch_size=source_batch_size,
        shuffle_seed=shuffle_seed,
        columns=TRAINING_COLUMNS,
    ):
        search_flags = np.asarray(
            batch.column(TRAINING_COLUMNS.index("has_search_targets")).to_numpy(
                zero_copy_only=False
            ),
            dtype=bool,
        )
        selected_indices: list[int] = []
        for row_index, is_searched in enumerate(search_flags):
            if is_searched:
                selected_indices.append(row_index)
            else:
                if anchor_number < searched_count and non_search_index == next_anchor:
                    selected_indices.append(row_index)
                    anchor_number += 1
                    if anchor_number < searched_count:
                        next_anchor = int(
                            np.floor(
                                (anchor_number + offset)
                                * non_search_count
                                / searched_count
                            )
                        )
                non_search_index += 1
        if not selected_indices:
            continue
        for row in batch.take(selected_indices).to_pylist():
            if row["has_search_targets"]:
                search_buffer.append(row)
            else:
                anchor_buffer.append(row)
            while len(search_buffer) >= half_batch and len(anchor_buffer) >= half_batch:
                yield emit(half_batch)
    if non_search_index != non_search_count:
        raise ValueError("balanced_search non-search count changed while streaming")
    if anchor_number != searched_count or len(search_buffer) != len(anchor_buffer):
        raise ValueError("balanced_search did not retain a one-to-one sample")
    while search_buffer:
        count = min(half_batch, len(search_buffer))
        yield emit(count)


def _rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    actions: list[list[float]] = []
    search_q: list[list[float]] = []
    offsets = [0]
    for row in rows:
        candidate_actions = row["candidate_action_features"]
        actions.extend(candidate_actions)
        offsets.append(len(actions))
        if row["has_search_targets"]:
            if len(row["search_q_values"]) != len(candidate_actions):
                raise ValueError("searched decision does not label every candidate")
            search_q.extend(row["search_q_values"])
        else:
            search_q.extend([[0.0] * 6 for _ in candidate_actions])
    return {
        "states": np.asarray([row["state_features"] for row in rows], dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "offsets": np.asarray(offsets, dtype=np.int32),
        "teacher": np.asarray(
            [row["teacher_action_index"] for row in rows], dtype=np.int32
        ),
        "terminal": np.asarray(
            [row["terminal_rank_values"] for row in rows], dtype=np.float32
        ),
        "player_masks": np.asarray([row["player_mask"] for row in rows], dtype=bool),
        "searched": np.asarray(
            [row["has_search_targets"] for row in rows], dtype=bool
        ),
        "search_q": np.asarray(search_q, dtype=np.float32),
    }


def _count_search_rows(root: Path, split: str) -> tuple[int, int]:
    searched = 0
    decisions = 0
    for batch in iter_rl_parquet_batches(
        root,
        split,
        batch_size=8192,
        columns=("has_search_targets",),
    ):
        values = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=bool)
        searched += int(np.sum(values))
        decisions += len(values)
    return searched, decisions - searched


def _stream_normalization(
    root: Path, *, batch_decisions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_sum = state_square = action_sum = action_square = None
    state_count = 0
    action_count = 0
    for arrays in _iter_array_batches(
        root, "train", batch_decisions=batch_decisions
    ):
        states = arrays["states"].astype(np.float64)
        actions = arrays["actions"].astype(np.float64)
        if state_sum is None:
            state_sum = np.zeros(states.shape[1], dtype=np.float64)
            state_square = np.zeros(states.shape[1], dtype=np.float64)
            action_sum = np.zeros(actions.shape[1], dtype=np.float64)
            action_square = np.zeros(actions.shape[1], dtype=np.float64)
        state_sum += states.sum(axis=0)
        state_square += np.square(states).sum(axis=0)
        action_sum += actions.sum(axis=0)
        action_square += np.square(actions).sum(axis=0)
        state_count += len(states)
        action_count += len(actions)
    if state_sum is None or state_square is None or action_sum is None or action_square is None:
        raise ValueError("RL train split contains no features")
    state_mean, state_scale = _moments(state_sum, state_square, state_count)
    action_mean, action_scale = _moments(action_sum, action_square, action_count)
    return state_mean, state_scale, action_mean, action_scale


def _moments(
    sums: np.ndarray, squares: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    mean = sums / max(1, count)
    variance = np.maximum(squares / max(1, count) - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    return mean.astype(np.float32), np.where(scale < 1e-6, 1.0, scale).astype(np.float32)


def _evaluate_split(
    model: NumpyRlPolicyQNetwork,
    root: Path,
    split: str,
    *,
    batch_decisions: int,
    policy_target_mode: str,
    search_policy_mix: float,
    search_temperature: float,
    improved_action_weight: float,
    min_search_advantage: float,
) -> dict[str, float]:
    totals = {
        "policy_loss": 0.0,
        "policy_target_top1_correct": 0.0,
        "policy_accuracy": 0.0,
        "q_mc_mae": 0.0,
        "q_search_mae": 0.0,
        "accepted_policy_top1_correct": 0.0,
        "searched_fallback_teacher_correct": 0.0,
        "non_search_teacher_correct": 0.0,
    }
    decisions = 0
    searched = 0
    q_mc_elements = 0
    q_search_elements = 0
    accepted = 0
    fallback = 0
    non_search = 0
    for arrays in _iter_array_batches(
        root, split, batch_decisions=batch_decisions
    ):
        result = model.evaluate_batch(
            arrays["states"],
            arrays["actions"],
            arrays["offsets"],
            arrays["teacher"],
            arrays["terminal"],
            arrays["player_masks"],
            arrays["searched"],
            arrays["search_q"],
            policy_target_mode=policy_target_mode,
            search_policy_mix=search_policy_mix,
            search_temperature=search_temperature,
            improved_action_weight=improved_action_weight,
            min_search_advantage=min_search_advantage,
        )
        count = len(arrays["states"])
        searched_count = int(np.sum(arrays["searched"]))
        totals["policy_loss"] += result["policy_loss"] * count
        totals["policy_accuracy"] += result["policy_accuracy"] * count
        totals["policy_target_top1_correct"] += (
            result["policy_target_top1_accuracy"] * count
        )
        batch_q_mc_elements = int(result["q_mc_elements"])
        batch_q_search_elements = int(result["q_search_elements"])
        totals["q_mc_mae"] += result["q_mc_mae"] * batch_q_mc_elements
        totals["q_search_mae"] += (
            result["q_search_mae"] * batch_q_search_elements
        )
        totals["accepted_policy_top1_correct"] += result[
            "accepted_policy_top1_correct"
        ]
        totals["searched_fallback_teacher_correct"] += result[
            "searched_fallback_teacher_correct"
        ]
        totals["non_search_teacher_correct"] += result[
            "non_search_teacher_correct"
        ]
        decisions += count
        searched += searched_count
        q_mc_elements += batch_q_mc_elements
        q_search_elements += batch_q_search_elements
        accepted += int(result["accepted_improvement_decisions"])
        fallback += int(result["searched_fallback_decisions"])
        non_search += int(result["non_search_decisions"])
    return {
        "policy_loss": totals["policy_loss"] / max(1, decisions),
        "gated_policy_cross_entropy": totals["policy_loss"] / max(1, decisions),
        "policy_accuracy": totals["policy_accuracy"] / max(1, decisions),
        "policy_target_top1_accuracy": totals["policy_target_top1_correct"]
        / max(1, decisions),
        "q_mc_mae": totals["q_mc_mae"] / max(1, q_mc_elements),
        "q_search_mae": totals["q_search_mae"] / max(1, q_search_elements),
        "q_mc_elements": float(q_mc_elements),
        "q_search_elements": float(q_search_elements),
        "decisions": float(decisions),
        "searched_decisions": float(searched),
        "accepted_improvement_decisions": float(accepted),
        "accepted_improvement_rate": accepted / max(1, searched),
        "accepted_policy_top1_accuracy": totals["accepted_policy_top1_correct"]
        / max(1, accepted),
        "searched_fallback_decisions": float(fallback),
        "searched_fallback_teacher_accuracy": totals[
            "searched_fallback_teacher_correct"
        ]
        / max(1, fallback),
        "non_search_decisions": float(non_search),
        "non_search_teacher_accuracy": totals["non_search_teacher_correct"]
        / max(1, non_search),
    }


def _validate_checkpoint_schema(
    model: NumpyRlPolicyQNetwork, metadata: dict[str, Any]
) -> None:
    if model.max_players != 6:
        raise ValueError("initial checkpoint max-player dimension does not match v1")
    if model.metadata.get("supported_map") != "germany" or int(
        model.metadata.get("supported_player_count", -1)
    ) != 3:
        raise ValueError("initial checkpoint support metadata does not match v1")
    if model.state_dim != int(metadata["state_dim"]) or model.action_dim != int(
        metadata["action_dim"]
    ):
        raise ValueError("initial checkpoint dimensions do not match dataset")
    if model.state_feature_names != tuple(metadata["state_feature_names"]):
        raise ValueError("initial checkpoint state schema does not match dataset")
    if model.action_feature_names != tuple(metadata["action_feature_names"]):
        raise ValueError("initial checkpoint action schema does not match dataset")


__all__ = [
    "TRAINING_SAMPLING_MODES",
    "RlTrainingProgress",
    "RlTrainingSummary",
    "train_rl_model",
]
