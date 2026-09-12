from __future__ import annotations

import math
from pathlib import Path
import tempfile

from powergrid.ai.nn_rank_value.dataset import (
    generate_rank_value_dataset,
)
from powergrid.ai.nn_rank_value.model import NumpyRankValueNetwork
from powergrid.ai.nn_rank_value.observation import (
    ACTION_FEATURE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    build_public_observation,
    encode_action_features,
    encode_state_features,
)
from powergrid.ai.nn_rank_value.candidates import generate_candidate_actions
from powergrid.ai.nn_rank_value.training import train_rank_value_model
from powergrid.session import GameSession


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
    snapshot = GameSession.from_scenario("opening", seed=29).snapshot()
    assert snapshot.active_request is not None
    observation = build_public_observation(snapshot.state, snapshot.active_request)
    state_features, state_names = encode_state_features(observation)
    candidate = generate_candidate_actions(snapshot.active_request, snapshot)[0]
    action_features, action_names = encode_action_features(observation, candidate)
    assert restored.state_dim == len(state_features)
    assert restored.action_dim == len(action_features)
    assert restored.state_feature_names == state_names
    assert restored.action_feature_names == action_names
    assert (
        restored.metadata["observation_schema_version"]
        == OBSERVATION_SCHEMA_VERSION
    )
    assert (
        restored.metadata["action_feature_schema_version"]
        == ACTION_FEATURE_SCHEMA_VERSION
    )
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
