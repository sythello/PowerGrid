from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ...model import ModelValidationError
from ...session_types import GameSnapshot, GuiIntent, TurnRequest
from ..base import BaseAiController
from .candidates import generate_candidate_actions
from .observation import (
    build_public_observation,
    encode_action_features,
    encode_state_features,
)

if TYPE_CHECKING:
    from .model import NumpyRankValueNetwork


CONTROLLER_NAME = "ai_nn_rank_value_v1"
CHECKPOINT_ENV_VAR = "POWERGRID_NN_RANK_VALUE_CHECKPOINT"
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ai_models"
    / "ai_nn_rank_value_v1.npz"
)


class NnRankValueAiController(BaseAiController):
    controller = CONTROLLER_NAME

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        win_weight: float = 0.7,
        rank_weight: float = 0.3,
    ) -> None:
        if win_weight < 0 or rank_weight < 0 or win_weight + rank_weight <= 0:
            raise ValueError("NN score weights must be non-negative with a positive sum")
        self._checkpoint_path = Path(
            checkpoint_path
            or os.environ.get(CHECKPOINT_ENV_VAR, "")
            or DEFAULT_CHECKPOINT_PATH
        )
        self._model: "NumpyRankValueNetwork | None" = None
        total = win_weight + rank_weight
        self._win_weight = win_weight / total
        self._rank_weight = rank_weight / total

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on the host environment
            raise ModelValidationError(
                f"{CONTROLLER_NAME} requires NumPy; install requirements-ml.txt"
            ) from exc
        model = self._load_model()
        observation = build_public_observation(snapshot.state, request)
        state_features, state_names = encode_state_features(observation)
        candidates = generate_candidate_actions(request, snapshot)
        action_rows: list[list[float]] = []
        action_names = None
        for candidate in candidates:
            action_features, current_names = encode_action_features(observation, candidate)
            action_rows.append(action_features)
            action_names = action_names or current_names
            if current_names != action_names:
                raise ModelValidationError("NN candidate action schema changed during scoring")
        if model.state_feature_names and tuple(state_names) != model.state_feature_names:
            raise ModelValidationError("NN checkpoint state feature schema does not match runtime")
        if model.action_feature_names and tuple(action_names or ()) != model.action_feature_names:
            raise ModelValidationError("NN checkpoint action feature schema does not match runtime")
        states = np.repeat(
            np.asarray(state_features, dtype=np.float32).reshape(1, -1),
            len(candidates),
            axis=0,
        )
        actions = np.asarray(action_rows, dtype=np.float32)
        predictions = model.predict(states, actions)
        normalized_rank = (predictions.rank_value + 1.0) / 2.0
        scores = (
            self._win_weight * predictions.win_probability
            + self._rank_weight * normalized_rank
        )
        best_index = max(
            range(len(candidates)),
            key=lambda index: (float(scores[index]), -index),
        )
        selected = candidates[best_index]
        ranked = sorted(
            (
                {
                    "candidate_index": index,
                    "intent": candidate.intent.to_dict(),
                    "score": round(float(scores[index]), 6),
                    "win_probability": round(float(predictions.win_probability[index]), 6),
                    "rank_value": round(float(predictions.rank_value[index]), 6),
                }
                for index, candidate in enumerate(candidates)
            ),
            key=lambda item: (-float(item["score"]), int(item["candidate_index"])),
        )
        self.log_state(
            snapshot,
            request,
            label="nn_rank_value_decision",
            state={
                "schema_version": 1,
                "checkpoint": str(self._checkpoint_path),
                "model_metadata": {
                    key: model.metadata[key]
                    for key in (
                        "model_name",
                        "training_dataset",
                        "training_epochs",
                        "train_samples",
                        "validation_samples",
                        "test_samples",
                        "training_dataset_manifest_sha256",
                    )
                    if key in model.metadata
                },
                "candidate_count": len(candidates),
                "score_weights": {
                    "win_probability": self._win_weight,
                    "normalized_rank_value": self._rank_weight,
                },
                "ranked_candidates": ranked,
                "selected_candidate_index": best_index,
                "selected_intent": selected.intent.to_dict(),
            },
            message="NN rank-value AI selected an intent.",
        )
        return selected.intent

    def _load_model(self) -> "NumpyRankValueNetwork":
        if self._model is not None:
            return self._model
        from .model import NumpyRankValueNetwork

        try:
            self._model = NumpyRankValueNetwork.load(self._checkpoint_path)
        except (FileNotFoundError, ValueError) as exc:
            raise ModelValidationError(
                f"cannot load {CONTROLLER_NAME} checkpoint {self._checkpoint_path}: {exc}"
            ) from exc
        return self._model


__all__ = [
    "CHECKPOINT_ENV_VAR",
    "CONTROLLER_NAME",
    "DEFAULT_CHECKPOINT_PATH",
    "NnRankValueAiController",
]
