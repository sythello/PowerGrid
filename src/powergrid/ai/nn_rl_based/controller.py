from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ...model import ModelValidationError
from ...session_types import GameSnapshot, GuiIntent, TurnRequest
from ..base import BaseAiController
from ..nn_rank_value.candidates import generate_candidate_actions
from ..nn_rank_value.observation import (
    build_public_observation,
    encode_action_features,
    encode_state_features,
    player_slot_ids,
)

if TYPE_CHECKING:
    from .model import NumpyRlPolicyQNetwork


CONTROLLER_NAME = "ai_nn_rl_based_v1"
CHECKPOINT_ENV_VAR = "POWERGRID_NN_RL_BASED_CHECKPOINT"
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ai_models"
    / "ai_nn_rl_based_v1.npz"
)


class NnRlBasedAiController(BaseAiController):
    controller = CONTROLLER_NAME

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self._checkpoint_path = Path(
            checkpoint_path
            or os.environ.get(CHECKPOINT_ENV_VAR, "")
            or DEFAULT_CHECKPOINT_PATH
        )
        self._model: "NumpyRlPolicyQNetwork | None" = None

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on host environment
            raise ModelValidationError(
                f"{CONTROLLER_NAME} requires NumPy; install requirements-ml.txt"
            ) from exc
        state = snapshot.state
        if state.game_map.id != "germany" or len(state.players) != 3:
            raise ModelValidationError(
                f"{CONTROLLER_NAME} v1 supports only 3-player Germany games"
            )
        model = self._load_model()
        observation = build_public_observation(state, request)
        state_features, state_names = encode_state_features(observation)
        candidates = generate_candidate_actions(request, snapshot)
        action_rows: list[list[float]] = []
        action_names: tuple[str, ...] | None = None
        for candidate in candidates:
            features, current_names = encode_action_features(observation, candidate)
            action_rows.append(features)
            action_names = action_names or current_names
            if current_names != action_names:
                raise ModelValidationError("RL action feature schema changed during scoring")
        if model.state_feature_names != state_names:
            raise ModelValidationError("RL checkpoint state feature schema mismatch")
        if model.action_feature_names != tuple(action_names or ()):
            raise ModelValidationError("RL checkpoint action feature schema mismatch")
        if model.max_players != 6:
            raise ModelValidationError("RL checkpoint max-player dimension is incompatible")
        if model.metadata.get("supported_map") != "germany" or int(
            model.metadata.get("supported_player_count", -1)
        ) != 3:
            raise ModelValidationError("RL checkpoint support metadata is incompatible")
        predictions = model.predict_one(
            np.asarray(state_features, dtype=np.float32),
            np.asarray(action_rows, dtype=np.float32),
        )
        best_index = max(
            range(len(candidates)),
            key=lambda index: (float(predictions.policy_logits[index]), -index),
        )
        slot_ids = player_slot_ids(observation)
        ranked = sorted(
            (
                {
                    "candidate_index": index,
                    "intent": candidate.intent.to_dict(),
                    "policy_probability": round(
                        float(predictions.policy_probabilities[index]), 6
                    ),
                    "actor_q": round(float(predictions.q_values[index, 0]), 6),
                    "q_by_player": {
                        player_id: round(float(predictions.q_values[index, slot]), 6)
                        for slot, player_id in enumerate(slot_ids)
                    },
                }
                for index, candidate in enumerate(candidates)
            ),
            key=lambda item: (
                -float(item["policy_probability"]),
                int(item["candidate_index"]),
            ),
        )
        selected = candidates[best_index]
        self.log_state(
            snapshot,
            request,
            label="nn_rl_based_decision",
            state={
                "schema_version": 1,
                "checkpoint": str(self._checkpoint_path),
                "model_metadata": {
                    key: model.metadata[key]
                    for key in (
                        "model_name",
                        "training_iteration",
                        "training_dataset",
                        "training_dataset_manifest_sha256",
                        "training_epochs",
                    )
                    if key in model.metadata
                },
                "player_slots": list(slot_ids),
                "candidate_count": len(candidates),
                "ranked_candidates": ranked,
                "selected_candidate_index": best_index,
                "selected_intent": selected.intent.to_dict(),
                "selection_source": "policy_logit",
            },
            message="NN RL Policy selected an intent.",
        )
        return selected.intent

    def _load_model(self) -> "NumpyRlPolicyQNetwork":
        if self._model is not None:
            return self._model
        from .model import NumpyRlPolicyQNetwork

        try:
            self._model = NumpyRlPolicyQNetwork.load(self._checkpoint_path)
        except (FileNotFoundError, OSError, ValueError, EOFError) as exc:
            raise ModelValidationError(
                f"cannot load {CONTROLLER_NAME} checkpoint {self._checkpoint_path}: {exc}"
            ) from exc
        return self._model


__all__ = [
    "CHECKPOINT_ENV_VAR",
    "CONTROLLER_NAME",
    "DEFAULT_CHECKPOINT_PATH",
    "NnRlBasedAiController",
]
