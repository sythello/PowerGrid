from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "ai_nn_rl_based_v1 requires NumPy; install requirements-ml.txt"
    ) from exc


MODEL_FORMAT_NAME = "powergrid.ai_nn_rl_based"
MODEL_FORMAT_VERSION = 1
MAX_PLAYERS = 6
POLICY_TARGET_MODES = ("legacy_soft_mix", "advantage_gate")


@dataclass(frozen=True)
class PolicyQPredictions:
    policy_logits: "np.ndarray"
    policy_probabilities: "np.ndarray"
    q_values: "np.ndarray"


class NumpyRlPolicyQNetwork:
    """Shared state/candidate MLP with listwise Policy and vector-Q heads."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_dims: tuple[int, int, int] = (128, 64, 64),
        max_players: int = MAX_PLAYERS,
        seed: int = 0,
        state_feature_names: tuple[str, ...] = (),
        action_feature_names: tuple[str, ...] = (),
    ) -> None:
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if len(hidden_dims) != 3 or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain three positive widths")
        if max_players <= 0:
            raise ValueError("max_players must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.max_players = int(max_players)
        self.state_feature_names = tuple(state_feature_names)
        self.action_feature_names = tuple(action_feature_names)
        h1, h2, hc = self.hidden_dims
        rng = np.random.default_rng(seed)
        self.parameters: dict[str, np.ndarray] = {
            "state_w1": _he_matrix(rng, self.state_dim, h1),
            "state_b1": np.zeros(h1, dtype=np.float32),
            "state_w2": _he_matrix(rng, h1, h2),
            "state_b2": np.zeros(h2, dtype=np.float32),
            "candidate_w": _he_matrix(rng, h2 + self.action_dim, hc),
            "candidate_b": np.zeros(hc, dtype=np.float32),
            "policy_w": rng.normal(0.0, np.sqrt(1.0 / hc), size=(hc, 1)).astype(np.float32),
            "policy_b": np.zeros(1, dtype=np.float32),
            "q_w": rng.normal(
                0.0, np.sqrt(1.0 / hc), size=(hc, self.max_players)
            ).astype(np.float32),
            "q_b": np.zeros(self.max_players, dtype=np.float32),
        }
        self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.state_scale = np.ones(self.state_dim, dtype=np.float32)
        self.action_mean = np.zeros(self.action_dim, dtype=np.float32)
        self.action_scale = np.ones(self.action_dim, dtype=np.float32)
        self._adam_m = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self._adam_step = 0
        self.metadata: dict[str, Any] = {}

    def set_normalization(
        self, state_features: "np.ndarray", action_features: "np.ndarray"
    ) -> None:
        states = np.asarray(state_features, dtype=np.float32)
        actions = np.asarray(action_features, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("normalization states have the wrong shape")
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError("normalization actions have the wrong shape")
        self.state_mean, self.state_scale = _mean_scale(states)
        self.action_mean, self.action_scale = _mean_scale(actions)

    def predict(
        self,
        state_features: "np.ndarray",
        action_features: "np.ndarray",
        decision_offsets: "np.ndarray | list[int] | tuple[int, ...]",
    ) -> PolicyQPredictions:
        states, actions, offsets, decision_ids = self._validate_inputs(
            state_features, action_features, decision_offsets
        )
        policy_logits, q_values, _ = self._forward(states, actions, decision_ids)
        return PolicyQPredictions(
            policy_logits=policy_logits,
            policy_probabilities=_segmented_softmax(policy_logits, offsets),
            q_values=q_values,
        )

    def predict_one(
        self, state_features: "np.ndarray", action_features: "np.ndarray"
    ) -> PolicyQPredictions:
        actions = np.asarray(action_features, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        return self.predict(
            np.asarray(state_features, dtype=np.float32).reshape(1, -1),
            actions,
            np.asarray([0, len(actions)], dtype=np.int32),
        )

    def train_batch(
        self,
        state_features: "np.ndarray",
        action_features: "np.ndarray",
        decision_offsets: "np.ndarray",
        teacher_action_indices: "np.ndarray",
        terminal_rank_values: "np.ndarray",
        player_masks: "np.ndarray",
        has_search_targets: "np.ndarray",
        search_q_values: "np.ndarray",
        *,
        learning_rate: float,
        policy_weight: float = 1.0,
        q_mc_weight: float = 1.0,
        q_search_weight: float = 1.0,
        policy_target_mode: str = "legacy_soft_mix",
        search_policy_mix: float = 0.5,
        search_temperature: float = 0.25,
        improved_action_weight: float = 0.75,
        min_search_advantage: float = 0.0,
        huber_delta: float = 1.0,
        weight_decay: float = 1e-5,
    ) -> dict[str, float]:
        states, actions, offsets, decision_ids = self._validate_inputs(
            state_features, action_features, decision_offsets
        )
        labels = self._validate_labels(
            offsets,
            teacher_action_indices,
            terminal_rank_values,
            player_masks,
            has_search_targets,
            search_q_values,
        )
        teacher, terminal, masks, searched, search_q = labels
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        _validate_policy_target_config(
            policy_target_mode=policy_target_mode,
            search_policy_mix=search_policy_mix,
            search_temperature=search_temperature,
            improved_action_weight=improved_action_weight,
            min_search_advantage=min_search_advantage,
        )
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")

        policy_logits, q_values, cache = self._forward(states, actions, decision_ids)
        policy_probs = _segmented_softmax(policy_logits, offsets)
        policy_targets, accepted, improved_indices = build_policy_targets(
            offsets,
            teacher,
            searched,
            search_q,
            actions,
            policy_target_mode=policy_target_mode,
            search_policy_mix=search_policy_mix,
            search_temperature=search_temperature,
            improved_action_weight=improved_action_weight,
            min_search_advantage=min_search_advantage,
        )
        decision_count = len(states)
        eps = 1e-7
        policy_loss = -sum(
            float(np.sum(policy_targets[start:end] * np.log(policy_probs[start:end] + eps)))
            for start, end in zip(offsets[:-1], offsets[1:])
        ) / max(1, decision_count)
        policy_gradient = policy_weight * (policy_probs - policy_targets)
        for start, end in zip(offsets[:-1], offsets[1:]):
            policy_gradient[start:end] /= max(1, decision_count)

        q_gradient = np.zeros_like(q_values)
        mc_loss_total = 0.0
        mc_count = 0
        for decision_index, local_index in enumerate(teacher):
            row = int(offsets[decision_index] + local_index)
            valid = masks[decision_index]
            differences = q_values[row] - terminal[decision_index]
            losses, gradients = _huber(differences, huber_delta)
            mc_loss_total += float(np.sum(losses[valid]))
            mc_count += int(np.sum(valid))
            q_gradient[row, valid] += q_mc_weight * gradients[valid]
        if mc_count:
            q_gradient /= mc_count
        q_mc_loss = mc_loss_total / max(1, mc_count)

        search_loss_total = 0.0
        search_count = 0
        search_gradient = np.zeros_like(q_values)
        for decision_index, is_searched in enumerate(searched):
            if not is_searched:
                continue
            start, end = int(offsets[decision_index]), int(offsets[decision_index + 1])
            valid = np.broadcast_to(masks[decision_index], (end - start, self.max_players))
            differences = q_values[start:end] - search_q[start:end]
            losses, gradients = _huber(differences, huber_delta)
            search_loss_total += float(np.sum(losses[valid]))
            search_count += int(np.sum(valid))
            search_gradient[start:end][valid] = gradients[valid]
        if search_count:
            search_gradient *= q_search_weight / search_count
        q_gradient += search_gradient
        q_search_loss = search_loss_total / max(1, search_count)

        q_pre_gradient = q_gradient * (1.0 - q_values**2)
        normalized_states, state_pre1, state_hidden1, state_pre2, state_hidden2, combined, candidate_pre, candidate_hidden = cache
        gradients: dict[str, np.ndarray] = {}
        policy_column = policy_gradient.reshape(-1, 1)
        gradients["policy_w"] = candidate_hidden.T @ policy_column + weight_decay * self.parameters["policy_w"]
        gradients["policy_b"] = policy_column.sum(axis=0)
        gradients["q_w"] = candidate_hidden.T @ q_pre_gradient + weight_decay * self.parameters["q_w"]
        gradients["q_b"] = q_pre_gradient.sum(axis=0)
        candidate_hidden_gradient = (
            policy_column @ self.parameters["policy_w"].T
            + q_pre_gradient @ self.parameters["q_w"].T
        )
        candidate_pre_gradient = candidate_hidden_gradient * (candidate_pre > 0)
        gradients["candidate_w"] = combined.T @ candidate_pre_gradient + weight_decay * self.parameters["candidate_w"]
        gradients["candidate_b"] = candidate_pre_gradient.sum(axis=0)
        combined_gradient = candidate_pre_gradient @ self.parameters["candidate_w"].T
        state_hidden2_by_candidate = combined_gradient[:, : self.hidden_dims[1]]
        state_hidden2_gradient = np.zeros_like(state_hidden2)
        np.add.at(state_hidden2_gradient, decision_ids, state_hidden2_by_candidate)
        state_pre2_gradient = state_hidden2_gradient * (state_pre2 > 0)
        gradients["state_w2"] = state_hidden1.T @ state_pre2_gradient + weight_decay * self.parameters["state_w2"]
        gradients["state_b2"] = state_pre2_gradient.sum(axis=0)
        state_hidden1_gradient = state_pre2_gradient @ self.parameters["state_w2"].T
        state_pre1_gradient = state_hidden1_gradient * (state_pre1 > 0)
        gradients["state_w1"] = normalized_states.T @ state_pre1_gradient + weight_decay * self.parameters["state_w1"]
        gradients["state_b1"] = state_pre1_gradient.sum(axis=0)
        self._apply_adam(gradients, learning_rate)

        total_loss = (
            policy_weight * policy_loss
            + q_mc_weight * q_mc_loss
            + q_search_weight * q_search_loss
        )
        return {
            "loss": float(total_loss),
            "policy_loss": float(policy_loss),
            "q_mc_loss": float(q_mc_loss),
            "q_search_loss": float(q_search_loss),
            "policy_accuracy": float(_policy_accuracy(policy_probs, offsets, teacher)),
            "searched_decisions": float(np.sum(searched)),
            "accepted_improvement_decisions": float(np.sum(accepted)),
            "accepted_improvement_rate": float(
                np.sum(accepted) / max(1, int(np.sum(searched)))
            ),
            "policy_target_top1_accuracy": float(
                _indexed_policy_accuracy(policy_probs, offsets, _policy_indices(policy_targets, offsets))
            ),
        }

    def evaluate_batch(
        self,
        state_features: "np.ndarray",
        action_features: "np.ndarray",
        decision_offsets: "np.ndarray",
        teacher_action_indices: "np.ndarray",
        terminal_rank_values: "np.ndarray",
        player_masks: "np.ndarray",
        has_search_targets: "np.ndarray",
        search_q_values: "np.ndarray",
        *,
        policy_target_mode: str = "legacy_soft_mix",
        search_policy_mix: float = 0.5,
        search_temperature: float = 0.25,
        improved_action_weight: float = 0.75,
        min_search_advantage: float = 0.0,
    ) -> dict[str, float]:
        states, actions, offsets, _ = self._validate_inputs(
            state_features, action_features, decision_offsets
        )
        teacher, terminal, masks, searched, search_q = self._validate_labels(
            offsets,
            teacher_action_indices,
            terminal_rank_values,
            player_masks,
            has_search_targets,
            search_q_values,
        )
        predictions = self.predict(states, actions, offsets)
        _validate_policy_target_config(
            policy_target_mode=policy_target_mode,
            search_policy_mix=search_policy_mix,
            search_temperature=search_temperature,
            improved_action_weight=improved_action_weight,
            min_search_advantage=min_search_advantage,
        )
        targets, accepted, improved_indices = build_policy_targets(
            offsets,
            teacher,
            searched,
            search_q,
            actions,
            policy_target_mode=policy_target_mode,
            search_policy_mix=search_policy_mix,
            search_temperature=search_temperature,
            improved_action_weight=improved_action_weight,
            min_search_advantage=min_search_advantage,
        )
        eps = 1e-7
        policy_loss = -sum(
            float(np.sum(targets[start:end] * np.log(predictions.policy_probabilities[start:end] + eps)))
            for start, end in zip(offsets[:-1], offsets[1:])
        ) / max(1, len(states))
        mc_errors: list[float] = []
        search_errors: list[float] = []
        for index, local_index in enumerate(teacher):
            row = int(offsets[index] + local_index)
            mc_errors.extend(
                np.abs(predictions.q_values[row][masks[index]] - terminal[index][masks[index]]).tolist()
            )
            if searched[index]:
                start, end = int(offsets[index]), int(offsets[index + 1])
                search_errors.extend(
                    np.abs(
                        predictions.q_values[start:end][:, masks[index]]
                        - search_q[start:end][:, masks[index]]
                    ).reshape(-1).tolist()
                )
        predicted = _policy_indices(predictions.policy_probabilities, offsets)
        fallback = searched & ~accepted
        non_search = ~searched
        target_indices = _policy_indices(targets, offsets)
        return {
            "policy_loss": float(policy_loss),
            "gated_policy_cross_entropy": float(policy_loss),
            "policy_accuracy": float(
                _policy_accuracy(predictions.policy_probabilities, offsets, teacher)
            ),
            "policy_target_top1_accuracy": float(np.mean(predicted == target_indices)),
            "q_mc_mae": float(np.mean(mc_errors)) if mc_errors else 0.0,
            "q_search_mae": float(np.mean(search_errors)) if search_errors else 0.0,
            "q_mc_elements": float(len(mc_errors)),
            "q_search_elements": float(len(search_errors)),
            "decisions": float(len(states)),
            "searched_decisions": float(np.sum(searched)),
            "accepted_improvement_decisions": float(np.sum(accepted)),
            "accepted_improvement_rate": float(
                np.sum(accepted) / max(1, int(np.sum(searched)))
            ),
            "accepted_policy_top1_correct": float(
                np.sum((predicted == improved_indices) & accepted)
            ),
            "accepted_policy_top1_accuracy": float(
                np.sum((predicted == improved_indices) & accepted)
                / max(1, int(np.sum(accepted)))
            ),
            "searched_fallback_decisions": float(np.sum(fallback)),
            "searched_fallback_teacher_correct": float(
                np.sum((predicted == teacher) & fallback)
            ),
            "searched_fallback_teacher_accuracy": float(
                np.sum((predicted == teacher) & fallback)
                / max(1, int(np.sum(fallback)))
            ),
            "non_search_decisions": float(np.sum(non_search)),
            "non_search_teacher_correct": float(
                np.sum((predicted == teacher) & non_search)
            ),
            "non_search_teacher_accuracy": float(
                np.sum((predicted == teacher) & non_search)
                / max(1, int(np.sum(non_search)))
            ),
        }

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata = {
            **self.metadata,
            **dict(metadata or {}),
            "format_name": MODEL_FORMAT_NAME,
            "format_version": MODEL_FORMAT_VERSION,
            "model_name": "ai_nn_rl_based_v1",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "max_players": self.max_players,
            "state_feature_names": list(self.state_feature_names),
            "action_feature_names": list(self.action_feature_names),
        }
        np.savez_compressed(
            output_path,
            **self.parameters,
            state_mean=self.state_mean,
            state_scale=self.state_scale,
            action_mean=self.action_mean,
            action_scale=self.action_scale,
            metadata=np.asarray(json.dumps(checkpoint_metadata, sort_keys=True)),
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "NumpyRlPolicyQNetwork":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"NN RL checkpoint does not exist: {checkpoint_path}")
        with np.load(checkpoint_path, allow_pickle=False) as payload:
            if "metadata" not in payload.files:
                raise ValueError("checkpoint metadata is missing")
            metadata = json.loads(str(payload["metadata"].item()))
            if metadata.get("format_name") != MODEL_FORMAT_NAME:
                raise ValueError("checkpoint is not an ai_nn_rl_based model")
            if int(metadata.get("format_version", -1)) != MODEL_FORMAT_VERSION:
                raise ValueError("unsupported NN RL checkpoint format")
            if metadata.get("model_name") != "ai_nn_rl_based_v1":
                raise ValueError("checkpoint model_name is not ai_nn_rl_based_v1")
            required_metadata = {
                "state_dim",
                "action_dim",
                "hidden_dims",
                "max_players",
            }
            missing_metadata = sorted(required_metadata - set(metadata))
            if missing_metadata:
                raise ValueError(
                    "checkpoint metadata is incomplete: " + ", ".join(missing_metadata)
                )
            model = cls(
                int(metadata["state_dim"]),
                int(metadata["action_dim"]),
                hidden_dims=tuple(int(value) for value in metadata["hidden_dims"]),
                max_players=int(metadata["max_players"]),
                state_feature_names=tuple(metadata.get("state_feature_names", [])),
                action_feature_names=tuple(metadata.get("action_feature_names", [])),
            )
            for name in model.parameters:
                if name not in payload.files:
                    raise ValueError(f"checkpoint parameter is missing: {name}")
                value = np.asarray(payload[name], dtype=np.float32)
                if value.shape != model.parameters[name].shape:
                    raise ValueError(f"checkpoint parameter has the wrong shape: {name}")
                model.parameters[name] = value
            for name, expected in (
                ("state_mean", (model.state_dim,)),
                ("state_scale", (model.state_dim,)),
                ("action_mean", (model.action_dim,)),
                ("action_scale", (model.action_dim,)),
            ):
                if name not in payload.files or payload[name].shape != expected:
                    raise ValueError(f"checkpoint normalization has the wrong shape: {name}")
            model.state_mean = np.asarray(payload["state_mean"], dtype=np.float32)
            model.state_scale = np.asarray(payload["state_scale"], dtype=np.float32)
            model.action_mean = np.asarray(payload["action_mean"], dtype=np.float32)
            model.action_scale = np.asarray(payload["action_scale"], dtype=np.float32)
            model.metadata = dict(metadata)
        return model

    def _validate_inputs(
        self,
        state_features: "np.ndarray",
        action_features: "np.ndarray",
        decision_offsets: "np.ndarray | list[int] | tuple[int, ...]",
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        states = np.asarray(state_features, dtype=np.float32)
        actions = np.asarray(action_features, dtype=np.float32)
        offsets = np.asarray(decision_offsets, dtype=np.int32).reshape(-1)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("state features have the wrong shape")
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError("action features have the wrong shape")
        if len(offsets) != len(states) + 1 or offsets[0] != 0 or offsets[-1] != len(actions):
            raise ValueError("decision offsets do not match states/actions")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("every decision must contain at least one candidate")
        decision_ids = np.repeat(np.arange(len(states), dtype=np.int32), np.diff(offsets))
        return states, actions, offsets, decision_ids

    def _validate_labels(
        self,
        offsets: "np.ndarray",
        teacher_action_indices: "np.ndarray",
        terminal_rank_values: "np.ndarray",
        player_masks: "np.ndarray",
        has_search_targets: "np.ndarray",
        search_q_values: "np.ndarray",
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        decisions = len(offsets) - 1
        teacher = np.asarray(teacher_action_indices, dtype=np.int32).reshape(-1)
        terminal = np.asarray(terminal_rank_values, dtype=np.float32)
        masks = np.asarray(player_masks, dtype=bool)
        searched = np.asarray(has_search_targets, dtype=bool).reshape(-1)
        search_q = np.asarray(search_q_values, dtype=np.float32)
        expected_player_shape = (decisions, self.max_players)
        if len(teacher) != decisions or len(searched) != decisions:
            raise ValueError("decision labels have the wrong length")
        if terminal.shape != expected_player_shape or masks.shape != expected_player_shape:
            raise ValueError("terminal values/player masks have the wrong shape")
        if search_q.shape != (int(offsets[-1]), self.max_players):
            raise ValueError("search Q labels have the wrong shape")
        counts = np.diff(offsets)
        if np.any(teacher < 0) or np.any(teacher >= counts):
            raise ValueError("teacher action index is outside its candidate group")
        return teacher, terminal, masks, searched, search_q

    def _forward(
        self, states: "np.ndarray", actions: "np.ndarray", decision_ids: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray", tuple["np.ndarray", ...]]:
        normalized_states = (states - self.state_mean) / self.state_scale
        normalized_actions = (actions - self.action_mean) / self.action_scale
        state_pre1 = normalized_states @ self.parameters["state_w1"] + self.parameters["state_b1"]
        state_hidden1 = np.maximum(state_pre1, 0.0)
        state_pre2 = state_hidden1 @ self.parameters["state_w2"] + self.parameters["state_b2"]
        state_hidden2 = np.maximum(state_pre2, 0.0)
        combined = np.concatenate([state_hidden2[decision_ids], normalized_actions], axis=1)
        candidate_pre = combined @ self.parameters["candidate_w"] + self.parameters["candidate_b"]
        candidate_hidden = np.maximum(candidate_pre, 0.0)
        policy_logits = (candidate_hidden @ self.parameters["policy_w"] + self.parameters["policy_b"]).reshape(-1)
        q_pre = candidate_hidden @ self.parameters["q_w"] + self.parameters["q_b"]
        q_values = np.tanh(q_pre)
        return policy_logits, q_values, (
            normalized_states,
            state_pre1,
            state_hidden1,
            state_pre2,
            state_hidden2,
            combined,
            candidate_pre,
            candidate_hidden,
        )

    def _apply_adam(self, gradients: dict[str, "np.ndarray"], learning_rate: float) -> None:
        self._adam_step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for name, gradient in gradients.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * gradient
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * gradient**2
            corrected_m = self._adam_m[name] / (1.0 - beta1**self._adam_step)
            corrected_v = self._adam_v[name] / (1.0 - beta2**self._adam_step)
            self.parameters[name] -= learning_rate * corrected_m / (np.sqrt(corrected_v) + epsilon)


def _he_matrix(rng: "np.random.Generator", inputs: int, outputs: int) -> "np.ndarray":
    return rng.normal(0.0, np.sqrt(2.0 / inputs), size=(inputs, outputs)).astype(np.float32)


def _mean_scale(values: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    return mean, np.where(scale < 1e-6, 1.0, scale).astype(np.float32)


def _segmented_softmax(logits: "np.ndarray", offsets: "np.ndarray") -> "np.ndarray":
    probabilities = np.empty_like(logits, dtype=np.float32)
    for start, end in zip(offsets[:-1], offsets[1:]):
        segment = logits[int(start) : int(end)]
        shifted = segment - np.max(segment)
        exponentials = np.exp(np.clip(shifted, -30.0, 30.0))
        probabilities[int(start) : int(end)] = exponentials / np.sum(exponentials)
    return probabilities


def build_policy_targets(
    offsets: "np.ndarray",
    teacher: "np.ndarray",
    searched: "np.ndarray",
    search_q: "np.ndarray",
    action_features: "np.ndarray",
    *,
    policy_target_mode: str = "legacy_soft_mix",
    search_policy_mix: float = 0.5,
    search_temperature: float = 0.25,
    improved_action_weight: float = 0.75,
    min_search_advantage: float = 0.0,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Build listwise Policy targets and report accepted improvement actions."""

    offsets = np.asarray(offsets, dtype=np.int32)
    teacher = np.asarray(teacher, dtype=np.int32)
    searched = np.asarray(searched, dtype=bool)
    search_q = np.asarray(search_q, dtype=np.float32)
    actions = np.asarray(action_features, dtype=np.float32)
    _validate_policy_target_config(
        policy_target_mode=policy_target_mode,
        search_policy_mix=search_policy_mix,
        search_temperature=search_temperature,
        improved_action_weight=improved_action_weight,
        min_search_advantage=min_search_advantage,
    )
    targets = np.zeros(int(offsets[-1]), dtype=np.float32)
    accepted = np.zeros(len(teacher), dtype=bool)
    improved_indices = teacher.copy()
    for index, (start_value, end_value) in enumerate(zip(offsets[:-1], offsets[1:])):
        start, end = int(start_value), int(end_value)
        teacher_index = int(teacher[index])
        teacher_row = start + teacher_index
        if (
            policy_target_mode == "legacy_soft_mix"
            and searched[index]
            and search_policy_mix > 0.0
        ):
            actor_q = search_q[start:end, 0] / search_temperature
            actor_q -= np.max(actor_q)
            soft = np.exp(np.clip(actor_q, -30.0, 30.0))
            soft /= np.sum(soft)
            targets[start:end] = search_policy_mix * soft
            targets[teacher_row] += 1.0 - search_policy_mix
            continue
        if policy_target_mode == "advantage_gate" and searched[index]:
            actor_q = search_q[start:end, 0]
            teacher_features = actions[teacher_row]
            best_index = -1
            best_q = -np.inf
            for local_index, value in enumerate(actor_q):
                if local_index == teacher_index or not np.isfinite(value):
                    continue
                if np.array_equal(actions[start + local_index], teacher_features):
                    continue
                if float(value) > best_q:
                    best_index = local_index
                    best_q = float(value)
            teacher_q = float(actor_q[teacher_index])
            if (
                best_index >= 0
                and np.isfinite(teacher_q)
                and best_q - teacher_q > min_search_advantage
            ):
                targets[teacher_row] = 1.0 - improved_action_weight
                targets[start + best_index] = improved_action_weight
                accepted[index] = True
                improved_indices[index] = best_index
                continue
        targets[teacher_row] = 1.0
    return targets, accepted, improved_indices


def _validate_policy_target_config(
    *,
    policy_target_mode: str,
    search_policy_mix: float,
    search_temperature: float,
    improved_action_weight: float,
    min_search_advantage: float,
) -> None:
    if policy_target_mode not in POLICY_TARGET_MODES:
        raise ValueError(
            "policy_target_mode must be one of " + ", ".join(POLICY_TARGET_MODES)
        )
    if not 0.0 <= search_policy_mix <= 1.0:
        raise ValueError("search_policy_mix must be between 0 and 1")
    if search_temperature <= 0:
        raise ValueError("search_temperature must be positive")
    if not 0.5 < improved_action_weight <= 1.0:
        raise ValueError("improved_action_weight must be in (0.5, 1.0]")
    if min_search_advantage < 0:
        raise ValueError("min_search_advantage may not be negative")


def _huber(differences: "np.ndarray", delta: float) -> tuple["np.ndarray", "np.ndarray"]:
    absolute = np.abs(differences)
    quadratic = absolute <= delta
    losses = np.where(quadratic, 0.5 * differences**2, delta * (absolute - 0.5 * delta))
    gradients = np.where(quadratic, differences, delta * np.sign(differences))
    return losses, gradients


def _policy_accuracy(
    probabilities: "np.ndarray", offsets: "np.ndarray", teacher: "np.ndarray"
) -> float:
    correct = 0
    for index, (start_value, end_value) in enumerate(zip(offsets[:-1], offsets[1:])):
        start, end = int(start_value), int(end_value)
        predicted = int(np.argmax(probabilities[start:end]))
        correct += int(predicted == int(teacher[index]))
    return correct / max(1, len(teacher))


def _policy_indices(probabilities: "np.ndarray", offsets: "np.ndarray") -> "np.ndarray":
    return np.asarray(
        [
            int(np.argmax(probabilities[int(start) : int(end)]))
            for start, end in zip(offsets[:-1], offsets[1:])
        ],
        dtype=np.int32,
    )


def _indexed_policy_accuracy(
    probabilities: "np.ndarray", offsets: "np.ndarray", indices: "np.ndarray"
) -> float:
    return float(np.mean(_policy_indices(probabilities, offsets) == indices))


__all__ = [
    "MAX_PLAYERS",
    "MODEL_FORMAT_NAME",
    "MODEL_FORMAT_VERSION",
    "POLICY_TARGET_MODES",
    "NumpyRlPolicyQNetwork",
    "PolicyQPredictions",
    "build_policy_targets",
]
