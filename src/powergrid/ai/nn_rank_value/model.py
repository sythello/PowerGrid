from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only in environments without the ML extra
    raise RuntimeError(
        "ai_nn_rank_value_v1 requires NumPy; install requirements-ml.txt"
    ) from exc


MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RankValuePredictions:
    win_probability: "np.ndarray"
    rank_value: "np.ndarray"


class NumpyRankValueNetwork:
    """Small two-head MLP for dynamic state/action candidate scoring."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_dims: tuple[int, int] = (128, 64),
        seed: int = 0,
        state_feature_names: tuple[str, ...] = (),
        action_feature_names: tuple[str, ...] = (),
    ) -> None:
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if len(hidden_dims) != 2 or any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain two positive widths")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.state_feature_names = tuple(state_feature_names)
        self.action_feature_names = tuple(action_feature_names)
        input_dim = self.state_dim + self.action_dim
        rng = np.random.default_rng(seed)
        h1, h2 = self.hidden_dims
        self.parameters: dict[str, np.ndarray] = {
            "w1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, h1)).astype(np.float32),
            "b1": np.zeros(h1, dtype=np.float32),
            "w2": rng.normal(0.0, np.sqrt(2.0 / h1), size=(h1, h2)).astype(np.float32),
            "b2": np.zeros(h2, dtype=np.float32),
            "w3": rng.normal(0.0, np.sqrt(1.0 / h2), size=(h2, 2)).astype(np.float32),
            "b3": np.zeros(2, dtype=np.float32),
        }
        self.input_mean = np.zeros(input_dim, dtype=np.float32)
        self.input_scale = np.ones(input_dim, dtype=np.float32)
        self._adam_m = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self._adam_step = 0
        self.metadata: dict[str, Any] = {}

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.action_dim

    def set_normalization(self, combined_features: "np.ndarray") -> None:
        features = np.asarray(combined_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("normalization features have the wrong shape")
        self.input_mean = features.mean(axis=0).astype(np.float32)
        scale = features.std(axis=0).astype(np.float32)
        self.input_scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)

    def predict(
        self,
        state_features: "np.ndarray",
        action_features: "np.ndarray",
    ) -> RankValuePredictions:
        combined = self._combine(state_features, action_features)
        logits, _ = self._forward(combined)
        return RankValuePredictions(
            win_probability=_sigmoid(logits[:, 0]),
            rank_value=np.tanh(logits[:, 1]),
        )

    def train_batch(
        self,
        state_features: "np.ndarray",
        action_features: "np.ndarray",
        win_labels: "np.ndarray",
        rank_labels: "np.ndarray",
        *,
        learning_rate: float,
        win_loss_weight: float = 1.0,
        rank_loss_weight: float = 1.0,
        weight_decay: float = 1e-5,
    ) -> dict[str, float]:
        combined = self._combine(state_features, action_features)
        win_targets = np.asarray(win_labels, dtype=np.float32).reshape(-1)
        rank_targets = np.asarray(rank_labels, dtype=np.float32).reshape(-1)
        if len(combined) != len(win_targets) or len(combined) != len(rank_targets):
            raise ValueError("feature and label batch sizes must match")
        logits, cache = self._forward(combined)
        win_probability = _sigmoid(logits[:, 0])
        rank_value = np.tanh(logits[:, 1])
        eps = 1e-7
        win_loss = -np.mean(
            win_targets * np.log(win_probability + eps)
            + (1.0 - win_targets) * np.log(1.0 - win_probability + eps)
        )
        rank_loss = np.mean((rank_value - rank_targets) ** 2)
        batch_size = max(1, len(combined))
        output_gradient = np.zeros_like(logits)
        output_gradient[:, 0] = (
            win_loss_weight * (win_probability - win_targets) / batch_size
        )
        output_gradient[:, 1] = (
            rank_loss_weight
            * 2.0
            * (rank_value - rank_targets)
            * (1.0 - rank_value**2)
            / batch_size
        )
        normalized, hidden1, hidden2, pre1, pre2 = cache
        gradients: dict[str, np.ndarray] = {}
        gradients["w3"] = hidden2.T @ output_gradient + weight_decay * self.parameters["w3"]
        gradients["b3"] = output_gradient.sum(axis=0)
        hidden2_gradient = output_gradient @ self.parameters["w3"].T
        pre2_gradient = hidden2_gradient * (pre2 > 0)
        gradients["w2"] = hidden1.T @ pre2_gradient + weight_decay * self.parameters["w2"]
        gradients["b2"] = pre2_gradient.sum(axis=0)
        hidden1_gradient = pre2_gradient @ self.parameters["w2"].T
        pre1_gradient = hidden1_gradient * (pre1 > 0)
        gradients["w1"] = normalized.T @ pre1_gradient + weight_decay * self.parameters["w1"]
        gradients["b1"] = pre1_gradient.sum(axis=0)
        self._apply_adam(gradients, learning_rate)
        return {
            "loss": float(win_loss_weight * win_loss + rank_loss_weight * rank_loss),
            "win_loss": float(win_loss),
            "rank_loss": float(rank_loss),
        }

    def evaluate(
        self,
        state_features: "np.ndarray",
        action_features: "np.ndarray",
        win_labels: "np.ndarray",
        rank_labels: "np.ndarray",
    ) -> dict[str, float]:
        predictions = self.predict(state_features, action_features)
        wins = np.asarray(win_labels, dtype=np.float32).reshape(-1)
        ranks = np.asarray(rank_labels, dtype=np.float32).reshape(-1)
        eps = 1e-7
        win_loss = -np.mean(
            wins * np.log(predictions.win_probability + eps)
            + (1.0 - wins) * np.log(1.0 - predictions.win_probability + eps)
        )
        rank_mse = np.mean((predictions.rank_value - ranks) ** 2)
        return {
            "loss": float(win_loss + rank_mse),
            "win_loss": float(win_loss),
            "rank_mse": float(rank_mse),
            "win_accuracy": float(
                np.mean((predictions.win_probability >= 0.5) == (wins >= 0.5))
            ),
            "rank_mae": float(np.mean(np.abs(predictions.rank_value - ranks))),
        }

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata = {
            "format_version": MODEL_FORMAT_VERSION,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "state_feature_names": list(self.state_feature_names),
            "action_feature_names": list(self.action_feature_names),
            **self.metadata,
            **dict(metadata or {}),
        }
        np.savez_compressed(
            output_path,
            **self.parameters,
            input_mean=self.input_mean,
            input_scale=self.input_scale,
            metadata=np.asarray(json.dumps(checkpoint_metadata, sort_keys=True)),
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "NumpyRankValueNetwork":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"NN rank-value checkpoint does not exist: {checkpoint_path}")
        with np.load(checkpoint_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            if int(metadata.get("format_version", -1)) != MODEL_FORMAT_VERSION:
                raise ValueError("unsupported NN rank-value checkpoint format")
            model = cls(
                int(metadata["state_dim"]),
                int(metadata["action_dim"]),
                hidden_dims=tuple(int(value) for value in metadata["hidden_dims"]),
                state_feature_names=tuple(metadata.get("state_feature_names", [])),
                action_feature_names=tuple(metadata.get("action_feature_names", [])),
            )
            for name in model.parameters:
                model.parameters[name] = np.asarray(payload[name], dtype=np.float32)
            model.input_mean = np.asarray(payload["input_mean"], dtype=np.float32)
            model.input_scale = np.asarray(payload["input_scale"], dtype=np.float32)
            model.metadata = dict(metadata)
        return model

    def _combine(
        self,
        state_features: "np.ndarray",
        action_features: "np.ndarray",
    ) -> "np.ndarray":
        states = np.asarray(state_features, dtype=np.float32)
        actions = np.asarray(action_features, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if states.shape[0] != actions.shape[0]:
            raise ValueError("state and action batch sizes must match")
        if states.shape[1] != self.state_dim or actions.shape[1] != self.action_dim:
            raise ValueError("state or action feature dimension does not match checkpoint")
        return np.concatenate([states, actions], axis=1)

    def _forward(
        self,
        combined_features: "np.ndarray",
    ) -> tuple["np.ndarray", tuple["np.ndarray", ...]]:
        normalized = (combined_features - self.input_mean) / self.input_scale
        pre1 = normalized @ self.parameters["w1"] + self.parameters["b1"]
        hidden1 = np.maximum(pre1, 0.0)
        pre2 = hidden1 @ self.parameters["w2"] + self.parameters["b2"]
        hidden2 = np.maximum(pre2, 0.0)
        logits = hidden2 @ self.parameters["w3"] + self.parameters["b3"]
        return logits, (normalized, hidden1, hidden2, pre1, pre2)

    def _apply_adam(
        self,
        gradients: dict[str, "np.ndarray"],
        learning_rate: float,
    ) -> None:
        self._adam_step += 1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        for name, gradient in gradients.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * gradient
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (gradient**2)
            corrected_m = self._adam_m[name] / (1.0 - beta1**self._adam_step)
            corrected_v = self._adam_v[name] / (1.0 - beta2**self._adam_step)
            self.parameters[name] -= learning_rate * corrected_m / (
                np.sqrt(corrected_v) + epsilon
            )


def _sigmoid(values: "np.ndarray") -> "np.ndarray":
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


__all__ = [
    "MODEL_FORMAT_VERSION",
    "NumpyRankValueNetwork",
    "RankValuePredictions",
]
