from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from powergrid.ai.nn_rank_value.model import NumpyRankValueNetwork


def main() -> None:
    rng = np.random.default_rng(23)
    states = rng.normal(size=(512, 6)).astype(np.float32)
    actions = rng.normal(size=(512, 3)).astype(np.float32)
    signal = states[:, 0] + 0.8 * actions[:, 0] - 0.5 * states[:, 1]
    wins = (signal > 0).astype(np.float32)
    ranks = np.tanh(signal).astype(np.float32)
    model = NumpyRankValueNetwork(6, 3, hidden_dims=(24, 12), seed=23)
    model.set_normalization(np.concatenate([states, actions], axis=1))
    before = model.evaluate(states, actions, wins, ranks)
    for _ in range(120):
        model.train_batch(states, actions, wins, ranks, learning_rate=3e-3)
    after = model.evaluate(states, actions, wins, ranks)
    assert after["loss"] < before["loss"] * 0.25
    assert after["win_accuracy"] > 0.95

    with tempfile.TemporaryDirectory(prefix="powergrid-nn-model-") as directory:
        checkpoint = Path(directory) / "roundtrip.npz"
        expected = model.predict(states[:16], actions[:16])
        model.save(checkpoint, metadata={"validation": True})
        restored = NumpyRankValueNetwork.load(checkpoint)
        actual = restored.predict(states[:16], actions[:16])
        max_delta = max(
            float(np.max(np.abs(expected.win_probability - actual.win_probability))),
            float(np.max(np.abs(expected.rank_value - actual.rank_value))),
        )
    assert max_delta == 0.0

    print("Model validation: PASS")
    print(f"  loss: {before['loss']:.6f} -> {after['loss']:.6f}")
    print(f"  win accuracy: {after['win_accuracy']:.6f}")
    print(f"  checkpoint round-trip max delta: {max_delta:.1f}")


if __name__ == "__main__":
    main()
