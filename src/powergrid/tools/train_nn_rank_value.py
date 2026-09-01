from __future__ import annotations

import argparse

from powergrid.ai.nn_rank_value.training import TrainingProgress, train_rank_value_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the NumPy rank-value network used by ai_nn_rank_value_v1."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, help="Output .npz checkpoint path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-dims",
        default="128,64",
        help="Two comma-separated hidden layer widths.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=8192,
        help="Streaming batch size used for normalization and split evaluation.",
    )
    args = parser.parse_args(argv)
    hidden_dims = tuple(int(value.strip()) for value in args.hidden_dims.split(","))
    if len(hidden_dims) != 2:
        parser.error("--hidden-dims must contain exactly two widths")
    def report_progress(progress: TrainingProgress) -> None:
        if progress.stage == "epoch":
            label = f"epoch {progress.epoch}/{progress.epochs}"
        else:
            label = progress.stage
        print(
            f"Progress: {label} samples={progress.samples} "
            f"elapsed={progress.elapsed_seconds:.1f}s",
            flush=True,
        )

    summary = train_rank_value_model(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=hidden_dims,
        seed=args.seed,
        scan_batch_size=args.scan_batch_size,
        progress_callback=report_progress,
    )
    print("NN Rank-Value Training Completed")
    print(f"Checkpoint: {summary.checkpoint_path}")
    print(
        f"Samples: train={summary.train_samples} "
        f"validation={summary.validation_samples} test={summary.test_samples}"
    )
    print(f"Epochs: {summary.epochs}")
    print(f"Elapsed: {summary.elapsed_seconds:.1f}s")
    print("Train metrics: " + _format_metrics(summary.final_train_metrics))
    print("Validation metrics: " + _format_metrics(summary.final_validation_metrics))
    print("Test metrics: " + _format_metrics(summary.final_test_metrics))
    return 0


def _format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))


if __name__ == "__main__":
    raise SystemExit(main())
