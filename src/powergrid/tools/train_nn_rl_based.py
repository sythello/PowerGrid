from __future__ import annotations

import argparse

from powergrid.ai.nn_rl_based.training import RlTrainingProgress, train_rl_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ai_nn_rl_based_v1 Policy/vector-Q.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-decisions", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="128,64,64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--q-mc-weight", type=float, default=1.0)
    parser.add_argument("--q-search-weight", type=float, default=1.0)
    parser.add_argument(
        "--policy-target-mode",
        choices=("legacy_soft_mix", "advantage_gate"),
        default="legacy_soft_mix",
    )
    parser.add_argument("--search-policy-mix", type=float, default=0.5)
    parser.add_argument("--search-temperature", type=float, default=0.25)
    parser.add_argument("--improved-action-weight", type=float, default=0.75)
    parser.add_argument("--min-search-advantage", type=float, default=0.0)
    parser.add_argument(
        "--training-sampling",
        choices=("all", "balanced_search"),
        default="all",
    )
    args = parser.parse_args(argv)
    hidden_dims = tuple(int(value.strip()) for value in args.hidden_dims.split(","))
    if len(hidden_dims) != 3:
        parser.error("--hidden-dims requires state1,state2,candidate widths")

    def progress(value: RlTrainingProgress) -> None:
        label = (
            f"epoch {value.epoch}/{value.epochs}"
            if value.stage == "epoch"
            else value.stage
        )
        print(
            f"Progress: {label} decisions={value.decisions} "
            f"elapsed={value.elapsed_seconds:.1f}s",
            flush=True,
        )

    summary = train_rl_model(
        args.dataset,
        args.output,
        init_checkpoint=args.init_checkpoint,
        epochs=args.epochs,
        batch_decisions=args.batch_decisions,
        learning_rate=args.learning_rate,
        hidden_dims=hidden_dims,  # type: ignore[arg-type]
        seed=args.seed,
        policy_weight=args.policy_weight,
        q_mc_weight=args.q_mc_weight,
        q_search_weight=args.q_search_weight,
        policy_target_mode=args.policy_target_mode,
        search_policy_mix=args.search_policy_mix,
        search_temperature=args.search_temperature,
        improved_action_weight=args.improved_action_weight,
        min_search_advantage=args.min_search_advantage,
        training_sampling=args.training_sampling,
        progress_callback=progress,
    )
    print("NN RL Training Completed")
    print(f"Checkpoint: {summary.checkpoint_path}")
    print(f"Epochs: {summary.epochs}; elapsed={summary.elapsed_seconds:.1f}s")
    for split, metrics in (
        ("train", summary.final_train_metrics),
        ("validation", summary.final_validation_metrics),
        ("test", summary.final_test_metrics),
    ):
        print(split + ": " + " ".join(f"{key}={value:.6f}" for key, value in sorted(metrics.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
