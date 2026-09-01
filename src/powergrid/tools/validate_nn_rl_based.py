from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import tempfile
import time
import tracemalloc
from typing import Any

import numpy as np

from powergrid.ai import (
    AiEvaluationBucketConfig,
    build_ai_controller,
    evaluate_ai_bucket,
)
from powergrid.ai.nn_rank_value.candidates import generate_candidate_actions
from powergrid.ai.nn_rank_value.observation import (
    build_public_observation,
    encode_action_features,
    encode_state_features,
)
from powergrid.ai.nn_rl_based.controller import NnRlBasedAiController
from powergrid.ai.nn_rl_based.dataset import (
    generate_rl_dataset,
    iter_rl_parquet_batches,
    load_rl_dataset_metadata,
    load_rl_dataset_records,
    verify_rl_dataset_manifest,
)
from powergrid.ai.nn_rl_based.model import (
    NumpyRlPolicyQNetwork,
    build_policy_targets,
)
from powergrid.ai.nn_rl_based.search import (
    FullActionSemanticSearcher,
    SearchConfig,
    terminal_rank_values,
)
from powergrid.ai.nn_rl_based.training import train_rl_model
from powergrid.model import GameConfig, ModelValidationError, SeatConfig
from powergrid.session import GameSession


SECTIONS = ("model", "search", "dataset", "training", "controller", "benchmark", "strength")
GATED_REFERENCE_SEARCH_Q_MAE = 0.1470
GATED_MAX_SEARCH_Q_MAE_REGRESSION = 0.05


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ai_nn_rl_based_v1 components.")
    parser.add_argument("--section", choices=("all", *SECTIONS), default="all")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--bootstrap-checkpoint")
    parser.add_argument("--dataset", help="Optional bootstrap dataset for release metrics.")
    parser.add_argument("--search-dataset", help="Optional searched dataset for release metrics.")
    parser.add_argument("--calibration-roots", type=int, default=0)
    parser.add_argument("--enforce-acceptance", action="store_true")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--strength-games-per-lineup", type=int, default=0)
    args = parser.parse_args(argv)
    selected = SECTIONS if args.section == "all" else (args.section,)
    report: dict[str, Any] = {
        "model_name": "ai_nn_rl_based_v1",
        "requested_section": args.section,
        "sections": {},
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bootstrap_checkpoint: Path | None = None
        final_checkpoint: Path | None = None
        bootstrap_dataset: Path | None = None
        external_acceptance_requested = any(
            (args.dataset, args.search_dataset, args.bootstrap_checkpoint)
        )
        if "model" in selected:
            report["sections"]["model"] = _validate_model()
        if "search" in selected:
            report["sections"]["search"] = _validate_search()
        needs_pipeline = any(
            section in selected for section in ("dataset", "controller", "benchmark")
        ) or ("training" in selected and not external_acceptance_requested)
        if needs_pipeline:
            bootstrap_dataset = root / "bootstrap"
            bootstrap_checkpoint = root / "bootstrap.npz"
            tracemalloc.start()
            started = time.perf_counter()
            summary = generate_rl_dataset(
                bootstrap_dataset,
                games=args.games,
                seed_start=9101,
                search_fraction=0.0,
                split_fractions=(1.0, 0.0, 0.0),
                target_shard_size_bytes=64 * 1024,
            )
            generation_seconds = time.perf_counter() - started
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            records = load_rl_dataset_records(bootstrap_dataset, split="train")
            if "dataset" in selected:
                report["sections"]["dataset"] = {
                    "status": "PASS",
                    "games": summary.games,
                    "decisions": summary.decisions,
                    "examples": len(summary.example_jsonl_paths),
                    "manifest": verify_rl_dataset_manifest(bootstrap_dataset),
                    "all_teacher_indices_valid": all(
                        0 <= int(row["teacher_action_index"])
                        < len(row["candidate_action_features"])
                        for row in records
                    ),
                    "all_player_vectors_six": all(
                        len(row["terminal_rank_values"]) == 6 for row in records
                    ),
                }
            training_started = time.perf_counter()
            training = train_rl_model(
                bootstrap_dataset,
                bootstrap_checkpoint,
                epochs=2,
                batch_decisions=128,
                hidden_dims=(32, 16, 16),
                q_search_weight=0.0,
            )
            training_seconds = time.perf_counter() - training_started
            search_dataset = root / "search"
            search_started = time.perf_counter()
            search_summary = generate_rl_dataset(
                search_dataset,
                games=max(1, min(args.games, 2)),
                seed_start=9201,
                target_checkpoint=bootstrap_checkpoint,
                search_fraction=0.05,
                split_fractions=(1.0, 0.0, 0.0),
                target_shard_size_bytes=64 * 1024,
            )
            search_generation_seconds = time.perf_counter() - search_started
            final_checkpoint = root / "search.npz"
            search_training_started = time.perf_counter()
            search_training = train_rl_model(
                search_dataset,
                final_checkpoint,
                init_checkpoint=bootstrap_checkpoint,
                epochs=1,
                batch_decisions=128,
            )
            search_training_seconds = time.perf_counter() - search_training_started
            if "training" in selected:
                report["sections"]["training"] = {
                    "status": "PASS",
                    "bootstrap_checkpoint": str(bootstrap_checkpoint),
                    "search_checkpoint": str(final_checkpoint),
                    "bootstrap_epochs": training.epochs,
                    "bootstrap_metrics": training.final_train_metrics,
                    "search_metrics": search_training.final_train_metrics,
                    "elapsed_seconds": training_seconds + search_training_seconds,
                }
            if "controller" in selected:
                report["sections"]["controller"] = _validate_controller(
                    final_checkpoint
                )
            if "benchmark" in selected:
                report["sections"]["benchmark"] = _benchmark(
                    records,
                    final_checkpoint,
                    games=summary.games,
                    generation_seconds=generation_seconds,
                    training_seconds=training_seconds,
                    peak_memory_bytes=peak_memory,
                    parquet_bytes=summary.parquet_bytes,
                    search_decisions=search_summary.searched_decisions,
                    search_nodes=search_summary.search_nodes,
                    search_depth_2_completed=search_summary.depth_2_completed,
                    search_games=search_summary.games,
                    search_decision_total=search_summary.decisions,
                    search_parquet_bytes=search_summary.parquet_bytes,
                    search_generation_seconds=search_generation_seconds,
                )
        if "training" in selected and external_acceptance_requested:
            if not all(
                (args.dataset, args.search_dataset, args.bootstrap_checkpoint, args.checkpoint)
            ):
                raise ValueError(
                    "external acceptance requires --dataset, --search-dataset, "
                    "--bootstrap-checkpoint, and --checkpoint"
                )
            acceptance = _validate_release_model_metrics(
                Path(args.dataset),
                Path(args.search_dataset),
                Path(args.bootstrap_checkpoint),
                Path(args.checkpoint),
                calibration_roots=args.calibration_roots,
            )
            training_section = report["sections"].setdefault(
                "training", {"status": "PASS"}
            )
            training_section["release_acceptance"] = acceptance
            if args.enforce_acceptance and not acceptance["all_checks_pass"]:
                training_section["status"] = "FAIL"
        if "strength" in selected:
            checkpoint = Path(args.checkpoint) if args.checkpoint else final_checkpoint
            report["sections"]["strength"] = _validate_strength(
                checkpoint,
                games_per_lineup=args.strength_games_per_lineup,
            )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print("ai_nn_rl_based_v1 validation")
    for name, result in report["sections"].items():
        print(f"  {name}: {result['status']}")
        for key, value in result.items():
            if key != "status":
                print(f"    {key}: {value}")
    if args.output:
        print(f"Wrote {args.output}")
    return int(any(result["status"] == "FAIL" for result in report["sections"].values()))


def _validate_model() -> dict[str, Any]:
    rng = np.random.default_rng(41)
    decisions = 24
    offsets = np.arange(0, (decisions + 1) * 3, 3, dtype=np.int32)
    states = rng.normal(size=(decisions, 5)).astype(np.float32)
    actions = rng.normal(size=(offsets[-1], 2)).astype(np.float32)
    teacher = np.argmax(actions[:, 0].reshape(decisions, 3), axis=1)
    terminal = np.zeros((decisions, 6), dtype=np.float32)
    terminal[:, 0] = np.tanh(states[:, 0])
    terminal[:, 1] = -terminal[:, 0]
    masks = np.zeros((decisions, 6), dtype=bool)
    masks[:, :2] = True
    searched = np.zeros(decisions, dtype=bool)
    search_q = np.zeros((len(actions), 6), dtype=np.float32)
    model = NumpyRlPolicyQNetwork(5, 2, hidden_dims=(24, 12, 12), seed=41)
    before = model.evaluate_batch(
        states, actions, offsets, teacher, terminal, masks, searched, search_q
    )
    for _ in range(100):
        model.train_batch(
            states,
            actions,
            offsets,
            teacher,
            terminal,
            masks,
            searched,
            search_q,
            learning_rate=3e-3,
            q_search_weight=0.0,
        )
    after = model.evaluate_batch(
        states, actions, offsets, teacher, terminal, masks, searched, search_q
    )
    if after["policy_accuracy"] <= before["policy_accuracy"] or after["q_mc_mae"] >= before["q_mc_mae"]:
        raise AssertionError("synthetic Policy/Q fit did not improve")
    return {"status": "PASS", "before": before, "after": after}


def _scenario_model() -> tuple[GameSession, NumpyRlPolicyQNetwork]:
    session = GameSession.from_scenario("opening", seed=7)
    snapshot = session.snapshot()
    assert snapshot.active_request is not None
    observation = build_public_observation(snapshot.state, snapshot.active_request)
    state_features, state_names = encode_state_features(observation)
    candidate = generate_candidate_actions(snapshot.active_request, snapshot)[0]
    action_features, action_names = encode_action_features(observation, candidate)
    model = NumpyRlPolicyQNetwork(
        len(state_features),
        len(action_features),
        hidden_dims=(16, 8, 8),
        state_feature_names=state_names,
        action_feature_names=action_names,
    )
    return session, model


def _validate_search() -> dict[str, Any]:
    session, model = _scenario_model()
    candidates = generate_candidate_actions(
        session.snapshot().active_request, session.snapshot()  # type: ignore[arg-type]
    )
    complete = FullActionSemanticSearcher(
        model, SearchConfig(max_search_nodes=512)
    ).search(session)
    fallback = FullActionSemanticSearcher(
        model, SearchConfig(max_search_nodes=len(candidates))
    ).search(session)
    if len(complete.q_values) != len(candidates) or len(fallback.q_values) != len(candidates):
        raise AssertionError("search did not label every root action")
    if not complete.depth_2_completed or fallback.depth_used != 1:
        raise AssertionError("adaptive depth behavior is incorrect")
    return {
        "status": "PASS",
        "candidate_count": len(candidates),
        "depth_2_nodes": complete.nodes_evaluated,
        "fallback_nodes": fallback.nodes_evaluated,
    }


def _validate_controller(checkpoint: Path) -> dict[str, Any]:
    session = GameSession.from_scenario("opening", seed=7)
    snapshot = session.snapshot()
    assert snapshot.active_request is not None
    controller = NnRlBasedAiController(checkpoint)
    intent = controller.choose_intent(snapshot.active_request, snapshot)
    result = session.submit_intent(intent, auto_advance=False)
    if result.event_log[-1].level == "error":
        raise AssertionError(result.event_log[-1].message)
    return {"status": "PASS", "intent": intent.to_dict()}


def _benchmark(
    records: list[dict[str, Any]],
    checkpoint: Path,
    *,
    games: int,
    generation_seconds: float,
    training_seconds: float,
    peak_memory_bytes: int,
    parquet_bytes: int,
    search_decisions: int,
    search_nodes: int,
    search_depth_2_completed: int,
    search_games: int,
    search_decision_total: int,
    search_parquet_bytes: int,
    search_generation_seconds: float,
) -> dict[str, Any]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for row in records:
        by_type[str(row["decision_type"])].append(len(row["candidate_action_features"]))
    distributions = {}
    for decision_type, values in sorted(by_type.items()):
        ordered = sorted(values)
        percentile = lambda fraction: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
        distributions[decision_type] = {
            "mean": mean(ordered),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": max(ordered),
        }
    controller = NnRlBasedAiController(checkpoint)
    session = GameSession.from_scenario("opening", seed=7)
    snapshot = session.snapshot()
    assert snapshot.active_request is not None
    samples = []
    for _ in range(100):
        started = time.perf_counter()
        controller.choose_intent(snapshot.active_request, snapshot)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    model = NumpyRlPolicyQNetwork.load(checkpoint)
    encoding_samples = []
    inference_samples = []
    for _ in range(100):
        started = time.perf_counter()
        observation = build_public_observation(snapshot.state, snapshot.active_request)
        state_features, _ = encode_state_features(observation)
        candidates = generate_candidate_actions(snapshot.active_request, snapshot)
        action_rows = [
            encode_action_features(observation, candidate)[0] for candidate in candidates
        ]
        encoding_samples.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        model.predict_one(
            np.asarray(state_features, dtype=np.float32),
            np.asarray(action_rows, dtype=np.float32),
        )
        inference_samples.append((time.perf_counter() - started) * 1000.0)
    encoding_samples.sort()
    inference_samples.sort()
    return {
        "status": "PASS",
        "candidate_counts": distributions,
        "generation_games_per_second": games / max(generation_seconds, 1e-9),
        "generation_decisions_per_second": len(records) / max(generation_seconds, 1e-9),
        "training_seconds": training_seconds,
        "parquet_bytes_per_decision": parquet_bytes / max(1, len(records)),
        "peak_memory_mib": peak_memory_bytes / (1024 * 1024),
        "policy_latency_p95_ms": samples[94],
        "candidate_encoding_p95_ms": encoding_samples[94],
        "model_inference_p95_ms": inference_samples[94],
        "search_generation_games_per_second": search_games
        / max(search_generation_seconds, 1e-9),
        "search_generation_decisions_per_second": search_decision_total
        / max(search_generation_seconds, 1e-9),
        "search_parquet_bytes_per_decision": search_parquet_bytes
        / max(1, search_decision_total),
        "search_decisions": search_decisions,
        "search_nodes": search_nodes,
        "search_nodes_per_second": search_nodes / max(search_generation_seconds, 1e-9),
        "search_depth_2_completion_rate": (
            search_depth_2_completed / max(1, search_decisions)
        ),
    }


def _validate_release_model_metrics(
    bootstrap_dataset: Path,
    search_dataset: Path,
    bootstrap_checkpoint: Path,
    final_checkpoint: Path,
    *,
    calibration_roots: int,
) -> dict[str, Any]:
    bootstrap_model = NumpyRlPolicyQNetwork.load(bootstrap_checkpoint)
    final_model = NumpyRlPolicyQNetwork.load(final_checkpoint)
    bootstrap_split = _preferred_evaluation_split(bootstrap_dataset)
    search_split = _preferred_evaluation_split(search_dataset)
    slot_means = _training_slot_means(bootstrap_dataset)
    bootstrap_metrics = _dataset_model_metrics(
        bootstrap_model,
        bootstrap_dataset,
        bootstrap_split,
        baseline_slot_means=slot_means,
    )
    init_search_metrics = _dataset_model_metrics(
        bootstrap_model,
        search_dataset,
        search_split,
        policy_target_metadata=final_model.metadata,
    )
    final_search_metrics = _dataset_model_metrics(
        final_model,
        search_dataset,
        search_split,
        policy_target_metadata=final_model.metadata,
    )
    q_baseline_reduction = 1.0 - (
        bootstrap_metrics["q_mc_mse"]
        / max(bootstrap_metrics["q_mc_slot_mean_baseline_mse"], 1e-12)
    )
    search_q_reduction = 1.0 - (
        final_search_metrics["q_search_mae"]
        / max(init_search_metrics["q_search_mae"], 1e-12)
    )
    frequent_types = {
        key: value
        for key, value in bootstrap_metrics["policy_accuracy_by_decision_type"].items()
        if int(value["decisions"])
        >= max(20, int(bootstrap_metrics["decisions"] * 0.01))
    }
    calibration = (
        _calibrate_q_pairwise_ordering(
            final_model, roots=calibration_roots, seed_start=9701
        )
        if calibration_roots > 0
        else {
            "status": "SKIP",
            "roots_requested": 0,
            "roots_evaluated": 0,
            "pairwise_comparisons": 0,
            "pairwise_ordering_accuracy": 0.0,
        }
    )
    checks = {
        "bootstrap_policy_top1_at_least_0_95": bootstrap_metrics[
            "policy_accuracy"
        ]
        >= 0.95,
        "frequent_decision_types_at_least_0_90": bool(frequent_types)
        and all(value["accuracy"] >= 0.90 for value in frequent_types.values()),
        "q_mc_mse_reduction_at_least_0_10": q_baseline_reduction >= 0.10,
        "calibration_has_50_roots": calibration["roots_evaluated"] >= 50,
        "q_pairwise_ordering_at_least_0_55": calibration[
            "pairwise_ordering_accuracy"
        ]
        >= 0.55,
    }
    policy_target_mode = str(
        final_model.metadata.get("policy_target_mode", "legacy_soft_mix")
    )
    if policy_target_mode == "advantage_gate":
        checks.update(
            {
                "accepted_policy_top1_improved_vs_stage0": final_search_metrics[
                    "accepted_policy_top1_accuracy"
                ]
                > init_search_metrics["accepted_policy_top1_accuracy"],
                "searched_fallback_teacher_accuracy_drop_at_most_0_01": (
                    final_search_metrics["searched_fallback_teacher_accuracy"]
                    >= init_search_metrics["searched_fallback_teacher_accuracy"] - 0.01
                ),
                "non_search_teacher_accuracy_drop_at_most_0_01": (
                    final_search_metrics["non_search_teacher_accuracy"]
                    >= init_search_metrics["non_search_teacher_accuracy"] - 0.01
                ),
                "search_q_mae_within_5_percent_of_0_1470": final_search_metrics[
                    "q_search_mae"
                ]
                <= GATED_REFERENCE_SEARCH_Q_MAE
                * (1.0 + GATED_MAX_SEARCH_Q_MAE_REGRESSION),
            }
        )
    else:
        checks.update(
            {
                "search_q_mae_reduction_at_least_0_20": search_q_reduction
                >= 0.20,
                "search_policy_cross_entropy_improved": final_search_metrics[
                    "policy_cross_entropy"
                ]
                < init_search_metrics["policy_cross_entropy"],
            }
        )
    return {
        "policy_target_mode": policy_target_mode,
        "bootstrap_split": bootstrap_split,
        "search_split": search_split,
        "bootstrap": bootstrap_metrics,
        "search_init_checkpoint": init_search_metrics,
        "search_final_checkpoint": final_search_metrics,
        "q_mc_mse_reduction_vs_slot_mean": q_baseline_reduction,
        "search_q_mae_reduction": search_q_reduction,
        "gated_reference_search_q_mae": GATED_REFERENCE_SEARCH_Q_MAE,
        "gated_max_search_q_mae_regression": GATED_MAX_SEARCH_Q_MAE_REGRESSION,
        "frequent_decision_types": frequent_types,
        "calibration": calibration,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _preferred_evaluation_split(dataset: Path) -> str:
    manifest = load_rl_dataset_metadata(dataset)
    for split in ("validation", "test", "train"):
        if int(manifest["splits"][split]["rows"]) > 0:
            return split
    raise ValueError(f"dataset contains no decisions: {dataset}")


def _training_slot_means(dataset: Path) -> np.ndarray:
    sums = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.int64)
    for batch in iter_rl_parquet_batches(
        dataset,
        "train",
        batch_size=1024,
        columns=("terminal_rank_values", "player_mask"),
    ):
        for row in batch.to_pylist():
            values = np.asarray(row["terminal_rank_values"], dtype=np.float64)
            mask = np.asarray(row["player_mask"], dtype=bool)
            sums[mask] += values[mask]
            counts[mask] += 1
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def _dataset_model_metrics(
    model: NumpyRlPolicyQNetwork,
    dataset: Path,
    split: str,
    *,
    baseline_slot_means: np.ndarray | None = None,
    policy_target_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    columns = (
        "decision_type",
        "state_features",
        "candidate_action_features",
        "teacher_action_index",
        "terminal_rank_values",
        "player_mask",
        "has_search_targets",
        "search_q_values",
    )
    decisions = policy_correct = searched_decisions = 0
    accepted_decisions = accepted_correct = 0
    fallback_decisions = fallback_correct = 0
    non_search_decisions = non_search_correct = 0
    policy_cross_entropy = 0.0
    q_mc_abs = q_mc_square = q_mc_baseline_square = 0.0
    q_mc_elements = 0
    q_search_abs = 0.0
    q_search_elements = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    target_metadata = policy_target_metadata or model.metadata
    mix = float(target_metadata.get("search_policy_mix", 0.5))
    temperature = float(target_metadata.get("search_temperature", 0.25))
    target_mode = str(target_metadata.get("policy_target_mode", "legacy_soft_mix"))
    improved_action_weight = float(
        target_metadata.get("improved_action_weight", 0.75)
    )
    min_search_advantage = float(
        target_metadata.get("min_search_advantage", 0.0)
    )
    for batch in iter_rl_parquet_batches(
        dataset, split, batch_size=512, columns=columns
    ):
        rows = batch.to_pylist()
        if not rows:
            continue
        offsets = [0]
        action_rows: list[list[float]] = []
        for row in rows:
            action_rows.extend(row["candidate_action_features"])
            offsets.append(len(action_rows))
        predictions = model.predict(
            np.asarray([row["state_features"] for row in rows], dtype=np.float32),
            np.asarray(action_rows, dtype=np.float32),
            np.asarray(offsets, dtype=np.int32),
        )
        for index, row in enumerate(rows):
            start, end = offsets[index], offsets[index + 1]
            teacher = int(row["teacher_action_index"])
            probabilities = predictions.policy_probabilities[start:end]
            candidate_features = np.asarray(
                row["candidate_action_features"], dtype=np.float32
            )
            search_labels = (
                np.asarray(row["search_q_values"], dtype=np.float32)
                if row["has_search_targets"]
                else np.zeros((end - start, 6), dtype=np.float32)
            )
            target, accepted, improved = build_policy_targets(
                np.asarray([0, end - start], dtype=np.int32),
                np.asarray([teacher], dtype=np.int32),
                np.asarray([row["has_search_targets"]], dtype=bool),
                search_labels,
                candidate_features,
                policy_target_mode=target_mode,
                search_policy_mix=mix,
                search_temperature=temperature,
                improved_action_weight=improved_action_weight,
                min_search_advantage=min_search_advantage,
            )
            policy_cross_entropy -= float(
                np.sum(target * np.log(np.clip(probabilities, 1e-7, 1.0)))
            )
            predicted = int(np.argmax(predictions.policy_logits[start:end]))
            correct = int(predicted == teacher)
            decision_type = str(row["decision_type"])
            by_type[decision_type][0] += correct
            by_type[decision_type][1] += 1
            policy_correct += correct
            decisions += 1
            if bool(accepted[0]):
                accepted_decisions += 1
                accepted_correct += int(predicted == int(improved[0]))
            elif row["has_search_targets"]:
                fallback_decisions += 1
                fallback_correct += correct
            else:
                non_search_decisions += 1
                non_search_correct += correct

            mask = np.asarray(row["player_mask"], dtype=bool)
            terminal = np.asarray(row["terminal_rank_values"], dtype=np.float64)
            chosen_q = predictions.q_values[start + teacher].astype(np.float64)
            differences = chosen_q[mask] - terminal[mask]
            q_mc_abs += float(np.abs(differences).sum())
            q_mc_square += float(np.square(differences).sum())
            q_mc_elements += int(mask.sum())
            if baseline_slot_means is not None:
                q_mc_baseline_square += float(
                    np.square(baseline_slot_means[mask] - terminal[mask]).sum()
                )
            if row["has_search_targets"]:
                search_labels = np.asarray(row["search_q_values"], dtype=np.float64)
                search_differences = (
                    predictions.q_values[start:end].astype(np.float64)[:, mask]
                    - search_labels[:, mask]
                )
                q_search_abs += float(np.abs(search_differences).sum())
                q_search_elements += int(search_differences.size)
                searched_decisions += 1
    if decisions <= 0:
        raise ValueError(f"dataset split {split!r} contains no decisions")
    return {
        "decisions": decisions,
        "searched_decisions": searched_decisions,
        "policy_accuracy": policy_correct / decisions,
        "policy_cross_entropy": policy_cross_entropy / decisions,
        "gated_policy_cross_entropy": policy_cross_entropy / decisions,
        "accepted_improvement_decisions": accepted_decisions,
        "accepted_improvement_rate": accepted_decisions
        / max(1, searched_decisions),
        "accepted_policy_top1_accuracy": accepted_correct
        / max(1, accepted_decisions),
        "searched_fallback_decisions": fallback_decisions,
        "searched_fallback_teacher_accuracy": fallback_correct
        / max(1, fallback_decisions),
        "non_search_decisions": non_search_decisions,
        "non_search_teacher_accuracy": non_search_correct
        / max(1, non_search_decisions),
        "policy_accuracy_by_decision_type": {
            key: {"accuracy": correct / count, "decisions": count}
            for key, (correct, count) in sorted(by_type.items())
        },
        "q_mc_mae": q_mc_abs / max(1, q_mc_elements),
        "q_mc_mse": q_mc_square / max(1, q_mc_elements),
        "q_mc_slot_mean_baseline_mse": q_mc_baseline_square
        / max(1, q_mc_elements),
        "q_search_mae": q_search_abs / max(1, q_search_elements),
        "q_mc_elements": q_mc_elements,
        "q_search_elements": q_search_elements,
    }


def _calibrate_q_pairwise_ordering(
    model: NumpyRlPolicyQNetwork, *, roots: int, seed_start: int
) -> dict[str, Any]:
    if roots <= 0:
        raise ValueError("calibration roots must be positive")
    root_sessions: list[GameSession] = []
    for game_offset in range(roots):
        seed = seed_start + game_offset
        config = GameConfig(
            map_id="germany",
            players=tuple(
                SeatConfig(f"p{index + 1}", f"Player {index + 1}", controller="human")
                for index in range(3)
            ),
            seed=seed,
        )
        session = GameSession.new_game(config)
        agents = {
            player.player_id: build_ai_controller("ai_deterministic")
            for player in session.snapshot().state.players
        }
        selected: GameSession | None = None
        selected_hash: bytes | None = None
        for decision_index in range(5000):
            snapshot = session.advance_until_blocked()
            if snapshot.winner_result is not None:
                break
            request = snapshot.active_request
            if request is None:
                raise ModelValidationError("calibration game stopped without a request")
            candidates = generate_candidate_actions(request, snapshot)
            if len(candidates) > 1:
                digest = hashlib.sha256(
                    f"{seed}:{decision_index}".encode("utf-8")
                ).digest()
                if selected_hash is None or digest < selected_hash:
                    selected_hash = digest
                    selected = session.fork()
            intent = agents[request.player_id].choose_intent(request, snapshot)
            result = session.submit_intent(intent, auto_advance=False)
            if result.event_log and result.event_log[-1].level == "error":
                raise ModelValidationError(result.event_log[-1].message)
        else:
            raise ModelValidationError("calibration game exceeded 5000 decisions")
        if selected is None:
            raise ModelValidationError("calibration game contained no multi-action state")
        root_sessions.append(selected)

    comparisons = 0
    ordering_credit = 0.0
    candidate_counts: list[int] = []
    for root in root_sessions:
        snapshot = root.snapshot()
        request = snapshot.active_request
        if request is None:
            raise ModelValidationError("calibration root has no request")
        observation = build_public_observation(snapshot.state, request)
        state_features, state_names = encode_state_features(observation)
        candidates = generate_candidate_actions(request, snapshot)
        action_rows = []
        action_names: tuple[str, ...] | None = None
        for candidate in candidates:
            features, names = encode_action_features(observation, candidate)
            action_rows.append(features)
            action_names = action_names or names
            if names != action_names:
                raise ModelValidationError("calibration action schema changed")
        if model.state_feature_names != state_names or model.action_feature_names != tuple(
            action_names or ()
        ):
            raise ModelValidationError("calibration checkpoint feature schema mismatch")
        predicted_q = model.predict_one(
            np.asarray(state_features, dtype=np.float32),
            np.asarray(action_rows, dtype=np.float32),
        ).q_values[:, 0]
        realized = np.asarray(
            [
                _rollout_candidate_terminal_value(root, candidate, request.player_id)
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        candidate_counts.append(len(candidates))
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                target_difference = realized[left] - realized[right]
                if abs(target_difference) < 1e-12:
                    continue
                comparisons += 1
                predicted_difference = float(predicted_q[left] - predicted_q[right])
                if abs(predicted_difference) < 1e-12:
                    ordering_credit += 0.5
                elif (predicted_difference > 0) == (target_difference > 0):
                    ordering_credit += 1.0
    return {
        "status": "PASS",
        "roots_requested": roots,
        "roots_evaluated": len(root_sessions),
        "candidate_count_mean": mean(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "pairwise_comparisons": comparisons,
        "pairwise_ordering_accuracy": ordering_credit / max(1, comparisons),
        "tie_targets_excluded": True,
        "continuation_controller": "ai_deterministic",
    }


def _rollout_candidate_terminal_value(
    root: GameSession, candidate: Any, actor_id: str
) -> float:
    rollout = root.fork()
    result = rollout.submit_intent(candidate.intent, auto_advance=False)
    if result.event_log and result.event_log[-1].level == "error":
        raise ModelValidationError(result.event_log[-1].message)
    agents = {
        player.player_id: build_ai_controller("ai_deterministic")
        for player in rollout.snapshot().state.players
    }
    for _ in range(5000):
        snapshot = rollout.advance_until_blocked()
        if snapshot.winner_result is not None:
            return terminal_rank_values(snapshot)[actor_id]
        request = snapshot.active_request
        if request is None:
            raise ModelValidationError("calibration rollout stopped without a request")
        intent = agents[request.player_id].choose_intent(request, snapshot)
        result = rollout.submit_intent(intent, auto_advance=False)
        if result.event_log and result.event_log[-1].level == "error":
            raise ModelValidationError(result.event_log[-1].message)
    raise ModelValidationError("calibration rollout exceeded 5000 decisions")


def _validate_strength(
    checkpoint: Path | None, *, games_per_lineup: int
) -> dict[str, Any]:
    if checkpoint is None or games_per_lineup <= 0:
        return {
            "status": "SKIP",
            "reason": "supply --checkpoint and positive --strength-games-per-lineup",
        }
    import os

    previous = os.environ.get("POWERGRID_NN_RL_BASED_CHECKPOINT")
    os.environ["POWERGRID_NN_RL_BASED_CHECKPOINT"] = str(checkpoint)
    try:
        report = evaluate_ai_bucket(
            AiEvaluationBucketConfig(
                controller_names=("ai_nn_rl_based_v1", "ai_deterministic"),
                games_per_lineup=games_per_lineup,
                seed_start=9501,
            )
        )
    finally:
        if previous is None:
            os.environ.pop("POWERGRID_NN_RL_BASED_CHECKPOINT", None)
        else:
            os.environ["POWERGRID_NN_RL_BASED_CHECKPOINT"] = previous
    summaries = {item.controller_name: item.to_dict() for item in report.controller_summaries}
    pairwise = _rl_pairwise_score_with_game_bootstrap(report.game_summaries)
    rl_finish = float(summaries["ai_nn_rl_based_v1"]["average_finish"])
    deterministic_finish = float(summaries["ai_deterministic"]["average_finish"])
    release_checks = {
        "at_least_400_completed_games": len(report.game_summaries) >= 400,
        "pairwise_score_at_least_0_50": pairwise["score"] >= 0.50,
        "average_finish_not_worse": rl_finish <= deterministic_finish,
        "bootstrap_ci_lower_at_least_0_48": pairwise["bootstrap_95_ci"][0] >= 0.48,
    }
    if len(report.game_summaries) < 400:
        status = "SMOKE_ONLY"
    else:
        status = "PASS" if all(release_checks.values()) else "FAIL"
    return {
        "status": status,
        "games": len(report.game_summaries),
        "controllers": summaries,
        "pair": report.controller_pair_summaries[0].to_dict(),
        "rl_pairwise": pairwise,
        "release_checks": release_checks,
        "release_eligible": all(release_checks.values()),
    }


def _rl_pairwise_score_with_game_bootstrap(
    games: tuple[Any, ...], *, samples: int = 2000, seed: int = 7301
) -> dict[str, Any]:
    per_game: list[tuple[float, int]] = []
    wins = draws = losses = 0
    for game in games:
        rl_seats = [
            standing
            for standing in game.standings
            if standing.controller_name == "ai_nn_rl_based_v1"
        ]
        deterministic_seats = [
            standing
            for standing in game.standings
            if standing.controller_name == "ai_deterministic"
        ]
        game_score = 0.0
        comparisons = 0
        for rl_standing in rl_seats:
            for deterministic_standing in deterministic_seats:
                comparisons += 1
                rl_signature = (
                    rl_standing.powered_cities,
                    rl_standing.money,
                    rl_standing.connected_cities,
                )
                deterministic_signature = (
                    deterministic_standing.powered_cities,
                    deterministic_standing.money,
                    deterministic_standing.connected_cities,
                )
                if rl_signature == deterministic_signature:
                    draws += 1
                    game_score += 0.5
                elif rl_standing.place < deterministic_standing.place:
                    wins += 1
                    game_score += 1.0
                else:
                    losses += 1
        if comparisons <= 0:
            raise AssertionError("strength game did not contain both controllers")
        per_game.append((game_score, comparisons))
    score_total = sum(item[0] for item in per_game)
    comparison_total = sum(item[1] for item in per_game)
    score = score_total / max(1, comparison_total)
    rng = np.random.default_rng(seed)
    bootstrap_scores = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        indices = rng.integers(0, len(per_game), size=len(per_game))
        sampled_score = sum(per_game[int(index)][0] for index in indices)
        sampled_comparisons = sum(per_game[int(index)][1] for index in indices)
        bootstrap_scores[sample_index] = sampled_score / sampled_comparisons
    lower, upper = np.quantile(bootstrap_scores, [0.025, 0.975])
    return {
        "score": float(score),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "comparisons": comparison_total,
        "bootstrap_unit": "game",
        "bootstrap_samples": samples,
        "bootstrap_95_ci": [float(lower), float(upper)],
        "significant_improvement": float(lower) > 0.50,
    }


if __name__ == "__main__":
    raise SystemExit(main())
