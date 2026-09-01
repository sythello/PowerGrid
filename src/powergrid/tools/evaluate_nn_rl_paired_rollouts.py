from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from powergrid.ai import build_ai_controller
from powergrid.ai.nn_rl_based.controller import NnRlBasedAiController
from powergrid.ai.nn_rl_based.search import terminal_rank_values
from powergrid.model import GameConfig, ModelValidationError, SeatConfig, legal_region_sets
from powergrid.session import GameSession
from powergrid.session_types import GuiIntent


@dataclass(frozen=True)
class PairedRolloutRecord:
    checkpoint: str
    game_index: int
    seed: int
    selected_regions: tuple[str, ...]
    decision_index: int
    round_number: int
    phase: str
    decision_type: str
    actor_id: str
    baseline_intent: dict[str, Any]
    rl_intent: dict[str, Any]
    baseline_rank_value: float
    rl_rank_value: float
    advantage: float


def evaluate_paired_terminal_rollouts(
    checkpoints: dict[str, str | Path],
    *,
    games: int,
    seed_start: int,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 7301,
    max_actions: int = 5000,
    progress_callback: Callable[[int, int, int, dict[str, int], float], None]
    | None = None,
) -> dict[str, Any]:
    """Evaluate one-step Policy deviations with paired baseline-continuation rollouts."""

    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if games <= 0 or bootstrap_samples <= 0 or max_actions <= 0:
        raise ValueError("games, bootstrap_samples, and max_actions must be positive")
    resolved = {name: Path(path) for name, path in checkpoints.items()}
    if any(not name for name in resolved):
        raise ValueError("checkpoint labels must be non-empty")
    controllers = {
        name: NnRlBasedAiController(checkpoint_path=path)
        for name, path in resolved.items()
    }
    baseline_agents = {
        f"p{index + 1}": build_ai_controller("ai_deterministic")
        for index in range(3)
    }
    regions = legal_region_sets("germany", 3)
    records: dict[str, list[PairedRolloutRecord]] = {
        name: [] for name in resolved
    }
    game_decisions: dict[int, int] = {}
    region_counts: Counter[tuple[str, ...]] = Counter()
    rollout_branches = 0
    started = time.perf_counter()

    for game_offset in range(games):
        game_index = game_offset + 1
        seed = seed_start + game_offset
        selected_regions = regions[game_offset % len(regions)]
        region_counts[selected_regions] += 1
        config = GameConfig(
            map_id="germany",
            players=tuple(
                SeatConfig(
                    player_id=f"p{index + 1}",
                    name=f"Player {index + 1}",
                    controller="human",
                )
                for index in range(3)
            ),
            seed=seed,
            selected_regions=selected_regions,
        )
        session = GameSession.new_game(config)
        for decision_index in range(max_actions):
            snapshot = session.advance_until_blocked()
            if snapshot.winner_result is not None:
                game_decisions[game_index] = decision_index
                break
            request = snapshot.active_request
            if request is None:
                raise ModelValidationError(
                    f"baseline game {game_index} stopped without a request"
                )
            baseline_intent = baseline_agents[request.player_id].choose_intent(
                request, snapshot
            )
            rl_intents = {
                name: controller.choose_intent(request, snapshot)
                for name, controller in controllers.items()
            }
            changed = {
                name: intent
                for name, intent in rl_intents.items()
                if intent.to_dict() != baseline_intent.to_dict()
            }
            if changed:
                baseline_values = _rollout_terminal_values(
                    session,
                    baseline_intent,
                    baseline_agents,
                    max_actions=max_actions,
                )
                rollout_branches += 1
                terminal_by_intent: dict[str, dict[str, float]] = {}
                for name, rl_intent in changed.items():
                    key = json.dumps(
                        rl_intent.to_dict(), sort_keys=True, separators=(",", ":")
                    )
                    if key not in terminal_by_intent:
                        terminal_by_intent[key] = _rollout_terminal_values(
                            session,
                            rl_intent,
                            baseline_agents,
                            max_actions=max_actions,
                        )
                        rollout_branches += 1
                    rl_values = terminal_by_intent[key]
                    actor_id = request.player_id
                    advantage = rl_values[actor_id] - baseline_values[actor_id]
                    records[name].append(
                        PairedRolloutRecord(
                            checkpoint=name,
                            game_index=game_index,
                            seed=seed,
                            selected_regions=selected_regions,
                            decision_index=decision_index,
                            round_number=snapshot.state.round_number,
                            phase=request.phase,
                            decision_type=request.decision_type,
                            actor_id=actor_id,
                            baseline_intent=baseline_intent.to_dict(),
                            rl_intent=rl_intent.to_dict(),
                            baseline_rank_value=baseline_values[actor_id],
                            rl_rank_value=rl_values[actor_id],
                            advantage=advantage,
                        )
                    )
            result = session.submit_intent(baseline_intent, auto_advance=False)
            _raise_last_error(result, f"baseline game {game_index}")
        else:
            raise ModelValidationError(
                f"baseline game {game_index} exceeded {max_actions} decisions"
            )
        if progress_callback is not None:
            progress_callback(
                game_index,
                games,
                sum(game_decisions.values()),
                {name: len(items) for name, items in records.items()},
                time.perf_counter() - started,
            )

    total_decisions = sum(game_decisions.values())
    checkpoint_results = {
        name: _summarize_checkpoint(
            items,
            games=games,
            total_decisions=total_decisions,
            game_decisions=game_decisions,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for name, items in records.items()
    }
    return {
        "format_name": "powergrid.nn_rl_paired_terminal_rollouts",
        "format_version": 1,
        "method": {
            "root_distribution": "ai_deterministic trajectories",
            "root_selection": "all exact intent deviations",
            "first_actions": "checkpoint Policy versus ai_deterministic",
            "continuation_policy": "ai_deterministic for every player",
            "hidden_state_pairing": "same GameSession fork and hidden deck order",
            "value": "root actor normalized terminal rank in [-1, 1]",
            "advantage": "RL branch value minus baseline branch value",
            "confidence_interval": "source-game cluster bootstrap",
        },
        "configuration": {
            "map": "germany",
            "players": 3,
            "games": games,
            "seed_start": seed_start,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "max_actions": max_actions,
            "checkpoint_paths": {name: str(path) for name, path in resolved.items()},
        },
        "baseline": {
            "games_completed": games,
            "decisions": total_decisions,
            "legal_region_sets_covered": len(region_counts),
            "region_game_counts": {
                ",".join(key): count for key, count in sorted(region_counts.items())
            },
        },
        "rollout_branches": rollout_branches,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoints": checkpoint_results,
        "records": {
            name: [asdict(item) for item in items] for name, items in records.items()
        },
    }


def _rollout_terminal_values(
    root: GameSession,
    first_intent: GuiIntent,
    baseline_agents: dict[str, Any],
    *,
    max_actions: int,
) -> dict[str, float]:
    rollout = root.fork()
    result = rollout.submit_intent(first_intent, auto_advance=False)
    _raise_last_error(result, "paired rollout first action")
    for _ in range(max_actions):
        snapshot = rollout.advance_until_blocked()
        if snapshot.winner_result is not None:
            return terminal_rank_values(snapshot)
        request = snapshot.active_request
        if request is None:
            raise ModelValidationError("paired rollout stopped without a request")
        intent = baseline_agents[request.player_id].choose_intent(request, snapshot)
        result = rollout.submit_intent(intent, auto_advance=False)
        _raise_last_error(result, "paired rollout continuation")
    raise ModelValidationError(
        f"paired rollout exceeded {max_actions} continuation actions"
    )


def _summarize_checkpoint(
    records: list[PairedRolloutRecord],
    *,
    games: int,
    total_decisions: int,
    game_decisions: dict[int, int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    advantages = np.asarray([item.advantage for item in records], dtype=np.float64)
    improved = int(np.sum(advantages > 0.0))
    harmed = int(np.sum(advantages < 0.0))
    tied = len(records) - improved - harmed
    score_total = improved + 0.5 * tied
    deviations = len(records)
    mean_advantage = float(np.mean(advantages)) if deviations else 0.0
    paired_score = score_total / deviations if deviations else 0.5
    grouped: dict[int, list[float]] = defaultdict(list)
    for item in records:
        grouped[item.game_index].append(item.advantage)
    intervals = _cluster_bootstrap_intervals(
        grouped,
        game_decisions=game_decisions,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    by_decision_type: dict[str, Any] = {}
    typed: dict[str, list[float]] = defaultdict(list)
    for item in records:
        typed[item.decision_type].append(item.advantage)
    for decision_type, values in sorted(typed.items()):
        array = np.asarray(values, dtype=np.float64)
        positives = int(np.sum(array > 0.0))
        negatives = int(np.sum(array < 0.0))
        ties = len(values) - positives - negatives
        by_decision_type[decision_type] = {
            "deviations": len(values),
            "mean_advantage": float(np.mean(array)),
            "paired_score": (positives + 0.5 * ties) / len(values),
            "improved": positives,
            "tied": ties,
            "harmed": negatives,
        }
    return {
        "games": games,
        "baseline_decisions": total_decisions,
        "deviations": deviations,
        "deviation_rate": deviations / max(1, total_decisions),
        "improved": improved,
        "tied": tied,
        "harmed": harmed,
        "harmful_switch_rate": harmed / max(1, deviations),
        "mean_advantage_on_deviations": mean_advantage,
        "mean_advantage_per_baseline_decision": float(np.sum(advantages))
        / max(1, total_decisions),
        "paired_score": paired_score,
        "mean_advantage_95_ci": intervals["mean_advantage"],
        "paired_score_95_ci": intervals["paired_score"],
        "mean_advantage_per_baseline_decision_95_ci": intervals[
            "advantage_per_baseline_decision"
        ],
        "point_estimate_positive": mean_advantage > 0.0 and paired_score > 0.5,
        "statistically_positive": intervals["mean_advantage"][0] > 0.0
        and intervals["paired_score"][0] > 0.5,
        "by_decision_type": by_decision_type,
    }


def _cluster_bootstrap_intervals(
    grouped_advantages: dict[int, list[float]],
    *,
    game_decisions: dict[int, int],
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    game_ids = np.asarray(sorted(game_decisions), dtype=np.int32)
    rng = np.random.default_rng(seed)
    mean_advantages = np.zeros(samples, dtype=np.float64)
    paired_scores = np.full(samples, 0.5, dtype=np.float64)
    per_decision = np.zeros(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = rng.choice(game_ids, size=len(game_ids), replace=True)
        values = [
            value
            for game_id in selected
            for value in grouped_advantages.get(int(game_id), ())
        ]
        decision_count = sum(game_decisions[int(game_id)] for game_id in selected)
        if values:
            array = np.asarray(values, dtype=np.float64)
            mean_advantages[sample_index] = float(np.mean(array))
            paired_scores[sample_index] = float(
                (np.sum(array > 0.0) + 0.5 * np.sum(array == 0.0)) / len(array)
            )
            per_decision[sample_index] = float(np.sum(array)) / max(
                1, decision_count
            )
    return {
        "mean_advantage": [
            float(value) for value in np.quantile(mean_advantages, [0.025, 0.975])
        ],
        "paired_score": [
            float(value) for value in np.quantile(paired_scores, [0.025, 0.975])
        ],
        "advantage_per_baseline_decision": [
            float(value) for value in np.quantile(per_decision, [0.025, 0.975])
        ],
    }


def _raise_last_error(snapshot: Any, context: str) -> None:
    if snapshot.event_log and snapshot.event_log[-1].level == "error":
        raise ModelValidationError(f"{context}: {snapshot.event_log[-1].message}")


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH or PATH")
    return label, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RL Policy deviations with paired terminal rollouts."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeatable LABEL=PATH checkpoint specification.",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=70001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=7301)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument(
        "--output",
        default="artifacts/validation/ai_nn_rl_based_v1_paired_rollouts.json",
    )
    args = parser.parse_args(argv)
    parsed = [_parse_checkpoint(value) for value in args.checkpoint]
    checkpoints = dict(parsed)
    if len(checkpoints) != len(parsed):
        parser.error("checkpoint labels must be unique")

    def progress(
        completed: int,
        total: int,
        decisions: int,
        deviations: dict[str, int],
        elapsed: float,
    ) -> None:
        print(
            f"Progress: games={completed}/{total} decisions={decisions} "
            f"deviations={deviations} elapsed={elapsed:.1f}s",
            flush=True,
        )

    report = evaluate_paired_terminal_rollouts(
        checkpoints,
        games=args.games,
        seed_start=args.seed_start,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        max_actions=args.max_actions,
        progress_callback=progress,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Paired terminal rollout evaluation")
    for name, result in report["checkpoints"].items():
        print(
            f"{name}: deviations={result['deviations']} "
            f"rate={result['deviation_rate']:.4f} "
            f"mean_advantage={result['mean_advantage_on_deviations']:.4f} "
            f"mean_ci={result['mean_advantage_95_ci']} "
            f"score={result['paired_score']:.4f} "
            f"score_ci={result['paired_score_95_ci']} "
            f"statistically_positive={result['statistically_positive']}"
        )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
