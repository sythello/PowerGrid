from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Any

from powergrid.ai import derive_final_standings
from powergrid.ai.nn_rank_value.dataset import sha256_file
from powergrid.ai.nn_rl_based.controller import NnRlBasedAiController
from powergrid.ai.nn_rl_based.search import terminal_rank_values
from powergrid.model import GameConfig, ModelValidationError, SeatConfig, legal_region_sets
from powergrid.session import GameSession
from powergrid.session_types import GameLogEntry, GuiIntent, HumanSeat
from powergrid.tools.evaluate_nn_rl_checkpoint_duel import (
    CANDIDATE,
    DUEL_LINEUPS,
    INCUMBENT,
)


def analyze_checkpoint_losses(
    candidate_checkpoint: str | Path,
    incumbent_checkpoint: str | Path,
    *,
    seeds: int,
    seed_start: int,
    output_dir: str | Path,
    max_actions: int = 5000,
) -> dict[str, Any]:
    """Run balanced checkpoint games and retain full logs for candidate losses."""

    if seeds <= 0 or max_actions <= 0:
        raise ValueError("seeds and max_actions must be positive")
    candidate_path = Path(candidate_checkpoint)
    incumbent_path = Path(incumbent_checkpoint)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate checkpoint does not exist: {candidate_path}")
    if not incumbent_path.is_file():
        raise FileNotFoundError(f"incumbent checkpoint does not exist: {incumbent_path}")

    destination = Path(output_dir)
    trace_dir = destination / "traces"
    loss_dir = destination / "losses"
    trace_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    controllers = {
        CANDIDATE: NnRlBasedAiController(candidate_path),
        INCUMBENT: NnRlBasedAiController(incumbent_path),
    }
    region_sets = legal_region_sets("germany", 3)
    games: list[dict[str, Any]] = []
    started = time.perf_counter()

    for seed_offset in range(seeds):
        seed = seed_start + seed_offset
        selected_regions = region_sets[seed_offset % len(region_sets)]
        for lineup_index, lineup in enumerate(DUEL_LINEUPS):
            game_index = len(games) + 1
            game, session, trace = _run_traced_game(
                controllers=controllers,
                lineup=lineup,
                lineup_index=lineup_index,
                game_index=game_index,
                seed=seed,
                selected_regions=selected_regions,
                max_actions=max_actions,
            )
            trace_path = trace_dir / f"game_{game_index:02d}_trace.json"
            _write_json(trace_path, trace)
            game["trace_path"] = str(trace_path)
            if float(game["pairwise_score"]) < 0.5:
                loss_path = loss_dir / f"game_{game_index:02d}_full_log.json"
                session.dump_game_log(loss_path)
                game["full_log_path"] = str(loss_path)
            games.append(game)
            print(
                f"game={game_index}/{seeds * len(DUEL_LINEUPS)} seed={seed} "
                f"lineup={','.join(lineup)} score={game['pairwise_score']:.3f} "
                f"disagreements={game['disagreements']}",
                flush=True,
            )

    report = _build_report(
        candidate_path=candidate_path,
        incumbent_path=incumbent_path,
        seeds=seeds,
        seed_start=seed_start,
        max_actions=max_actions,
        games=games,
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json(destination / "summary.json", report)
    return report


def _run_traced_game(
    *,
    controllers: dict[str, NnRlBasedAiController],
    lineup: tuple[str, str, str],
    lineup_index: int,
    game_index: int,
    seed: int,
    selected_regions: tuple[str, ...],
    max_actions: int,
) -> tuple[dict[str, Any], GameSession, dict[str, Any]]:
    config = GameConfig(
        map_id="germany",
        players=tuple(
            SeatConfig(f"p{index + 1}", f"Player {index + 1}", controller="human")
            for index in range(3)
        ),
        seed=seed,
        selected_regions=selected_regions,
    )
    roles_by_player = {
        f"p{index + 1}": role for index, role in enumerate(lineup)
    }
    human_seats = {player_id: HumanSeat() for player_id in roles_by_player}
    session = GameSession.new_game(config, seat_agents=human_seats)
    snapshot = session.advance_until_blocked()
    decisions: list[dict[str, Any]] = []

    for decision_index in range(1, max_actions + 1):
        if snapshot.winner_result is not None:
            break
        request = session.current_request()
        if request is None:
            raise ModelValidationError(
                f"traced game {game_index} ended without a request or winner"
            )
        before_summary = _state_summary(snapshot.state, roles_by_player)

        candidate_intent, candidate_choice = _choose_and_capture(
            controllers[CANDIDATE], request, snapshot, session
        )
        incumbent_intent, incumbent_choice = _choose_and_capture(
            controllers[INCUMBENT], request, snapshot, session
        )
        _add_cross_choice(candidate_choice, incumbent_intent)
        _add_cross_choice(incumbent_choice, candidate_intent)
        differs = _intent_key(candidate_intent) != _intent_key(incumbent_intent)
        actual_role = roles_by_player[request.player_id]
        actual_intent = (
            candidate_intent if actual_role == CANDIDATE else incumbent_intent
        )
        paired_continuation = (
            _paired_continuation(
                session=session,
                controllers=controllers,
                roles_by_player=roles_by_player,
                actor_id=request.player_id,
                candidate_intent=candidate_intent,
                incumbent_intent=incumbent_intent,
            )
            if differs and actual_role == CANDIDATE
            else None
        )
        decisions.append(
            {
                "decision_index": decision_index,
                "round_number": snapshot.state.round_number,
                "step": snapshot.state.step,
                "phase": request.phase,
                "decision_type": request.decision_type,
                "actor_id": request.player_id,
                "actor_role": actual_role,
                "request_metadata": request.metadata,
                "state": before_summary,
                "candidate": candidate_choice,
                "incumbent": incumbent_choice,
                "models_disagree": differs,
                "actual_intent": actual_intent.to_dict(),
                "paired_continuation": paired_continuation,
            }
        )
        snapshot = session.submit_intent(actual_intent, auto_advance=False)
        if snapshot.event_log and snapshot.event_log[-1].level == "error":
            raise ModelValidationError(
                f"traced game {game_index} action failed: "
                f"{snapshot.event_log[-1].message}"
            )
        snapshot = session.advance_until_blocked()
    else:
        raise ModelValidationError(
            f"traced game {game_index} exceeded {max_actions} decisions"
        )

    if snapshot.winner_result is None:
        raise ModelValidationError(f"traced game {game_index} has no winner")
    standings = derive_final_standings(snapshot.state, snapshot.winner_result)
    candidate_standings = [
        standing
        for standing in standings
        if roles_by_player[standing.player_id] == CANDIDATE
    ]
    incumbent_standings = [
        standing
        for standing in standings
        if roles_by_player[standing.player_id] == INCUMBENT
    ]
    score = 0.0
    wins = draws = losses = 0
    for candidate in candidate_standings:
        for incumbent in incumbent_standings:
            candidate_signature = _standing_signature(candidate)
            incumbent_signature = _standing_signature(incumbent)
            if candidate_signature == incumbent_signature:
                score += 0.5
                draws += 1
            elif candidate.place < incumbent.place:
                score += 1.0
                wins += 1
            else:
                losses += 1
    comparisons = wins + draws + losses
    disagreement_counts = Counter(
        decision["decision_type"]
        for decision in decisions
        if decision["models_disagree"]
    )
    candidate_actual_disagreements = Counter(
        decision["decision_type"]
        for decision in decisions
        if decision["models_disagree"] and decision["actor_role"] == CANDIDATE
    )
    paired_decisions = [
        decision for decision in decisions if decision["paired_continuation"] is not None
    ]
    paired_rank_advantages = [
        float(decision["paired_continuation"]["candidate_action_advantage"])
        for decision in paired_decisions
    ]
    paired_signature_comparisons = [
        _compare_terminal_signature(decision["paired_continuation"])
        for decision in paired_decisions
    ]
    standing_payloads = []
    for standing in standings:
        payload = standing.to_dict()
        payload["duel_role"] = roles_by_player[standing.player_id]
        standing_payloads.append(payload)
    game = {
        "game_index": game_index,
        "lineup_index": lineup_index,
        "seed": seed,
        "selected_regions": list(selected_regions),
        "lineup": list(lineup),
        "composition": (
            "candidate_majority"
            if lineup.count(CANDIDATE) == 2
            else "incumbent_majority"
        ),
        "score": score,
        "comparisons": comparisons,
        "pairwise_score": score / comparisons,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "decisions": len(decisions),
        "disagreements": sum(disagreement_counts.values()),
        "disagreements_by_type": dict(sorted(disagreement_counts.items())),
        "candidate_actual_disagreements": sum(
            candidate_actual_disagreements.values()
        ),
        "candidate_actual_disagreements_by_type": dict(
            sorted(candidate_actual_disagreements.items())
        ),
        "candidate_policy_chose_lower_own_q": sum(
            float(decision["candidate"]["selected_minus_other_actor_q"]) < 0.0
            for decision in paired_decisions
        ),
        "paired_continuation_rank": {
            "harmful": sum(value < 0.0 for value in paired_rank_advantages),
            "neutral": sum(value == 0.0 for value in paired_rank_advantages),
            "helpful": sum(value > 0.0 for value in paired_rank_advantages),
        },
        "paired_continuation_terminal_signature": {
            "worse": sum(value < 0 for value in paired_signature_comparisons),
            "same": sum(value == 0 for value in paired_signature_comparisons),
            "better": sum(value > 0 for value in paired_signature_comparisons),
        },
        "standings": standing_payloads,
    }
    trace = {
        "format_name": "powergrid.nn_rl_checkpoint_loss_trace",
        "format_version": 1,
        "game": game,
        "decisions": decisions,
    }
    return game, session, trace


def _paired_continuation(
    *,
    session: GameSession,
    controllers: dict[str, NnRlBasedAiController],
    roles_by_player: dict[str, str],
    actor_id: str,
    candidate_intent: GuiIntent,
    incumbent_intent: GuiIntent,
) -> dict[str, Any]:
    agents = {
        player_id: controllers[role]
        for player_id, role in roles_by_player.items()
    }
    branches = {}
    for name, intent in (
        ("candidate_action", candidate_intent),
        ("incumbent_action", incumbent_intent),
    ):
        branch = session.fork(seat_agents=agents)
        snapshot = branch.submit_intent(intent, auto_advance=False)
        if snapshot.event_log and snapshot.event_log[-1].level == "error":
            raise ModelValidationError(
                f"paired continuation {name} failed: {snapshot.event_log[-1].message}"
            )
        snapshot = branch.advance_until_blocked()
        if snapshot.winner_result is None:
            raise ModelValidationError(f"paired continuation {name} has no winner")
        values = terminal_rank_values(snapshot)
        standings = derive_final_standings(snapshot.state, snapshot.winner_result)
        actor_standing = next(
            standing for standing in standings if standing.player_id == actor_id
        )
        branches[name] = {
            "actor_rank_value": float(values[actor_id]),
            "actor_standing": actor_standing.to_dict(),
            "values": values,
        }
    branches["candidate_action_advantage"] = (
        branches["candidate_action"]["actor_rank_value"]
        - branches["incumbent_action"]["actor_rank_value"]
    )
    return branches


def _choose_and_capture(
    controller: NnRlBasedAiController,
    request: Any,
    snapshot: Any,
    session: GameSession,
) -> tuple[GuiIntent, dict[str, Any]]:
    before = len(session.game_log_entries())
    intent = controller.choose_intent(request, snapshot)
    appended = session.game_log_entries()[before:]
    entries = [
        entry
        for entry in appended
        if entry.event_type == "ai_state"
        and entry.payload.get("label") == "nn_rl_based_decision"
    ]
    if len(entries) != 1:
        raise ModelValidationError(
            f"expected one NN RL decision log, observed {len(entries)}"
        )
    choice = _choice_from_log(entries[0])
    if _intent_key(intent) != _intent_key_dict(choice["selected_intent"]):
        raise ModelValidationError("controller result does not match its decision log")
    return intent, choice


def _choice_from_log(entry: GameLogEntry) -> dict[str, Any]:
    state = dict(entry.payload["state"])
    ranked = list(state["ranked_candidates"])
    selected_key = _intent_key_dict(state["selected_intent"])
    selected_rank = next(
        index
        for index, item in enumerate(ranked, start=1)
        if _intent_key_dict(item["intent"]) == selected_key
    )
    selected = ranked[selected_rank - 1]
    probabilities = [float(item["policy_probability"]) for item in ranked]
    entropy = -sum(
        probability * math.log(max(probability, 1e-12))
        for probability in probabilities
    )
    return {
        "checkpoint": state["checkpoint"],
        "selected_intent": state["selected_intent"],
        "selected_rank": selected_rank,
        "selected_probability": float(selected["policy_probability"]),
        "selected_actor_q": float(selected["actor_q"]),
        "runner_up_probability": (
            float(ranked[1]["policy_probability"]) if len(ranked) > 1 else None
        ),
        "probability_margin": (
            float(ranked[0]["policy_probability"])
            - float(ranked[1]["policy_probability"])
            if len(ranked) > 1
            else float(ranked[0]["policy_probability"])
        ),
        "policy_entropy": entropy,
        "candidate_count": len(ranked),
        "top_candidates": ranked[:5],
        "_ranked_candidates": ranked,
    }


def _add_cross_choice(choice: dict[str, Any], other_intent: GuiIntent) -> None:
    other_key = _intent_key(other_intent)
    ranked = choice.pop("_ranked_candidates")
    match = next(
        (
            (index, item)
            for index, item in enumerate(ranked, start=1)
            if _intent_key_dict(item["intent"]) == other_key
        ),
        None,
    )
    if match is None:
        raise ModelValidationError("checkpoint candidate sets do not match")
    rank, item = match
    choice["other_model_intent_rank"] = rank
    choice["other_model_intent_probability"] = float(item["policy_probability"])
    choice["other_model_intent_actor_q"] = float(item["actor_q"])
    choice["selected_minus_other_actor_q"] = (
        choice["selected_actor_q"] - float(item["actor_q"])
    )


def _state_summary(state: Any, roles_by_player: dict[str, str]) -> dict[str, Any]:
    return {
        "round_number": state.round_number,
        "step": state.step,
        "phase": state.phase,
        "player_order": list(state.player_order),
        "current_market": [plant.to_dict() for plant in state.current_market],
        "future_market": [plant.to_dict() for plant in state.future_market],
        "auction_state": (
            state.auction_state.to_dict() if state.auction_state is not None else None
        ),
        "last_powered_cities": dict(state.last_powered_cities),
        "players": [
            {
                "player_id": player.player_id,
                "duel_role": roles_by_player[player.player_id],
                "elektro": player.elektro,
                "connected_cities": list(player.network_city_ids),
                "power_plants": [plant.to_dict() for plant in player.power_plants],
                "resources": player.resource_storage.to_dict(),
                "turn_order_position": player.turn_order_position,
            }
            for player in state.players
        ],
    }


def _compare_terminal_signature(paired: dict[str, Any]) -> int:
    candidate = paired["candidate_action"]["actor_standing"]
    incumbent = paired["incumbent_action"]["actor_standing"]
    candidate_signature = (
        int(candidate["powered_cities"]),
        int(candidate["money"]),
        int(candidate["connected_cities"]),
    )
    incumbent_signature = (
        int(incumbent["powered_cities"]),
        int(incumbent["money"]),
        int(incumbent["connected_cities"]),
    )
    return (candidate_signature > incumbent_signature) - (
        candidate_signature < incumbent_signature
    )


def _build_report(
    *,
    candidate_path: Path,
    incumbent_path: Path,
    seeds: int,
    seed_start: int,
    max_actions: int,
    games: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    total_score = sum(float(game["score"]) for game in games)
    total_comparisons = sum(int(game["comparisons"]) for game in games)
    loss_games = [game for game in games if float(game["pairwise_score"]) < 0.5]
    disagreement_types = Counter()
    candidate_actual_types = Counter()
    for game in games:
        disagreement_types.update(game["disagreements_by_type"])
        candidate_actual_types.update(game["candidate_actual_disagreements_by_type"])
    paired_rank = {
        outcome: sum(
            int(game["paired_continuation_rank"][outcome]) for game in games
        )
        for outcome in ("harmful", "neutral", "helpful")
    }
    paired_signature = {
        outcome: sum(
            int(game["paired_continuation_terminal_signature"][outcome])
            for game in games
        )
        for outcome in ("worse", "same", "better")
    }
    return {
        "format_name": "powergrid.nn_rl_checkpoint_loss_analysis",
        "format_version": 1,
        "method": {
            "schedule": "both 2v1 compositions and all six seat placements per seed",
            "counterfactual_choice": (
                "both checkpoints score the exact same public decision state; only the "
                "controller assigned to the acting seat is applied"
            ),
            "loss_game": "candidate cross-checkpoint pairwise score below 0.5",
            "retained_logs": "full GameSession logs for loss games; compact traces for all games",
        },
        "configuration": {
            "candidate_checkpoint": str(candidate_path),
            "candidate_checkpoint_sha256": sha256_file(candidate_path),
            "incumbent_checkpoint": str(incumbent_path),
            "incumbent_checkpoint_sha256": sha256_file(incumbent_path),
            "seeds": seeds,
            "seed_start": seed_start,
            "games": len(games),
            "max_actions": max_actions,
        },
        "overall": {
            "pairwise_score": total_score / total_comparisons,
            "seat_pair_comparisons": total_comparisons,
            "wins": sum(int(game["wins"]) for game in games),
            "draws": sum(int(game["draws"]) for game in games),
            "losses": sum(int(game["losses"]) for game in games),
            "loss_games": len(loss_games),
            "decisions": sum(int(game["decisions"]) for game in games),
            "disagreements": sum(int(game["disagreements"]) for game in games),
            "disagreements_by_type": dict(sorted(disagreement_types.items())),
            "candidate_actual_disagreements": sum(candidate_actual_types.values()),
            "candidate_actual_disagreements_by_type": dict(
                sorted(candidate_actual_types.items())
            ),
            "candidate_policy_chose_lower_own_q": sum(
                int(game["candidate_policy_chose_lower_own_q"])
                for game in games
            ),
            "paired_continuation_rank": paired_rank,
            "paired_continuation_terminal_signature": paired_signature,
        },
        "elapsed_seconds": elapsed_seconds,
        "games": games,
    }


def _standing_signature(standing: Any) -> tuple[int, int, int]:
    return (
        int(standing.powered_cities),
        int(standing.money),
        int(standing.connected_cities),
    )


def _intent_key(intent: GuiIntent) -> str:
    return _intent_key_dict(intent.to_dict())


def _intent_key_dict(intent: dict[str, Any]) -> str:
    return json.dumps(intent, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small balanced NN RL checkpoint duel, log both models' choices "
            "at every state, and retain full logs for candidate losses."
        )
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=75001)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        default="artifacts/validation/ai_nn_rl_checkpoint_loss_analysis",
    )
    args = parser.parse_args(argv)
    report = analyze_checkpoint_losses(
        args.candidate,
        args.incumbent,
        seeds=args.seeds,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        max_actions=args.max_actions,
    )
    overall = report["overall"]
    print(
        f"score={overall['pairwise_score']:.4f} "
        f"W/D/L={overall['wins']}/{overall['draws']}/{overall['losses']} "
        f"loss_games={overall['loss_games']} disagreements={overall['disagreements']}"
    )
    print(f"Wrote {Path(args.output_dir) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
