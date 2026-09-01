from __future__ import annotations

import argparse
from pathlib import Path

from powergrid.ai import DeterministicAiController
from powergrid.ai.nn_rank_value.controller import (
    DEFAULT_CHECKPOINT_PATH,
    NnRankValueAiController,
)
from powergrid.model import GameConfig, SeatConfig
from powergrid.session import GameSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a full game using the NN controller.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--max-actions", type=int, default=5000)
    args = parser.parse_args()

    config = GameConfig(
        map_id="germany",
        players=(
            SeatConfig("p1", "NN", controller="ai_nn_rank_value_v1"),
            SeatConfig("p2", "Baseline 1", controller="ai_deterministic"),
            SeatConfig("p3", "Baseline 2", controller="ai_deterministic"),
        ),
        seed=args.seed,
    )
    session = GameSession.new_game(
        config,
        {
            "p1": NnRankValueAiController(args.checkpoint),
            "p2": DeterministicAiController(),
            "p3": DeterministicAiController(),
        },
    )
    actions = 0
    for actions in range(1, args.max_actions + 1):
        snapshot, applied = session.advance_one_ai_action()
        if snapshot.winner_result is not None:
            break
        assert applied, "AI game stopped before reaching a terminal state"
    else:
        raise AssertionError("AI game exceeded max-actions")
    snapshot = session.snapshot()
    assert snapshot.winner_result is not None
    errors = [entry for entry in session.game_log_entries() if entry.event_type == "intent_error"]
    nn_decisions = [
        entry
        for entry in session.game_log_entries()
        if entry.payload.get("label") == "nn_rank_value_decision"
    ]
    assert not errors
    assert nn_decisions

    print("Controller validation: PASS")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  applied AI actions: {actions}")
    print(f"  NN decisions scored: {len(nn_decisions)}")
    print(f"  winner ids: {', '.join(snapshot.winner_result.winner_ids)}")
    print("  intent errors: 0")


if __name__ == "__main__":
    main()
