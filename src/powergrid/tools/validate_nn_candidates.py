from __future__ import annotations

from powergrid.ai.nn_rank_value.candidates import generate_candidate_actions
from powergrid.session import GameSession, GuiIntent


def _validate_snapshot_candidates(session: GameSession) -> int:
    snapshot = session.snapshot()
    assert snapshot.active_request is not None
    candidates = generate_candidate_actions(snapshot.active_request, snapshot)
    assert candidates, f"no candidates for {snapshot.active_request.phase}"
    for candidate in candidates:
        rollout = session.fork()
        result = rollout.submit_intent(candidate.intent, auto_advance=False)
        assert result.event_log
        last_event = result.event_log[-1]
        assert last_event.level != "error", (
            f"illegal generated candidate {candidate.intent.to_dict()}: {last_event.message}"
        )
    return len(candidates)


def _validate_cash_limited_bid() -> None:
    session = GameSession.from_scenario("opening", seed=7)
    request = session.current_request()
    assert request is not None
    start = next(action for action in request.legal_actions if action.action_type == "auction_start")
    session.submit_intent(
        GuiIntent.auction_start(request.player_id, int(start.payload["plant_price"]), 49),
        auto_advance=False,
    )
    request = session.current_request()
    assert request is not None
    session.submit_intent(GuiIntent.auction_bid(request.player_id, 50), auto_advance=False)
    snapshot = session.snapshot()
    assert snapshot.active_request is not None
    candidates = generate_candidate_actions(snapshot.active_request, snapshot)
    assert [candidate.intent.intent_type for candidate in candidates] == ["auction_pass"]


def main() -> None:
    results = {
        scenario: _validate_snapshot_candidates(GameSession.from_scenario(scenario, seed=7))
        for scenario in ("opening", "resource", "build_test", "endgame")
    }
    _validate_cash_limited_bid()
    print("Candidate validation: PASS")
    for scenario, count in results.items():
        print(f"  {scenario}: {count} legal candidates replayed")
    print("  unaffordable auction raise filtered: PASS")


if __name__ == "__main__":
    main()
