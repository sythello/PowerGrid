from __future__ import annotations

import argparse

from powergrid.cli import render_game_state, render_round_summary
from powergrid.model import GameConfig, ModelValidationError, PlantRunPlan, SeatConfig
from powergrid.scenarios import SCENARIO_NAMES
from powergrid.session import GameSession, GuiIntent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive manual test for the frontend-neutral Power Grid session layer."
    )
    parser.add_argument("--scenario", choices=SCENARIO_NAMES)
    parser.add_argument("--map", dest="map_id", default="germany")
    parser.add_argument("--players", type=int, choices=range(3, 7), default=3)
    parser.add_argument("--ai-players", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.scenario:
        session = GameSession.from_scenario(args.scenario, seed=args.seed)
        print(f"Loaded scenario: {args.scenario}")
    else:
        seats = tuple(
            SeatConfig(
                player_id=f"p{index + 1}",
                name=f"Player {index + 1}",
                controller="ai" if index < args.ai_players else "human",
            )
            for index in range(args.players)
        )
        session = GameSession.new_game(
            GameConfig(
                map_id=args.map_id,
                players=seats,
                seed=args.seed,
            )
        )
        print(f"Started new game: map={args.map_id} players={args.players} seed={args.seed}")
    print()
    session.advance_until_blocked()

    while True:
        snapshot = session.snapshot()
        print_snapshot(snapshot)
        if snapshot.winner_result is not None:
            print("Winner: " + ", ".join(snapshot.winner_result.winner_ids))
            return
        if snapshot.active_request is None:
            print("Session is not waiting on a player action.")
            return
        raw = input("> ").strip()
        if not raw:
            continue
        if raw.lower() == "quit":
            return
        if raw.lower() == "status":
            continue
        if raw.lower() == "events":
            print_events(snapshot)
            continue
        if raw.lower() == "help":
            print_help(snapshot.active_request.phase, snapshot.active_request.decision_type)
            continue
        try:
            intent = parse_intent(
                snapshot.active_request.player_id,
                snapshot.active_request.phase,
                raw,
            )
        except (ModelValidationError, ValueError) as exc:
            print(f"Rejected: {exc}")
            continue
        session.submit_intent(intent)
        print()


def print_snapshot(snapshot) -> None:
    print(render_game_state(snapshot.state, active_player_id=snapshot.active_request.player_id if snapshot.active_request else None))
    if snapshot.last_round_summary is not None:
        print()
        print(render_round_summary(snapshot.last_round_summary, snapshot.state))
    if snapshot.active_request is not None:
        print()
        print(
            "Active request: "
            f"{snapshot.active_request.phase} / {snapshot.active_request.decision_type} / {snapshot.active_request.player_id}"
        )


def print_events(snapshot) -> None:
    if not snapshot.event_log:
        print("No session events yet.")
        return
    print("Session events:")
    for event in snapshot.event_log[-10:]:
        location = f" phase={event.phase}" if event.phase else ""
        actor = f" player={event.player_id}" if event.player_id else ""
        print(f"- [{event.level}]{location}{actor} {event.message}")


def print_help(phase: str, decision_type: str) -> None:
    if decision_type == "discard_power_plant":
        print("Commands: discard <plant_price>, status, events, help, quit")
        return
    if decision_type == "discard_hybrid_resources":
        print("Commands: discard coal=<amount> oil=<amount>, status, events, help, quit")
        return
    if phase == "auction":
        print("Commands: start <plant_price> <bid>, bid <amount>, pass, status, events, help, quit")
        return
    if phase == "buy_resources":
        print("Commands: buy <resource> <amount>, done, status, events, help, quit")
        return
    if phase == "build_houses":
        print("Commands: quote <city_id> [city_id ...], build <city_id> [city_id ...], done, status, events, help, quit")
        return
    if phase == "bureaucracy":
        print("Commands: run <plant_price>[:resource=amount,...] ..., skip, status, events, help, quit")
        return


def parse_intent(player_id: str, phase: str, command: str) -> GuiIntent:
    tokens = command.split()
    lowered = tokens[0].lower()
    if lowered == "start" and len(tokens) == 3:
        return GuiIntent.auction_start(player_id, int(tokens[1]), int(tokens[2]))
    if lowered == "bid" and len(tokens) == 2:
        return GuiIntent.auction_bid(player_id, int(tokens[1]))
    if lowered == "pass" and len(tokens) == 1:
        return GuiIntent.auction_pass(player_id)
    if lowered == "buy" and len(tokens) == 3:
        return GuiIntent.buy_resource(player_id, tokens[1].lower(), int(tokens[2]))
    if lowered == "done" and len(tokens) == 1 and phase == "buy_resources":
        return GuiIntent.finish_buying(player_id)
    if lowered == "done" and len(tokens) == 1 and phase == "build_houses":
        return GuiIntent.finish_building(player_id)
    if lowered == "quote" and len(tokens) >= 2:
        return GuiIntent.quote_build(player_id, tokens[1:])
    if lowered == "build" and len(tokens) >= 2:
        return GuiIntent.commit_build(player_id, tokens[1:])
    if lowered == "skip" and len(tokens) == 1:
        return GuiIntent.skip_bureaucracy(player_id)
    if lowered == "run" and len(tokens) >= 2:
        return GuiIntent.run_plants(player_id, _parse_run_plans(tokens[1:]))
    if lowered == "discard" and len(tokens) == 2 and "=" not in tokens[1]:
        return GuiIntent.discard_plant(player_id, int(tokens[1]))
    if lowered == "discard" and len(tokens) >= 2:
        mix = _parse_named_mix(tokens[1:])
        return GuiIntent.discard_hybrid_resources(
            player_id,
            coal=int(mix.get("coal", 0)),
            oil=int(mix.get("oil", 0)),
        )
    raise ValueError(f"unsupported command {command!r}")


def _parse_named_mix(tokens: list[str]) -> dict[str, int]:
    mix: dict[str, int] = {}
    for token in tokens:
        name, amount = token.split("=", 1)
        mix[name.strip().lower()] = int(amount)
    return mix


def _parse_run_plans(tokens: list[str]) -> tuple[PlantRunPlan, ...]:
    plans = []
    for token in tokens:
        if ":" not in token:
            plans.append(PlantRunPlan(int(token), {}))
            continue
        plant_text, mix_text = token.split(":", 1)
        mix = {}
        for entry in mix_text.split(","):
            resource, amount = entry.split("=", 1)
            mix[resource.strip().lower()] = int(amount)
        plans.append(PlantRunPlan(int(plant_text), mix))
    return tuple(plans)


if __name__ == "__main__":
    main()
