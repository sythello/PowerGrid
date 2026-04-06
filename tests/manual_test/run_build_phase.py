from __future__ import annotations

import argparse
from dataclasses import replace

from powergrid.model import (
    apply_builds,
    build_city,
    GameConfig,
    legal_build_targets,
    make_default_seat_configs,
    ModelValidationError,
    create_initial_state,
    GameState,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive manual test for Power Grid network expansion and building.")
    parser.add_argument("--map", dest="map_id", default="test", choices=("test", "germany"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--step", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()

    state = build_manual_build_state(args.map_id, args.seed, args.step)
    build_order = tuple(reversed(state.player_order))
    active_index = 0

    print("Build Phase Manual Test")
    print(f"Map: {state.game_map.name} ({state.game_map.id})")
    print(f"Seed: {args.seed}")
    print(f"Step: {state.step}")
    print("Build order: " + ", ".join(build_order))
    print()
    print_state(state, build_order, active_index)

    while True:
        print()
        print(turn_prompt(build_order[active_index]))
        raw = input("> ").strip()
        if not raw:
            continue

        lowered = raw.lower()
        if lowered == "quit":
            print("Manual build-phase test stopped.")
            return
        if lowered == "help":
            print_help()
            continue
        if lowered == "status":
            print_state(state, build_order, active_index)
            continue
        if lowered == "options":
            print_options(state, build_order[active_index])
            continue
        if lowered.startswith("quote "):
            quote_build(state, build_order[active_index], raw.split()[1:])
            continue
        if lowered == "done":
            if active_index + 1 < len(build_order):
                active_index += 1
                print(f"Moved to next builder: {build_order[active_index]}")
            else:
                print("All builders have acted for this manual test.")
            continue

        try:
            state = apply_build_command(state, build_order[active_index], raw)
        except (ValueError, ModelValidationError) as exc:
            print(f"Rejected: {exc}")
            continue

        print("Accepted.")
        print_state(state, build_order, active_index)


def build_manual_build_state(map_id: str, seed: int, step: int) -> GameState:
    selected_regions = ("alpha", "beta", "gamma") if map_id == "test" else ("black", "blue", "magenta")
    base_state = create_initial_state(
        GameConfig(
            map_id=map_id,
            players=make_default_seat_configs(3),
            seed=seed,
            selected_regions=selected_regions,
        )
    )

    if map_id == "test":
        assignments = {
            "p1": ("amber_falls",),
            "p2": ("brass_harbor",),
            "p3": (),
        }
    else:
        assignments = {
            "p1": ("bremen",),
            "p2": ("dortmund",),
            "p3": ("berlin",),
        }

    updated_players = []
    for player in base_state.players:
        owned_cities = assignments[player.player_id]
        updated_players.append(
            replace(
                player,
                elektro=80,
                houses_in_supply=22 - len(owned_cities),
                network_city_ids=owned_cities,
            )
        )

    return replace(
        base_state,
        players=tuple(updated_players),
        phase="build_houses",
        step=step,
        auction_state=None,
    )


def apply_build_command(state: GameState, player_id: str, raw: str) -> GameState:
    tokens = raw.split()
    if tokens[0].lower() != "build" or len(tokens) < 2:
        raise ValueError("expected: build <city_id> [city_id ...]")
    city_ids = tuple(tokens[1:])
    if len(city_ids) == 1:
        return build_city(state, player_id, city_ids[0])
    return apply_builds(state, player_id, city_ids)


def turn_prompt(active_player: str) -> str:
    return (
        f"Active builder: {active_player}. "
        "Use: options, quote <city_id> [city_id ...], build <city_id> [city_id ...], done, status, help, quit"
    )


def print_state(state: GameState, build_order: tuple[str, ...], active_index: int) -> None:
    print(f"Phase={state.phase} Step={state.step}")
    print("Selected regions: " + ", ".join(state.selected_regions))
    print("Turn order: " + ", ".join(state.player_order))
    print("Build order: " + ", ".join(build_order))
    print(f"Active builder: {build_order[active_index]}")
    print("Players:")
    for player in sorted(state.players, key=lambda item: item.turn_order_position):
        cities = ", ".join(player.network_city_ids) if player.network_city_ids else "-"
        print(
            f"  {player.turn_order_position}. {player.player_id} elektro={player.elektro} "
            f"houses_left={player.houses_in_supply} cities=[{cities}]"
        )


def print_options(state: GameState, player_id: str) -> None:
    actions = legal_build_targets(state, player_id)
    if not actions:
        print(f"No legal single-city builds for {player_id}.")
        return
    print(f"Legal single-city builds for {player_id}:")
    for action in sorted(actions, key=lambda item: (item.payload["total_cost"], item.payload["city_id"])):
        payload = action.payload
        print(
            f"  {payload['city_id']} ({payload['city_name']}): "
            f"connection={payload['connection_cost']} build={payload['build_cost']} total={payload['total_cost']}"
        )


def quote_build(state: GameState, player_id: str, city_ids: list[str]) -> None:
    if not city_ids:
        print("Rejected: expected at least one city id after quote")
        return
    simulated = GameState.from_dict(state.to_dict())
    player_before = next(player for player in simulated.players if player.player_id == player_id)
    try:
        simulated = apply_builds(simulated, player_id, tuple(city_ids))
    except ModelValidationError as exc:
        print(f"Rejected: {exc}")
        return
    player_after = next(player for player in simulated.players if player.player_id == player_id)
    print(
        f"Quote for {player_id}: cities={', '.join(city_ids)} "
        f"cost={player_before.elektro - player_after.elektro} "
        f"new_network={', '.join(player_after.network_city_ids)}"
    )


def print_help() -> None:
    print("Commands:")
    print("  options")
    print("  quote <city_id> [city_id ...]")
    print("  build <city_id> [city_id ...]")
    print("  done")
    print("  status")
    print("  help")
    print("  quit")


if __name__ == "__main__":
    main()
