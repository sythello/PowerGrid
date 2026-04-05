from __future__ import annotations

import argparse
from dataclasses import replace

from powergrid.model import (
    GameConfig,
    legal_resource_purchases,
    make_default_seat_configs,
    ModelValidationError,
    purchase_resources,
    refill_resource_market,
)
from powergrid.model import PowerPlantCard, create_initial_state
from powergrid.rules_data import load_power_plants


PLANT_ASSIGNMENTS = (
    (5, 10),
    (7, 11),
    (6, 13),
    (12, 14),
    (15, 17),
    (8, 16),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive manual test for Power Grid resource buying.")
    parser.add_argument("--map", dest="map_id", default="germany")
    parser.add_argument("--players", type=int, choices=range(3, 7))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--step", type=int, choices=(1, 2, 3))
    args = parser.parse_args()

    player_count = (
        args.players if args.players is not None else prompt_int("Player count", default=3, allowed=range(3, 7))
    )
    seed = args.seed if args.seed is not None else prompt_int("Seed", default=7)
    step = args.step if args.step is not None else prompt_int("Step", default=1, allowed=(1, 2, 3))

    state = build_manual_resource_state(args.map_id, player_count, seed, step)
    buy_order = tuple(reversed(state.player_order))
    active_index = 0

    print("Resource Buying Manual Test")
    print(f"Map: {state.game_map.name} ({state.game_map.id})")
    print(f"Players: {len(state.players)}")
    print(f"Seed: {seed}")
    print(f"Step: {state.step}")
    print("Buy order: " + ", ".join(buy_order))
    print()
    print_state(state, buy_order, active_index)

    while True:
        print()
        print(turn_prompt(state, buy_order, active_index))
        raw = input("> ").strip()
        if not raw:
            continue

        lowered = raw.lower()
        if lowered == "quit":
            print("Manual resource-buying test stopped.")
            return
        if lowered == "help":
            print_help()
            continue
        if lowered == "status":
            print_state(state, buy_order, active_index)
            continue
        if lowered == "options":
            print_options(state, buy_order[active_index])
            continue
        if lowered == "done":
            if active_index + 1 < len(buy_order):
                active_index += 1
                print(f"Moved to next buyer: {buy_order[active_index]}")
            else:
                print("All buyers have acted for this manual test.")
            continue
        if lowered == "refill":
            state = replace(
                state,
                resource_market=refill_resource_market(
                    state.resource_market,
                    state.rules,
                    step=state.step,
                    player_count=len(state.players),
                ),
            )
            print("Applied market refill.")
            print_state(state, buy_order, active_index)
            continue

        try:
            state = apply_buy_command(state, buy_order[active_index], raw)
        except (ValueError, ModelValidationError) as exc:
            print(f"Rejected: {exc}")
            continue
        print("Accepted.")
        print_state(state, buy_order, active_index)


def build_manual_resource_state(map_id: str, player_count: int, seed: int, step: int):
    config = GameConfig(map_id=map_id, players=make_default_seat_configs(player_count), seed=seed)
    state = create_initial_state(config)
    definitions = {definition.price: definition for definition in load_power_plants()}
    updated_players = []
    for index, player in enumerate(state.players):
        plants = tuple(
            PowerPlantCard.from_definition(definitions[price])
            for price in PLANT_ASSIGNMENTS[index]
        )
        updated_players.append(replace(player, power_plants=plants))
    return replace(state, players=tuple(updated_players), phase="buy_resources", step=step, auction_state=None)


def apply_buy_command(state, player_id: str, raw: str):
    tokens = raw.split()
    if tokens[0].lower() != "buy" or len(tokens) != 3:
        raise ValueError("expected: buy <resource> <amount>")
    resource = tokens[1].lower()
    amount = int(tokens[2])
    return purchase_resources(state, player_id, {resource: amount})


def turn_prompt(state, buy_order: tuple[str, ...], active_index: int) -> str:
    active_player = buy_order[active_index]
    return (
        f"Active buyer: {active_player}. "
        "Use: options, buy <resource> <amount>, done, refill, status, help, quit"
    )


def print_state(state, buy_order: tuple[str, ...], active_index: int) -> None:
    print(f"Phase={state.phase} Step={state.step}")
    print("Turn order: " + ", ".join(state.player_order))
    print("Buy order: " + ", ".join(buy_order))
    print(f"Active buyer: {buy_order[active_index]}")
    print("Players:")
    for player in sorted(state.players, key=lambda item: item.turn_order_position):
        print(
            f"  {player.turn_order_position}. {player.player_id} elektro={player.elektro} "
            f"plants=[{format_plants(player)}] storage=[{format_storage(player)}]"
        )
    print(
        "Resource market: "
        + ", ".join(
            f"{resource}={state.resource_market.total_in_market(resource)}"
            for resource in state.resource_market.market
        )
    )
    print(
        "Supply: "
        + ", ".join(
            f"{resource}={state.resource_market.supply[resource]}"
            for resource in state.resource_market.supply
        )
    )


def print_options(state, player_id: str) -> None:
    options = legal_resource_purchases(state, player_id)
    if not options:
        print(f"No legal purchases for {player_id}.")
        return
    print(f"Legal purchases for {player_id}:")
    for action in options:
        payload = action.payload
        print(
            f"  {payload['resource']}: "
            f"max_units={payload['max_units']} "
            f"max_affordable={payload['max_affordable_units']} "
            f"unit_prices={payload['unit_prices']}"
        )


def print_help() -> None:
    print("Commands:")
    print("  options")
    print("  buy <resource> <amount>")
    print("  done")
    print("  refill")
    print("  status")
    print("  help")
    print("  quit")


def format_plants(player) -> str:
    return "; ".join(str(plant.price) for plant in sorted(player.power_plants, key=lambda item: item.price))


def format_storage(player) -> str:
    storage = player.resource_storage
    parts = [
        f"coal={storage.coal}",
        f"oil={storage.oil}",
        f"hybrid_coal={storage.hybrid_coal}",
        f"hybrid_oil={storage.hybrid_oil}",
        f"garbage={storage.garbage}",
        f"uranium={storage.uranium}",
    ]
    return ", ".join(parts)


def prompt_int(label: str, default: int, allowed=None) -> int:
    allowed_set = set(allowed) if allowed is not None else None
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue
        if allowed_set is None or value in allowed_set:
            return value
        print(f"Please enter one of: {', '.join(str(item) for item in sorted(allowed_set))}")


if __name__ == "__main__":
    main()
