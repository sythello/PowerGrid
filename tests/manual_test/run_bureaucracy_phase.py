from __future__ import annotations

import argparse
from dataclasses import replace

from powergrid.model import (
    BureaucracySummary,
    choose_plants_to_run,
    compute_powered_cities,
    create_initial_state,
    GameConfig,
    GameState,
    make_default_seat_configs,
    ModelValidationError,
    pay_income,
    PlantRunPlan,
    PlayerState,
    PowerPlantCard,
    resolve_bureaucracy,
    ResourceStorage,
)
from powergrid.rules_data import load_power_plants


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive manual test for Power Grid Bureaucracy, step changes, and winner resolution."
    )
    parser.add_argument("--scenario", choices=("normal", "step2", "step3", "endgame"), default="normal")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    state = build_manual_bureaucracy_state(args.scenario, args.seed)
    choices: dict[str, tuple[PlantRunPlan, ...]] = {}
    active_index = 0

    print("Bureaucracy Phase Manual Test")
    print(f"Scenario: {args.scenario}")
    print(f"Map: {state.game_map.name} ({state.game_map.id})")
    print(f"Seed: {args.seed}")
    print()
    print_state(state, choices, active_index)

    while True:
        print()
        print(turn_prompt(state.player_order, active_index))
        raw = input("> ").strip()
        if not raw:
            continue

        lowered = raw.lower()
        if lowered == "quit":
            print("Manual bureaucracy test stopped.")
            return
        if lowered == "help":
            print_help()
            continue
        if lowered == "status":
            print_state(state, choices, active_index)
            continue
        if lowered == "options":
            print_options(state, state.player_order[active_index])
            continue
        if lowered == "resolve":
            if len(choices) != len(state.player_order):
                print("Rejected: every player must choose `run ...` or `skip` before resolving.")
                continue
            resolved_state, summary = resolve_bureaucracy(state, generation_choices=choices)
            print("Resolved bureaucracy.")
            print_summary(resolved_state, summary)
            return
        if lowered == "done":
            active_player = state.player_order[active_index]
            if active_player not in choices:
                print("Rejected: choose `run ...` or `skip` first.")
                continue
            if active_index + 1 < len(state.player_order):
                active_index += 1
                print(f"Moved to next player: {state.player_order[active_index]}")
            else:
                print("All players have entered their choices. Use `resolve` to apply the phase.")
            continue
        if lowered == "skip":
            active_player = state.player_order[active_index]
            choices[active_player] = ()
            print(f"{active_player} will power 0 cities and receive {pay_income(state.rules, 0)} Elektro.")
            continue
        if lowered.startswith("run "):
            active_player = state.player_order[active_index]
            try:
                selection = parse_run_selection(raw.split()[1:])
                plans = choose_plants_to_run(state, active_player, selection)
                powered = compute_powered_cities(state, active_player, plans)
            except (ValueError, ModelValidationError) as exc:
                print(f"Rejected: {exc}")
                continue
            choices[active_player] = plans
            print(
                f"Accepted for {active_player}: "
                + (", ".join(format_plan(plan) for plan in plans) if plans else "(no plants)")
            )
            print(f"Projected powered cities: {powered}")
            print(f"Projected income: {pay_income(state.rules, powered)}")
            continue

        print("Rejected: expected one of options, run, skip, done, status, resolve, help, quit")


def build_manual_bureaucracy_state(scenario: str, seed: int) -> GameState:
    base_state = create_initial_state(
        GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=seed)
    )
    definitions = {definition.price: definition for definition in load_power_plants()}
    available_city_ids = [
        city.id for city in base_state.game_map.cities if city.region in base_state.selected_regions
    ]

    scenario_specs = {
        "normal": {
            "step": 1,
            "players": {
                "p1": {"cities": 3, "plants": (10, 13), "storage": {"coal": 2}, "elektro": 40},
                "p2": {"cities": 4, "plants": (11, 18), "storage": {"uranium": 1}, "elektro": 35},
                "p3": {"cities": 2, "plants": (6, 15), "storage": {"garbage": 1, "coal": 2}, "elektro": 30},
            },
            "draw_stack": tuple(plant.price for plant in base_state.power_plant_draw_stack),
            "bottom_stack": (),
            "market_adjustment": None,
        },
        "step2": {
            "step": 1,
            "players": {
                "p1": {"cities": 7, "plants": (13,), "storage": {}, "elektro": 40},
                "p2": {"cities": 4, "plants": (18,), "storage": {}, "elektro": 35},
                "p3": {"cities": 2, "plants": (22,), "storage": {}, "elektro": 30},
            },
            "draw_stack": tuple(plant.price for plant in base_state.power_plant_draw_stack),
            "bottom_stack": (),
            "market_adjustment": None,
        },
        "step3": {
            "step": 2,
            "players": {
                "p1": {"cities": 3, "plants": (13,), "storage": {}, "elektro": 40},
                "p2": {"cities": 4, "plants": (18,), "storage": {}, "elektro": 35},
                "p3": {"cities": 2, "plants": (22,), "storage": {}, "elektro": 30},
            },
            "draw_stack": (),
            "bottom_stack": (25, 31, 33),
            "market_adjustment": ("oil", 4),
        },
        "endgame": {
            "step": 2,
            "players": {
                "p1": {"cities": 17, "plants": (25, 13), "storage": {"coal": 2}, "elektro": 40},
                "p2": {"cities": 15, "plants": (20, 23), "storage": {"coal": 3, "uranium": 1}, "elektro": 20},
                "p3": {"cities": 10, "plants": (18, 22), "storage": {}, "elektro": 50},
            },
            "draw_stack": tuple(plant.price for plant in base_state.power_plant_draw_stack),
            "bottom_stack": (),
            "market_adjustment": None,
        },
    }
    spec = scenario_specs[scenario]

    updated_players = []
    for player in base_state.players:
        player_spec = spec["players"][player.player_id]
        city_count = int(player_spec["cities"])
        updated_players.append(
            build_player(
                player,
                city_ids=tuple(available_city_ids[:city_count]),
                plant_prices=player_spec["plants"],
                storage=player_spec["storage"],
                elektro=int(player_spec["elektro"]),
                definitions=definitions,
            )
        )

    draw_stack = tuple(
        PowerPlantCard.from_definition(definitions[price]) for price in spec["draw_stack"]
    )
    bottom_stack = tuple(
        PowerPlantCard.from_definition(definitions[price]) for price in spec["bottom_stack"]
    )
    state = replace(
        base_state,
        players=tuple(updated_players),
        phase="bureaucracy",
        step=int(spec["step"]),
        round_number=2,
        auction_state=None,
        pending_decision=None,
        power_plant_draw_stack=draw_stack,
        power_plant_bottom_stack=bottom_stack,
        last_powered_cities={},
        last_income_paid={},
    )
    if spec["market_adjustment"] is not None:
        resource, amount = spec["market_adjustment"]
        state = replace(state, resource_market=state.resource_market.remove_from_market(resource, amount))
    return state


def build_player(
    player: PlayerState,
    *,
    city_ids: tuple[str, ...],
    plant_prices: tuple[int, ...],
    storage: dict[str, int],
    elektro: int,
    definitions: dict[int, object],
) -> PlayerState:
    plants = tuple(PowerPlantCard.from_definition(definitions[price]) for price in plant_prices)
    return replace(
        player,
        elektro=elektro,
        houses_in_supply=22 - len(city_ids),
        network_city_ids=city_ids,
        power_plants=plants,
        resource_storage=ResourceStorage.from_dict(storage),
    )


def parse_run_selection(tokens: list[str]) -> tuple[PlantRunPlan, ...]:
    plans = []
    for token in tokens:
        if ":" not in token:
            plans.append(PlantRunPlan(plant_price=int(token), resource_mix={}))
            continue
        plant_text, mix_text = token.split(":", 1)
        mix: dict[str, int] = {}
        for entry in mix_text.split(","):
            resource, amount = entry.split("=", 1)
            mix[resource.strip()] = int(amount)
        plans.append(PlantRunPlan(plant_price=int(plant_text), resource_mix=mix))
    return tuple(plans)


def turn_prompt(player_order: tuple[str, ...], active_index: int) -> str:
    return (
        f"Active player: {player_order[active_index]}. "
        "Use: options, run <plant_price>[:resource=amount,...] ..., skip, done, status, resolve, help, quit"
    )


def print_state(
    state: GameState,
    choices: dict[str, tuple[PlantRunPlan, ...]],
    active_index: int,
) -> None:
    print(f"Phase={state.phase} Step={state.step} Round={state.round_number}")
    print("Player order: " + ", ".join(state.player_order))
    print(
        "Current market: "
        + ", ".join(str(plant.price) for plant in state.current_market)
    )
    print(
        "Future market: "
        + (", ".join(str(plant.price) for plant in state.future_market) if state.future_market else "(none)")
    )
    print(
        "Draw stack top: "
        + (", ".join(str(plant.price) for plant in state.power_plant_draw_stack[:6]) if state.power_plant_draw_stack else "(empty)")
    )
    print(
        "Bottom stack: "
        + (", ".join(str(plant.price) for plant in state.power_plant_bottom_stack) if state.power_plant_bottom_stack else "(empty)")
    )
    print("Players:")
    for player in sorted(state.players, key=lambda item: item.turn_order_position):
        plants = ", ".join(str(plant.price) for plant in player.power_plants) or "-"
        cities = len(player.network_city_ids)
        chosen = choices.get(player.player_id)
        choice_text = ", ".join(format_plan(plan) for plan in chosen) if chosen else "-"
        print(
            f"  {player.turn_order_position}. {player.player_id} elektro={player.elektro} "
            f"cities={cities} plants=[{plants}] storage=[{format_storage(player.resource_storage)}] "
            f"choice=[{choice_text}]"
        )
    print(f"Active player: {state.player_order[active_index]}")


def print_options(state: GameState, player_id: str) -> None:
    player = next(player for player in state.players if player.player_id == player_id)
    if not player.power_plants:
        print(f"{player_id} owns no power plants.")
        return
    print(f"Power plants for {player_id}:")
    for plant in player.power_plants:
        if plant.is_ecological:
            requirement = "eco"
        elif plant.is_hybrid:
            requirement = f"{plant.resource_cost} coal/oil in any mix"
        else:
            requirement = f"{plant.resource_cost} {plant.resource_types[0]}"
        print(
            f"  {plant.price}: powers {plant.output_cities} cities, requires {requirement}"
        )
    print("Examples:")
    print("  run 13")
    print("  run 10 13")
    print("  run 5:coal=1,oil=1")


def print_summary(state: GameState, summary: BureaucracySummary) -> None:
    print(f"Step after resolution: {state.step}")
    print(f"Phase after resolution: {state.phase}")
    print("Powered cities: " + ", ".join(f"{player}={count}" for player, count in summary.powered_cities.items()))
    print("Income paid: " + ", ".join(f"{player}={count}" for player, count in summary.income_paid.items()))
    print(f"Triggered Step 2: {summary.triggered_step_2}")
    print(f"Triggered Step 3: {summary.triggered_step_3}")
    print(f"Refill step used: {summary.refill_step_used}")
    print(f"Game end triggered: {summary.game_end_triggered}")
    print(
        "Current market: "
        + ", ".join(str(plant.price) for plant in state.current_market)
    )
    print(
        "Future market: "
        + (", ".join(str(plant.price) for plant in state.future_market) if state.future_market else "(none)")
    )
    print(
        "Draw stack top: "
        + (", ".join(str(plant.price) for plant in state.power_plant_draw_stack[:6]) if state.power_plant_draw_stack else "(empty)")
    )
    print(
        "Bottom stack: "
        + (", ".join(str(plant.price) for plant in state.power_plant_bottom_stack) if state.power_plant_bottom_stack else "(empty)")
    )
    for player in sorted(state.players, key=lambda item: item.turn_order_position):
        print(
            f"  {player.player_id}: elektro={player.elektro} "
            f"cities={len(player.network_city_ids)} storage=[{format_storage(player.resource_storage)}]"
        )
    if summary.winner_result is not None:
        print("Winner: " + ", ".join(summary.winner_result.winner_ids))


def format_plan(plan: PlantRunPlan) -> str:
    if not plan.resource_mix:
        return str(plan.plant_price)
    return (
        f"{plan.plant_price}:"
        + ",".join(f"{resource}={amount}" for resource, amount in plan.resource_mix.items())
    )


def format_storage(storage: ResourceStorage) -> str:
    return (
        f"coal={storage.coal} oil={storage.oil} hybrid_coal={storage.hybrid_coal} "
        f"hybrid_oil={storage.hybrid_oil} garbage={storage.garbage} uranium={storage.uranium}"
    )


def print_help() -> None:
    print("Commands:")
    print("  options")
    print("  run <plant_price>[:resource=amount,...] ...")
    print("  skip")
    print("  done")
    print("  status")
    print("  resolve")
    print("  help")
    print("  quit")


if __name__ == "__main__":
    main()
