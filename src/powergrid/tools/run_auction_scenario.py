from __future__ import annotations

import argparse
from dataclasses import replace

from powergrid.model import (
    AuctionState,
    GameConfig,
    PlayerState,
    PowerPlantCard,
    advance_phase,
    create_initial_state,
    make_default_seat_configs,
    pass_auction,
    replace_plant_if_needed,
    start_auction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scripted Power Grid auction scenarios.")
    parser.add_argument(
        "--scenario",
        choices=("first-round", "replacement"),
        required=True,
        help="Scenario to run.",
    )
    args = parser.parse_args()

    if args.scenario == "first-round":
        run_first_round_scenario()
        return
    run_replacement_scenario()


def run_first_round_scenario() -> None:
    state = advance_phase(
        create_initial_state(
            GameConfig(
                map_id="germany",
                players=make_default_seat_configs(3),
                seed=7,
            )
        )
    )
    print("Scenario: first-round")
    print(f"Opening chooser: {state.auction_state.current_chooser_id}")
    print("Opening market: " + ", ".join(str(plant.price) for plant in state.current_market))

    state = start_auction(state, "p3", 6, 1)
    state = pass_auction(state, "p1")
    state = pass_auction(state, "p2")
    print(
        "After p3 purchase: "
        f"phase={state.phase} chooser={state.auction_state.current_chooser_id} "
        f"p3_plants={player_prices(state, 'p3')} p3_elektro={player_elektro(state, 'p3')}"
    )

    state = start_auction(state, "p1", 7, 7)
    state = pass_auction(state, "p2")
    print(
        "After p1 purchase: "
        f"phase={state.phase} chooser={state.auction_state.current_chooser_id} "
        f"p1_plants={player_prices(state, 'p1')} p1_elektro={player_elektro(state, 'p1')}"
    )

    state = start_auction(state, "p2", 10, 10)
    print(
        "After p2 purchase: "
        f"phase={state.phase} order={','.join(state.player_order)} "
        f"p2_plants={player_prices(state, 'p2')} p2_elektro={player_elektro(state, 'p2')}"
    )
    print("Current market: " + ", ".join(str(plant.price) for plant in state.current_market))


def run_replacement_scenario() -> None:
    state = advance_phase(
        create_initial_state(
            GameConfig(
                map_id="germany",
                players=make_default_seat_configs(3),
                seed=7,
            )
        )
    )
    player = next(existing for existing in state.players if existing.player_id == "p3")
    seeded_player = PlayerState(
        player_id=player.player_id,
        name=player.name,
        controller=player.controller,
        color=player.color,
        elektro=60,
        houses_in_supply=player.houses_in_supply,
        network_city_ids=player.network_city_ids,
        power_plants=tuple(
            PowerPlantCard.from_dict(power_plant.to_dict()) for power_plant in state.current_market[1:4]
        ),
        turn_order_position=player.turn_order_position,
    )
    state = replace(
        state,
        players=tuple(
            seeded_player if existing.player_id == "p3" else existing for existing in state.players
        ),
        round_number=2,
        auction_state=AuctionState(
            current_chooser_id="p3",
            players_passed_phase=("p1", "p2"),
        ),
    )

    print("Scenario: replacement")
    print("Starting p3 plants: " + ", ".join(str(price) for price in player_prices(state, "p3")))
    state = start_auction(state, "p3", 6, 1)
    print(
        "After 4th plant purchase: "
        f"phase={state.phase} pending={state.pending_decision.decision_type} "
        f"plants={player_prices(state, 'p3')}"
    )
    state = replace_plant_if_needed(state, "p3", 6)
    print(
        "After discard: "
        f"phase={state.phase} pending={state.pending_decision} "
        f"plants={player_prices(state, 'p3')}"
    )


def player_prices(state, player_id: str) -> tuple[int, ...]:
    player = next(player for player in state.players if player.player_id == player_id)
    return tuple(plant.price for plant in player.power_plants)


def player_elektro(state, player_id: str) -> int:
    player = next(player for player in state.players if player.player_id == player_id)
    return player.elektro


if __name__ == "__main__":
    main()
