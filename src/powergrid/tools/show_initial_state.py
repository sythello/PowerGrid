from __future__ import annotations

import argparse

from powergrid.model import advance_phase, GameConfig, initialize_game, make_default_seat_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a seeded Power Grid initial state.")
    parser.add_argument("--map", dest="map_id", default="germany")
    parser.add_argument("--players", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ai-players", type=int, default=0)
    parser.add_argument("--advance-phases", type=int, default=0)
    parser.add_argument(
        "--selected-regions",
        default="",
        help="Comma-separated region ids to store on the initial state.",
    )
    args = parser.parse_args()

    selected_regions = tuple(
        region.strip() for region in args.selected_regions.split(",") if region.strip()
    )
    config = GameConfig(
        map_id=args.map_id,
        players=make_default_seat_configs(args.players, ai_players=args.ai_players),
        seed=args.seed,
        selected_regions=selected_regions,
    )
    controllers = {seat.player_id: seat.controller for seat in config.players}
    state = initialize_game(config, controllers)
    for _ in range(args.advance_phases):
        state = advance_phase(state)

    print("Initial state summary")
    print(f"Map: {state.game_map.name} ({state.game_map.id})")
    print(f"Seed: {state.config.seed}")
    print(f"Phase: {state.phase}")
    print(f"Step: {state.step}")
    print(
        "Selected regions: "
        + (", ".join(state.selected_regions) if state.selected_regions else "(none)")
    )
    print("Turn order: " + ", ".join(state.player_order))
    print("Players:")
    for player in state.players:
        print(
            f"  {player.turn_order_position}. {player.name} "
            f"[{player.player_id}] controller={player.controller} color={player.color} "
            f"elektro={player.elektro} cities={player.connected_city_count} "
            f"plants={len(player.power_plants)}"
        )
    print(
        "Current market: "
        + ", ".join(str(plant.price) for plant in state.current_market)
    )
    print(
        "Future market: "
        + ", ".join(str(plant.price) for plant in state.future_market)
    )
    print(
        "Draw stack top: "
        + ", ".join(str(plant.price) for plant in state.power_plant_draw_stack[:8])
    )
    print(f"Draw stack remaining: {len(state.power_plant_draw_stack)}")
    print(f"Step 3 pending: {state.step_3_card_pending}")
    if state.auction_state is not None:
        print(
            "Auction state: "
            f"chooser={state.auction_state.current_chooser_id or '(none)'} "
            f"discount={state.auction_state.discount_token_plant_price or '(off)'} "
            f"buyers={','.join(state.auction_state.players_with_plants) or '(none)'} "
            f"phase_passes={','.join(state.auction_state.players_passed_phase) or '(none)'}"
        )
        if state.auction_state.has_active_auction:
            print(
                "Active auction: "
                f"plant={state.auction_state.active_plant_price} "
                f"bid={state.auction_state.current_bid} "
                f"leader={state.auction_state.highest_bidder_id} "
                f"next={state.auction_state.next_bidder_id}"
            )
    if state.pending_decision is not None:
        print(
            "Pending decision: "
            f"{state.pending_decision.decision_type} for {state.pending_decision.player_id}"
        )
    print(
        "Resource market totals: "
        + ", ".join(
            f"{resource}={state.resource_market.total_in_market(resource)}"
            for resource in state.resource_market.market
        )
    )
    print(
        "Supply totals: "
        + ", ".join(
            f"{resource}={state.resource_market.supply[resource]}"
            for resource in state.resource_market.supply
        )
    )


if __name__ == "__main__":
    main()
