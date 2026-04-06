from __future__ import annotations

import argparse
from dataclasses import replace

from powergrid.model import (
    GameConfig,
    ModelValidationError,
    PowerPlantCard,
    advance_phase,
    create_initial_state,
    list_auctionable_plants,
    make_default_seat_configs,
    pass_auction,
    raise_bid,
    replace_plant_if_needed,
    start_auction,
)
from powergrid.rules_data import load_power_plants


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive manual test for the Power Grid auction phase.")
    parser.add_argument("--map", dest="map_id", default="germany")
    parser.add_argument("--players", type=int, choices=range(3, 7))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--step", type=int, choices=(1, 2, 3))
    parser.add_argument(
        "--first-round",
        choices=("yes", "no"),
        help="Use round 1 mandatory-purchase behavior or a later round.",
    )
    parser.add_argument(
        "--step3-on-next-refill",
        action="store_true",
        help="Force the next auction-time market refill to discover the Step 3 card.",
    )
    args = parser.parse_args()

    players = args.players if args.players is not None else prompt_int("Player count", default=3, allowed=range(3, 7))
    seed = args.seed if args.seed is not None else prompt_int("Seed", default=7)
    step = args.step if args.step is not None else prompt_int("Step", default=1, allowed=(1, 2, 3))
    first_round = (
        parse_yes_no(args.first_round)
        if args.first_round is not None
        else prompt_yes_no("First round", default=True)
    )

    state = build_manual_auction_state(
        map_id=args.map_id,
        player_count=players,
        seed=seed,
        step=step,
        first_round=first_round,
        step3_on_next_refill=args.step3_on_next_refill,
    )

    print("Auction Phase Manual Test")
    print(f"Map: {state.game_map.name} ({state.game_map.id})")
    print(f"Players: {len(state.players)}")
    print(f"Seed: {seed}")
    print(f"Round: {state.round_number}")
    print(f"Step: {state.step}")
    print()
    print_state(state)

    while True:
        if state.phase != "auction":
            print()
            print(f"Auction phase finished. Next phase: {state.phase}")
            print_state(state)
            return

        print()
        print(turn_prompt(state))
        raw = input("> ").strip()
        if not raw:
            continue

        lowered = raw.lower()
        if lowered == "quit":
            print("Manual auction test stopped.")
            return
        if lowered == "help":
            print_help(state)
            continue
        if lowered == "status":
            print_state(state)
            continue
        if lowered == "options":
            print_options(state)
            continue

        try:
            next_state = apply_command(state, raw)
        except ModelValidationError as exc:
            print(f"Rejected: {exc}")
            continue
        except ValueError as exc:
            print(f"Rejected: {exc}")
            continue

        state = next_state
        print("Accepted.")
        print_state(state)


def build_manual_auction_state(
    map_id: str,
    player_count: int,
    seed: int,
    step: int,
    first_round: bool,
    step3_on_next_refill: bool,
):
    config = GameConfig(
        map_id=map_id,
        players=make_default_seat_configs(player_count),
        seed=seed,
    )
    state = advance_phase(create_initial_state(config))
    state = replace(
        state,
        round_number=1 if first_round else 2,
        step=step,
    )
    if step3_on_next_refill:
        definitions = {definition.price: definition for definition in load_power_plants()}
        state = replace(
            state,
            power_plant_draw_stack=(),
            power_plant_bottom_stack=tuple(
                PowerPlantCard.from_definition(definitions[price]) for price in (25, 31, 33)
            ),
        )
    return state


def apply_command(state, raw: str):
    tokens = raw.split()
    command = tokens[0].lower()

    if state.pending_decision is not None:
        if command != "discard" or len(tokens) != 2:
            raise ValueError("expected: discard <plant_price>")
        return replace_plant_if_needed(state, state.pending_decision.player_id, int(tokens[1]))

    auction_state = state.auction_state
    if auction_state is None:
        raise ValueError("auction state is missing")

    if auction_state.has_active_auction:
        acting_player = auction_state.next_bidder_id
        if command == "bid":
            if len(tokens) != 2:
                raise ValueError("expected: bid <amount>")
            return raise_bid(state, str(acting_player), int(tokens[1]))
        if command == "pass":
            if len(tokens) != 1:
                raise ValueError("expected: pass")
            return pass_auction(state, str(acting_player))
        raise ValueError("allowed commands: bid <amount>, pass, options, status, help, quit")

    acting_player = auction_state.current_chooser_id
    if command == "start":
        if len(tokens) != 3:
            raise ValueError("expected: start <plant_price> <bid>")
        return start_auction(state, str(acting_player), int(tokens[1]), int(tokens[2]))
    if command == "pass":
        if len(tokens) != 1:
            raise ValueError("expected: pass")
        return pass_auction(state, str(acting_player))
    raise ValueError("allowed commands: start <plant_price> <bid>, pass, options, status, help, quit")


def turn_prompt(state) -> str:
    if state.pending_decision is not None:
        legal_discards = ", ".join(
            str(action.payload["price"]) for action in state.pending_decision.legal_actions
        )
        return (
            f"{state.pending_decision.player_id} must discard a plant. "
            f"Allowed: discard <plant_price> where plant_price is one of [{legal_discards}]"
        )

    auction_state = state.auction_state
    if auction_state is None:
        return "Auction state missing."
    if auction_state.has_active_auction:
        bidders = ", ".join(auction_state.active_bidders)
        return (
            f"Active bidder: {auction_state.next_bidder_id}. "
            f"Plant {auction_state.active_plant_price}, current bid {auction_state.current_bid}, "
            f"leader {auction_state.highest_bidder_id}, bidders [{bidders}]. "
            "Use: bid <amount>, pass, or options"
        )

    chooser = auction_state.current_chooser_id
    auctionable = ", ".join(format_plant(plant) for plant in list_auctionable_plants(state))
    pass_note = "pass allowed" if state.round_number > 1 else "pass not allowed in round 1"
    discount_note = (
        f"$1 token on plant {auction_state.discount_token_plant_price}"
        if auction_state.discount_token_plant_price is not None
        else "$1 token is off the market for this round"
    )
    return (
        f"Chooser: {chooser}. Auctionable plants [{auctionable}]. "
        f"{discount_note}. Use: start <plant_price> <bid>, pass, or options ({pass_note})"
    )


def print_state(state) -> None:
    print(f"Phase={state.phase} Round={state.round_number} Step={state.step}")
    print("Turn order: " + ", ".join(state.player_order))
    print("Players:")
    for player in sorted(state.players, key=lambda item: item.turn_order_position):
        plants = ", ".join(str(plant.price) for plant in player.power_plants) or "-"
        print(
            f"  {player.turn_order_position}. {player.player_id} elektro={player.elektro} "
            f"plants=[{plants}] cities={player.connected_city_count}"
        )
    print("Current market: " + ", ".join(format_plant(plant) for plant in state.current_market))
    print(
        "Future market: "
        + (", ".join(format_plant(plant) for plant in state.future_market) if state.future_market else "(none)")
    )
    stack_preview = ", ".join(format_plant(plant) for plant in state.power_plant_draw_stack[:8]) or "-"
    print(f"Draw stack preview: {stack_preview}")
    bottom_preview = ", ".join(format_plant(plant) for plant in state.power_plant_bottom_stack) or "-"
    print(f"Bottom stack: {bottom_preview}")
    print(
        "Step 3 status: "
        f"card_pending={state.step_3_card_pending} "
        f"auction_pending={state.auction_step_3_pending}"
    )
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


def print_help(state) -> None:
    print("Utility commands: options, status, help, quit")
    if state.pending_decision is not None:
        print("Decision command: discard <plant_price>")
        return
    if state.auction_state is not None and state.auction_state.has_active_auction:
        print("Auction commands: bid <amount>, pass")
        return
    print("Chooser commands: start <plant_price> <bid>, pass")


def print_options(state) -> None:
    if state.pending_decision is not None:
        legal_discards = ", ".join(
            str(action.payload["price"]) for action in state.pending_decision.legal_actions
        )
        print(
            f"Discard options for {state.pending_decision.player_id}: "
            f"[{legal_discards}]"
        )
        return

    auction_state = state.auction_state
    if auction_state is None:
        print("Auction state is missing.")
        return

    if auction_state.has_active_auction:
        acting_player = str(auction_state.next_bidder_id)
        player = next(player for player in state.players if player.player_id == acting_player)
        minimum_raise = int(auction_state.current_bid) + 1
        print(f"Bid options for {acting_player}:")
        print(f"  pass")
        if minimum_raise <= player.elektro:
            print(f"  bid <amount> where {minimum_raise} <= amount <= {player.elektro}")
        else:
            print(
                f"  no legal raise: current bid is {auction_state.current_bid}, "
                f"but {acting_player} only has {player.elektro} Elektro"
            )
        return

    chooser = str(auction_state.current_chooser_id)
    player = next(player for player in state.players if player.player_id == chooser)
    print(f"Auction start options for {chooser}:")
    for plant in list_auctionable_plants(state):
        minimum_bid = 1 if auction_state.discount_token_plant_price == plant.price else plant.price
        if minimum_bid <= player.elektro:
            print(
                f"  start {plant.price} <bid> where {minimum_bid} <= bid <= {player.elektro}"
            )
        else:
            print(
                f"  plant {plant.price} unavailable to start: minimum bid {minimum_bid}, "
                f"but {chooser} only has {player.elektro} Elektro"
            )
    if state.round_number > 1:
        print("  pass")
    else:
        print("  pass unavailable in round 1")


def format_plant(plant) -> str:
    return "STEP3" if getattr(plant, "is_step_3_placeholder", False) else str(plant.price)


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


def prompt_yes_no(label: str, default: bool) -> bool:
    default_label = "yes" if default else "no"
    while True:
        raw = input(f"{label} [{default_label}]: ").strip()
        if not raw:
            return default
        try:
            return parse_yes_no(raw)
        except ValueError as exc:
            print(exc)


def parse_yes_no(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"y", "yes", "true", "1"}:
        return True
    if lowered in {"n", "no", "false", "0"}:
        return False
    raise ValueError("Please answer yes or no.")


if __name__ == "__main__":
    main()
