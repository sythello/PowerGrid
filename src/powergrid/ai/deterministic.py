from __future__ import annotations

from ..model import (
    GameState,
    ModelValidationError,
    PlantRunPlan,
    choose_plants_to_run,
    compute_powered_cities,
    legal_build_targets,
    legal_resource_purchases,
)
from ..session_types import GameSnapshot, GuiIntent, TurnRequest
from .base import BaseAiController


class DeterministicAiController(BaseAiController):
    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        state = snapshot.state
        if state.pending_decision is not None:
            return _choose_pending_intent(state)
        if request.phase == "auction":
            return _choose_auction_intent(request)
        if request.phase == "buy_resources":
            return _choose_resource_intent(state, request.player_id)
        if request.phase == "build_houses":
            return _choose_build_intent(state, request.player_id)
        if request.phase == "bureaucracy":
            plans = _choose_best_generation_plans(state, request.player_id)
            if plans:
                return GuiIntent.run_plants(request.player_id, plans)
            return GuiIntent.skip_bureaucracy(request.player_id)
        raise ModelValidationError(f"unsupported request phase {request.phase!r}")


def _get_player(state: GameState, player_id: str):
    for player in state.players:
        if player.player_id == player_id:
            return player
    raise ModelValidationError(f"unknown player {player_id!r}")


def _choose_pending_intent(state: GameState) -> GuiIntent:
    assert state.pending_decision is not None
    if state.pending_decision.decision_type == "discard_power_plant":
        prices = sorted(int(action.payload["price"]) for action in state.pending_decision.legal_actions)
        return GuiIntent.discard_plant(state.pending_decision.player_id, prices[0])
    legal = sorted(
        (
            int(action.payload.get("coal", 0)),
            int(action.payload.get("oil", 0)),
        )
        for action in state.pending_decision.legal_actions
    )
    coal, oil = legal[0]
    return GuiIntent.discard_hybrid_resources(state.pending_decision.player_id, coal=coal, oil=oil)


def _choose_auction_intent(request: TurnRequest) -> GuiIntent:
    if request.decision_type == "auction_start":
        start_actions = [
            action for action in request.legal_actions if action.action_type == "auction_start"
        ]
        if not start_actions:
            return GuiIntent.auction_pass(request.player_id)
        cheapest = min(start_actions, key=lambda action: int(action.payload["plant_price"]))
        return GuiIntent.auction_start(
            request.player_id,
            plant_price=int(cheapest.payload["plant_price"]),
            bid=int(cheapest.payload["min_bid"]),
        )
    bid_action = next(
        action for action in request.legal_actions if action.action_type == "auction_bid"
    )
    min_bid = int(bid_action.payload["min_bid"])
    max_bid = int(bid_action.payload["max_bid"])
    plant_price = int(bid_action.payload["plant_price"])
    bid_cap = min(max_bid, plant_price + 2)
    if min_bid > bid_cap:
        return GuiIntent.auction_pass(request.player_id)
    return GuiIntent.auction_bid(request.player_id, min_bid)


def _resource_need_by_type(state: GameState, player_id: str) -> dict[str, int]:
    player = _get_player(state, player_id)
    stored = player.resource_storage.resource_totals()
    needed = {"coal": 0, "oil": 0, "garbage": 0, "uranium": 0}
    hybrid_cost = 0
    for plant in player.power_plants:
        if plant.is_ecological or plant.is_step_3_placeholder:
            continue
        if plant.is_hybrid:
            hybrid_cost += plant.resource_cost
            continue
        needed[plant.resource_types[0]] += plant.resource_cost
    hybrid_remaining = max(0, hybrid_cost - (stored["coal"] + stored["oil"]))
    deficits = {
        resource: max(0, needed[resource] - stored[resource])
        for resource in needed
    }
    if hybrid_remaining > 0:
        if stored["coal"] <= stored["oil"]:
            deficits["coal"] += hybrid_remaining
        else:
            deficits["oil"] += hybrid_remaining
    return deficits


def _choose_resource_intent(state: GameState, player_id: str) -> GuiIntent:
    actions = legal_resource_purchases(state, player_id)
    if not actions:
        return GuiIntent.finish_buying(player_id)
    deficits = _resource_need_by_type(state, player_id)
    candidate_actions = [
        action
        for action in actions
        if deficits.get(str(action.payload["resource"]), 0) > 0
    ]
    if not candidate_actions:
        return GuiIntent.finish_buying(player_id)
    chosen = min(
        candidate_actions,
        key=lambda action: (
            int(action.payload["unit_prices"][0]),
            str(action.payload["resource"]),
        ),
    )
    resource = str(chosen.payload["resource"])
    amount = min(
        int(chosen.payload["max_affordable_units"]),
        max(1, deficits[resource]),
    )
    return GuiIntent.buy_resource(player_id, resource=resource, amount=amount)


def _choose_build_intent(state: GameState, player_id: str) -> GuiIntent:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return GuiIntent.finish_building(player_id)
    chosen = min(
        actions,
        key=lambda action: (
            int(action.payload["total_cost"]),
            str(action.payload["city_id"]),
        ),
    )
    return GuiIntent.commit_build(player_id, [str(chosen.payload["city_id"])])


def _choose_best_generation_plans(state: GameState, player_id: str) -> tuple[PlantRunPlan, ...]:
    player = _get_player(state, player_id)
    resource_totals = player.resource_storage.resource_totals()
    plant_choices = []
    for plant in sorted(player.power_plants, key=lambda item: item.price):
        if plant.is_step_3_placeholder:
            continue
        options = [None]
        if plant.is_ecological:
            options.append(PlantRunPlan(plant.price, {}))
        elif plant.is_hybrid:
            for coal in range(plant.resource_cost + 1):
                oil = plant.resource_cost - coal
                if coal <= resource_totals["coal"] and oil <= resource_totals["oil"]:
                    options.append(PlantRunPlan(plant.price, {"coal": coal, "oil": oil}))
        else:
            resource = plant.resource_types[0]
            if plant.resource_cost <= resource_totals[resource]:
                options.append(PlantRunPlan(plant.price, {resource: plant.resource_cost}))
        plant_choices.append((plant, tuple(options)))

    best: tuple[PlantRunPlan, ...] = ()
    best_signature = (-1, 999999, ())

    def backtrack(
        index: int,
        remaining: dict[str, int],
        selected: list[PlantRunPlan],
    ) -> None:
        nonlocal best
        nonlocal best_signature
        if index >= len(plant_choices):
            plans = tuple(selected)
            try:
                choose_plants_to_run(state, player_id, plans)
            except ModelValidationError:
                return
            powered = compute_powered_cities(state, player_id, plans)
            spent = sum(sum(plan.resource_mix.values()) for plan in plans)
            signature = (
                powered,
                -spent,
                tuple(plan.plant_price for plan in plans),
            )
            if signature > best_signature:
                best_signature = signature
                best = plans
            return

        plant, options = plant_choices[index]
        for option in options:
            if option is None:
                backtrack(index + 1, dict(remaining), selected)
                continue
            next_remaining = dict(remaining)
            for resource, amount in option.resource_mix.items():
                if next_remaining[resource] < amount:
                    break
                next_remaining[resource] -= amount
            else:
                selected.append(option)
                backtrack(index + 1, next_remaining, selected)
                selected.pop()

    backtrack(0, dict(resource_totals), [])
    return best
