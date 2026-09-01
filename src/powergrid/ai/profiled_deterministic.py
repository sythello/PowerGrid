from __future__ import annotations

from dataclasses import dataclass

from ..model import (
    GameState,
    ModelValidationError,
    PowerPlantCard,
    legal_build_targets,
    legal_resource_purchases,
)
from ..session_types import GameSnapshot, GuiIntent, TurnRequest
from .base import BaseAiController
from .deterministic import (
    _choose_best_generation_plans,
    _choose_pending_intent,
    _get_player,
)


RESOURCE_TYPES = ("coal", "oil", "garbage", "uranium")


@dataclass(frozen=True)
class DeterministicStrategyProfile:
    controller_name: str
    strategy_angle: str
    output_weight: float
    run_cost_weight: float
    purchase_price_weight: float
    ecological_bonus: float
    hybrid_bonus: float
    bid_markup: int
    fuel_runs: int
    cheap_fuel_buffer: int
    build_capacity_buffer: int
    cash_reserve: int
    build_priority: str
    resource_priority: str


EFFICIENCY_PROFILE = DeterministicStrategyProfile(
    controller_name="ai_deterministic_efficiency",
    strategy_angle="plant_and_fuel_efficiency",
    output_weight=6.5,
    run_cost_weight=1.75,
    purchase_price_weight=0.2,
    ecological_bonus=7.0,
    hybrid_bonus=2.0,
    bid_markup=3,
    fuel_runs=1,
    cheap_fuel_buffer=1,
    build_capacity_buffer=3,
    cash_reserve=2,
    build_priority="total_cost",
    resource_priority="unit_cost",
)


EXPANSION_PROFILE = DeterministicStrategyProfile(
    controller_name="ai_deterministic_expansion",
    strategy_angle="generation_capacity_and_network_growth",
    output_weight=8.0,
    run_cost_weight=0.8,
    purchase_price_weight=0.16,
    ecological_bonus=3.0,
    hybrid_bonus=1.0,
    bid_markup=2,
    fuel_runs=1,
    cheap_fuel_buffer=0,
    build_capacity_buffer=3,
    cash_reserve=0,
    build_priority="connection_cost",
    resource_priority="largest_deficit",
)


RESERVE_PROFILE = DeterministicStrategyProfile(
    controller_name="ai_deterministic_reserve",
    strategy_angle="cash_preservation_and_low_running_cost",
    output_weight=4.5,
    run_cost_weight=2.8,
    purchase_price_weight=1.8,
    ecological_bonus=10.0,
    hybrid_bonus=2.0,
    bid_markup=2,
    fuel_runs=1,
    cheap_fuel_buffer=0,
    build_capacity_buffer=2,
    cash_reserve=10,
    build_priority="cash_preserving",
    resource_priority="cash_preserving",
)


class ProfiledDeterministicAiController(BaseAiController):
    profile: DeterministicStrategyProfile

    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        state = snapshot.state
        if state.pending_decision is not None:
            intent = _choose_pending_intent(state)
        elif request.phase == "auction":
            intent = _choose_profiled_auction_intent(state, request, self.profile)
        elif request.phase == "buy_resources":
            intent = _choose_profiled_resource_intent(state, request.player_id, self.profile)
        elif request.phase == "build_houses":
            intent = _choose_profiled_build_intent(state, request.player_id, self.profile)
        elif request.phase == "bureaucracy":
            plans = _choose_best_generation_plans(state, request.player_id)
            intent = (
                GuiIntent.run_plants(request.player_id, plans)
                if plans
                else GuiIntent.skip_bureaucracy(request.player_id)
            )
        else:
            raise ModelValidationError(f"unsupported request phase {request.phase!r}")
        self.log_state(
            snapshot,
            request,
            label="profiled_deterministic_decision",
            state={
                "schema_version": 1,
                "controller": self.profile.controller_name,
                "strategy_angle": self.profile.strategy_angle,
                "decision_type": request.decision_type,
                "intent": intent.to_dict(),
            },
            message="Profiled deterministic AI selected an intent.",
        )
        return intent


class EfficiencyDeterministicAiController(ProfiledDeterministicAiController):
    controller = EFFICIENCY_PROFILE.controller_name
    profile = EFFICIENCY_PROFILE


class ExpansionDeterministicAiController(ProfiledDeterministicAiController):
    controller = EXPANSION_PROFILE.controller_name
    profile = EXPANSION_PROFILE


class ReserveDeterministicAiController(ProfiledDeterministicAiController):
    controller = RESERVE_PROFILE.controller_name
    profile = RESERVE_PROFILE


def _choose_profiled_auction_intent(
    state: GameState,
    request: TurnRequest,
    profile: DeterministicStrategyProfile,
) -> GuiIntent:
    if request.decision_type == "auction_start":
        actions = [
            action for action in request.legal_actions if action.action_type == "auction_start"
        ]
        if not actions:
            return GuiIntent.auction_pass(request.player_id)
        chosen = max(
            actions,
            key=lambda action: (
                _plant_profile_score(
                    state,
                    _market_plant(state, int(action.payload["plant_price"])),
                    int(action.payload["min_bid"]),
                    profile,
                ),
                -int(action.payload["min_bid"]),
                -int(action.payload["plant_price"]),
            ),
        )
        return GuiIntent.auction_start(
            request.player_id,
            int(chosen.payload["plant_price"]),
            int(chosen.payload["min_bid"]),
        )

    bid_action = next(
        action for action in request.legal_actions if action.action_type == "auction_bid"
    )
    minimum_bid = int(bid_action.payload["min_bid"])
    maximum_bid = int(bid_action.payload["max_bid"])
    plant_price = int(bid_action.payload["plant_price"])
    plant = _market_plant(state, plant_price)
    quality_bonus = max(
        0,
        round(
            _plant_profile_score(state, plant, plant_price, profile)
            / max(4.0, profile.output_weight * 2.0)
        ),
    )
    bid_cap = min(maximum_bid, plant_price + profile.bid_markup + quality_bonus)
    if minimum_bid > bid_cap:
        return GuiIntent.auction_pass(request.player_id)
    return GuiIntent.auction_bid(request.player_id, minimum_bid)


def _plant_profile_score(
    state: GameState,
    plant: PowerPlantCard,
    purchase_price: int,
    profile: DeterministicStrategyProfile,
) -> float:
    return (
        plant.output_cities * profile.output_weight
        - _estimated_run_cost(state, plant) * profile.run_cost_weight
        - purchase_price * profile.purchase_price_weight
        + (profile.ecological_bonus if plant.is_ecological else 0.0)
        + (profile.hybrid_bonus if plant.is_hybrid else 0.0)
    )


def _estimated_run_cost(state: GameState, plant: PowerPlantCard) -> int:
    if plant.is_ecological:
        return 0
    if plant.is_hybrid:
        prices = sorted(
            (*state.resource_market.available_unit_prices("coal"),
             *state.resource_market.available_unit_prices("oil"))
        )
        return sum(prices[: plant.resource_cost]) if len(prices) >= plant.resource_cost else 99
    resource = plant.resource_types[0]
    prices = state.resource_market.available_unit_prices(resource)
    return sum(prices[: plant.resource_cost]) if len(prices) >= plant.resource_cost else 99


def _market_plant(state: GameState, price: int) -> PowerPlantCard:
    for plant in (*state.current_market, *state.future_market):
        if plant.price == price:
            return plant
    for player in state.players:
        for plant in player.power_plants:
            if plant.price == price:
                return plant
    raise ModelValidationError(f"unknown visible plant {price}")


def _choose_profiled_resource_intent(
    state: GameState,
    player_id: str,
    profile: DeterministicStrategyProfile,
) -> GuiIntent:
    actions = legal_resource_purchases(state, player_id)
    if not actions:
        return GuiIntent.finish_buying(player_id)
    deficits = _profile_resource_deficits(state, player_id, profile)
    candidates = [
        action
        for action in actions
        if deficits.get(str(action.payload["resource"]), 0) > 0
    ]
    if not candidates:
        return GuiIntent.finish_buying(player_id)

    def priority(action) -> tuple[float, ...]:
        resource = str(action.payload["resource"])
        first_price = int(action.payload["unit_prices"][0])
        deficit = deficits[resource]
        if profile.resource_priority == "largest_deficit":
            return (-deficit, first_price, RESOURCE_TYPES.index(resource))
        if profile.resource_priority == "cash_preserving":
            return (first_price * max(1, deficit), first_price, RESOURCE_TYPES.index(resource))
        return (first_price, -deficit, RESOURCE_TYPES.index(resource))

    chosen = min(candidates, key=priority)
    resource = str(chosen.payload["resource"])
    amount = min(int(chosen.payload["max_affordable_units"]), deficits[resource])
    if profile.resource_priority == "cash_preserving":
        amount = 1
    return GuiIntent.buy_resource(player_id, resource, max(1, amount))


def _profile_resource_deficits(
    state: GameState,
    player_id: str,
    profile: DeterministicStrategyProfile,
) -> dict[str, int]:
    player = _get_player(state, player_id)
    stored = player.resource_storage.resource_totals()
    targets = {resource: 0 for resource in RESOURCE_TYPES}
    hybrid_units = 0
    for plant in player.power_plants:
        if plant.is_ecological or plant.is_step_3_placeholder:
            continue
        if plant.is_hybrid:
            hybrid_units += plant.resource_cost * profile.fuel_runs
        else:
            targets[plant.resource_types[0]] += plant.resource_cost * profile.fuel_runs

    fossil_surplus = max(0, stored["coal"] - targets["coal"]) + max(
        0, stored["oil"] - targets["oil"]
    )
    hybrid_deficit = max(0, hybrid_units - fossil_surplus)
    if hybrid_deficit:
        coal_price = _next_unit_price(state, "coal")
        oil_price = _next_unit_price(state, "oil")
        preferred = "coal" if (coal_price, "coal") <= (oil_price, "oil") else "oil"
        targets[preferred] += hybrid_deficit

    if profile.cheap_fuel_buffer:
        for resource in RESOURCE_TYPES:
            if targets[resource] and _next_unit_price(state, resource) <= 2:
                targets[resource] += profile.cheap_fuel_buffer
    return {
        resource: max(0, targets[resource] - stored[resource])
        for resource in RESOURCE_TYPES
    }


def _next_unit_price(state: GameState, resource: str) -> int:
    prices = state.resource_market.available_unit_prices(resource)
    return int(prices[0]) if prices else 99


def _choose_profiled_build_intent(
    state: GameState,
    player_id: str,
    profile: DeterministicStrategyProfile,
) -> GuiIntent:
    player = _get_player(state, player_id)
    total_output = sum(
        plant.output_cities
        for plant in player.power_plants
        if not plant.is_step_3_placeholder
    )
    end_threshold = int(state.rules.player_count_rules[len(state.players)]["end_game_cities"])
    target = min(end_threshold, total_output + profile.build_capacity_buffer)
    if player.connected_city_count >= target:
        return GuiIntent.finish_building(player_id)
    actions = legal_build_targets(state, player_id)
    if not actions:
        return GuiIntent.finish_building(player_id)

    if profile.build_priority == "connection_cost":
        chosen = min(
            actions,
            key=lambda action: (
                int(action.payload["connection_cost"]),
                int(action.payload["total_cost"]),
                str(action.payload["city_id"]),
            ),
        )
    else:
        chosen = min(
            actions,
            key=lambda action: (
                int(action.payload["total_cost"]),
                int(action.payload["connection_cost"]),
                str(action.payload["city_id"]),
            ),
        )
    total_cost = int(chosen.payload["total_cost"])
    effective_reserve = 0 if player.connected_city_count + 1 >= end_threshold else profile.cash_reserve
    if player.elektro - total_cost < effective_reserve:
        return GuiIntent.finish_building(player_id)
    return GuiIntent.commit_build(player_id, [str(chosen.payload["city_id"])])


__all__ = [
    "DeterministicStrategyProfile",
    "EfficiencyDeterministicAiController",
    "ExpansionDeterministicAiController",
    "ReserveDeterministicAiController",
]
