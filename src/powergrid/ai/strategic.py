from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from statistics import mean

from ..model import (
    GameState,
    ModelValidationError,
    PlantRunPlan,
    PlayerState,
    PowerPlantCard,
    add_power_plant_to_player,
    apply_builds,
    choose_plants_to_run,
    compute_all_targets_connection_cost,
    compute_powered_cities,
    consume_resources,
    discard_resources_to_fit_storage,
    legal_build_targets,
    legal_resource_purchases,
    list_auctionable_plants,
    pay_income,
    purchase_resources,
    replace_plant_if_needed,
)
from ..session_types import GameSnapshot, GuiIntent, TurnRequest
from .base import BaseAiController


RESOURCE_TYPES = ("coal", "oil", "garbage", "uranium")

PLANT_STATIC_ADJUSTMENTS = {
    5: 1.5,
    6: -1.5,
    7: -2.5,
    10: 1.0,
    11: 1.5,
    13: 0.5,
    15: 2.5,
    17: -0.5,
    18: 0.5,
    20: 3.0,
    21: 1.5,
    23: 1.0,
    25: 3.0,
    27: 1.0,
    29: 3.5,
    31: 1.5,
    34: 1.0,
    35: 1.0,
    37: 1.0,
    42: 0.5,
    44: 3.0,
    46: 1.5,
    50: 3.0,
}


@dataclass(frozen=True)
class _StageWeights:
    connected: float
    powered: float
    income: float
    cash: float
    plants: float
    frontier: float
    resources: float
    order: float
    exposure: float
    overbuild: float
    unused_capacity: float


STAGE_WEIGHTS = {
    "opening": _StageWeights(
        connected=5.0,
        powered=6.0,
        income=2.0,
        cash=0.45,
        plants=3.8,
        frontier=2.2,
        resources=1.0,
        order=3.0,
        exposure=1.8,
        overbuild=4.0,
        unused_capacity=1.2,
    ),
    "step1": _StageWeights(
        connected=6.0,
        powered=6.8,
        income=2.4,
        cash=0.55,
        plants=3.2,
        frontier=2.3,
        resources=1.1,
        order=2.8,
        exposure=2.1,
        overbuild=5.2,
        unused_capacity=1.0,
    ),
    "pre_step2": _StageWeights(
        connected=6.4,
        powered=7.4,
        income=2.8,
        cash=0.6,
        plants=3.3,
        frontier=2.0,
        resources=1.3,
        order=2.4,
        exposure=2.2,
        overbuild=6.2,
        unused_capacity=0.8,
    ),
    "step2": _StageWeights(
        connected=6.2,
        powered=8.2,
        income=3.2,
        cash=0.65,
        plants=3.5,
        frontier=1.6,
        resources=1.4,
        order=2.0,
        exposure=2.4,
        overbuild=4.8,
        unused_capacity=0.7,
    ),
    "sprint": _StageWeights(
        connected=4.0,
        powered=10.5,
        income=4.0,
        cash=0.75,
        plants=3.0,
        frontier=0.9,
        resources=1.0,
        order=1.2,
        exposure=1.6,
        overbuild=2.8,
        unused_capacity=0.4,
    ),
}


@dataclass(frozen=True)
class _GenerationSummary:
    plans: tuple[PlantRunPlan, ...]
    powered: int
    income: int
    spent_units: int
    spent_value: float


@dataclass(frozen=True)
class _BuildPlan:
    city_ids: tuple[str, ...]
    total_cost: int
    score: float


@dataclass(frozen=True)
class _AuctionStartOption:
    action_type: str
    payload: dict[str, int]


class StrategicAiController(BaseAiController):
    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        state = snapshot.state
        if state.pending_decision is not None:
            return _choose_pending_intent(state)
        if request.phase == "auction":
            return _choose_auction_intent(request, snapshot)
        if request.phase == "buy_resources":
            return _choose_resource_intent(state, request.player_id)
        if request.phase == "build_houses":
            return _choose_build_intent(state, request.player_id)
        if request.phase == "bureaucracy":
            return _choose_bureaucracy_intent(state, request.player_id)
        raise ModelValidationError(f"unsupported request phase {request.phase!r}")


def _choose_pending_intent(state: GameState) -> GuiIntent:
    decision = state.pending_decision
    assert decision is not None
    if decision.decision_type == "discard_power_plant":
        best_price = None
        best_score = float("-inf")
        for action in decision.legal_actions:
            plant_price = int(action.payload["price"])
            candidate = replace_plant_if_needed(_clone_state(state), decision.player_id, plant_price)
            score = _evaluate_relative_state(candidate, decision.player_id)
            if score > best_score or (
                score == best_score and (best_price is None or plant_price < best_price)
            ):
                best_score = score
                best_price = plant_price
        assert best_price is not None
        return GuiIntent.discard_plant(decision.player_id, best_price)

    best_choice = None
    for action in decision.legal_actions:
        coal = int(action.payload.get("coal", 0))
        oil = int(action.payload.get("oil", 0))
        candidate = discard_resources_to_fit_storage(
            _clone_state(state),
            decision.player_id,
            {"coal": coal, "oil": oil},
        )
        score = _evaluate_relative_state(candidate, decision.player_id)
        signature = (score, -(coal + oil), oil, coal)
        if best_choice is None or signature > best_choice[0]:
            best_choice = (signature, coal, oil)
    assert best_choice is not None
    return GuiIntent.discard_hybrid_resources(
        decision.player_id,
        coal=best_choice[1],
        oil=best_choice[2],
    )


def _choose_auction_intent(request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
    state = snapshot.state
    if request.decision_type == "auction_start":
        return _choose_auction_start_intent(state, request.player_id)
    return _choose_auction_bid_intent(state, request.player_id)


def _choose_auction_start_intent(state: GameState, player_id: str) -> GuiIntent:
    start_actions = [
        action
        for action in _legal_auction_start_actions(state, player_id)
        if action.action_type == "auction_start"
    ]
    if not start_actions:
        return GuiIntent.auction_pass(player_id)

    player = _get_player(state, player_id)
    current_score = _evaluate_relative_state(state, player_id)
    best_choice = None
    best_delta = float("-inf")
    for action in start_actions:
        plant_price = int(action.payload["plant_price"])
        min_bid = int(action.payload["min_bid"])
        plant = _get_market_plant(state, plant_price)
        reserve = _auction_reserve(state, player_id, plant)
        if reserve < min_bid and state.round_number == 1:
            reserve = min_bid
        own_interest = reserve - min_bid
        opponent_reserves = [
            _auction_reserve(state, opponent.player_id, plant)
            for opponent in state.players
            if opponent.player_id != player_id and _can_participate_in_auction(state, opponent.player_id)
        ]
        strongest_opponent = max(opponent_reserves, default=0)
        contest_pressure = max(0, strongest_opponent - reserve)
        market_bonus = _market_roll_bonus(state, plant)
        bait_score = 0.0
        if strongest_opponent > min_bid and reserve + 1 < strongest_opponent:
            bait_score = (strongest_opponent - min_bid) * 0.75 + market_bonus * 0.6
        if own_interest >= 0:
            price_guess = min(reserve, max(min_bid, strongest_opponent))
            post_buy = _simulate_hypothetical_purchase(state, player_id, plant, price_guess)
            gain = _evaluate_relative_state(post_buy, player_id) - current_score
            delta = gain + market_bonus - (contest_pressure * 0.35)
            opening_bid = min_bid
            if _is_sprint_state(state, player_id) and reserve - min_bid >= 4:
                opening_bid = min(reserve, min_bid + 2, player.elektro)
        else:
            delta = bait_score - 1.5
            opening_bid = min_bid
        signature = (
            delta,
            own_interest,
            bait_score,
            _plant_purchase_gain(state, player_id, plant),
            -plant.price,
        )
        if best_choice is None or signature > best_choice[0]:
            best_choice = (signature, plant_price, opening_bid)
            best_delta = delta

    if state.round_number > 1 and best_delta <= 0:
        return GuiIntent.auction_pass(player_id)

    assert best_choice is not None
    return GuiIntent.auction_start(
        player_id,
        plant_price=best_choice[1],
        bid=best_choice[2],
    )


def _choose_auction_bid_intent(state: GameState, player_id: str) -> GuiIntent:
    auction_state = state.auction_state
    assert auction_state is not None
    plant = _get_market_plant(state, int(auction_state.active_plant_price))
    min_bid = int(auction_state.current_bid) + 1
    reserve = _auction_reserve(state, player_id, plant)
    hard_cap = _auction_hard_cap(state, player_id)
    if min_bid > reserve or min_bid > hard_cap:
        return GuiIntent.auction_pass(player_id)
    return GuiIntent.auction_bid(player_id, min_bid)


def _choose_resource_intent(state: GameState, player_id: str) -> GuiIntent:
    depth = 3
    actions = legal_resource_purchases(state, player_id)
    if len(actions) > 8:
        depth = 2
    score, move = _search_resource_purchase(state, player_id, depth=depth)
    if move is None:
        return GuiIntent.finish_buying(player_id)
    return GuiIntent.buy_resource(player_id, resource=move[0], amount=move[1])


def _choose_build_intent(state: GameState, player_id: str) -> GuiIntent:
    plan = _search_best_build_plan(state, player_id)
    if plan is None:
        return GuiIntent.finish_building(player_id)
    current_score = _evaluate_relative_state(state, player_id)
    if plan.score <= current_score + 0.25:
        return GuiIntent.finish_building(player_id)
    return GuiIntent.commit_build(player_id, list(plan.city_ids))


def _choose_bureaucracy_intent(state: GameState, player_id: str) -> GuiIntent:
    player = _get_player(state, player_id)
    plans = _enumerate_generation_summaries(state, player_id)
    if not plans:
        return GuiIntent.skip_bureaucracy(player_id)

    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    final_round = any(candidate.connected_city_count >= end_threshold for candidate in state.players)
    if final_round:
        best = max(
            plans,
            key=lambda summary: (
                summary.powered,
                player.elektro + summary.income,
                player.connected_city_count,
                -summary.spent_value,
                tuple(plan.plant_price for plan in summary.plans),
            ),
        )
    else:
        best = max(
            plans,
            key=lambda summary: _evaluate_bureaucracy_summary(state, player_id, summary),
        )
    if not best.plans:
        return GuiIntent.skip_bureaucracy(player_id)
    return GuiIntent.run_plants(player_id, best.plans)


def _legal_auction_start_actions(state: GameState, player_id: str) -> tuple[_AuctionStartOption, ...]:
    auction_state = state.auction_state
    assert auction_state is not None
    player = _get_player(state, player_id)
    actions: list[_AuctionStartOption] = []
    for plant in list_auctionable_plants(state):
        minimum_bid = 1 if auction_state.discount_token_plant_price == plant.price else plant.price
        if minimum_bid <= player.elektro:
            actions.append(
                _AuctionStartOption(
                    action_type="auction_start",
                    payload={
                        "plant_price": plant.price,
                        "min_bid": minimum_bid,
                        "max_bid": player.elektro,
                    },
                )
            )
    return tuple(actions)


def _search_resource_purchase(
    state: GameState,
    player_id: str,
    *,
    depth: int,
) -> tuple[float, tuple[str, int] | None]:
    best_score = _resource_finish_score(state, player_id)
    best_move = None
    if depth <= 0:
        return best_score, None

    for action in legal_resource_purchases(state, player_id):
        resource = str(action.payload["resource"])
        for amount in _candidate_resource_amounts(state, player_id, action):
            try:
                next_state = purchase_resources(
                    _clone_state(state),
                    player_id,
                    {resource: amount},
                )
            except ModelValidationError:
                continue
            next_score, _ = _search_resource_purchase(next_state, player_id, depth=depth - 1)
            signature = (next_score, amount, -RESOURCE_TYPES.index(resource))
            if best_move is None or signature > (best_score, best_move[1], -RESOURCE_TYPES.index(best_move[0])):
                if next_score > best_score + 0.15:
                    best_score = next_score
                    best_move = (resource, amount)
    return best_score, best_move


def _candidate_resource_amounts(state: GameState, player_id: str, action) -> tuple[int, ...]:
    resource = str(action.payload["resource"])
    max_affordable = int(action.payload["max_affordable_units"])
    max_units = min(int(action.payload["max_units"]), max_affordable)
    deficits = _resource_need_by_type(state, player_id)
    market_prices = tuple(int(price) for price in action.payload["unit_prices"])
    scarce = _resource_pressure(state)[resource] >= 0.7
    cheap_run = 0
    for price in market_prices:
        if price <= 2:
            cheap_run += 1
        else:
            break
    amounts = {
        1,
        max_units,
        min(max_units, max(1, deficits.get(resource, 0))),
        min(max_units, 2),
        min(max_units, cheap_run) if cheap_run else 1,
    }
    if scarce:
        amounts.add(min(max_units, max(1, deficits.get(resource, 0) + 1)))
    return tuple(sorted(amount for amount in amounts if 1 <= amount <= max_units))


def _resource_finish_score(state: GameState, player_id: str) -> float:
    score = _evaluate_relative_state(state, player_id)
    score += _quick_build_potential(state, player_id) * 0.6
    deficits = _resource_need_by_type(state, player_id)
    shortfall_penalty = 0.0
    for resource, amount in deficits.items():
        if amount <= 0:
            continue
        shortfall_penalty += amount * (_resource_unit_price(state, resource) + _resource_pressure(state)[resource] * 2.0)
    score -= shortfall_penalty * 0.35
    return score


def _search_best_build_plan(state: GameState, player_id: str) -> _BuildPlan | None:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return None

    player = _get_player(state, player_id)
    powered_now = _best_generation_summary(state, player_id).powered
    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    max_depth = 4
    if _is_sprint_state(state, player_id):
        max_depth = 5
    else:
        max_depth = max(1, min(4, max(1, powered_now - player.connected_city_count + 2)))
    max_depth = min(max_depth, len(actions), player.houses_in_supply)

    ranked = sorted(
        actions,
        key=lambda action: (
            _quick_city_score(state, player_id, str(action.payload["city_id"]), int(action.payload["total_cost"])),
            -int(action.payload["total_cost"]),
            str(action.payload["city_id"]),
        ),
        reverse=True,
    )
    candidate_ids = [str(action.payload["city_id"]) for action in ranked[: min(10, len(ranked))]]
    current_score = _evaluate_relative_state(state, player_id)
    current_player = _get_player(state, player_id)
    current_connected = current_player.connected_city_count
    current_powered = _best_generation_summary(state, player_id).powered
    seen: set[frozenset[str]] = set()
    beam: list[tuple[str, ...]] = [()]
    best_plan: _BuildPlan | None = None

    for _ in range(max_depth):
        expansions: list[tuple[float, float, tuple[str, ...], int]] = []
        for base in beam:
            used = set(base)
            for city_id in candidate_ids:
                if city_id in used:
                    continue
                proposal = tuple((*base, city_id))
                key = frozenset(proposal)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    next_state = apply_builds(_clone_state(state), player_id, proposal)
                except ModelValidationError:
                    continue
                next_score = _evaluate_relative_state(next_state, player_id)
                next_player = _get_player(next_state, player_id)
                next_powered = _best_generation_summary(next_state, player_id).powered
                total_cost = player.elektro - _get_player(next_state, player_id).elektro
                trigger_bonus = _build_trigger_adjustment(next_state, player_id)
                build_bonus = (
                    (next_player.connected_city_count - current_connected) * 14.0
                    + max(0, next_powered - current_powered) * 4.0
                    - (total_cost * 0.08)
                )
                score = next_score + trigger_bonus + build_bonus
                expansions.append((score, -total_cost, proposal, total_cost))
                if score > current_score + 0.75:
                    plan = _BuildPlan(proposal, total_cost, score)
                    if best_plan is None or (plan.score, -plan.total_cost, plan.city_ids) > (
                        best_plan.score,
                        -best_plan.total_cost,
                        best_plan.city_ids,
                    ):
                        best_plan = plan
        if not expansions:
            break
        expansions.sort(reverse=True)
        beam = [proposal for _, _, proposal, _ in expansions[:6]]
        if best_plan is not None and len(best_plan.city_ids) >= end_threshold:
            break
    return best_plan


def _quick_city_score(state: GameState, player_id: str, city_id: str, total_cost: int) -> float:
    savings = _future_connection_savings(state, player_id, city_id)
    pressure = _city_contest_pressure(state, player_id, city_id)
    stage = STAGE_WEIGHTS[_stage_name(state)]
    return savings * 0.18 - total_cost * 0.42 - pressure * 1.4 + stage.frontier * 3.0


def _future_connection_savings(state: GameState, player_id: str, city_id: str) -> float:
    player = _get_player(state, player_id)
    current = compute_all_targets_connection_cost(state, player_id)
    updated = compute_all_targets_connection_cost(
        state,
        player_id,
        source_city_ids=tuple((*player.network_city_ids, city_id)),
    )
    savings = 0
    deltas = []
    for target, current_cost in current.items():
        if target == city_id or target in player.network_city_ids:
            continue
        next_cost = updated.get(target, current_cost)
        if next_cost < current_cost:
            deltas.append(current_cost - next_cost)
    for delta in sorted(deltas, reverse=True)[:5]:
        savings += delta
    return float(savings)


def _city_contest_pressure(state: GameState, player_id: str, city_id: str) -> int:
    player_costs = compute_all_targets_connection_cost(state, player_id)
    own_cost = player_costs.get(city_id, 999)
    pressure = 0
    for opponent in state.players:
        if opponent.player_id == player_id or city_id in opponent.network_city_ids:
            continue
        opponent_costs = compute_all_targets_connection_cost(state, opponent.player_id)
        candidate_cost = opponent_costs.get(city_id)
        if candidate_cost is None:
            continue
        if candidate_cost <= own_cost + 10 and opponent.elektro >= 10:
            pressure += 1
    return pressure


def _build_trigger_adjustment(state: GameState, player_id: str) -> float:
    player = _get_player(state, player_id)
    step_2_threshold = state.rules.player_count_rules[len(state.players)]["step_2_cities"]
    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    if player.connected_city_count >= end_threshold:
        own_power = _best_generation_summary(state, player_id)
        own_tuple = (
            own_power.powered,
            player.elektro + own_power.income,
            player.connected_city_count,
        )
        opponent_best = max(
            (
                _best_generation_summary(state, opponent.player_id).powered,
                opponent.elektro + _best_generation_summary(state, opponent.player_id).income,
                opponent.connected_city_count,
            )
            for opponent in state.players
            if opponent.player_id != player_id
        )
        return 80.0 if own_tuple > opponent_best else -120.0
    if state.step == 1 and player.connected_city_count >= step_2_threshold:
        own_powered = _best_generation_summary(state, player_id).powered
        rival_power = max(
            _best_generation_summary(state, opponent.player_id).powered
            for opponent in state.players
            if opponent.player_id != player_id
        )
        return 18.0 if own_powered >= rival_power else -26.0
    return 0.0


def _evaluate_bureaucracy_summary(
    state: GameState,
    player_id: str,
    summary: _GenerationSummary,
) -> float:
    simulated = _clone_state(state)
    simulated = consume_resources(simulated, player_id, summary.plans)
    player = _get_player(simulated, player_id)
    simulated = _replace_player_on_state(
        simulated,
        replace(player, elektro=player.elektro + summary.income),
    )
    return _evaluate_relative_state(simulated, player_id)


def _best_generation_summary(state: GameState, player_id: str) -> _GenerationSummary:
    options = _enumerate_generation_summaries(state, player_id)
    if not options:
        return _GenerationSummary((), 0, pay_income(state.rules, 0), 0, 0.0)
    return max(
        options,
        key=lambda summary: (
            summary.powered,
            summary.income,
            -summary.spent_value,
            -summary.spent_units,
            tuple(plan.plant_price for plan in summary.plans),
        ),
    )


def _enumerate_generation_summaries(state: GameState, player_id: str) -> tuple[_GenerationSummary, ...]:
    player = _get_player(state, player_id)
    totals = player.resource_storage.resource_totals()
    plants = [plant for plant in sorted(player.power_plants, key=lambda item: item.price) if not plant.is_step_3_placeholder]
    if not plants:
        return (_GenerationSummary((), 0, pay_income(state.rules, 0), 0, 0.0),)

    choices: list[tuple[PowerPlantCard, tuple[PlantRunPlan | None, ...]]] = []
    for plant in plants:
        options: list[PlantRunPlan | None] = [None]
        if plant.is_ecological:
            options.append(PlantRunPlan(plant.price, {}))
        elif plant.is_hybrid:
            for coal in range(plant.resource_cost + 1):
                oil = plant.resource_cost - coal
                if coal <= totals["coal"] and oil <= totals["oil"]:
                    options.append(PlantRunPlan(plant.price, {"coal": coal, "oil": oil}))
        else:
            resource = plant.resource_types[0]
            if plant.resource_cost <= totals[resource]:
                options.append(PlantRunPlan(plant.price, {resource: plant.resource_cost}))
        choices.append((plant, tuple(options)))

    summaries: list[_GenerationSummary] = []

    def backtrack(index: int, remaining: dict[str, int], selected: list[PlantRunPlan]) -> None:
        if index >= len(choices):
            plans = tuple(selected)
            try:
                validated = choose_plants_to_run(state, player_id, plans)
            except ModelValidationError:
                return
            powered = compute_powered_cities(state, player_id, validated)
            spent_units = sum(sum(plan.resource_mix.values()) for plan in validated)
            spent_value = 0.0
            for plan in validated:
                for resource, amount in plan.resource_mix.items():
                    spent_value += amount * (_resource_unit_price(state, resource) + _resource_pressure(state)[resource] * 0.8)
            summaries.append(
                _GenerationSummary(
                    plans=validated,
                    powered=powered,
                    income=pay_income(state.rules, powered),
                    spent_units=spent_units,
                    spent_value=spent_value,
                )
            )
            return

        _, options = choices[index]
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

    backtrack(0, dict(totals), [])
    return tuple(summaries)


def _evaluate_relative_state(state: GameState, player_id: str) -> float:
    own = _evaluate_player_strength(state, player_id)
    opponent_strengths = [
        _evaluate_player_strength(state, opponent.player_id)
        for opponent in state.players
        if opponent.player_id != player_id
    ]
    if not opponent_strengths:
        return own
    return own - (0.65 * max(opponent_strengths)) - (0.2 * mean(opponent_strengths))


def _evaluate_player_strength(state: GameState, player_id: str) -> float:
    stage_name = _stage_name(state)
    weights = STAGE_WEIGHTS[stage_name]
    player = _get_player(state, player_id)
    generation = _best_generation_summary(state, player_id)
    portfolio = _portfolio_score(state, player.power_plants)
    frontier = _frontier_score(state, player_id)
    resource_value = _stored_resource_value(state, player_id)
    exposure = _resource_exposure_penalty(state, player_id)
    order_value = len(state.players) - 1 - state.player_order.index(player_id)
    total_output = sum(plant.output_cities for plant in player.power_plants)
    overbuild = max(0, player.connected_city_count - total_output)
    unused_capacity = max(0, total_output - player.connected_city_count)
    trigger_score = _trigger_timing_score(state, player_id, generation, player)

    return (
        weights.connected * player.connected_city_count
        + weights.powered * generation.powered
        + weights.income * generation.income
        + weights.cash * player.elektro
        + weights.plants * portfolio
        + weights.frontier * frontier
        + weights.resources * resource_value
        + weights.order * order_value
        - weights.exposure * exposure
        - weights.overbuild * overbuild
        - weights.unused_capacity * unused_capacity
        + trigger_score
    )


def _frontier_score(state: GameState, player_id: str) -> float:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return 0.0
    cheapest = [int(action.payload["total_cost"]) for action in sorted(actions, key=lambda item: int(item.payload["total_cost"]))[:4]]
    return (len(actions) * 0.65) + max(0.0, 18.0 - mean(cheapest))


def _trigger_timing_score(
    state: GameState,
    player_id: str,
    generation: _GenerationSummary,
    player: PlayerState,
) -> float:
    step_2_threshold = state.rules.player_count_rules[len(state.players)]["step_2_cities"]
    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    if player.connected_city_count >= end_threshold:
        own_tuple = (
            generation.powered,
            player.elektro + generation.income,
            player.connected_city_count,
        )
        opponent_tuples = [
            (
                _best_generation_summary(state, opponent.player_id).powered,
                opponent.elektro + _best_generation_summary(state, opponent.player_id).income,
                opponent.connected_city_count,
            )
            for opponent in state.players
            if opponent.player_id != player_id
        ]
        if own_tuple > max(opponent_tuples):
            return 120.0
        return -160.0
    if state.step == 1 and player.connected_city_count >= step_2_threshold:
        best_opponent_power = max(
            _best_generation_summary(state, opponent.player_id).powered
            for opponent in state.players
            if opponent.player_id != player_id
        )
        return 18.0 if generation.powered >= best_opponent_power else -20.0
    return 0.0


def _stage_name(state: GameState) -> str:
    step_2_threshold = state.rules.player_count_rules[len(state.players)]["step_2_cities"]
    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    highest_connected = max(player.connected_city_count for player in state.players)
    if state.round_number <= 1:
        return "opening"
    if state.step == 3 or highest_connected >= end_threshold - 2:
        return "sprint"
    if state.step == 2:
        return "step2"
    if highest_connected >= step_2_threshold - 1:
        return "pre_step2"
    return "step1"


def _is_sprint_state(state: GameState, player_id: str) -> bool:
    return _stage_name(state) == "sprint"


def _stored_resource_value(state: GameState, player_id: str) -> float:
    player = _get_player(state, player_id)
    totals = player.resource_storage.resource_totals()
    pressure = _resource_pressure(state)
    useful = {resource: 0 for resource in RESOURCE_TYPES}
    for plant in player.power_plants:
        if plant.is_step_3_placeholder or plant.is_ecological:
            continue
        if plant.is_hybrid:
            useful["coal"] += plant.max_storage
            useful["oil"] += plant.max_storage
        else:
            useful[plant.resource_types[0]] += plant.max_storage
    value = 0.0
    for resource, amount in totals.items():
        if amount <= 0:
            continue
        utility = 1.0 if useful[resource] > 0 else 0.2
        value += amount * ((_resource_unit_price(state, resource) * 0.55) + (pressure[resource] * 1.8)) * utility
    return value


def _resource_exposure_penalty(state: GameState, player_id: str) -> float:
    demand = _estimated_round_resource_demand(state, player_id)
    total = sum(demand.values())
    if total <= 0:
        return 0.0
    pressure = _resource_pressure(state)
    penalty = 0.0
    for resource, amount in demand.items():
        if amount <= 0:
            continue
        share = amount / total
        if share <= 0.5:
            continue
        penalty += (share - 0.5) * amount * (1.0 + pressure[resource])
    return penalty


def _resource_pressure(state: GameState) -> dict[str, float]:
    refill_key = f"step_{state.step}"
    refill_table = state.rules.player_count_rules[len(state.players)]["resource_refill"][refill_key]
    pressure: dict[str, float] = {}
    demand_by_resource = {resource: 0.0 for resource in RESOURCE_TYPES}
    for player in state.players:
        player_demand = _estimated_round_resource_demand(state, player.player_id)
        for resource, amount in player_demand.items():
            demand_by_resource[resource] += amount
    for resource in RESOURCE_TYPES:
        available = len(state.resource_market.available_unit_prices(resource))
        refill = int(refill_table[resource])
        pressure[resource] = demand_by_resource[resource] / max(1.0, available + refill)
    return pressure


def _estimated_round_resource_demand(state: GameState, player_id: str) -> dict[str, float]:
    player = _get_player(state, player_id)
    demand = {resource: 0.0 for resource in RESOURCE_TYPES}
    for plant in player.power_plants:
        if plant.is_step_3_placeholder or plant.is_ecological:
            continue
        if plant.is_hybrid:
            coal_price = _resource_unit_price(state, "coal")
            oil_price = _resource_unit_price(state, "oil")
            preferred = "coal" if coal_price <= oil_price else "oil"
            secondary = "oil" if preferred == "coal" else "coal"
            demand[preferred] += plant.resource_cost * 0.7
            demand[secondary] += plant.resource_cost * 0.3
            continue
        demand[plant.resource_types[0]] += plant.resource_cost
    return demand


def _portfolio_score(state: GameState, plants: tuple[PowerPlantCard, ...]) -> float:
    if not plants:
        return 0.0
    values = [_plant_value(state, plant) for plant in plants if not plant.is_step_3_placeholder]
    total_output = sum(plant.output_cities for plant in plants if not plant.is_step_3_placeholder)
    resource_counts = {resource: 0 for resource in RESOURCE_TYPES}
    hybrids = 0
    ecological = 0
    for plant in plants:
        if plant.is_step_3_placeholder:
            continue
        if plant.is_ecological:
            ecological += 1
        elif plant.is_hybrid:
            hybrids += 1
            resource_counts["coal"] += 1
            resource_counts["oil"] += 1
        else:
            resource_counts[plant.resource_types[0]] += 1
    diversity = sum(1 for count in resource_counts.values() if count > 0)
    concentration = max(resource_counts.values(), default=0)
    return (
        sum(values)
        + (total_output * 1.7)
        + (diversity * 1.5)
        + (hybrids * 1.2)
        + (ecological * 1.8)
        - max(0, concentration - 2) * 2.0
    )


def _plant_value(state: GameState, plant: PowerPlantCard) -> float:
    if plant.is_step_3_placeholder:
        return 0.0
    if plant.is_ecological:
        run_cost = 0.0
        pressure_penalty = 0.0
        efficiency = plant.output_cities * 2.5
        flexibility = 3.5
    elif plant.is_hybrid:
        run_cost = _hybrid_run_cost(state, plant.resource_cost)
        pressure = (_resource_pressure(state)["coal"] + _resource_pressure(state)["oil"]) / 2.0
        pressure_penalty = pressure * plant.resource_cost * 0.9
        efficiency = (plant.output_cities / max(1, plant.resource_cost)) * 4.5
        flexibility = 2.8
    else:
        resource = plant.resource_types[0]
        run_cost = _resource_purchase_cost(state, resource, plant.resource_cost)
        pressure_penalty = _resource_pressure(state)[resource] * plant.resource_cost * 1.1
        efficiency = (plant.output_cities / max(1, plant.resource_cost)) * 4.2
        flexibility = 0.0
    return (
        plant.output_cities * 5.5
        + efficiency
        + (plant.max_storage * 0.2)
        + flexibility
        + PLANT_STATIC_ADJUSTMENTS.get(plant.price, 0.0)
        - (run_cost * 0.8)
        - pressure_penalty
    )


def _auction_reserve(state: GameState, player_id: str, plant: PowerPlantCard) -> int:
    player = _get_player(state, player_id)
    min_bid = 1 if state.auction_state and state.auction_state.discount_token_plant_price == plant.price else plant.price
    hard_cap = _auction_hard_cap(state, player_id)
    if hard_cap < min_bid:
        return 0
    gain = _plant_purchase_gain(state, player_id, plant)
    stage = _stage_name(state)
    reserve = min_bid + int(max(0.0, gain * (1.6 if stage == "opening" else 1.25)))
    if state.round_number == 1 and not player.power_plants:
        reserve = max(reserve, min_bid)
    if _is_sprint_state(state, player_id) and plant.output_cities >= 4:
        reserve += 2
    return max(0, min(reserve, player.elektro, hard_cap))


def _auction_hard_cap(state: GameState, player_id: str) -> int:
    player = _get_player(state, player_id)
    build_floor = _recommended_build_cash_floor(state, player_id)
    resource_floor = _resource_purchase_floor(state, player_id)
    floor = build_floor + resource_floor
    if _is_sprint_state(state, player_id):
        floor = max(0, floor - 4)
    return max(0, player.elektro - floor)


def _recommended_build_cash_floor(state: GameState, player_id: str) -> int:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return 0
    player = _get_player(state, player_id)
    powered = _best_generation_summary(state, player_id).powered
    desired = 1 if player.connected_city_count >= powered else 2
    ranked = sorted(actions, key=lambda action: int(action.payload["total_cost"]))[:5]
    best_cost = int(ranked[0].payload["total_cost"])
    if desired <= 1:
        return best_cost
    city_ids = [str(action.payload["city_id"]) for action in ranked]
    for size in range(2, min(desired, len(city_ids)) + 1):
        for combo in combinations(city_ids, size):
            try:
                next_state = apply_builds(_clone_state(state), player_id, combo)
            except ModelValidationError:
                continue
            total_cost = player.elektro - _get_player(next_state, player_id).elektro
            if total_cost > best_cost:
                best_cost = total_cost
    return best_cost


def _quick_build_potential(state: GameState, player_id: str) -> float:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return 0.0
    ranked = sorted(int(action.payload["total_cost"]) for action in actions)[:3]
    return max(0.0, 20.0 - mean(ranked))


def _resource_purchase_floor(state: GameState, player_id: str) -> int:
    deficits = _resource_need_by_type(state, player_id)
    cost = 0
    for resource, amount in deficits.items():
        if amount <= 0:
            continue
        cost += _resource_purchase_cost(state, resource, amount)
    return cost


def _plant_purchase_gain(state: GameState, player_id: str, plant: PowerPlantCard) -> float:
    player = _get_player(state, player_id)
    before = _portfolio_score(state, player.power_plants)
    after_plants = _best_portfolio_after_purchase(state, player, plant)
    after = _portfolio_score(state, after_plants)
    before_output = sum(card.output_cities for card in player.power_plants if not card.is_step_3_placeholder)
    after_output = sum(card.output_cities for card in after_plants if not card.is_step_3_placeholder)
    capacity_bonus = max(0, after_output - before_output) * 1.8
    return after - before + capacity_bonus


def _best_portfolio_after_purchase(
    state: GameState,
    player: PlayerState,
    plant: PowerPlantCard,
) -> tuple[PowerPlantCard, ...]:
    candidates = tuple(sorted((*player.power_plants, plant), key=lambda card: card.price))
    max_plants = state.rules.player_count_rules[len(state.players)]["max_power_plants"]
    if len(candidates) <= max_plants:
        return candidates
    best = max(
        combinations(candidates, max_plants),
        key=lambda combo: (
            _portfolio_score(state, tuple(combo)),
            sum(card.output_cities for card in combo),
            tuple(card.price for card in combo),
        ),
    )
    return tuple(sorted(best, key=lambda card: card.price))


def _simulate_hypothetical_purchase(
    state: GameState,
    player_id: str,
    plant: PowerPlantCard,
    price_paid: int,
) -> GameState:
    simulated = _clone_state(state)
    player = _get_player(simulated, player_id)
    simulated = _replace_player_on_state(
        simulated,
        replace(player, elektro=max(0, player.elektro - price_paid)),
    )
    simulated = add_power_plant_to_player(simulated, player_id, plant.price)
    if simulated.pending_decision is not None and simulated.pending_decision.decision_type == "discard_power_plant":
        best_state = None
        best_score = float("-inf")
        for action in simulated.pending_decision.legal_actions:
            discard_price = int(action.payload["price"])
            candidate = replace_plant_if_needed(_clone_state(simulated), player_id, discard_price)
            score = _evaluate_relative_state(candidate, player_id)
            if score > best_score:
                best_score = score
                best_state = candidate
        assert best_state is not None
        simulated = best_state
    return simulated


def _market_roll_bonus(state: GameState, plant: PowerPlantCard) -> float:
    future_values = [_plant_value(state, candidate) for candidate in state.future_market if candidate.price != plant.price]
    if not future_values:
        return 0.0
    return mean(future_values) * 0.08


def _resource_need_by_type(state: GameState, player_id: str) -> dict[str, int]:
    player = _get_player(state, player_id)
    stored = player.resource_storage.resource_totals()
    needed = {resource: 0 for resource in RESOURCE_TYPES}
    hybrid_cost = 0
    for plant in player.power_plants:
        if plant.is_ecological or plant.is_step_3_placeholder:
            continue
        if plant.is_hybrid:
            hybrid_cost += plant.resource_cost
            continue
        needed[plant.resource_types[0]] += plant.resource_cost
    deficits = {
        resource: max(0, needed[resource] - stored[resource])
        for resource in RESOURCE_TYPES
    }
    hybrid_remaining = max(0, hybrid_cost - (stored["coal"] + stored["oil"]))
    if hybrid_remaining > 0:
        coal_price = _resource_unit_price(state, "coal")
        oil_price = _resource_unit_price(state, "oil")
        preferred = "coal" if coal_price <= oil_price else "oil"
        deficits[preferred] += hybrid_remaining
    return deficits


def _resource_purchase_cost(state: GameState, resource: str, amount: int) -> int:
    prices = state.resource_market.available_unit_prices(resource)
    if amount <= 0:
        return 0
    if len(prices) >= amount:
        return sum(prices[:amount])
    if not prices:
        return amount * 8
    return sum(prices) + ((amount - len(prices)) * max(prices[-1], 8))


def _resource_unit_price(state: GameState, resource: str) -> int:
    prices = state.resource_market.available_unit_prices(resource)
    if not prices:
        return 8
    return int(prices[0])


def _hybrid_run_cost(state: GameState, amount: int) -> float:
    best = None
    coal_prices = state.resource_market.available_unit_prices("coal")
    oil_prices = state.resource_market.available_unit_prices("oil")
    for coal in range(amount + 1):
        oil = amount - coal
        if len(coal_prices) < coal or len(oil_prices) < oil:
            continue
        cost = sum(coal_prices[:coal]) + sum(oil_prices[:oil])
        if best is None or cost < best:
            best = cost
    if best is None:
        return float(amount * 8)
    return float(best)


def _can_participate_in_auction(state: GameState, player_id: str) -> bool:
    auction_state = state.auction_state
    assert auction_state is not None
    if player_id in auction_state.players_with_plants:
        return False
    if player_id in auction_state.players_passed_phase:
        return False
    return True


def _get_market_plant(state: GameState, plant_price: int) -> PowerPlantCard:
    for plant in (*state.current_market, *state.future_market):
        if plant.price == plant_price:
            return plant
    raise ModelValidationError(f"power plant {plant_price} is not visible in the market")


def _get_player(state: GameState, player_id: str) -> PlayerState:
    for player in state.players:
        if player.player_id == player_id:
            return player
    raise ModelValidationError(f"unknown player {player_id!r}")


def _replace_player_on_state(state: GameState, updated_player: PlayerState) -> GameState:
    players = tuple(
        updated_player if player.player_id == updated_player.player_id else player
        for player in state.players
    )
    return replace(state, players=players)


def _clone_state(state: GameState) -> GameState:
    return GameState.from_dict(state.to_dict())
