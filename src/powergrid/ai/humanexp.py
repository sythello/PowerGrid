from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
from math import ceil, inf
import random

from ..model import (
    RESOURCE_TYPES, GameState, ModelValidationError, PlayerState, PowerPlantCard,
    ResourceMarket, add_power_plant_to_player, build_city, can_store_resources,
    discard_resources_to_fit_storage, legal_build_targets, pay_income,
    purchase_resources, remove_power_plant_from_player,
)
from ..rules_data import load_power_plants
from ..session_types import GameSnapshot, GuiIntent, TurnRequest
from .base import BaseAiController
from .deterministic import _choose_best_generation_plans, _get_player


CONTROLLER = "ai_humanexp_heuristics_v1"
DISPLAY_LABEL = "经验启发式v1"


@lru_cache(maxsize=1)
def _catalog() -> tuple[PowerPlantCard, ...]:
    return tuple(PowerPlantCard.from_definition(item) for item in load_power_plants())


def _with_player(state: GameState, player: PlayerState) -> GameState:
    return replace(state, players=tuple(
        player if item.player_id == player.player_id else item for item in state.players
    ))


def _capacity(plants: tuple[PowerPlantCard, ...]) -> int:
    return sum(plant.output_cities for plant in plants)


def _worst(plants: tuple[PowerPlantCard, ...]) -> PowerPlantCard:
    return min(plants, key=lambda plant: plant.price)


def _replacement_plant(plants: tuple[PowerPlantCard, ...], *, terminal_complete: bool) -> PowerPlantCard:
    if terminal_complete:
        return min(plants, key=lambda plant: (plant.output_cities, plant.price))
    return _worst(plants)


def _later_players(state: GameState, player_id: str) -> tuple[str, ...]:
    return state.player_order[state.player_order.index(player_id) + 1:]


def _fuel_demands(plants: tuple[PowerPlantCard, ...], multiplier: float = 1.0):
    """Enumerate all aggregate hybrid splits, without double-using stored fuel."""
    demand = dict.fromkeys(RESOURCE_TYPES, 0)
    hybrid = 0
    for plant in plants:
        amount = ceil(plant.resource_cost * multiplier)
        if plant.is_hybrid:
            hybrid += amount
        elif plant.resource_types:
            demand[plant.resource_types[0]] += amount
    for coal in range(hybrid + 1):
        yield {**demand, "coal": demand["coal"] + coal,
               "oil": demand["oil"] + hybrid - coal}


def _quote(market: ResourceMarket, basket: dict[str, int]) -> float:
    try:
        return sum(market.quote_purchase_cost(resource, amount)
                   for resource, amount in basket.items())
    except ModelValidationError:
        return inf


def _fuel_options(
    player: PlayerState, plants: tuple[PowerPlantCard, ...], market: ResourceMarket,
    *, available: tuple[str, ...] = RESOURCE_TYPES,
):
    stored = player.resource_storage.resource_totals()
    seen = set()
    for demand in _fuel_demands(plants):
        basket = {resource: max(0, demand[resource] - stored[resource])
                  for resource in RESOURCE_TYPES}
        key = tuple(basket.values())
        if key in seen or any(basket[r] and r not in available for r in RESOURCE_TYPES):
            continue
        seen.add(key)
        cost = _quote(market, basket)
        if cost != inf and can_store_resources(player, basket):
            yield basket, int(cost)


def _consume_estimated(market: ResourceMarket, basket: dict[str, int]) -> ResourceMarket:
    for resource, amount in basket.items():
        amount = min(amount, market.total_in_market(resource))
        if amount:
            market = market.remove_from_market(resource, amount)
    return market


def _demand_purchase(
    plants: tuple[PowerPlantCard, ...], market: ResourceMarket,
    multiplier: float, stored: dict[str, int] | None = None,
) -> dict[str, int]:
    stored = stored or dict.fromkeys(RESOURCE_TYPES, 0)
    baskets = ({r: max(0, demand[r] - stored[r]) for r in RESOURCE_TYPES}
               for demand in _fuel_demands(plants, multiplier))
    # Forecasts may exhaust a track. Prefer the split with the smallest shortage.
    return min(baskets, key=lambda basket: (
        sum(max(0, n - market.total_in_market(r)) for r, n in basket.items()),
        _quote(market, {r: min(n, market.total_in_market(r)) for r, n in basket.items()}),
        tuple(basket.values()),
    ))


def _contest_adjustments(state: GameState, player_id: str) -> dict[str, int]:
    adjustments: dict[str, int] = {}
    for opponent in state.players:
        if opponent.player_id == player_id or not opponent.network_city_ids:
            continue
        projected = _with_player(state, replace(opponent, elektro=100000))
        targets = sorted(legal_build_targets(projected, opponent.player_id), key=lambda action: (
            int(action.payload["total_cost"]), str(action.payload["city_id"]),
        ))
        for action, adjustment in zip(targets[:3], (2, 1, 1)):
            city = str(action.payload["city_id"])
            adjustments[city] = adjustments.get(city, 0) + adjustment
    return adjustments


@dataclass(frozen=True)
class BuildProjection:
    cities: tuple[str, ...]
    costs: tuple[int, ...]
    starting_count: int

    def affordable(self, budget: int, limit: int | None = None) -> tuple[int, int]:
        count, spent = self.starting_count, 0
        for cost in self.costs:
            if spent + cost > budget or (limit is not None and count >= limit):
                break
            spent += cost
            count += 1
        return count, spent


def _build_projection(
    state: GameState, player_id: str, *, surcharge: int = 0,
    adjustments: dict[str, int] | None = None, budget: int | None = None,
) -> BuildProjection:
    player = _get_player(state, player_id)
    projected = _with_player(state, replace(player, elektro=100000 if budget is None else budget))
    cities, costs = [], []
    adjustments = adjustments or {}
    while True:
        targets = tuple(action for action in legal_build_targets(projected, player_id)
                        if int(action.payload["total_cost"]) + surcharge <= _get_player(projected, player_id).elektro)
        if not targets:
            break
        chosen = min(targets, key=lambda action: (
            int(action.payload["total_cost"]) - adjustments.get(str(action.payload["city_id"]), 0),
            int(action.payload["total_cost"]), str(action.payload["city_id"]),
        ))
        city = str(chosen.payload["city_id"])
        cities.append(city)
        costs.append(int(chosen.payload["total_cost"]) + surcharge)
        projected = build_city(projected, player_id, city)
        holder = _get_player(projected, player_id)
        projected = _with_player(projected, replace(holder, elektro=holder.elektro - surcharge))
    return BuildProjection(tuple(cities), tuple(costs), player.connected_city_count)


def _can_overbuild(state: GameState, player_id: str, final_city_count: int) -> bool:
    if state.player_order[0] != player_id:
        return False
    player = _get_player(state, player_id)
    return all((final_city_count, player.largest_power_plant) >=
               (other.connected_city_count, other.largest_power_plant)
               for other in state.players if other.player_id != player_id)


def _hybrid_discard(state: GameState) -> tuple[int, int]:
    assert state.pending_decision is not None
    def value(action):
        mix = {r: int(action.payload.get(r, 0)) for r in ("coal", "oil")}
        prices = {r: state.resource_market.available_unit_prices(r) for r in mix}
        return (sum(n * (prices[r][0] if prices[r] else 8) for r, n in mix.items()),
                mix["coal"], mix["oil"])
    chosen = min(state.pending_decision.legal_actions, key=value)
    return int(chosen.payload.get("coal", 0)), int(chosen.payload.get("oil", 0))


def _portfolio_after_purchase(
    state: GameState, player_id: str, plant: PowerPlantCard, *, terminal_complete: bool = False,
) -> PlayerState:
    player = _get_player(state, player_id)
    limit = state.rules.player_count_rules[len(state.players)]["max_power_plants"]
    # Acquire first so fuel can transfer into the new plant's storage.
    state = add_power_plant_to_player(state, player_id, plant.price)
    if len(player.power_plants) >= limit:
        discard = _replacement_plant(player.power_plants, terminal_complete=terminal_complete)
        state = remove_power_plant_from_player(state, player_id, discard.price)
        if state.pending_decision is not None:
            coal, oil = _hybrid_discard(state)
            state = discard_resources_to_fit_storage(state, player_id, {"coal": coal, "oil": oil})
    return _get_player(state, player_id)


@dataclass(frozen=True)
class Competition:
    current: tuple[int, ...]
    future: tuple[int, ...]
    opponents: int

    def has(self, *prices: int) -> bool:
        return (len(prices) <= self.opponents and
                set(prices) <= set((*self.current, *self.future[:self.opponents])))


def _competition(state: GameState, player_id: str, plant_price: int) -> Competition:
    auction = state.auction_state
    unavailable = set(auction.players_with_plants + auction.players_passed_phase) if auction else set()
    opponents = sum(pid != player_id and pid not in unavailable for pid in state.player_order)
    return Competition(
        tuple(p.price for p in state.current_market if p.price != plant_price),
        tuple(p.price for p in state.future_market if not p.is_step_3_placeholder), opponents,
    )


def _opening_range(price: int, competition: Competition) -> tuple[int, int]:
    c = competition.has
    fixed = {3: (3, 5), 4: (4, 8), 6: (1, 2), 9: (9, 11),
             11: (0, 0), 12: (12, 14), 13: (15, 20)}
    if price in fixed:
        return fixed[price]
    if price == 5:
        return (5, 6) if c(4) else (6, 9)
    if price == 7:
        return (0, 0) if c(3) else (5, 8)
    if price == 8:
        return (0, 0) if c(4, 5) else ((8, 10) if c(4) or c(5) else (10, 12))
    if price == 10:
        if c(4, 8) or c(5, 8):
            return (0, 0)
        if c(8) or c(4, 5):
            return (9, 10)
        return (10, 12) if c(4) or c(5) else (13, 16)
    # Socket plants are normally outside the opening preference list. A long
    # six-player auction can expose one; allow only a mandatory minimum purchase.
    return (0, 0)


def _opening_priority(price: int, competition: Competition) -> int:
    c = competition.has
    if price == 13:
        return 0
    if price == 10 and not (c(4) or c(5) or c(8)):
        return 1
    if price == 8 and not (c(4) or c(5)):
        return 2
    if price in (9, 4, 5, 12):
        return {9: 3, 4: 4, 5: 5, 12: 6}[price]
    if price == 10 and (c(4) or c(5)) and not c(4, 5) and not c(8):
        return 7
    if price == 8 and (c(4) or c(5)) and not c(4, 5):
        return 8
    if price == 3:
        return 9
    if price == 7 and not c(3):
        return 10
    if price == 10 and ((c(4, 5) and not c(8)) or (c(8) and not (c(4) or c(5)))):
        return 11
    if price == 6:
        return 12
    return 99


def _endgame_plant(
    state: GameState, history: tuple[PowerPlantCard, ...], candidate: PowerPlantCard,
    *, owned: tuple[PowerPlantCard, ...] = (),
) -> bool:
    if len(history) >= 3:
        return bool(owned) and candidate.output_cities > min(p.output_cities for p in owned)
    end_cities = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    return candidate.output_cities >= ceil((end_cities - _capacity(history)) / (3 - len(history)) - 1)


def _estimated_market(state: GameState, player_id: str, plant_price: int | None = None) -> ResourceMarket:
    market = state.resource_market
    for opponent_id in reversed(_later_players(state, player_id)):
        opponent = _get_player(state, opponent_id)
        basket = _demand_purchase(opponent.power_plants, market, 1.5,
                                  opponent.resource_storage.resource_totals())
        market = _consume_estimated(market, basket)
    if state.phase == "auction":
        competition = _competition(state, player_id, plant_price or -1)
        for plant in (*state.current_market, *state.future_market):
            if plant.price != plant_price and competition.has(plant.price):
                market = _consume_estimated(market, _demand_purchase((plant,), market, 0.5))
    return market


@dataclass(frozen=True)
class AuctionProjection:
    plant: PowerPlantCard
    base: int
    cities: int
    endgame: bool
    discounted: bool
    discount_eligible: bool
    eligible: bool
    margin: int
    efficiency: float
    priority: int


class HumanExpHeuristicsAiController(BaseAiController):
    controller = CONTROLLER

    def __init__(self) -> None:
        self._seen: set[int] = set()
        self._game_key: tuple | None = None
        self._resource_key: tuple | None = None
        self._resource_basket: dict[str, int] = {}
        self._resource_details: dict[str, object] = {}
        self._discard_choices: dict[str, int] = {}
        self._observed_owned_plants: dict[str, list[int]] = {}
        self._terminal_history: dict[str, list[PowerPlantCard]] = {}

    def _observe_terminal_history(self, state: GameState, pid: str) -> None:
        # Observe actual ownership, including an awarded plant awaiting discard.
        # Hypothetical auction portfolios never enter this method. Old plants can
        # qualify as soon as a new terminal plant lowers the next threshold.
        observed = self._observed_owned_plants.setdefault(pid, [])
        history = self._terminal_history.setdefault(pid, [])
        owned = {plant.price: plant for plant in _get_player(state, pid).power_plants}
        observed.extend(price for price in owned if price not in observed)
        while len(history) < 3:
            recognized = {plant.price for plant in history}
            chosen = next((owned[price] for price in observed
                           if price in owned and price not in recognized
                           and _endgame_plant(state, tuple(history), owned[price])), None)
            if chosen is None:
                break
            history.append(chosen)

    def _sample(self, state: GameState, player_id: str, key: str, low: int, high: int) -> int:
        # A private, reproducible stream per seat/plant; never touch deck RNG.
        return random.Random(f"{CONTROLLER}:{state.config.seed}:{player_id}:{key}").randint(low, high)

    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        state, pid = snapshot.state, request.player_id
        game_key = (state.config.seed, state.config.map_id, tuple(p.player_id for p in state.players))
        if game_key != self._game_key:
            self._seen.clear()
            self._resource_key = None
            self._discard_choices.clear()
            self._observed_owned_plants.clear()
            self._terminal_history.clear()
            self._game_key = game_key
        self._observe_terminal_history(state, pid)
        self._seen.update(p.price for p in (*state.current_market, *state.future_market))
        self._seen.update(p.price for player in state.players for p in player.power_plants)
        details: dict[str, object] = {}
        if state.pending_decision is not None:
            if state.pending_decision.decision_type == "discard_power_plant":
                plants = _get_player(state, pid).power_plants
                fallback = _replacement_plant(plants, terminal_complete=len(self._terminal_history.get(pid, [])) >= 3)
                price = self._discard_choices.get(pid, fallback.price)
                if price not in {plant.price for plant in plants}:
                    price = fallback.price
                intent = GuiIntent.discard_plant(pid, price)
            else:
                coal, oil = _hybrid_discard(state)
                intent = GuiIntent.discard_hybrid_resources(pid, coal, oil)
        elif request.phase == "auction":
            if state.round_number == 1:
                intent, details = self._opening_auction(state, request)
            else:
                intent, details = self._later_auction(state, request)
        elif request.phase == "buy_resources":
            intent, details = self._resources(state, request)
        elif request.phase == "build_houses":
            intent, details = self._build(state, pid)
        elif request.phase == "bureaucracy":
            plans = _choose_best_generation_plans(state, pid)
            intent = GuiIntent.run_plants(pid, plans) if plans else GuiIntent.skip_bureaucracy(pid)
        else:
            raise ModelValidationError(f"unsupported request phase {request.phase!r}")
        if intent.intent_type in {"auction_start", "auction_bid"}:
            plants = _get_player(state, pid).power_plants
            if len(plants) >= state.rules.player_count_rules[len(state.players)]["max_power_plants"]:
                self._discard_choices[pid] = _replacement_plant(
                    plants, terminal_complete=len(self._terminal_history.get(pid, [])) >= 3,
                ).price
        self.log_state(snapshot, request, label="humanexp_heuristics_decision",
                       state={"decision_type": request.decision_type, **details,
                              "terminal_history": [{"plant": p.price, "output": p.output_cities}
                                                   for p in self._terminal_history.get(pid, [])],
                              "intent": intent.to_dict()}, message="经验启发式v1选择行动。")
        return intent

    def _better_next_probability(self, state: GameState, pid: str, current: int) -> float:
        if not state.future_market or not state.power_plant_draw_stack:
            return 0.0
        competition = _competition(state, pid, current)
        current_priority = _opening_priority(current, competition)
        # Only the public back of the top card is observed, never its identity.
        back = state.power_plant_draw_stack[0].deck_back
        unseen = [p.price for p in _catalog() if p.deck_back == "plug" and p.price not in self._seen]
        draws = unseen if back == "plug" else [10000]
        if not draws:
            return 0.0
        better = 0
        for drawn in draws:
            visible = sorted((*competition.current, *competition.future, drawn))
            next_price = min(drawn, competition.future[0])
            # Passing awards the current plant to an opponent, who leaves this
            # round's auctions before we evaluate the newly available plant.
            next_competition = Competition(tuple(p for p in visible[:4] if p != next_price),
                                           tuple(visible[4:]), max(0, competition.opponents - 1))
            better += _opening_priority(next_price, next_competition) < current_priority
        return better / len(draws)

    def _opening_auction(self, state: GameState, request: TurnRequest):
        pid = request.player_id
        if request.decision_type == "auction_start":
            candidates = []
            for action in request.legal_actions:
                if action.action_type != "auction_start":
                    continue
                price, base = int(action.payload["plant_price"]), int(action.payload["min_bid"])
                competition = _competition(state, pid, price)
                low, high = _opening_range(price, competition)
                cap = self._sample(state, pid, f"opening:{price}", low, high)
                candidates.append((price, base, _opening_priority(price, competition), cap, competition.opponents))
            acceptable = [item for item in candidates if item[1] <= item[3]]
            # Every first-round buyer obeys the cap, including the last one.
            # If none qualify, the mandatory purchase uses the cheapest legal
            # opening price rather than the preference ranking.
            if acceptable:
                chosen = min(acceptable, key=lambda item: (item[2], item[0]))
            else:
                chosen = min(candidates, key=lambda item: (item[1], item[0]))
            return GuiIntent.auction_start(pid, chosen[0], chosen[1]), {
                "price_cap": chosen[3], "priority": chosen[2], "forced_purchase": not acceptable,
            }
        action = next((a for a in request.legal_actions if a.action_type == "auction_bid"), None)
        if action is None:
            return GuiIntent.auction_pass(pid), {"reason": "cannot_afford_bid"}
        price, bid = int(action.payload["plant_price"]), int(action.payload["min_bid"])
        low, high = _opening_range(price, _competition(state, pid, price))
        cap = self._sample(state, pid, f"opening:{price}", low, high)
        standing_bid = int(state.auction_state.current_bid) if state.auction_state else bid - 1
        probability = self._better_next_probability(state, pid, price) if cap - 2 <= standing_bid <= cap - 1 else 0.0
        passes = bid > cap or probability >= 0.5
        return (GuiIntent.auction_pass(pid) if passes else GuiIntent.auction_bid(pid, bid)), {
            "price_cap": cap, "better_next_probability": probability,
        }

    def _auction_projection(self, state: GameState, pid: str, plant: PowerPlantCard,
                            curve: BuildProjection) -> AuctionProjection:
        player = _get_player(state, pid)
        self._observe_terminal_history(state, pid)
        history = tuple(self._terminal_history.get(pid, []))
        terminal_complete = len(history) >= 3
        power = _capacity(player.power_plants)
        limit = state.rules.player_count_rules[len(state.players)]["max_power_plants"]
        full = len(player.power_plants) >= limit
        worst = _worst(player.power_plants) if full else None
        discounted = bool(state.auction_state and state.auction_state.discount_token_plant_price == plant.price)
        base = 1 if discounted else plant.price
        portfolio = _portfolio_after_purchase(state, pid, plant, terminal_complete=terminal_complete)
        market = _estimated_market(state, pid, plant.price)
        fuel = min((cost for _, cost in _fuel_options(portfolio, portfolio.power_plants, market)), default=inf)
        cities = curve.affordable(int(player.elektro - base - fuel))[0] if fuel <= player.elektro - base else -1
        endgame = _endgame_plant(state, history, plant, owned=player.power_plants)
        required = power if endgame and not terminal_complete else power + 1
        upgrade = (terminal_complete or not worst or
                   (plant.output_cities >= worst.output_cities + 2 and plant.price > worst.price * 1.5))
        qualifies = (cities >= required and (discounted or upgrade)
                     and (not terminal_complete or endgame))
        target_cost = sum(curve.costs[:max(0, required - curve.starting_count)])
        margin = max(0, int(player.elektro - base - fuel - target_cost)) if qualifies else 0
        run_cost = min((_quote(market, demand) for demand in _fuel_demands((plant,))), default=inf)
        efficiency = inf if run_cost == 0 else plant.output_cities / run_cost
        special = discounted and (qualifies if terminal_complete else
                                   (not full or plant.price >= worst.price * 1.5))
        priority = (0 if qualifies and endgame and cities >= power + 1 else
                    1 if special else 2 if qualifies and not endgame else 3 if qualifies else 99)
        return AuctionProjection(plant, base, cities, endgame, discounted, special, qualifies, margin, efficiency, priority)

    def _later_auction(self, state: GameState, request: TurnRequest):
        pid = request.player_id
        player = _get_player(state, pid)
        power = _capacity(player.power_plants)
        curve = _build_projection(state, pid, surcharge=len(_later_players(state, pid)))
        if request.decision_type == "auction_start":
            self._observe_terminal_history(state, pid)
            terminal_complete = len(self._terminal_history.get(pid, [])) >= 3
            market = _estimated_market(state, pid)
            fuel = min((cost for _, cost in _fuel_options(player, player.power_plants, market)), default=inf)
            cities = curve.affordable(int(player.elektro - fuel))[0] if fuel <= player.elektro else -1
            candidates = []
            for action in request.legal_actions:
                if action.action_type != "auction_start":
                    continue
                plant = next(p for p in state.current_market if p.price == int(action.payload["plant_price"]))
                candidate = self._auction_projection(state, pid, plant, curve)
                if terminal_complete:
                    eligible = candidate.eligible
                else:
                    eligible = candidate.discount_eligible or (not candidate.discounted and power <= cities - 2 and candidate.eligible)
                if eligible:
                    candidates.append(candidate)
            details = {"capacity": power, "projected_cities": cities,
                       "candidates": [{"plant": c.plant.price, "cities": c.cities, "endgame": c.endgame,
                                       "priority": c.priority, "margin": c.margin} for c in candidates]}
            if not candidates:
                return GuiIntent.auction_pass(pid), details
            chosen = min(candidates, key=lambda c: (c.priority, -c.efficiency, c.plant.price))
            return GuiIntent.auction_start(pid, chosen.plant.price, chosen.base), details
        action = next((a for a in request.legal_actions if a.action_type == "auction_bid"), None)
        if action is None:
            return GuiIntent.auction_pass(pid), {"reason": "cannot_afford_bid"}
        plant = next(p for p in state.current_market if p.price == int(action.payload["plant_price"]))
        candidate = self._auction_projection(state, pid, plant, curve)
        premium = 10 if candidate.endgame and candidate.cities >= power + 1 else 3
        cap = self._sample(state, pid, f"later:{plant.price}", candidate.base,
                           candidate.base + min(candidate.margin, premium))
        bid = int(action.payload["min_bid"])
        intent = (GuiIntent.auction_bid(pid, bid) if candidate.eligible and bid <= cap else GuiIntent.auction_pass(pid))
        return intent, {"capacity": power, "projected_cities": candidate.cities,
                        "endgame": candidate.endgame, "margin": candidate.margin, "price_cap": cap,
                        "eligible": candidate.eligible}

    def _plan_resources(self, state: GameState, pid: str, available: tuple[str, ...]):
        player = _get_player(state, pid)
        adjustments = _contest_adjustments(state, pid)
        curves: dict[int, BuildProjection] = {}
        best_key = None
        best_basket: dict[str, int] = dict.fromkeys(RESOURCE_TYPES, 0)
        best_details: dict[str, object] = {}
        reserve = 0
        for count in range(len(player.power_plants) + 1):
            for plants in combinations(player.power_plants, count):
                for basket, fuel_cost in _fuel_options(player, plants, state.resource_market, available=available):
                    if fuel_cost > player.elektro:
                        continue
                    cash = player.elektro - fuel_cost
                    if cash not in curves:
                        curves[cash] = _build_projection(state, pid, surcharge=len(_later_players(state, pid)),
                                                        adjustments=adjustments, budget=cash)
                    curve = curves[cash]
                    potential, _ = curve.affordable(cash)
                    limit = None if _can_overbuild(state, pid, potential) else _capacity(player.power_plants)
                    cities, build_cost = curve.affordable(cash, limit)
                    powered = min(cities, _capacity(plants))
                    income = pay_income(state.rules, powered)
                    key = (income - fuel_cost, -fuel_cost, cities, -sum(basket.values()), tuple(-basket[r] for r in RESOURCE_TYPES))
                    if best_key is None or key > best_key:
                        best_key, best_basket, reserve = key, basket, build_cost
                        best_details = {"plants": [p.price for p in plants], "fuel_cost": fuel_cost,
                                        "projected_cities": cities, "income": income, "build_reserve": reserve,
                                        "net_income": income - fuel_cost}
        projected = purchase_resources(state, pid, best_basket) if any(best_basket.values()) else state
        consumption = dict.fromkeys(RESOURCE_TYPES, 0)
        for other in state.players:
            demand = _demand_purchase(other.power_plants, state.resource_market, 1.0)
            for resource in RESOURCE_TYPES:
                consumption[resource] += demand[resource]
        refill = state.rules.player_count_rules[len(state.players)]["resource_refill"][f"step_{state.step}"]
        gaps = {r: consumption[r] - int(refill[r]) for r in RESOURCE_TYPES}
        result = dict(best_basket)
        for resource in sorted(available, key=lambda r: (-gaps[r], RESOURCE_TYPES.index(r))):
            if gaps[resource] <= 0:
                continue
            while True:
                holder = _get_player(projected, pid)
                prices = projected.resource_market.available_unit_prices(resource)
                if (not prices or holder.elektro - reserve < prices[0] or
                        not can_store_resources(holder, {resource: 1})):
                    break
                projected = purchase_resources(projected, pid, {resource: 1})
                result[resource] += 1
        return result, {**best_details, "resource_gaps": gaps, "basket": dict(result)}

    def _resources(self, state: GameState, request: TurnRequest):
        resource = str(request.metadata["resource"])
        index = RESOURCE_TYPES.index(resource)
        key = (self._game_key, state.round_number, request.player_id)
        if key != self._resource_key:
            self._resource_basket, self._resource_details = self._plan_resources(state, request.player_id, RESOURCE_TYPES[index:])
            self._resource_key = key
        action = next(a for a in request.legal_actions if a.action_type == "buy_resource")
        amount = min(self._resource_basket.get(resource, 0), int(action.payload["max_affordable_units"]))
        return GuiIntent.buy_resource(request.player_id, resource, amount), self._resource_details

    def _build(self, state: GameState, pid: str):
        player = _get_player(state, pid)
        adjustments = _contest_adjustments(state, pid)
        curve = _build_projection(state, pid, adjustments=adjustments, budget=player.elektro)
        count, _ = curve.affordable(player.elektro)
        overbuild = _can_overbuild(state, pid, count)
        details = {"capacity": _capacity(player.power_plants), "can_overbuild": overbuild,
                   "contest_adjustments": adjustments}
        targets = legal_build_targets(state, pid)
        if not targets or (not overbuild and player.connected_city_count >= _capacity(player.power_plants)):
            return GuiIntent.finish_building(pid), details
        chosen = min(targets, key=lambda action: (
            int(action.payload["total_cost"]) - adjustments.get(str(action.payload["city_id"]), 0),
            int(action.payload["total_cost"]), str(action.payload["city_id"]),
        ))
        return GuiIntent.commit_build(pid, [str(chosen.payload["city_id"])]), details
