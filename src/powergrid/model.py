from __future__ import annotations

from itertools import combinations, permutations
import heapq
import random
from dataclasses import dataclass, field, replace
from typing import Any

from .rules_data import (
    CityDefinition,
    ConnectionDefinition,
    MapDefinition,
    PowerPlantDefinition,
    RegionDefinition,
    RuleTables,
    load_map,
    load_power_plants,
    load_rule_tables,
)


RESOURCE_TYPES = ("coal", "oil", "garbage", "uranium")
CONTROLLER_TYPES = ("human", "ai")
GAME_PHASES = ("setup", "determine_order", "auction", "buy_resources", "build_houses", "bureaucracy")
ROUND_PHASES = ("determine_order", "auction", "buy_resources", "build_houses", "bureaucracy")
PLAYER_COLORS = ("black", "purple", "green", "blue", "yellow", "red")
HOUSES_PER_PLAYER = 22
STEP_3_PLACEHOLDER_PRICE = 10000


class ModelValidationError(ValueError):
    """Raised when a model object is internally inconsistent."""


@dataclass
class SeatConfig:
    player_id: str
    name: str
    controller: str = "human"

    def __post_init__(self) -> None:
        if not self.player_id:
            raise ModelValidationError("player_id must be non-empty")
        if not self.name:
            raise ModelValidationError("player name must be non-empty")
        if self.controller not in CONTROLLER_TYPES:
            raise ModelValidationError(
                f"controller must be one of {CONTROLLER_TYPES}, got {self.controller!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "controller": self.controller,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SeatConfig":
        return cls(
            player_id=payload["player_id"],
            name=payload["name"],
            controller=payload["controller"],
        )


@dataclass
class GameConfig:
    map_id: str
    players: tuple[SeatConfig, ...]
    seed: int = 0
    selected_regions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.players = tuple(self.players)
        self.selected_regions = tuple(self.selected_regions)
        if not self.map_id:
            raise ModelValidationError("map_id must be non-empty")
        if not 3 <= len(self.players) <= 6:
            raise ModelValidationError("Power Grid v1 supports 3-6 players")
        player_ids = [player.player_id for player in self.players]
        if len(player_ids) != len(set(player_ids)):
            raise ModelValidationError("player ids must be unique")
        if len(self.selected_regions) != len(set(self.selected_regions)):
            raise ModelValidationError("selected regions must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "players": [player.to_dict() for player in self.players],
            "seed": self.seed,
            "selected_regions": list(self.selected_regions),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GameConfig":
        return cls(
            map_id=payload["map_id"],
            players=tuple(SeatConfig.from_dict(player) for player in payload["players"]),
            seed=int(payload["seed"]),
            selected_regions=tuple(payload.get("selected_regions", [])),
        )


@dataclass
class PowerPlantCard:
    price: int
    resource_types: tuple[str, ...]
    resource_cost: int
    output_cities: int
    deck_back: str
    is_hybrid: bool
    is_ecological: bool

    def __post_init__(self) -> None:
        self.resource_types = tuple(self.resource_types)
        if self.price <= 0:
            raise ModelValidationError("power plant price must be positive")
        if self.resource_cost < 0:
            raise ModelValidationError("power plant resource cost cannot be negative")
        if self.output_cities <= 0 and not self.is_step_3_placeholder:
            raise ModelValidationError("power plant output must be positive")
        if self.deck_back not in {"plug", "socket", "step3"}:
            raise ModelValidationError("deck_back must be 'plug' or 'socket'")
        if self.is_step_3_placeholder:
            if self.resource_types or self.resource_cost != 0 or self.output_cities != 0:
                raise ModelValidationError("Step 3 placeholder may not consume resources or power cities")
            if self.deck_back != "step3":
                raise ModelValidationError("Step 3 placeholder must use deck_back='step3'")
            if self.is_hybrid or self.is_ecological:
                raise ModelValidationError("Step 3 placeholder may not be hybrid or ecological")
            return
        if self.is_ecological != (len(self.resource_types) == 0):
            raise ModelValidationError("ecological flag must match resource types")
        if self.is_hybrid and set(self.resource_types) != {"coal", "oil"}:
            raise ModelValidationError("hybrid plants must use coal and oil")
        if not self.is_hybrid and len(self.resource_types) > 1:
            raise ModelValidationError("non-hybrid plants may only use one resource type")
        if any(resource not in RESOURCE_TYPES for resource in self.resource_types):
            raise ModelValidationError("unknown power plant resource type")

    @property
    def max_storage(self) -> int:
        return self.resource_cost * 2

    @property
    def is_step_3_placeholder(self) -> bool:
        return self.deck_back == "step3"

    @classmethod
    def from_definition(cls, definition: PowerPlantDefinition) -> "PowerPlantCard":
        return cls(
            price=definition.price,
            resource_types=definition.resource_types,
            resource_cost=definition.resource_cost,
            output_cities=definition.output_cities,
            deck_back=definition.deck_back,
            is_hybrid=definition.is_hybrid,
            is_ecological=definition.is_ecological,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "resource_types": list(self.resource_types),
            "resource_cost": self.resource_cost,
            "output_cities": self.output_cities,
            "deck_back": self.deck_back,
            "is_hybrid": self.is_hybrid,
            "is_ecological": self.is_ecological,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PowerPlantCard":
        return cls(
            price=int(payload["price"]),
            resource_types=tuple(payload["resource_types"]),
            resource_cost=int(payload["resource_cost"]),
            output_cities=int(payload["output_cities"]),
            deck_back=payload["deck_back"],
            is_hybrid=bool(payload["is_hybrid"]),
            is_ecological=bool(payload["is_ecological"]),
        )

    @classmethod
    def step_3_placeholder(cls) -> "PowerPlantCard":
        return cls(
            price=STEP_3_PLACEHOLDER_PRICE,
            resource_types=(),
            resource_cost=0,
            output_cities=0,
            deck_back="step3",
            is_hybrid=False,
            is_ecological=False,
        )


@dataclass
class ResourceStorage:
    coal: int = 0
    oil: int = 0
    hybrid_coal: int = 0
    hybrid_oil: int = 0
    garbage: int = 0
    uranium: int = 0

    def __post_init__(self) -> None:
        self.coal = int(self.coal)
        self.oil = int(self.oil)
        self.hybrid_coal = int(self.hybrid_coal)
        self.hybrid_oil = int(self.hybrid_oil)
        self.garbage = int(self.garbage)
        self.uranium = int(self.uranium)
        if min(
            self.coal,
            self.oil,
            self.hybrid_coal,
            self.hybrid_oil,
            self.garbage,
            self.uranium,
        ) < 0:
            raise ModelValidationError("resource storage values cannot be negative")

    @property
    def hybrid_used(self) -> int:
        return self.hybrid_coal + self.hybrid_oil

    def total(self, resource: str) -> int:
        _validate_resource_name(resource)
        if resource == "coal":
            return self.coal + self.hybrid_coal
        if resource == "oil":
            return self.oil + self.hybrid_oil
        return getattr(self, resource)

    def resource_totals(self) -> dict[str, int]:
        return {resource: self.total(resource) for resource in RESOURCE_TYPES}

    def to_dict(self) -> dict[str, int]:
        return {
            "coal": self.coal,
            "oil": self.oil,
            "hybrid_coal": self.hybrid_coal,
            "hybrid_oil": self.hybrid_oil,
            "garbage": self.garbage,
            "uranium": self.uranium,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResourceStorage":
        return cls(
            coal=int(payload.get("coal", 0)),
            oil=int(payload.get("oil", 0)),
            hybrid_coal=int(payload.get("hybrid_coal", 0)),
            hybrid_oil=int(payload.get("hybrid_oil", 0)),
            garbage=int(payload.get("garbage", 0)),
            uranium=int(payload.get("uranium", 0)),
        )


@dataclass
class PreparedDeck:
    current_market: tuple[PowerPlantCard, ...]
    future_market: tuple[PowerPlantCard, ...]
    draw_stack: tuple[PowerPlantCard, ...]
    step_3_card_pending: bool
    removed_plant_prices: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.current_market = tuple(self.current_market)
        self.future_market = tuple(self.future_market)
        self.draw_stack = tuple(self.draw_stack)
        self.removed_plant_prices = tuple(self.removed_plant_prices)
        if len(self.current_market) != 4 or len(self.future_market) != 4:
            raise ModelValidationError("prepared deck must expose 4 current and 4 future plants")
        if not self.draw_stack:
            raise ModelValidationError("prepared deck draw stack cannot be empty")
        all_prices = (
            [plant.price for plant in self.current_market]
            + [plant.price for plant in self.future_market]
            + [plant.price for plant in self.draw_stack]
        )
        if len(all_prices) != len(set(all_prices)):
            raise ModelValidationError("prepared deck may not duplicate visible or draw-stack plants")
        if [plant.price for plant in self.current_market] != sorted(
            plant.price for plant in self.current_market
        ):
            raise ModelValidationError("current market must be sorted by price")
        if [plant.price for plant in self.future_market] != sorted(
            plant.price for plant in self.future_market
        ):
            raise ModelValidationError("future market must be sorted by price")


@dataclass
class ResourceMarket:
    market: dict[str, dict[int, int]]
    supply: dict[str, int]

    def __post_init__(self) -> None:
        self.market = {
            resource: {int(price): int(amount) for price, amount in price_bands.items()}
            for resource, price_bands in self.market.items()
        }
        self.supply = {resource: int(amount) for resource, amount in self.supply.items()}
        if set(self.market) != set(RESOURCE_TYPES):
            raise ModelValidationError("resource market must cover coal, oil, garbage, and uranium")
        if set(self.supply) != set(RESOURCE_TYPES):
            raise ModelValidationError("resource supply must cover coal, oil, garbage, and uranium")
        for resource in RESOURCE_TYPES:
            if any(amount < 0 for amount in self.market[resource].values()):
                raise ModelValidationError(f"{resource} market counts cannot be negative")
            if self.supply[resource] < 0:
                raise ModelValidationError(f"{resource} supply cannot be negative")

    @classmethod
    def from_rule_tables(cls, rules: RuleTables) -> "ResourceMarket":
        market: dict[str, dict[int, int]] = {}
        supply: dict[str, int] = {}
        for resource in RESOURCE_TYPES:
            track = rules.resource_market_tracks[resource]
            capacities = {int(price): int(amount) for price, amount in track["capacity_by_price"].items()}
            starting_prices = {int(price) for price in track["starting_prices"]}
            market[resource] = {
                price: capacity if price in starting_prices else 0
                for price, capacity in sorted(capacities.items())
            }
            supply[resource] = int(rules.resource_supply[resource]) - sum(market[resource].values())
        return cls(market=market, supply=supply)

    def total_in_market(self, resource: str) -> int:
        return sum(self.market[resource].values())

    def available_unit_prices(self, resource: str) -> tuple[int, ...]:
        _validate_resource_name(resource)
        prices: list[int] = []
        for price, amount in sorted(self.market[resource].items()):
            prices.extend([price] * amount)
        return tuple(prices)

    def quote_purchase_cost(self, resource: str, amount: int) -> int:
        _validate_resource_name(resource)
        if amount < 0:
            raise ModelValidationError("resource purchase amount cannot be negative")
        available_prices = self.available_unit_prices(resource)
        if amount > len(available_prices):
            raise ModelValidationError(f"not enough {resource} available in the market")
        return sum(available_prices[:amount])

    def remove_from_market(self, resource: str, amount: int) -> "ResourceMarket":
        _validate_resource_name(resource)
        if amount < 0:
            raise ModelValidationError("resource removal amount cannot be negative")
        if amount == 0:
            return self
        if amount > self.total_in_market(resource):
            raise ModelValidationError(f"not enough {resource} available in the market")
        updated_market = {
            name: dict(price_bands) for name, price_bands in self.market.items()
        }
        remaining = amount
        for price in sorted(updated_market[resource]):
            if remaining == 0:
                break
            available = updated_market[resource][price]
            if available == 0:
                continue
            removed = min(available, remaining)
            updated_market[resource][price] -= removed
            remaining -= removed
        return ResourceMarket(market=updated_market, supply=dict(self.supply))

    def add_to_supply(self, resource: str, amount: int) -> "ResourceMarket":
        _validate_resource_name(resource)
        if amount < 0:
            raise ModelValidationError("resource supply addition amount cannot be negative")
        if amount == 0:
            return self
        updated_supply = dict(self.supply)
        updated_supply[resource] += amount
        return ResourceMarket(
            market={name: dict(price_bands) for name, price_bands in self.market.items()},
            supply=updated_supply,
        )

    def add_resources_to_supply(self, resources: dict[str, int]) -> "ResourceMarket":
        updated_market: ResourceMarket = self
        for resource, amount in resources.items():
            updated_market = updated_market.add_to_supply(resource, int(amount))
        return updated_market

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": {
                resource: {str(price): amount for price, amount in price_bands.items()}
                for resource, price_bands in self.market.items()
            },
            "supply": dict(self.supply),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResourceMarket":
        return cls(
            market={
                resource: {int(price): int(amount) for price, amount in price_bands.items()}
                for resource, price_bands in payload["market"].items()
            },
            supply={resource: int(amount) for resource, amount in payload["supply"].items()},
        )


@dataclass
class PlayerState:
    player_id: str
    name: str
    controller: str
    color: str
    elektro: int
    houses_in_supply: int
    network_city_ids: tuple[str, ...] = field(default_factory=tuple)
    power_plants: tuple[PowerPlantCard, ...] = field(default_factory=tuple)
    resource_storage: ResourceStorage = field(default_factory=ResourceStorage)
    turn_order_position: int = 0

    def __post_init__(self) -> None:
        self.network_city_ids = tuple(self.network_city_ids)
        self.power_plants = tuple(self.power_plants)
        if not isinstance(self.resource_storage, ResourceStorage):
            self.resource_storage = ResourceStorage.from_dict(dict(self.resource_storage))
        if self.controller not in CONTROLLER_TYPES:
            raise ModelValidationError("player controller must be human or ai")
        if self.elektro < 0:
            raise ModelValidationError("player elektro cannot be negative")
        if self.houses_in_supply < 0:
            raise ModelValidationError("player houses_in_supply cannot be negative")
        if self.houses_in_supply + len(self.network_city_ids) != HOUSES_PER_PLAYER:
            raise ModelValidationError("player houses must add up to 22")
        if len(set(self.network_city_ids)) != len(self.network_city_ids):
            raise ModelValidationError("player network cannot contain duplicate cities")
        plant_prices = [plant.price for plant in self.power_plants]
        if len(plant_prices) != len(set(plant_prices)):
            raise ModelValidationError("player power plants must be unique by price")
        if self.turn_order_position < 0:
            raise ModelValidationError("turn order position cannot be negative")
        if not _resource_storage_fits_power_plants(self.resource_storage, self.power_plants):
            raise ModelValidationError("player resource storage exceeds available storage spaces")

    @property
    def connected_city_count(self) -> int:
        return len(self.network_city_ids)

    @property
    def largest_power_plant(self) -> int:
        return max((plant.price for plant in self.power_plants), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "controller": self.controller,
            "color": self.color,
            "elektro": self.elektro,
            "houses_in_supply": self.houses_in_supply,
            "network_city_ids": list(self.network_city_ids),
            "power_plants": [plant.to_dict() for plant in self.power_plants],
            "resource_storage": self.resource_storage.to_dict(),
            "turn_order_position": self.turn_order_position,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlayerState":
        return cls(
            player_id=payload["player_id"],
            name=payload["name"],
            controller=payload["controller"],
            color=payload["color"],
            elektro=int(payload["elektro"]),
            houses_in_supply=int(payload["houses_in_supply"]),
            network_city_ids=tuple(payload.get("network_city_ids", [])),
            power_plants=tuple(
                PowerPlantCard.from_dict(plant) for plant in payload.get("power_plants", [])
            ),
            resource_storage=ResourceStorage.from_dict(payload.get("resource_storage", {})),
            turn_order_position=int(payload["turn_order_position"]),
        )


@dataclass
class Action:
    action_type: str
    player_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_type:
            raise ModelValidationError("action_type must be non-empty")
        if not self.player_id:
            raise ModelValidationError("player_id must be non-empty")
        self.payload = dict(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "player_id": self.player_id,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Action":
        return cls(
            action_type=payload["action_type"],
            player_id=payload["player_id"],
            payload=dict(payload.get("payload", {})),
        )


@dataclass
class DecisionRequest:
    player_id: str
    decision_type: str
    prompt: str
    legal_actions: tuple[Action, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.legal_actions = tuple(self.legal_actions)
        self.metadata = dict(self.metadata)
        if not self.player_id:
            raise ModelValidationError("decision request player_id must be non-empty")
        if not self.decision_type:
            raise ModelValidationError("decision_type must be non-empty")
        if not self.prompt:
            raise ModelValidationError("decision prompt must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "decision_type": self.decision_type,
            "prompt": self.prompt,
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecisionRequest":
        return cls(
            player_id=payload["player_id"],
            decision_type=payload["decision_type"],
            prompt=payload["prompt"],
            legal_actions=tuple(Action.from_dict(action) for action in payload.get("legal_actions", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class AuctionState:
    current_chooser_id: str | None = None
    discount_token_plant_price: int | None = None
    players_with_plants: tuple[str, ...] = field(default_factory=tuple)
    players_passed_phase: tuple[str, ...] = field(default_factory=tuple)
    active_plant_price: int | None = None
    current_bid: int | None = None
    highest_bidder_id: str | None = None
    active_bidders: tuple[str, ...] = field(default_factory=tuple)
    next_bidder_id: str | None = None

    def __post_init__(self) -> None:
        self.players_with_plants = tuple(self.players_with_plants)
        self.players_passed_phase = tuple(self.players_passed_phase)
        self.active_bidders = tuple(self.active_bidders)
        if len(set(self.players_with_plants)) != len(self.players_with_plants):
            raise ModelValidationError("auction buyers must be unique")
        if len(set(self.players_passed_phase)) != len(self.players_passed_phase):
            raise ModelValidationError("auction phase passes must be unique")
        if set(self.players_with_plants) & set(self.players_passed_phase):
            raise ModelValidationError("auction buyers and phase passes may not overlap")
        has_active_auction = self.active_plant_price is not None
        if has_active_auction:
            if self.current_bid is None or self.highest_bidder_id is None:
                raise ModelValidationError("active auction must define a bid and highest bidder")
            if self.current_bid < 1:
                raise ModelValidationError("active auction bid must be positive")
            if len(self.active_bidders) < 1:
                raise ModelValidationError("active auction must have at least one bidder")
            if self.highest_bidder_id not in self.active_bidders:
                raise ModelValidationError("highest bidder must still be active in the auction")
            if len(self.active_bidders) > 1 and self.next_bidder_id is None:
                raise ModelValidationError("multi-player auction must define the next bidder")
            if self.next_bidder_id is not None and self.next_bidder_id not in self.active_bidders:
                raise ModelValidationError("next bidder must still be active in the auction")
        else:
            if any(
                value is not None
                for value in (self.current_bid, self.highest_bidder_id, self.next_bidder_id)
            ):
                raise ModelValidationError("inactive auction may not define bid or bidder metadata")
            if self.active_bidders:
                raise ModelValidationError("inactive auction may not carry active bidders")

    @property
    def has_active_auction(self) -> bool:
        return self.active_plant_price is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_chooser_id": self.current_chooser_id,
            "discount_token_plant_price": self.discount_token_plant_price,
            "players_with_plants": list(self.players_with_plants),
            "players_passed_phase": list(self.players_passed_phase),
            "active_plant_price": self.active_plant_price,
            "current_bid": self.current_bid,
            "highest_bidder_id": self.highest_bidder_id,
            "active_bidders": list(self.active_bidders),
            "next_bidder_id": self.next_bidder_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AuctionState":
        return cls(
            current_chooser_id=payload.get("current_chooser_id"),
            discount_token_plant_price=payload.get("discount_token_plant_price"),
            players_with_plants=tuple(payload.get("players_with_plants", [])),
            players_passed_phase=tuple(payload.get("players_passed_phase", [])),
            active_plant_price=payload.get("active_plant_price"),
            current_bid=payload.get("current_bid"),
            highest_bidder_id=payload.get("highest_bidder_id"),
            active_bidders=tuple(payload.get("active_bidders", [])),
            next_bidder_id=payload.get("next_bidder_id"),
        )


@dataclass
class PlantRunPlan:
    plant_price: int
    resource_mix: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.plant_price = int(self.plant_price)
        if self.plant_price <= 0:
            raise ModelValidationError("plant run plan must reference a positive plant price")
        self.resource_mix = _normalize_resource_mix(self.resource_mix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plant_price": self.plant_price,
            "resource_mix": dict(self.resource_mix),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlantRunPlan":
        return cls(
            plant_price=int(payload["plant_price"]),
            resource_mix=dict(payload.get("resource_mix", {})),
        )


@dataclass
class WinnerResult:
    winner_ids: tuple[str, ...]
    powered_cities: dict[str, int]
    money: dict[str, int]
    connected_cities: dict[str, int]

    def __post_init__(self) -> None:
        self.winner_ids = tuple(self.winner_ids)
        self.powered_cities = {player_id: int(value) for player_id, value in self.powered_cities.items()}
        self.money = {player_id: int(value) for player_id, value in self.money.items()}
        self.connected_cities = {
            player_id: int(value) for player_id, value in self.connected_cities.items()
        }
        if not self.winner_ids:
            raise ModelValidationError("winner result must include at least one winner")

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_ids": list(self.winner_ids),
            "powered_cities": dict(self.powered_cities),
            "money": dict(self.money),
            "connected_cities": dict(self.connected_cities),
        }


@dataclass
class BureaucracySummary:
    powered_cities: dict[str, int]
    income_paid: dict[str, int]
    triggered_step_2: bool
    triggered_step_3: bool
    refill_step_used: int
    game_end_triggered: bool
    winner_result: WinnerResult | None = None

    def __post_init__(self) -> None:
        self.powered_cities = {player_id: int(value) for player_id, value in self.powered_cities.items()}
        self.income_paid = {player_id: int(value) for player_id, value in self.income_paid.items()}
        self.refill_step_used = int(self.refill_step_used)

    def to_dict(self) -> dict[str, Any]:
        return {
            "powered_cities": dict(self.powered_cities),
            "income_paid": dict(self.income_paid),
            "triggered_step_2": self.triggered_step_2,
            "triggered_step_3": self.triggered_step_3,
            "refill_step_used": self.refill_step_used,
            "game_end_triggered": self.game_end_triggered,
            "winner_result": self.winner_result.to_dict() if self.winner_result is not None else None,
        }


@dataclass
class GameState:
    config: GameConfig
    game_map: MapDefinition
    rules: RuleTables
    players: tuple[PlayerState, ...]
    player_order: tuple[str, ...]
    resource_market: ResourceMarket
    current_market: tuple[PowerPlantCard, ...]
    future_market: tuple[PowerPlantCard, ...]
    power_plant_draw_stack: tuple[PowerPlantCard, ...] = field(default_factory=tuple)
    power_plant_bottom_stack: tuple[PowerPlantCard, ...] = field(default_factory=tuple)
    step_3_card_pending: bool = True
    auction_step_3_pending: bool = False
    round_number: int = 0
    step: int = 1
    phase: str = "setup"
    selected_regions: tuple[str, ...] = field(default_factory=tuple)
    auction_state: AuctionState | None = None
    pending_decision: DecisionRequest | None = None
    last_powered_cities: dict[str, int] = field(default_factory=dict)
    last_income_paid: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.players = tuple(self.players)
        self.player_order = tuple(self.player_order)
        self.current_market = tuple(self.current_market)
        self.future_market = tuple(self.future_market)
        self.power_plant_draw_stack = tuple(self.power_plant_draw_stack)
        self.power_plant_bottom_stack = tuple(self.power_plant_bottom_stack)
        self.selected_regions = tuple(self.selected_regions)
        self.last_powered_cities = {
            player_id: int(value) for player_id, value in self.last_powered_cities.items()
        }
        self.last_income_paid = {
            player_id: int(value) for player_id, value in self.last_income_paid.items()
        }
        if self.phase not in GAME_PHASES:
            raise ModelValidationError(f"phase must be one of {GAME_PHASES}")
        if self.step not in {1, 2, 3}:
            raise ModelValidationError("step must be 1, 2, or 3")
        if self.round_number < 0:
            raise ModelValidationError("round number cannot be negative")
        player_ids = [player.player_id for player in self.players]
        if len(player_ids) != len(set(player_ids)):
            raise ModelValidationError("game state player ids must be unique")
        if tuple(sorted(self.player_order)) != tuple(sorted(player_ids)):
            raise ModelValidationError("player_order must contain each player exactly once")
        order_positions = {player_id: index + 1 for index, player_id in enumerate(self.player_order)}
        for player in self.players:
            if player.turn_order_position != order_positions[player.player_id]:
                raise ModelValidationError("player turn_order_position must match player_order")
        region_ids = {region.id for region in self.game_map.regions}
        if any(region not in region_ids for region in self.selected_regions):
            raise ModelValidationError("selected regions must exist on the chosen map")
        required_regions = self.rules.player_count_rules[len(self.players)]["areas"]
        if self.selected_regions and len(self.selected_regions) != required_regions:
            raise ModelValidationError("selected regions must match the required area count")
        if self.step in {1, 2}:
            if self.auction_step_3_pending:
                total_visible = len(self.current_market) + len(self.future_market)
                if len(self.current_market) > 4 or len(self.future_market) > 4:
                    raise ModelValidationError(
                        "auction Step 3 pending state may expose at most 4 current and 4 future plants"
                    )
                if not 1 <= total_visible <= 8:
                    raise ModelValidationError(
                        "auction Step 3 pending state must expose between 1 and 8 visible power plants"
                    )
                if self.step_3_card_pending:
                    raise ModelValidationError(
                        "auction Step 3 pending state may not also keep the Step 3 card pending in the draw stack"
                    )
                if sum(1 for plant in (*self.current_market, *self.future_market) if plant.is_step_3_placeholder) != 1:
                    raise ModelValidationError(
                        "auction Step 3 pending state must contain exactly one Step 3 placeholder"
                    )
            elif len(self.current_market) != 4 or len(self.future_market) != 4:
                raise ModelValidationError("steps 1 and 2 must contain 4 current and 4 future plants")
        else:
            if self.auction_step_3_pending:
                raise ModelValidationError("auction Step 3 pending state is only valid before Step 3 begins")
            if self.future_market:
                raise ModelValidationError("step 3 may not contain a future market")
            if not 0 <= len(self.current_market) <= 6:
                raise ModelValidationError("step 3 current market must contain between 0 and 6 plants")
        current_prices = [plant.price for plant in self.current_market]
        future_prices = [plant.price for plant in self.future_market]
        if current_prices != sorted(current_prices) or future_prices != sorted(future_prices):
            raise ModelValidationError("power plant markets must be sorted by price")
        if set(current_prices) & set(future_prices):
            raise ModelValidationError("current and future markets may not overlap")
        stack_prices = [plant.price for plant in self.power_plant_draw_stack]
        bottom_prices = [plant.price for plant in self.power_plant_bottom_stack]
        visible_prices = set(current_prices) | set(future_prices)
        if visible_prices & set(stack_prices):
            raise ModelValidationError("draw stack may not overlap with visible markets")
        if visible_prices & set(bottom_prices):
            raise ModelValidationError("bottom stack may not overlap with visible markets")
        if set(stack_prices) & set(bottom_prices):
            raise ModelValidationError("draw stack may not overlap with bottom stack")
        if self.auction_state is not None:
            if (
                self.auction_state.discount_token_plant_price is not None
                and self.auction_state.discount_token_plant_price not in current_prices
            ):
                raise ModelValidationError("auction discount token must mark a current-market plant")
            known_ids = set(player_ids)
            referenced_ids = set(self.auction_state.players_with_plants) | set(
                self.auction_state.players_passed_phase
            )
            if self.auction_state.current_chooser_id is not None:
                referenced_ids.add(self.auction_state.current_chooser_id)
            if self.auction_state.highest_bidder_id is not None:
                referenced_ids.add(self.auction_state.highest_bidder_id)
            if self.auction_state.next_bidder_id is not None:
                referenced_ids.add(self.auction_state.next_bidder_id)
            referenced_ids.update(self.auction_state.active_bidders)
            if not referenced_ids <= known_ids:
                raise ModelValidationError("auction state may only reference known players")
        if self.pending_decision and self.pending_decision.player_id not in player_ids:
            raise ModelValidationError("pending decision must belong to a known player")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "game_map": _serialize_map_definition(self.game_map),
            "rules": _serialize_rule_tables(self.rules),
            "players": [player.to_dict() for player in self.players],
            "player_order": list(self.player_order),
            "resource_market": self.resource_market.to_dict(),
            "current_market": [plant.to_dict() for plant in self.current_market],
            "future_market": [plant.to_dict() for plant in self.future_market],
            "power_plant_draw_stack": [plant.to_dict() for plant in self.power_plant_draw_stack],
            "power_plant_bottom_stack": [plant.to_dict() for plant in self.power_plant_bottom_stack],
            "step_3_card_pending": self.step_3_card_pending,
            "auction_step_3_pending": self.auction_step_3_pending,
            "round_number": self.round_number,
            "step": self.step,
            "phase": self.phase,
            "selected_regions": list(self.selected_regions),
            "auction_state": self.auction_state.to_dict() if self.auction_state is not None else None,
            "pending_decision": (
                self.pending_decision.to_dict() if self.pending_decision is not None else None
            ),
            "last_powered_cities": dict(self.last_powered_cities),
            "last_income_paid": dict(self.last_income_paid),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GameState":
        return cls(
            config=GameConfig.from_dict(payload["config"]),
            game_map=_deserialize_map_definition(payload["game_map"]),
            rules=_deserialize_rule_tables(payload["rules"]),
            players=tuple(PlayerState.from_dict(player) for player in payload["players"]),
            player_order=tuple(payload["player_order"]),
            resource_market=ResourceMarket.from_dict(payload["resource_market"]),
            current_market=tuple(
                PowerPlantCard.from_dict(plant) for plant in payload["current_market"]
            ),
            future_market=tuple(
                PowerPlantCard.from_dict(plant) for plant in payload["future_market"]
            ),
            power_plant_draw_stack=tuple(
                PowerPlantCard.from_dict(plant)
                for plant in payload.get("power_plant_draw_stack", [])
            ),
            power_plant_bottom_stack=tuple(
                PowerPlantCard.from_dict(plant)
                for plant in payload.get("power_plant_bottom_stack", [])
            ),
            step_3_card_pending=bool(payload.get("step_3_card_pending", True)),
            auction_step_3_pending=bool(payload.get("auction_step_3_pending", False)),
            round_number=int(payload["round_number"]),
            step=int(payload["step"]),
            phase=payload["phase"],
            selected_regions=tuple(payload.get("selected_regions", [])),
            auction_state=(
                AuctionState.from_dict(payload["auction_state"])
                if payload.get("auction_state") is not None
                else None
            ),
            pending_decision=(
                DecisionRequest.from_dict(payload["pending_decision"])
                if payload.get("pending_decision") is not None
                else None
            ),
            last_powered_cities=dict(payload.get("last_powered_cities", {})),
            last_income_paid=dict(payload.get("last_income_paid", {})),
        )


def create_initial_state(
    config: GameConfig,
    data_root: str | None = None,
) -> GameState:
    game_map = load_map(config.map_id, data_root=data_root)
    rules = load_rule_tables(data_root=data_root)
    selected_regions = select_play_areas(
        game_map,
        len(config.players),
        chosen_region_ids=config.selected_regions or None,
        data_root=data_root,
    )

    rng = random.Random(config.seed)
    player_order = [player.player_id for player in config.players]
    rng.shuffle(player_order)
    turn_order_positions = {player_id: index + 1 for index, player_id in enumerate(player_order)}

    players = tuple(
        PlayerState(
            player_id=seat.player_id,
            name=seat.name,
            controller=seat.controller,
            color=PLAYER_COLORS[index],
            elektro=rules.starting_money,
            houses_in_supply=rules.houses_per_player,
            network_city_ids=(),
            power_plants=(),
            turn_order_position=turn_order_positions[seat.player_id],
        )
        for index, seat in enumerate(config.players)
    )

    resource_market = ResourceMarket.from_rule_tables(rules)
    prepared_deck = prepare_plant_deck(len(config.players), config.seed, data_root=data_root)

    return GameState(
        config=config,
        game_map=game_map,
        rules=rules,
        players=players,
        player_order=tuple(player_order),
        resource_market=resource_market,
        current_market=prepared_deck.current_market,
        future_market=prepared_deck.future_market,
        power_plant_draw_stack=prepared_deck.draw_stack,
        step_3_card_pending=prepared_deck.step_3_card_pending,
        round_number=0,
        step=1,
        phase="setup",
        selected_regions=selected_regions,
        pending_decision=None,
    )


def make_default_seat_configs(player_count: int, ai_players: int = 0) -> tuple[SeatConfig, ...]:
    if player_count < 1:
        raise ModelValidationError("player_count must be positive")
    if not 0 <= ai_players <= player_count:
        raise ModelValidationError("ai_players must be between 0 and player_count")
    seats = []
    for index in range(player_count):
        controller = "ai" if index < ai_players else "human"
        seats.append(
            SeatConfig(
                player_id=f"p{index + 1}",
                name=f"Player {index + 1}",
                controller=controller,
            )
        )
    return tuple(seats)


def prepare_plant_deck(
    player_count: int,
    seed: int,
    data_root: str | None = None,
) -> PreparedDeck:
    if not 3 <= player_count <= 6:
        raise ModelValidationError("prepare_plant_deck supports 3-6 players")

    rules = load_rule_tables(data_root=data_root)
    definitions = load_power_plants(data_root=data_root)
    rng = random.Random(seed)

    plug_plants = [
        PowerPlantCard.from_definition(definition)
        for definition in definitions
        if definition.deck_back == "plug"
    ]
    socket_plants = [
        PowerPlantCard.from_definition(definition)
        for definition in definitions
        if definition.deck_back == "socket"
    ]

    rng.shuffle(plug_plants)
    opening_market = tuple(sorted(plug_plants[:8], key=lambda plant: plant.price))
    set_aside_top_card = plug_plants[8]
    remaining_plugs = plug_plants[9:]

    rng.shuffle(socket_plants)
    player_rules = rules.player_count_rules[player_count]
    removed_plugs = remaining_plugs[: player_rules["remove_plug_plants"]]
    remaining_plugs = remaining_plugs[player_rules["remove_plug_plants"] :]
    removed_sockets = socket_plants[: player_rules["remove_socket_plants"]]
    remaining_sockets = socket_plants[player_rules["remove_socket_plants"] :]

    shuffled_tail = list(remaining_plugs + remaining_sockets)
    rng.shuffle(shuffled_tail)

    return PreparedDeck(
        current_market=opening_market[:4],
        future_market=opening_market[4:8],
        draw_stack=tuple([set_aside_top_card] + shuffled_tail),
        step_3_card_pending=True,
        removed_plant_prices=tuple(
            sorted(plant.price for plant in (*removed_plugs, *removed_sockets))
        ),
    )


def select_play_areas(
    game_map: MapDefinition,
    player_count: int,
    chosen_region_ids: tuple[str, ...] | list[str] | None = None,
    data_root: str | None = None,
) -> tuple[str, ...]:
    if not 3 <= player_count <= 6:
        raise ModelValidationError("select_play_areas supports 3-6 players")
    rules = load_rule_tables(data_root=data_root)
    required_count = rules.player_count_rules[player_count]["areas"]
    available_regions = tuple(sorted(region.id for region in game_map.regions))

    if chosen_region_ids is not None:
        selected = tuple(chosen_region_ids)
        _validate_play_areas(game_map, selected, required_count)
        return tuple(sorted(selected))

    for candidate in combinations(available_regions, required_count):
        if _are_regions_contiguous(game_map, candidate):
            return tuple(candidate)
    raise ModelValidationError("no contiguous playing zone could be selected for this map")


def initialize_game(
    config: GameConfig,
    controllers: dict[str, Any] | None,
    data_root: str | None = None,
) -> GameState:
    expected_player_ids = {seat.player_id for seat in config.players}
    if controllers is not None:
        controller_ids = set(controllers)
        if controller_ids != expected_player_ids:
            raise ModelValidationError("controllers must match the configured player ids exactly")
        if any(controller is None for controller in controllers.values()):
            raise ModelValidationError("controllers may not contain None values")
    return create_initial_state(config, data_root=data_root)


def list_auctionable_plants(state: GameState) -> tuple[PowerPlantCard, ...]:
    if state.step == 3:
        return tuple(sorted((*state.current_market, *state.future_market), key=lambda plant: plant.price))
    return state.current_market


def start_auction(
    state: GameState,
    player_id: str,
    plant_id: int,
    bid: int,
) -> GameState:
    state = _require_auction_phase(state)
    auction_state = state.auction_state
    assert auction_state is not None
    if state.pending_decision is not None:
        raise ModelValidationError("resolve the pending decision before starting a new auction")
    if auction_state.has_active_auction:
        raise ModelValidationError("an auction is already in progress")
    if player_id != auction_state.current_chooser_id:
        raise ModelValidationError("only the current chooser may start an auction")
    if player_id in auction_state.players_with_plants:
        raise ModelValidationError("a player may buy only one power plant per round")
    if player_id in auction_state.players_passed_phase:
        raise ModelValidationError("a player who passed the phase may not start an auction")

    plant = _get_auctionable_plant(state, plant_id)
    minimum_bid = _minimum_opening_bid(state, plant)
    if bid < minimum_bid:
        raise ModelValidationError(
            f"opening bid must be at least {minimum_bid} for power plant {plant.price}"
        )
    player = _get_player(state, player_id)
    if bid > player.elektro:
        raise ModelValidationError("opening bid may not exceed the player's Elektro")

    eligible_bidders = _eligible_auction_participants(state, auction_state)
    bidder_order = _rotate_player_order(eligible_bidders, player_id)
    if len(bidder_order) == 1:
        return _award_auction_purchase(
            state,
            winner_id=player_id,
            plant=plant,
            price_paid=bid,
            chooser_id=player_id,
        )

    updated_auction = replace(
        auction_state,
        active_plant_price=plant.price,
        current_bid=bid,
        highest_bidder_id=player_id,
        active_bidders=bidder_order,
        next_bidder_id=_next_player_in_sequence(bidder_order, player_id),
    )
    return replace(state, auction_state=updated_auction)


def raise_bid(state: GameState, player_id: str, amount: int) -> GameState:
    state = _require_auction_phase(state)
    auction_state = state.auction_state
    assert auction_state is not None
    if state.pending_decision is not None:
        raise ModelValidationError("resolve the pending decision before raising the bid")
    if not auction_state.has_active_auction:
        raise ModelValidationError("there is no active auction to bid in")
    if player_id != auction_state.next_bidder_id:
        raise ModelValidationError("it is not this player's turn to bid")
    if amount <= int(auction_state.current_bid):
        raise ModelValidationError("bid raises must exceed the current bid")
    player = _get_player(state, player_id)
    if amount > player.elektro:
        raise ModelValidationError("bid raises may not exceed the player's Elektro")

    updated_auction = replace(
        auction_state,
        current_bid=amount,
        highest_bidder_id=player_id,
        next_bidder_id=_next_player_in_sequence(auction_state.active_bidders, player_id),
    )
    return replace(state, auction_state=updated_auction)


def pass_auction(state: GameState, player_id: str) -> GameState:
    state = _require_auction_phase(state)
    auction_state = state.auction_state
    assert auction_state is not None
    if state.pending_decision is not None:
        raise ModelValidationError("resolve the pending decision before passing")

    if not auction_state.has_active_auction:
        if player_id != auction_state.current_chooser_id:
            raise ModelValidationError("only the current chooser may pass the auction phase")
        if state.round_number == 1:
            raise ModelValidationError("players must buy a power plant in the first round")
        updated_auction = replace(
            auction_state,
            players_passed_phase=tuple((*auction_state.players_passed_phase, player_id)),
            current_chooser_id=_next_auction_chooser(
                state.player_order,
                player_id,
                players_with_plants=auction_state.players_with_plants,
                players_passed_phase=tuple((*auction_state.players_passed_phase, player_id)),
            ),
        )
        updated_state = replace(state, auction_state=updated_auction)
        return _maybe_finish_auction_phase(updated_state)

    if player_id != auction_state.next_bidder_id:
        raise ModelValidationError("only the next bidder may pass in an active auction")

    remaining_bidders = tuple(
        bidder_id for bidder_id in auction_state.active_bidders if bidder_id != player_id
    )
    if len(remaining_bidders) == 1:
        winning_plant = _get_plant_from_visible_market(state, int(auction_state.active_plant_price))
        return _award_auction_purchase(
            state,
            winner_id=str(auction_state.highest_bidder_id),
            plant=winning_plant,
            price_paid=int(auction_state.current_bid),
            chooser_id=str(auction_state.current_chooser_id),
        )

    updated_auction = replace(
        auction_state,
        active_bidders=remaining_bidders,
        next_bidder_id=_next_remaining_player(auction_state.active_bidders, player_id, remaining_bidders),
    )
    return replace(state, auction_state=updated_auction)


def resolve_auction_round(state: GameState) -> GameState:
    state = _require_auction_phase(state)
    auction_state = state.auction_state
    assert auction_state is not None
    if state.pending_decision is not None:
        raise ModelValidationError("resolve the pending decision before ending the auction phase")
    if auction_state.has_active_auction:
        raise ModelValidationError("cannot end the auction phase while an auction is active")

    eligible = _eligible_auction_participants(state, auction_state)
    if eligible:
        raise ModelValidationError("the auction phase is not complete yet")

    visible_market_state = state
    if auction_state.discount_token_plant_price is not None:
        visible_market_state = _remove_visible_plant_and_refill(
            state,
            auction_state.discount_token_plant_price,
            clear_discount_before_refill=True,
        )
    if visible_market_state.auction_step_3_pending:
        visible_market_state = _apply_auction_step_3_transition(visible_market_state)

    completed_state = replace(
        visible_market_state,
        phase="buy_resources",
        auction_state=None,
        pending_decision=None,
    )
    if completed_state.round_number == 1:
        completed_state = _recalculate_player_order(completed_state)
    return completed_state


def replace_plant_if_needed(
    state: GameState,
    player_id: str,
    discard_choice: int,
) -> GameState:
    if state.pending_decision is None:
        raise ModelValidationError("there is no pending plant replacement decision")
    if state.pending_decision.decision_type != "discard_power_plant":
        raise ModelValidationError("the pending decision is not a power plant discard")
    if state.pending_decision.player_id != player_id:
        raise ModelValidationError("the pending replacement decision belongs to another player")

    player = _get_player(state, player_id)
    max_plants = state.rules.player_count_rules[len(state.players)]["max_power_plants"]
    if len(player.power_plants) <= max_plants:
        return replace(state, pending_decision=None)
    if discard_choice not in {plant.price for plant in player.power_plants}:
        raise ModelValidationError("discard choice must be one of the player's power plants")

    kept_plants = tuple(
        sorted(
            (plant for plant in player.power_plants if plant.price != discard_choice),
            key=lambda plant: plant.price,
        )
    )
    storage_result = _resolve_resource_storage_after_power_plant_discard(
        player,
        kept_plants=kept_plants,
        discarded_plant_price=discard_choice,
    )
    if isinstance(storage_result, DecisionRequest):
        return replace(state, pending_decision=storage_result)
    updated_player = replace(
        player,
        power_plants=kept_plants,
        resource_storage=storage_result,
    )
    updated_state = _replace_player_on_state(state, updated_player)
    updated_state = replace(updated_state, pending_decision=None)
    if updated_state.phase == "auction" and updated_state.auction_state is not None:
        return _maybe_finish_auction_phase(updated_state)
    return updated_state


def discard_resources_to_fit_storage(
    state: GameState,
    player_id: str,
    discard_mix: dict[str, int],
) -> GameState:
    if state.pending_decision is None:
        raise ModelValidationError("there is no pending resource discard decision")
    if state.pending_decision.decision_type != "discard_hybrid_resources":
        raise ModelValidationError("the pending decision is not a hybrid resource discard")
    if state.pending_decision.player_id != player_id:
        raise ModelValidationError("the pending resource discard decision belongs to another player")

    normalized_mix = _normalize_resource_mix(discard_mix)
    if set(normalized_mix) - {"coal", "oil"}:
        raise ModelValidationError("only coal and oil may be chosen in a hybrid resource discard")
    chosen = {
        "coal": int(normalized_mix.get("coal", 0)),
        "oil": int(normalized_mix.get("oil", 0)),
    }
    legal_choices = {
        (
            int(action.payload.get("coal", 0)),
            int(action.payload.get("oil", 0)),
        )
        for action in state.pending_decision.legal_actions
    }
    if (chosen["coal"], chosen["oil"]) not in legal_choices:
        raise ModelValidationError("discard choice must match one of the legal coal/oil discard options")

    player = _get_player(state, player_id)
    current_totals = player.resource_storage.resource_totals()
    auto_discards = {
        resource: int(amount)
        for resource, amount in state.pending_decision.metadata.get("auto_discard_resources", {}).items()
    }
    capacities = {
        resource: int(amount)
        for resource, amount in state.pending_decision.metadata["storage_capacities"].items()
    }
    kept_plants = tuple(
        PowerPlantCard.from_dict(payload)
        for payload in state.pending_decision.metadata["kept_power_plants"]
    )

    final_totals = dict(current_totals)
    for resource, amount in {**auto_discards, **chosen}.items():
        if amount < 0:
            raise ModelValidationError("discard amounts may not be negative")
        if amount > final_totals.get(resource, 0):
            raise ModelValidationError(f"cannot discard more {resource} than the player currently stores")
        final_totals[resource] -= amount

    if not _coal_oil_totals_fit_storage(final_totals["coal"], final_totals["oil"], capacities):
        raise ModelValidationError("the chosen coal/oil discard does not fit the remaining hybrid storage")
    if final_totals["garbage"] > capacities["garbage"] or final_totals["uranium"] > capacities["uranium"]:
        raise ModelValidationError("the chosen discard does not remove enough non-hybrid resources")

    updated_player = replace(
        player,
        power_plants=kept_plants,
        resource_storage=_normalize_resource_totals_into_storage(final_totals, capacities),
    )
    updated_state = _replace_player_on_state(state, updated_player)
    updated_state = replace(updated_state, pending_decision=None)
    if updated_state.phase == "auction" and updated_state.auction_state is not None:
        return _maybe_finish_auction_phase(updated_state)
    return updated_state


def plant_storage_capacity(plant: PowerPlantCard) -> int:
    return plant.max_storage


def can_store_resources(player: PlayerState, mix: dict[str, int]) -> bool:
    try:
        normalized = _normalize_resource_mix(mix)
    except ModelValidationError:
        return False
    if not normalized:
        return True
    totals = player.resource_storage.resource_totals()
    for resource, amount in normalized.items():
        totals[resource] += amount
    normalized_storage = _normalize_resource_totals_into_storage(
        totals,
        _storage_space_capacities(player.power_plants),
    )
    return normalized_storage.resource_totals() == totals


def legal_resource_purchases(state: GameState, player_id: str) -> tuple[Action, ...]:
    player = _get_player(state, player_id)
    actions: list[Action] = []
    for resource in RESOURCE_TYPES:
        market_prices = state.resource_market.available_unit_prices(resource)
        if not market_prices:
            continue
        max_units = 0
        for amount in range(1, len(market_prices) + 1):
            if not can_store_resources(player, {resource: amount}):
                break
            max_units = amount
        if max_units == 0:
            continue
        unit_prices = market_prices[:max_units]
        max_affordable = 0
        running_cost = 0
        for index, price in enumerate(unit_prices, start=1):
            running_cost += price
            if running_cost <= player.elektro:
                max_affordable = index
            else:
                break
        if max_affordable == 0:
            continue
        actions.append(
            Action(
                action_type="buy_resource",
                player_id=player_id,
                payload={
                    "resource": resource,
                    "max_units": max_units,
                    "max_affordable_units": max_affordable,
                    "unit_prices": list(unit_prices),
                },
            )
        )
    return tuple(actions)


def purchase_resources(
    state: GameState,
    player_id: str,
    basket: dict[str, int] | dict[int | str, dict[str, int]],
) -> GameState:
    player = _get_player(state, player_id)
    normalized_purchase = _normalize_purchase_request(basket)
    if not normalized_purchase:
        raise ModelValidationError("resource purchase basket cannot be empty")
    if not can_store_resources(player, normalized_purchase):
        raise ModelValidationError("player cannot store the requested resources")

    total_cost = 0
    updated_market = state.resource_market
    for resource, amount in normalized_purchase.items():
        if amount == 0:
            continue
        total_cost += updated_market.quote_purchase_cost(resource, amount)
        updated_market = updated_market.remove_from_market(resource, amount)
    if total_cost > player.elektro:
        raise ModelValidationError("player cannot afford the requested resources")

    updated_totals = player.resource_storage.resource_totals()
    for resource, amount in normalized_purchase.items():
        updated_totals[resource] += amount
    updated_storage = _normalize_resource_totals_into_storage(
        updated_totals,
        _storage_space_capacities(player.power_plants),
    )
    updated_player = replace(
        player,
        elektro=player.elektro - total_cost,
        resource_storage=updated_storage,
    )
    updated_state = _replace_player_on_state(state, updated_player)
    return replace(updated_state, resource_market=updated_market)


def refill_resource_market(
    resource_market: ResourceMarket,
    rules: RuleTables,
    step: int,
    player_count: int,
) -> ResourceMarket:
    if step not in {1, 2, 3}:
        raise ModelValidationError("resource refill step must be 1, 2, or 3")
    if player_count not in rules.player_count_rules:
        raise ModelValidationError("unknown player count for resource refill")

    refill_key = f"step_{step}"
    refill_amounts = rules.player_count_rules[player_count]["resource_refill"][refill_key]
    updated_market = {resource: dict(price_bands) for resource, price_bands in resource_market.market.items()}
    updated_supply = dict(resource_market.supply)

    for resource in RESOURCE_TYPES:
        capacities = {
            int(price): int(amount)
            for price, amount in rules.resource_market_tracks[resource]["capacity_by_price"].items()
        }
        to_add = min(int(refill_amounts[resource]), updated_supply[resource])
        for price in sorted(capacities, reverse=True):
            if to_add == 0:
                break
            open_slots = capacities[price] - updated_market[resource].get(price, 0)
            if open_slots <= 0:
                continue
            added = min(open_slots, to_add)
            updated_market[resource][price] = updated_market[resource].get(price, 0) + added
            updated_supply[resource] -= added
            to_add -= added

    return ResourceMarket(market=updated_market, supply=updated_supply)


def choose_plants_to_run(
    state: GameState,
    player_id: str,
    selection: (
        tuple[int | PlantRunPlan, ...]
        | list[int | PlantRunPlan]
        | dict[int | str, dict[str, int] | None]
    ),
) -> tuple[PlantRunPlan, ...]:
    player = _get_player(state, player_id)
    return _normalize_run_selection(player, selection)


def consume_resources(
    state: GameState,
    player_id: str,
    selection: (
        tuple[int | PlantRunPlan, ...]
        | list[int | PlantRunPlan]
        | dict[int | str, dict[str, int] | None]
    ),
) -> GameState:
    player = _get_player(state, player_id)
    plans = _normalize_run_selection(player, selection)
    required = _summarize_resource_usage(plans)
    remaining = player.resource_storage.resource_totals()
    for resource, amount in required.items():
        if amount > remaining[resource]:
            raise ModelValidationError(f"player does not have enough {resource} to run the selected plants")
        remaining[resource] -= amount
    updated_player = replace(
        player,
        resource_storage=_normalize_resource_totals_into_storage(
            remaining,
            _storage_space_capacities(player.power_plants),
        ),
    )
    updated_state = _replace_player_on_state(state, updated_player)
    return replace(
        updated_state,
        resource_market=updated_state.resource_market.add_resources_to_supply(required),
    )


def compute_powered_cities(
    state: GameState,
    player_id: str,
    selection: (
        tuple[int | PlantRunPlan, ...]
        | list[int | PlantRunPlan]
        | dict[int | str, dict[str, int] | None]
    ),
) -> int:
    player = _get_player(state, player_id)
    plans = _normalize_run_selection(player, selection)
    output = 0
    for plan in plans:
        plant = _get_player_power_plant(player, plan.plant_price)
        output += plant.output_cities
    return min(output, player.connected_city_count)


def pay_income(rules: RuleTables, powered_cities: int) -> int:
    bounded = max(0, min(20, int(powered_cities)))
    return int(rules.payment_schedule[bounded])


def update_plant_market_after_bureaucracy(state: GameState) -> GameState:
    return _update_plant_market_after_bureaucracy(state)


def check_step_2_trigger(state: GameState) -> bool:
    if state.step != 1:
        return False
    threshold = state.rules.player_count_rules[len(state.players)]["step_2_cities"]
    return any(player.connected_city_count >= threshold for player in state.players)


def check_step_3_trigger(state: GameState) -> bool:
    return state.step_3_card_pending and not state.power_plant_draw_stack


def resolve_winner(
    state: GameState,
    powered_cities: dict[str, int] | None = None,
) -> WinnerResult:
    powered = (
        {player_id: int(value) for player_id, value in powered_cities.items()}
        if powered_cities is not None
        else dict(state.last_powered_cities)
    )
    if not powered:
        raise ModelValidationError("winner resolution requires powered-city totals for the round")

    money = {player.player_id: player.elektro for player in state.players}
    connected = {player.player_id: player.connected_city_count for player in state.players}
    prior_index = {player_id: index for index, player_id in enumerate(state.player_order)}
    ordered_player_ids = sorted(
        powered,
        key=lambda player_id: (
            -powered[player_id],
            -money[player_id],
            -connected[player_id],
            prior_index.get(player_id, 0),
        ),
    )
    best_id = ordered_player_ids[0]
    best_signature = (powered[best_id], money[best_id], connected[best_id])
    winner_ids = tuple(
        player_id
        for player_id in ordered_player_ids
        if (powered[player_id], money[player_id], connected[player_id]) == best_signature
    )
    return WinnerResult(
        winner_ids=winner_ids,
        powered_cities=powered,
        money=money,
        connected_cities=connected,
    )


def resolve_bureaucracy(
    state: GameState,
    generation_choices: dict[str, tuple[int | PlantRunPlan, ...] | list[int | PlantRunPlan] | dict[int | str, dict[str, int] | None]] | None = None,
) -> tuple[GameState, BureaucracySummary]:
    if state.phase != "bureaucracy":
        raise ModelValidationError("bureaucracy resolution may only be used during the bureaucracy phase")

    choices = generation_choices or {}
    working_state = state
    triggered_step_2 = check_step_2_trigger(working_state)
    if triggered_step_2:
        working_state = _apply_step_2_transition(working_state)

    refill_step_used = working_state.step
    powered_cities: dict[str, int] = {}
    income_paid: dict[str, int] = {}

    for player_id in working_state.player_order:
        selection = choices.get(player_id, ())
        powered = compute_powered_cities(working_state, player_id, selection)
        working_state = consume_resources(working_state, player_id, selection)
        player = _get_player(working_state, player_id)
        income = pay_income(working_state.rules, powered)
        powered_cities[player_id] = powered
        income_paid[player_id] = income
        working_state = _replace_player_on_state(
            working_state,
            replace(player, elektro=player.elektro + income),
        )

    working_state = replace(
        working_state,
        last_powered_cities=powered_cities,
        last_income_paid=income_paid,
    )
    working_state = update_plant_market_after_bureaucracy(working_state)
    triggered_step_3 = working_state.step == 3 and state.step != 3
    working_state = replace(
        working_state,
        resource_market=refill_resource_market(
            working_state.resource_market,
            working_state.rules,
            step=refill_step_used,
            player_count=len(working_state.players),
        ),
    )

    game_end_triggered = _check_end_game_trigger(working_state)
    winner_result = resolve_winner(working_state, powered_cities) if game_end_triggered else None
    if not game_end_triggered:
        working_state = advance_round(working_state)

    summary = BureaucracySummary(
        powered_cities=powered_cities,
        income_paid=income_paid,
        triggered_step_2=triggered_step_2,
        triggered_step_3=triggered_step_3,
        refill_step_used=refill_step_used,
        game_end_triggered=game_end_triggered,
        winner_result=winner_result,
    )
    return working_state, summary


def compute_connection_cost(
    state: GameState,
    player_id: str,
    target_city: str,
    source_city_ids: tuple[str, ...] | list[str] | None = None,
) -> int:
    allowed_cities = _allowed_city_ids(state)
    if target_city not in allowed_cities:
        raise ModelValidationError(f"city {target_city!r} is not in the selected play area")
    all_costs = compute_all_targets_connection_cost(state, player_id, source_city_ids=source_city_ids)
    if target_city not in all_costs:
        raise ModelValidationError(f"city {target_city!r} is not reachable from the player's network")
    return all_costs[target_city]


def compute_all_targets_connection_cost(
    state: GameState,
    player_id: str,
    source_city_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    player = _get_player(state, player_id)
    sources = tuple(source_city_ids) if source_city_ids is not None else player.network_city_ids
    return _all_connection_costs_from_sources(state, sources)


def legal_build_targets(state: GameState, player_id: str) -> tuple[Action, ...]:
    player = _get_player(state, player_id)
    allowed_cities = _allowed_city_ids(state)
    connection_costs = compute_all_targets_connection_cost(state, player_id)
    actions: list[Action] = []
    for city_id in sorted(allowed_cities):
        if city_id in player.network_city_ids:
            continue
        occupant_count = _city_occupant_count(state, city_id)
        if occupant_count >= _city_occupancy_limit(state.step):
            continue
        connection_cost = connection_costs.get(city_id)
        if connection_cost is None:
            continue
        build_cost = _city_build_cost(state, city_id)
        total_cost = connection_cost + build_cost
        if total_cost > player.elektro or player.houses_in_supply <= 0:
            continue
        actions.append(
            Action(
                action_type="build_city",
                player_id=player_id,
                payload={
                    "city_id": city_id,
                    "city_name": _city_name(state, city_id),
                    "connection_cost": connection_cost,
                    "build_cost": build_cost,
                    "total_cost": total_cost,
                },
            )
        )
    return tuple(actions)


def build_city(state: GameState, player_id: str, city_id: str) -> GameState:
    return apply_builds(state, player_id, (city_id,))


def apply_builds(
    state: GameState,
    player_id: str,
    city_ids: tuple[str, ...] | list[str],
) -> GameState:
    player = _get_player(state, player_id)
    targets = tuple(city_ids)
    if not targets:
        raise ModelValidationError("build selection cannot be empty")
    if len(set(targets)) != len(targets):
        raise ModelValidationError("build selection cannot contain duplicate cities")
    if player.houses_in_supply < len(targets):
        raise ModelValidationError("player does not have enough houses left to build there")

    allowed_cities = _allowed_city_ids(state)
    for city_id in targets:
        if city_id not in allowed_cities:
            raise ModelValidationError(f"city {city_id!r} is not in the selected play area")
        if city_id in player.network_city_ids:
            raise ModelValidationError(f"player already has a house in {city_id!r}")
        if _city_occupant_count(state, city_id) >= _city_occupancy_limit(state.step):
            raise ModelValidationError(f"city {city_id!r} is already full for step {state.step}")

    total_cost, ordered_targets = _best_build_sequence(state, player, targets)
    if total_cost > player.elektro:
        raise ModelValidationError("player cannot afford the requested city builds")

    updated_player = replace(
        player,
        elektro=player.elektro - total_cost,
        houses_in_supply=player.houses_in_supply - len(ordered_targets),
        network_city_ids=tuple(sorted((*player.network_city_ids, *ordered_targets))),
    )
    return _replace_player_on_state(state, updated_player)


def advance_phase(state: GameState) -> GameState:
    if state.phase == "setup":
        auction_state = _create_auction_state(state, state.player_order[0])
        return replace(
            state,
            phase="auction",
            round_number=max(1, state.round_number or 1),
            auction_state=auction_state,
            pending_decision=None,
        )
    if state.phase == "determine_order":
        auction_state = _create_auction_state(state, state.player_order[0])
        return replace(state, phase="auction", auction_state=auction_state, pending_decision=None)
    if state.phase == "auction":
        return resolve_auction_round(state)
    if state.phase == "buy_resources":
        return replace(state, phase="build_houses")
    if state.phase == "build_houses":
        return replace(state, phase="bureaucracy")
    if state.phase == "bureaucracy":
        return advance_round(state)
    raise ModelValidationError(f"cannot advance unknown phase {state.phase!r}")


def advance_round(state: GameState) -> GameState:
    if state.phase == "setup":
        auction_state = _create_auction_state(state, state.player_order[0])
        return replace(
            state,
            phase="auction",
            round_number=1,
            auction_state=auction_state,
            pending_decision=None,
        )
    if state.phase != "bureaucracy":
        raise ModelValidationError("advance_round may only be called from setup or bureaucracy")
    ordered_state = _recalculate_player_order(state)
    return replace(
        ordered_state,
        phase="determine_order",
        round_number=state.round_number + 1,
        auction_state=None,
        pending_decision=None,
    )


def _require_auction_phase(state: GameState) -> GameState:
    if state.phase != "auction":
        raise ModelValidationError("auction actions may only be used during the auction phase")
    if state.auction_state is not None:
        return state
    return replace(state, auction_state=_create_auction_state(state, state.player_order[0]))


def _get_player(state: GameState, player_id: str) -> PlayerState:
    for player in state.players:
        if player.player_id == player_id:
            return player
    raise ModelValidationError(f"unknown player {player_id!r}")


def _validate_resource_name(resource: str) -> None:
    if resource not in RESOURCE_TYPES:
        raise ModelValidationError(f"unknown resource type {resource!r}")


def _create_auction_state(state: GameState, chooser_id: str | None) -> AuctionState:
    return AuctionState(
        current_chooser_id=chooser_id,
        discount_token_plant_price=min(
            (plant.price for plant in state.current_market),
            default=None,
        ),
    )


def _replace_player_on_state(state: GameState, updated_player: PlayerState) -> GameState:
    players = tuple(
        updated_player if player.player_id == updated_player.player_id else player
        for player in state.players
    )
    return replace(state, players=players)


def _normalize_resource_mix(mix: dict[str, int]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for resource, amount in mix.items():
        _validate_resource_name(resource)
        normalized_amount = int(amount)
        if normalized_amount < 0:
            raise ModelValidationError("resource amounts cannot be negative")
        if normalized_amount:
            normalized[resource] = normalized_amount
    return normalized


def _storage_space_capacities(power_plants: tuple[PowerPlantCard, ...]) -> dict[str, int]:
    capacities = {
        "coal": 0,
        "oil": 0,
        "hybrid": 0,
        "garbage": 0,
        "uranium": 0,
    }
    for plant in power_plants:
        contribution = plant_storage_capacity(plant)
        if plant.is_ecological or contribution == 0:
            continue
        if plant.is_hybrid:
            capacities["hybrid"] += contribution
        else:
            capacities[plant.resource_types[0]] += contribution
    return capacities


def _normalize_resource_totals_into_storage(
    totals: dict[str, int],
    capacities: dict[str, int],
) -> ResourceStorage:
    normalized = _normalize_resource_mix(totals)
    coal_total = normalized.get("coal", 0)
    oil_total = normalized.get("oil", 0)
    garbage_total = normalized.get("garbage", 0)
    uranium_total = normalized.get("uranium", 0)

    coal_spaces = min(coal_total, capacities["coal"])
    oil_spaces = min(oil_total, capacities["oil"])
    garbage_spaces = min(garbage_total, capacities["garbage"])
    uranium_spaces = min(uranium_total, capacities["uranium"])

    remaining_coal = coal_total - coal_spaces
    remaining_oil = oil_total - oil_spaces
    hybrid_remaining = capacities["hybrid"]
    hybrid_coal = min(remaining_coal, hybrid_remaining)
    hybrid_remaining -= hybrid_coal
    hybrid_oil = min(remaining_oil, hybrid_remaining)

    return ResourceStorage(
        coal=coal_spaces,
        oil=oil_spaces,
        hybrid_coal=hybrid_coal,
        hybrid_oil=hybrid_oil,
        garbage=garbage_spaces,
        uranium=uranium_spaces,
    )


def _normalize_resource_storage(
    storage: ResourceStorage,
    power_plants: tuple[PowerPlantCard, ...],
) -> ResourceStorage:
    return _normalize_resource_totals_into_storage(
        storage.resource_totals(),
        _storage_space_capacities(power_plants),
    )


def _resource_storage_fits_power_plants(
    storage: ResourceStorage,
    power_plants: tuple[PowerPlantCard, ...],
) -> bool:
    return storage == _normalize_resource_storage(storage, power_plants)


def _coal_oil_totals_fit_storage(
    coal_total: int,
    oil_total: int,
    capacities: dict[str, int],
) -> bool:
    overflow_into_hybrid = max(0, coal_total - capacities["coal"]) + max(0, oil_total - capacities["oil"])
    return overflow_into_hybrid <= capacities["hybrid"]


def _minimal_hybrid_discard_options(
    coal_total: int,
    oil_total: int,
    capacities: dict[str, int],
) -> tuple[dict[str, int], ...]:
    best_kept_total = -1
    discard_options: list[dict[str, int]] = []
    for kept_coal in range(coal_total + 1):
        for kept_oil in range(oil_total + 1):
            if not _coal_oil_totals_fit_storage(kept_coal, kept_oil, capacities):
                continue
            kept_total = kept_coal + kept_oil
            discard = {"coal": coal_total - kept_coal, "oil": oil_total - kept_oil}
            if kept_total > best_kept_total:
                best_kept_total = kept_total
                discard_options = [discard]
            elif kept_total == best_kept_total:
                discard_options.append(discard)
    unique_options = {
        (option["coal"], option["oil"]): option for option in discard_options
    }
    return tuple(
        unique_options[key]
        for key in sorted(unique_options)
    )


def _resolve_resource_storage_after_power_plant_discard(
    player: PlayerState,
    *,
    kept_plants: tuple[PowerPlantCard, ...],
    discarded_plant_price: int,
) -> ResourceStorage | DecisionRequest:
    current_totals = player.resource_storage.resource_totals()
    capacities = _storage_space_capacities(kept_plants)
    auto_discards = {
        "garbage": max(0, current_totals["garbage"] - capacities["garbage"]),
        "uranium": max(0, current_totals["uranium"] - capacities["uranium"]),
    }
    hybrid_options = _minimal_hybrid_discard_options(
        current_totals["coal"],
        current_totals["oil"],
        capacities,
    )
    if not hybrid_options:
        raise ModelValidationError("discarding this power plant leaves no valid storage configuration")

    if len(hybrid_options) > 1:
        return DecisionRequest(
            player_id=player.player_id,
            decision_type="discard_hybrid_resources",
            prompt=(
                "Choose how many coal and oil resources to discard so the remaining plants "
                "can legally store your fuel."
            ),
            legal_actions=tuple(
                Action(
                    action_type="discard_hybrid_resources",
                    player_id=player.player_id,
                    payload={"coal": option["coal"], "oil": option["oil"]},
                )
                for option in hybrid_options
            ),
            metadata={
                "discarded_power_plant_price": discarded_plant_price,
                "kept_power_plants": [plant.to_dict() for plant in kept_plants],
                "current_resource_totals": dict(current_totals),
                "storage_capacities": dict(capacities),
                "auto_discard_resources": {
                    resource: amount for resource, amount in auto_discards.items() if amount
                },
            },
        )

    final_totals = dict(current_totals)
    chosen_option = hybrid_options[0]
    for resource, amount in {**auto_discards, **chosen_option}.items():
        final_totals[resource] -= amount
    return _normalize_resource_totals_into_storage(final_totals, capacities)


def _normalize_purchase_request(
    basket: dict[str, int] | dict[int | str, dict[str, int]],
) -> dict[str, int]:
    if not basket:
        return {}
    first_value = next(iter(basket.values()))
    if isinstance(first_value, dict):
        aggregated = {resource: 0 for resource in RESOURCE_TYPES}
        for mix in basket.values():
            normalized = _normalize_resource_mix(mix)
            for resource, amount in normalized.items():
                aggregated[resource] += amount
        return {resource: amount for resource, amount in aggregated.items() if amount}
    return _normalize_resource_mix(basket)  # type: ignore[arg-type]


def _get_player_power_plant(player: PlayerState, plant_price: int) -> PowerPlantCard:
    for plant in player.power_plants:
        if plant.price == int(plant_price):
            return plant
    raise ModelValidationError(f"player does not own power plant {plant_price}")


def _normalize_run_selection(
    player: PlayerState,
    selection: (
        tuple[int | PlantRunPlan, ...]
        | list[int | PlantRunPlan]
        | dict[int | str, dict[str, int] | None]
    ),
) -> tuple[PlantRunPlan, ...]:
    if isinstance(selection, dict):
        raw_items = [(int(plant_price), mix) for plant_price, mix in selection.items()]
    else:
        raw_items = []
        for item in selection:
            if isinstance(item, PlantRunPlan):
                raw_items.append((item.plant_price, dict(item.resource_mix)))
            else:
                raw_items.append((int(item), None))

    remaining_resources = player.resource_storage.resource_totals()
    seen_prices: set[int] = set()
    raw_plans: list[tuple[PowerPlantCard, dict[str, int] | None]] = []
    fixed_usage = {resource: 0 for resource in RESOURCE_TYPES}
    unresolved_hybrids: list[PowerPlantCard] = []

    for plant_price, requested_mix in raw_items:
        if plant_price in seen_prices:
            raise ModelValidationError("a power plant may only be selected once per bureaucracy phase")
        seen_prices.add(plant_price)
        plant = _get_player_power_plant(player, plant_price)

        if plant.is_ecological:
            normalized_mix: dict[str, int] | None = {}
        elif plant.is_hybrid:
            if requested_mix:
                normalized_mix = _normalize_resource_mix(requested_mix)
                if set(normalized_mix) - {"coal", "oil"}:
                    raise ModelValidationError("hybrid plants may only consume coal and oil")
                if sum(normalized_mix.values()) != plant.resource_cost:
                    raise ModelValidationError(
                        f"hybrid plant {plant.price} must consume exactly {plant.resource_cost} resources"
                    )
            else:
                normalized_mix = None
                unresolved_hybrids.append(plant)
        else:
            resource = plant.resource_types[0]
            expected_mix = {resource: plant.resource_cost}
            if requested_mix:
                normalized_mix = _normalize_resource_mix(requested_mix)
                if normalized_mix != expected_mix:
                    raise ModelValidationError(
                        f"power plant {plant.price} must consume exactly {expected_mix}"
                    )
            else:
                normalized_mix = expected_mix

        if normalized_mix is not None:
            for resource, amount in normalized_mix.items():
                fixed_usage[resource] += amount
        raw_plans.append((plant, normalized_mix))

    for resource, amount in fixed_usage.items():
        if amount > remaining_resources[resource]:
            raise ModelValidationError(f"player does not have enough {resource} to run the selected plants")
        remaining_resources[resource] -= amount

    resolved_hybrid_mixes: dict[int, dict[str, int]] = {}
    if unresolved_hybrids:
        feasible_assignments = _enumerate_hybrid_run_assignments(unresolved_hybrids, remaining_resources)
        if not feasible_assignments:
            raise ModelValidationError("player does not have enough coal and oil to run the selected hybrid plants")
        for plant in unresolved_hybrids:
            plant_options = {
                _resource_mix_key(assignment[plant.price])
                for assignment in feasible_assignments
            }
            if len(plant_options) > 1:
                valid_mixes = "; ".join(
                    _format_resource_mix(_resource_mix_from_key(key))
                    for key in sorted(plant_options)
                )
                raise ModelValidationError(
                    "hybrid plant "
                    f"{plant.price} requires an explicit coal/oil mix after accounting for the other selected plants; "
                    f"valid mixes: {valid_mixes}"
                )
            chosen_mix = _resource_mix_from_key(next(iter(plant_options)))
            resolved_hybrid_mixes[plant.price] = chosen_mix
            for assignment in feasible_assignments:
                if assignment[plant.price] == chosen_mix:
                    feasible_assignments = tuple(
                        candidate
                        for candidate in feasible_assignments
                        if candidate[plant.price] == chosen_mix
                    )
                    break

    plans: list[PlantRunPlan] = []
    for plant, normalized_mix in raw_plans:
        if normalized_mix is None:
            normalized_mix = resolved_hybrid_mixes[plant.price]
        plans.append(PlantRunPlan(plant_price=plant.price, resource_mix=normalized_mix))

    return tuple(plans)


def _default_hybrid_mix(
    available_resources: dict[str, int],
    required_amount: int,
) -> dict[str, int]:
    coal_used = min(available_resources["coal"], required_amount)
    oil_used = min(available_resources["oil"], required_amount - coal_used)
    if coal_used + oil_used != required_amount:
        raise ModelValidationError("player does not have enough coal and oil to run the hybrid plant")
    mix: dict[str, int] = {}
    if coal_used:
        mix["coal"] = coal_used
    if oil_used:
        mix["oil"] = oil_used
    return mix


def _enumerate_hybrid_run_assignments(
    hybrid_plants: list[PowerPlantCard],
    remaining_resources: dict[str, int],
) -> tuple[dict[int, dict[str, int]], ...]:
    assignments: list[dict[int, dict[str, int]]] = []

    def search(index: int, coal_left: int, oil_left: int, chosen: dict[int, dict[str, int]]) -> None:
        if index == len(hybrid_plants):
            assignments.append({price: dict(mix) for price, mix in chosen.items()})
            return
        plant = hybrid_plants[index]
        minimum_coal = max(0, plant.resource_cost - oil_left)
        maximum_coal = min(plant.resource_cost, coal_left)
        for coal_used in range(minimum_coal, maximum_coal + 1):
            oil_used = plant.resource_cost - coal_used
            chosen[plant.price] = _normalize_resource_mix({"coal": coal_used, "oil": oil_used})
            search(index + 1, coal_left - coal_used, oil_left - oil_used, chosen)
        chosen.pop(plant.price, None)

    search(0, remaining_resources["coal"], remaining_resources["oil"], {})
    return tuple(assignments)


def _resource_mix_key(mix: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((resource, int(amount)) for resource, amount in mix.items() if amount))


def _resource_mix_from_key(key: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return {resource: amount for resource, amount in key}


def _format_resource_mix(mix: dict[str, int]) -> str:
    if not mix:
        return "(none)"
    return ", ".join(f"{resource}={amount}" for resource, amount in sorted(mix.items()))


def _summarize_resource_usage(plans: tuple[PlantRunPlan, ...]) -> dict[str, int]:
    usage = {resource: 0 for resource in RESOURCE_TYPES}
    for plan in plans:
        for resource, amount in plan.resource_mix.items():
            usage[resource] += amount
    return usage


def _highest_connected_city_count(state: GameState) -> int:
    return max((player.connected_city_count for player in state.players), default=0)


def _check_end_game_trigger(state: GameState) -> bool:
    threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    return any(player.connected_city_count >= threshold for player in state.players)


def _visible_market(state: GameState) -> list[PowerPlantCard]:
    return sorted((*state.current_market, *state.future_market), key=lambda plant: plant.price)


def _replace_visible_market(
    state: GameState,
    visible_market: list[PowerPlantCard] | tuple[PowerPlantCard, ...],
    *,
    step_override: int | None = None,
    draw_stack: tuple[PowerPlantCard, ...] | None = None,
    bottom_stack: tuple[PowerPlantCard, ...] | None = None,
    step_3_card_pending: bool | None = None,
    auction_step_3_pending: bool | None = None,
) -> GameState:
    step = state.step if step_override is None else step_override
    ordered = tuple(sorted(visible_market, key=lambda plant: plant.price))
    replace_kwargs: dict[str, Any] = {
        "step": step,
        "power_plant_draw_stack": state.power_plant_draw_stack if draw_stack is None else draw_stack,
        "power_plant_bottom_stack": (
            state.power_plant_bottom_stack if bottom_stack is None else bottom_stack
        ),
        "step_3_card_pending": (
            state.step_3_card_pending if step_3_card_pending is None else step_3_card_pending
        ),
        "auction_step_3_pending": (
            state.auction_step_3_pending
            if auction_step_3_pending is None
            else auction_step_3_pending
        ),
    }
    if step == 3:
        return replace(state, current_market=ordered, future_market=(), **replace_kwargs)
    if replace_kwargs["auction_step_3_pending"]:
        if not 1 <= len(ordered) <= 8:
            raise ModelValidationError(
                "auction Step 3 pending state must keep between 1 and 8 visible plants including the Step 3 placeholder"
            )
        current_count = min(4, len(ordered))
        return replace(
            state,
            current_market=ordered[:current_count],
            future_market=ordered[current_count:],
            **replace_kwargs,
        )
    if len(ordered) != 8:
        raise ModelValidationError("steps 1 and 2 must expose exactly 8 visible power plants")
    return replace(
        state,
        current_market=ordered[:4],
        future_market=ordered[4:8],
        **replace_kwargs,
    )


def _draw_valid_market_plant(
    draw_stack: tuple[PowerPlantCard, ...],
    minimum_allowed_price: int,
    *,
    trigger_step_3_if_exhausted: bool,
) -> tuple[PowerPlantCard | None, tuple[PowerPlantCard, ...], bool]:
    remaining = list(draw_stack)
    while remaining:
        candidate = remaining.pop(0)
        if candidate.price <= minimum_allowed_price:
            continue
        return candidate, tuple(remaining), False
    return None, tuple(remaining), trigger_step_3_if_exhausted


def _shuffle_bottom_stack_for_step_3(
    state: GameState,
    bottom_stack: tuple[PowerPlantCard, ...],
) -> tuple[PowerPlantCard, ...]:
    shuffled = list(bottom_stack)
    rng = random.Random(f"{state.config.seed}:step3:{state.round_number}:{len(bottom_stack)}")
    rng.shuffle(shuffled)
    return tuple(shuffled)


def _apply_auction_step_3_transition(state: GameState) -> GameState:
    if not state.auction_step_3_pending:
        return state
    visible_market = [
        plant for plant in _visible_market(state) if not plant.is_step_3_placeholder
    ]
    if visible_market:
        visible_market.pop(0)
    return _replace_visible_market(
        state,
        visible_market,
        step_override=3,
        draw_stack=state.power_plant_draw_stack,
        bottom_stack=(),
        step_3_card_pending=False,
        auction_step_3_pending=False,
    )


def _apply_step_2_transition(state: GameState) -> GameState:
    if not check_step_2_trigger(state):
        return state

    lowest_current = min(state.current_market, key=lambda plant: plant.price)
    visible_market = [
        plant
        for plant in _visible_market(state)
        if plant.price != lowest_current.price
    ]
    replacement, draw_stack, trigger_step_3 = _draw_valid_market_plant(
        state.power_plant_draw_stack,
        _highest_connected_city_count(state),
        trigger_step_3_if_exhausted=state.step_3_card_pending,
    )
    if replacement is not None:
        visible_market.append(replacement)
        return _replace_visible_market(
            state,
            visible_market,
            step_override=2,
            draw_stack=draw_stack,
        )
    if trigger_step_3:
        if visible_market:
            visible_market = visible_market[1:]
        return _replace_visible_market(
            state,
            visible_market,
            step_override=3,
            draw_stack=_shuffle_bottom_stack_for_step_3(
                state,
                state.power_plant_bottom_stack,
            ),
            bottom_stack=(),
            step_3_card_pending=False,
        )
    return _replace_visible_market(
        state,
        visible_market,
        step_override=2,
        draw_stack=draw_stack,
    )


def _update_plant_market_after_bureaucracy(state: GameState) -> GameState:
    visible_market = _visible_market(state)
    if not visible_market:
        return state

    if state.step in {1, 2}:
        cycled_highest = visible_market.pop(-1)
        bottom_stack = tuple((*state.power_plant_bottom_stack, cycled_highest))
        replacement, draw_stack, trigger_step_3 = _draw_valid_market_plant(
            state.power_plant_draw_stack,
            _highest_connected_city_count(state),
            trigger_step_3_if_exhausted=state.step_3_card_pending,
        )
        if replacement is not None:
            visible_market.append(replacement)
            return _replace_visible_market(
                state,
                visible_market,
                step_override=state.step,
                draw_stack=draw_stack,
                bottom_stack=bottom_stack,
            )
        if trigger_step_3:
            if visible_market:
                visible_market = visible_market[1:]
            return _replace_visible_market(
                state,
                visible_market,
                step_override=3,
                draw_stack=_shuffle_bottom_stack_for_step_3(
                    state,
                    bottom_stack,
                ),
                bottom_stack=(),
                step_3_card_pending=False,
            )
        return _replace_visible_market(
            state,
            visible_market,
            step_override=state.step,
            draw_stack=draw_stack,
            bottom_stack=bottom_stack,
        )

    visible_market = visible_market[1:]
    replacement, draw_stack, _ = _draw_valid_market_plant(
        state.power_plant_draw_stack,
        _highest_connected_city_count(state),
        trigger_step_3_if_exhausted=False,
    )
    if replacement is not None:
        visible_market.append(replacement)
    return _replace_visible_market(
        state,
        visible_market,
        step_override=3,
        draw_stack=draw_stack,
    )


def _allowed_city_ids(state: GameState) -> set[str]:
    selected_regions = set(state.selected_regions)
    if not selected_regions:
        return {city.id for city in state.game_map.cities}
    return {city.id for city in state.game_map.cities if city.region in selected_regions}


def _city_name(state: GameState, city_id: str) -> str:
    for city in state.game_map.cities:
        if city.id == city_id:
            return city.name
    raise ModelValidationError(f"unknown city {city_id!r}")


def _city_occupancy_limit(step: int) -> int:
    if step not in {1, 2, 3}:
        raise ModelValidationError("step must be 1, 2, or 3")
    return step


def _city_occupant_count(state: GameState, city_id: str) -> int:
    return sum(1 for player in state.players if city_id in player.network_city_ids)


def _city_build_cost(state: GameState, city_id: str) -> int:
    occupants = _city_occupant_count(state, city_id)
    if occupants >= _city_occupancy_limit(state.step):
        raise ModelValidationError(f"city {city_id!r} is already full for step {state.step}")
    return 10 + (occupants * 5)


def _selected_area_adjacency(state: GameState) -> dict[str, dict[str, int]]:
    allowed = _allowed_city_ids(state)
    adjacency: dict[str, dict[str, int]] = {city_id: {} for city_id in allowed}
    for connection in state.game_map.connections:
        if connection.city_1 not in allowed or connection.city_2 not in allowed:
            continue
        adjacency[connection.city_1][connection.city_2] = connection.cost
        adjacency[connection.city_2][connection.city_1] = connection.cost
    return adjacency


def _shortest_connection_cost(
    state: GameState,
    source_city_ids: tuple[str, ...],
    target_city: str,
) -> int:
    return _all_connection_costs_from_sources(state, source_city_ids)[target_city]


def _all_connection_costs_from_sources(
    state: GameState,
    source_city_ids: tuple[str, ...],
) -> dict[str, int]:
    adjacency = _selected_area_adjacency(state)
    if not source_city_ids:
        return {city_id: 0 for city_id in adjacency}

    queue: list[tuple[int, str]] = []
    best_costs: dict[str, int] = {}
    for city_id in source_city_ids:
        if city_id not in adjacency:
            raise ModelValidationError(f"city {city_id!r} is not in the selected play area")
        best_costs[city_id] = 0
        heapq.heappush(queue, (0, city_id))

    while queue:
        cost, city_id = heapq.heappop(queue)
        if cost > best_costs.get(city_id, cost):
            continue
        for neighbor, edge_cost in adjacency[city_id].items():
            next_cost = cost + edge_cost
            if next_cost < best_costs.get(neighbor, next_cost + 1):
                best_costs[neighbor] = next_cost
                heapq.heappush(queue, (next_cost, neighbor))
    return best_costs


def _best_build_sequence(
    state: GameState,
    player: PlayerState,
    targets: tuple[str, ...],
) -> tuple[int, tuple[str, ...]]:
    if len(targets) == 1:
        target = targets[0]
        connection_cost = compute_connection_cost(state, player.player_id, target)
        return connection_cost + _city_build_cost(state, target), targets

    best_total: int | None = None
    best_order: tuple[str, ...] | None = None
    for order in permutations(targets):
        current_sources = tuple(player.network_city_ids)
        running_total = 0
        built_sources = list(current_sources)
        for city_id in order:
            connection_cost = (
                0 if not built_sources else _shortest_connection_cost(state, tuple(built_sources), city_id)
            )
            running_total += connection_cost + _city_build_cost(state, city_id)
            built_sources.append(city_id)
            if best_total is not None and running_total >= best_total:
                break
        else:
            if best_total is None or running_total < best_total:
                best_total = running_total
                best_order = tuple(order)
    assert best_total is not None and best_order is not None
    return best_total, best_order


def _get_auctionable_plant(state: GameState, plant_id: int) -> PowerPlantCard:
    for plant in list_auctionable_plants(state):
        if plant.price == int(plant_id):
            return plant
    raise ModelValidationError(f"power plant {plant_id} is not currently auctionable")


def _get_plant_from_visible_market(state: GameState, plant_price: int) -> PowerPlantCard:
    for plant in (*state.current_market, *state.future_market):
        if plant.price == plant_price:
            return plant
    raise ModelValidationError(f"power plant {plant_price} is not visible in the market")


def _minimum_opening_bid(state: GameState, plant: PowerPlantCard) -> int:
    if (
        state.auction_state is not None
        and state.auction_state.discount_token_plant_price == plant.price
    ):
        return 1
    return plant.price


def _eligible_auction_participants(
    state: GameState,
    auction_state: AuctionState,
) -> tuple[str, ...]:
    excluded = set(auction_state.players_with_plants) | set(auction_state.players_passed_phase)
    return tuple(player_id for player_id in state.player_order if player_id not in excluded)


def _rotate_player_order(player_ids: tuple[str, ...], start_player_id: str) -> tuple[str, ...]:
    if start_player_id not in player_ids:
        raise ModelValidationError("rotation start player must be in the provided order")
    start_index = player_ids.index(start_player_id)
    return tuple((*player_ids[start_index:], *player_ids[:start_index]))


def _next_player_in_sequence(player_ids: tuple[str, ...], current_player_id: str) -> str | None:
    if len(player_ids) <= 1:
        return None
    if current_player_id not in player_ids:
        raise ModelValidationError("current player must be in the bidder order")
    current_index = player_ids.index(current_player_id)
    return player_ids[(current_index + 1) % len(player_ids)]


def _next_remaining_player(
    prior_order: tuple[str, ...],
    current_player_id: str,
    remaining_players: tuple[str, ...],
) -> str | None:
    if not remaining_players:
        return None
    if current_player_id not in prior_order:
        raise ModelValidationError("current player must be in the prior bidder order")
    start_index = prior_order.index(current_player_id)
    for offset in range(1, len(prior_order) + 1):
        candidate = prior_order[(start_index + offset) % len(prior_order)]
        if candidate in remaining_players:
            return candidate
    return None


def _next_auction_chooser(
    player_order: tuple[str, ...],
    current_player_id: str | None,
    players_with_plants: tuple[str, ...],
    players_passed_phase: tuple[str, ...],
) -> str | None:
    excluded = set(players_with_plants) | set(players_passed_phase)
    eligible = tuple(player_id for player_id in player_order if player_id not in excluded)
    if not eligible:
        return None
    if current_player_id is None:
        return eligible[0]
    for offset in range(1, len(player_order) + 1):
        candidate = player_order[(player_order.index(current_player_id) + offset) % len(player_order)]
        if candidate not in excluded:
            return candidate
    return None


def _award_auction_purchase(
    state: GameState,
    winner_id: str,
    plant: PowerPlantCard,
    price_paid: int,
    chooser_id: str,
) -> GameState:
    player = _get_player(state, winner_id)
    purchased_plant = PowerPlantCard.from_dict(plant.to_dict())
    updated_power_plants = tuple(
        sorted((*player.power_plants, purchased_plant), key=lambda card: card.price)
    )
    updated_player = replace(
        player,
        elektro=player.elektro - price_paid,
        power_plants=updated_power_plants,
        resource_storage=_normalize_resource_storage(player.resource_storage, updated_power_plants),
    )
    updated_state = _replace_player_on_state(state, updated_player)
    updated_state = _remove_visible_plant_and_refill(
        updated_state,
        plant.price,
        clear_discount_before_refill=(
            updated_state.auction_state is not None
            and updated_state.auction_state.discount_token_plant_price == plant.price
        ),
    )

    auction_state = updated_state.auction_state
    assert auction_state is not None
    buyers = tuple((*auction_state.players_with_plants, winner_id))
    if winner_id == chooser_id:
        next_chooser = _next_auction_chooser(
            updated_state.player_order,
            chooser_id,
            players_with_plants=buyers,
            players_passed_phase=auction_state.players_passed_phase,
        )
    else:
        next_chooser = chooser_id

    cleared_auction = replace(
        auction_state,
        current_chooser_id=next_chooser,
        players_with_plants=buyers,
        active_plant_price=None,
        current_bid=None,
        highest_bidder_id=None,
        active_bidders=(),
        next_bidder_id=None,
    )
    updated_state = replace(updated_state, auction_state=cleared_auction)

    max_plants = updated_state.rules.player_count_rules[len(updated_state.players)]["max_power_plants"]
    if len(updated_player.power_plants) > max_plants:
        legal_discards = tuple(
            Action(
                action_type="discard_power_plant",
                player_id=winner_id,
                payload={"price": power_plant.price},
            )
            for power_plant in updated_player.power_plants
        )
        updated_state = replace(
            updated_state,
            pending_decision=DecisionRequest(
                player_id=winner_id,
                decision_type="discard_power_plant",
                prompt="Choose a power plant to discard.",
                legal_actions=legal_discards,
                metadata={"max_power_plants": max_plants},
            ),
        )
        return updated_state

    return _maybe_finish_auction_phase(updated_state)


def _maybe_finish_auction_phase(state: GameState) -> GameState:
    if state.phase != "auction" or state.auction_state is None or state.pending_decision is not None:
        return state
    auction_state = state.auction_state
    if auction_state.has_active_auction:
        return state
    if auction_state.current_chooser_id is not None:
        return state
    return resolve_auction_round(state)


def _remove_visible_plant_and_refill(
    state: GameState,
    sold_price: int,
    clear_discount_before_refill: bool = False,
) -> GameState:
    visible_market = [plant for plant in (*state.current_market, *state.future_market) if plant.price != sold_price]
    if len(visible_market) != len(state.current_market) + len(state.future_market) - 1:
        raise ModelValidationError("expected to remove exactly one visible power plant")
    draw_stack = list(state.power_plant_draw_stack)
    auction_state = state.auction_state
    auction_step_3_pending = state.auction_step_3_pending
    if auction_state is not None and clear_discount_before_refill:
        auction_state = replace(auction_state, discount_token_plant_price=None)

    if state.step == 3:
        drawn_plant, remaining_stack, _ = _draw_valid_market_plant(
            tuple(draw_stack),
            _highest_connected_city_count(state),
            trigger_step_3_if_exhausted=False,
        )
        if drawn_plant is not None:
            visible_market.append(drawn_plant)
        visible_market.sort(key=lambda plant: plant.price)
        return replace(
            state,
            current_market=tuple(visible_market),
            future_market=(),
            power_plant_draw_stack=remaining_stack,
            auction_state=auction_state,
        )

    if auction_step_3_pending:
        if draw_stack:
            visible_market.append(draw_stack.pop(0))
        visible_market.sort(key=lambda plant: plant.price)
        return _replace_visible_market(
            replace(state, auction_state=auction_state),
            visible_market,
            draw_stack=tuple(draw_stack),
            bottom_stack=(),
            step_3_card_pending=False,
            auction_step_3_pending=True,
        )

    if draw_stack:
        drawn_plant = draw_stack.pop(0)
        if (
            auction_state is not None
            and auction_state.discount_token_plant_price is not None
            and drawn_plant.price < auction_state.discount_token_plant_price
        ):
            auction_state = replace(auction_state, discount_token_plant_price=None)
            if draw_stack:
                drawn_plant = draw_stack.pop(0)
                visible_market.append(drawn_plant)
            elif state.step_3_card_pending:
                visible_market.append(PowerPlantCard.step_3_placeholder())
                auction_step_3_pending = True
        else:
            visible_market.append(drawn_plant)
    elif state.step_3_card_pending:
        visible_market.append(PowerPlantCard.step_3_placeholder())
        auction_step_3_pending = True
    visible_market.sort(key=lambda plant: plant.price)
    if auction_step_3_pending:
        shuffled_bottom = _shuffle_bottom_stack_for_step_3(state, state.power_plant_bottom_stack)
        return _replace_visible_market(
            replace(state, auction_state=auction_state),
            visible_market,
            draw_stack=shuffled_bottom if not state.auction_step_3_pending else tuple(draw_stack),
            step_3_card_pending=False,
            auction_step_3_pending=True,
            bottom_stack=(),
        )
    if len(visible_market) < 8:
        raise ModelValidationError("power plant market must contain eight visible plants before Step 3")
    return replace(
        state,
        current_market=tuple(visible_market[:4]),
        future_market=tuple(visible_market[4:8]),
        power_plant_draw_stack=tuple(draw_stack),
        auction_state=auction_state,
        auction_step_3_pending=False,
    )


def _recalculate_player_order(state: GameState) -> GameState:
    previous_index = {player_id: index for index, player_id in enumerate(state.player_order)}
    sorted_players = sorted(
        state.players,
        key=lambda player: (
            -player.connected_city_count,
            -player.largest_power_plant,
            previous_index[player.player_id],
        ),
    )
    ordered_ids = tuple(player.player_id for player in sorted_players)
    positioned_players = tuple(
        replace(player, turn_order_position=ordered_ids.index(player.player_id) + 1)
        for player in state.players
    )
    return replace(state, players=positioned_players, player_order=ordered_ids)


def _validate_play_areas(
    game_map: MapDefinition,
    selected_regions: tuple[str, ...],
    required_count: int,
) -> None:
    region_ids = {region.id for region in game_map.regions}
    if len(selected_regions) != required_count:
        raise ModelValidationError(
            f"expected exactly {required_count} selected regions, got {len(selected_regions)}"
        )
    if len(set(selected_regions)) != len(selected_regions):
        raise ModelValidationError("selected regions must be unique")
    if any(region not in region_ids for region in selected_regions):
        raise ModelValidationError("selected regions must exist on the chosen map")
    if not _are_regions_contiguous(game_map, selected_regions):
        raise ModelValidationError("selected regions must form one contiguous playing zone")


def _are_regions_contiguous(game_map: MapDefinition, selected_regions: tuple[str, ...] | list[str]) -> bool:
    selected = tuple(selected_regions)
    if not selected:
        return False
    allowed = set(selected)
    frontier = [selected[0]]
    visited: set[str] = set()
    while frontier:
        region = frontier.pop()
        if region in visited:
            continue
        visited.add(region)
        for neighbor in game_map.region_adjacency.get(region, ()):
            if neighbor in allowed and neighbor not in visited:
                frontier.append(neighbor)
    return visited == allowed


def _serialize_map_definition(game_map: MapDefinition) -> dict[str, Any]:
    return {
        "id": game_map.id,
        "name": game_map.name,
        "regions": [
            {"id": region.id, "label": region.label, "color": region.color}
            for region in game_map.regions
        ],
        "cities": [
            {"id": city.id, "name": city.name, "region": city.region}
            for city in game_map.cities
        ],
        "connections": [
            {"city_1": connection.city_1, "city_2": connection.city_2, "cost": connection.cost}
            for connection in game_map.connections
        ],
        "region_adjacency": {
            region: list(neighbors) for region, neighbors in game_map.region_adjacency.items()
        },
        "special_rules": list(game_map.special_rules),
    }


def _deserialize_map_definition(payload: dict[str, Any]) -> MapDefinition:
    return MapDefinition(
        id=payload["id"],
        name=payload["name"],
        regions=tuple(
            RegionDefinition(
                id=region["id"],
                label=region["label"],
                color=region["color"],
            )
            for region in payload["regions"]
        ),
        cities=tuple(
            CityDefinition(
                id=city["id"],
                name=city["name"],
                region=city["region"],
            )
            for city in payload["cities"]
        ),
        connections=tuple(
            ConnectionDefinition(
                city_1=connection["city_1"],
                city_2=connection["city_2"],
                cost=int(connection["cost"]),
            )
            for connection in payload["connections"]
        ),
        region_adjacency={
            region: tuple(neighbors)
            for region, neighbors in payload["region_adjacency"].items()
        },
        special_rules=tuple(payload["special_rules"]),
    )


def _serialize_rule_tables(rules: RuleTables) -> dict[str, Any]:
    return {
        "starting_money": rules.starting_money,
        "houses_per_player": rules.houses_per_player,
        "resource_supply": dict(rules.resource_supply),
        "resource_market_tracks": rules.resource_market_tracks,
        "payment_schedule": {str(key): value for key, value in rules.payment_schedule.items()},
        "player_count_rules": {str(key): value for key, value in rules.player_count_rules.items()},
        "setup": rules.setup,
    }


def _deserialize_rule_tables(payload: dict[str, Any]) -> RuleTables:
    return RuleTables(
        starting_money=int(payload["starting_money"]),
        houses_per_player=int(payload["houses_per_player"]),
        resource_supply={key: int(value) for key, value in payload["resource_supply"].items()},
        resource_market_tracks=payload["resource_market_tracks"],
        payment_schedule={int(key): int(value) for key, value in payload["payment_schedule"].items()},
        player_count_rules={
            int(key): value for key, value in payload["player_count_rules"].items()
        },
        setup=payload["setup"],
    )
