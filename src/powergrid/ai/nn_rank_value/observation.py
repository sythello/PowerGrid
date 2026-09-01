from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from statistics import mean
from typing import Any

from ...model import GameState, RESOURCE_TYPES, PowerPlantCard, legal_build_targets
from ...session_types import TurnRequest
from .candidates import CandidateAction


OBSERVATION_SCHEMA_VERSION = 1
ACTION_FEATURE_SCHEMA_VERSION = 1
MAX_PLAYERS = 6
MAX_PLAYER_PLANTS = 4
MAX_CURRENT_MARKET = 6
MAX_FUTURE_MARKET = 4
MAX_RESOURCE_PRICE_SLOTS = 8
MAX_BUILD_COST_SLOTS = 8

PHASES = ("auction", "buy_resources", "build_houses", "bureaucracy")
DECISION_TYPES = (
    "auction_start",
    "auction_bid",
    "buy_resources",
    "build_houses",
    "bureaucracy",
    "discard_power_plant",
    "discard_hybrid_resources",
)
ACTION_TYPES = (
    "auction_start",
    "auction_bid",
    "auction_pass",
    "buy_resource",
    "finish_buying",
    "commit_build",
    "finish_building",
    "run_plants",
    "skip_bureaucracy",
    "discard_power_plant",
    "discard_hybrid_resources",
)


@dataclass(frozen=True)
class PublicObservation:
    """AI-visible state with hidden deck order and the game seed removed."""

    payload: dict[str, Any]
    schema_version: int = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", deepcopy(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "payload": deepcopy(self.payload),
        }


class _FeatureBuilder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.values: list[float] = []

    def add(self, name: str, value: int | float | bool) -> None:
        self.names.append(name)
        self.values.append(float(value))


def build_public_observation(
    state: GameState,
    request: TurnRequest,
    player_id: str | None = None,
) -> PublicObservation:
    actor_id = player_id or request.player_id
    allowed_city_ids = {
        city.id for city in state.game_map.cities if city.region in state.selected_regions
    }
    city_occupancy = {
        city_id: sum(city_id in player.network_city_ids for player in state.players)
        for city_id in sorted(allowed_city_ids)
    }
    allowed_connections = [
        connection
        for connection in state.game_map.connections
        if connection.city_1 in allowed_city_ids and connection.city_2 in allowed_city_ids
    ]
    build_targets = []
    try:
        build_targets = [
            {
                "city_id": str(action.payload["city_id"]),
                "connection_cost": int(action.payload["connection_cost"]),
                "build_cost": int(action.payload["build_cost"]),
                "total_cost": int(action.payload["total_cost"]),
            }
            for action in legal_build_targets(state, actor_id)
        ]
    except (StopIteration, ValueError):
        build_targets = []

    auction = state.auction_state
    auction_payload: dict[str, Any] | None = None
    if auction is not None:
        auction_payload = {
            "current_chooser_id": auction.current_chooser_id,
            "discount_token_plant_price": auction.discount_token_plant_price,
            "players_with_plants": list(auction.players_with_plants),
            "players_passed_phase": list(auction.players_passed_phase),
            "active_plant_price": auction.active_plant_price,
            "current_bid": auction.current_bid,
            "highest_bidder_id": auction.highest_bidder_id,
            "active_bidders": list(auction.active_bidders),
            "next_bidder_id": auction.next_bidder_id,
        }

    payload: dict[str, Any] = {
        "actor_id": actor_id,
        "map_id": state.game_map.id,
        "selected_regions": list(state.selected_regions),
        "round_number": state.round_number,
        "step": state.step,
        "phase": state.phase,
        "decision_type": request.decision_type,
        "player_count": len(state.players),
        "end_game_cities": int(
            state.rules.player_count_rules[len(state.players)]["end_game_cities"]
        ),
        # Counts are public. Card identities/order and the seed are deliberately absent.
        "hidden_plant_count": (
            len(state.power_plant_draw_stack)
            + len(state.power_plant_bottom_stack)
            + int(state.step_3_card_pending)
        ),
        "step_3_card_pending": state.step_3_card_pending,
        "auction_step_3_pending": state.auction_step_3_pending,
        "current_market": [_public_plant(plant) for plant in state.current_market],
        "future_market": [_public_plant(plant) for plant in state.future_market],
        "resource_market": {
            resource: {
                "price_bands": {
                    str(price): int(amount)
                    for price, amount in state.resource_market.market[resource].items()
                },
                "supply": int(state.resource_market.supply[resource]),
                "unit_prices": list(state.resource_market.available_unit_prices(resource)),
            }
            for resource in RESOURCE_TYPES
        },
        "players": [
            {
                "player_id": player.player_id,
                "is_actor": player.player_id == actor_id,
                "elektro": player.elektro,
                "houses_in_supply": player.houses_in_supply,
                "network_city_ids": list(player.network_city_ids),
                "turn_order_position": player.turn_order_position,
                "power_plants": [_public_plant(plant) for plant in player.power_plants],
                "resources": player.resource_storage.resource_totals(),
                "last_powered_cities": int(state.last_powered_cities.get(player.player_id, 0)),
                "last_income_paid": int(state.last_income_paid.get(player.player_id, 0)),
            }
            for player in state.players
        ],
        "player_order": list(state.player_order),
        "auction": auction_payload,
        "map_summary": {
            "allowed_city_count": len(allowed_city_ids),
            "city_occupancy": city_occupancy,
            "connection_costs": [int(connection.cost) for connection in allowed_connections],
            "build_targets": build_targets,
        },
    }
    return PublicObservation(payload)


def encode_state_features(observation: PublicObservation) -> tuple[list[float], tuple[str, ...]]:
    payload = observation.payload
    builder = _FeatureBuilder()
    phase = str(payload["phase"])
    decision_type = str(payload["decision_type"])
    builder.add("global.round", int(payload["round_number"]) / 20.0)
    for step in (1, 2, 3):
        builder.add(f"global.step_{step}", int(payload["step"]) == step)
    builder.add("global.player_count", int(payload["player_count"]) / MAX_PLAYERS)
    builder.add("global.end_game_cities", int(payload["end_game_cities"]) / 22.0)
    builder.add("global.hidden_plant_count", int(payload["hidden_plant_count"]) / 42.0)
    builder.add("global.step_3_pending", bool(payload["step_3_card_pending"]))
    builder.add("global.auction_step_3_pending", bool(payload["auction_step_3_pending"]))
    for known_phase in PHASES:
        builder.add(f"global.phase.{known_phase}", phase == known_phase)
    for known_decision in DECISION_TYPES:
        builder.add(f"global.decision.{known_decision}", decision_type == known_decision)
    for map_id in ("germany", "usa", "test"):
        builder.add(f"global.map.{map_id}", payload["map_id"] == map_id)
    builder.add("global.selected_region_count", len(payload["selected_regions"]) / 6.0)

    resource_market = payload["resource_market"]
    for resource in RESOURCE_TYPES:
        market = resource_market[resource]
        unit_prices = [int(value) for value in market["unit_prices"]]
        builder.add(f"market.{resource}.available", len(unit_prices) / 24.0)
        builder.add(f"market.{resource}.supply", int(market["supply"]) / 24.0)
        for slot in range(MAX_RESOURCE_PRICE_SLOTS):
            price = unit_prices[slot] if slot < len(unit_prices) else 0
            builder.add(f"market.{resource}.unit_price_{slot}", price / 20.0)

    _append_plant_slots(
        builder,
        "current_market",
        list(payload["current_market"]),
        MAX_CURRENT_MARKET,
    )
    _append_plant_slots(
        builder,
        "future_market",
        list(payload["future_market"]),
        MAX_FUTURE_MARKET,
    )

    actor_id = str(payload["actor_id"])
    players = list(payload["players"])
    actor = next(player for player in players if player["player_id"] == actor_id)
    opponents = sorted(
        (player for player in players if player["player_id"] != actor_id),
        key=lambda item: (int(item["turn_order_position"]), str(item["player_id"])),
    )
    ordered_players = [actor, *opponents]
    for slot in range(MAX_PLAYERS):
        player = ordered_players[slot] if slot < len(ordered_players) else None
        _append_player_slot(builder, slot, player)

    auction = payload["auction"]
    builder.add("auction.present", auction is not None)
    builder.add(
        "auction.active",
        auction is not None and auction.get("active_plant_price") is not None,
    )
    active_price = int(auction.get("active_plant_price") or 0) if auction else 0
    current_bid = int(auction.get("current_bid") or 0) if auction else 0
    builder.add("auction.plant_price", active_price / 50.0)
    builder.add("auction.current_bid", current_bid / 100.0)
    builder.add(
        "auction.actor_is_highest",
        auction is not None and auction.get("highest_bidder_id") == actor_id,
    )
    builder.add(
        "auction.actor_is_chooser",
        auction is not None and auction.get("current_chooser_id") == actor_id,
    )
    builder.add(
        "auction.active_bidder_count",
        len(auction.get("active_bidders", [])) / MAX_PLAYERS if auction else 0,
    )
    builder.add(
        "auction.passed_player_count",
        len(auction.get("players_passed_phase", [])) / MAX_PLAYERS if auction else 0,
    )

    map_summary = payload["map_summary"]
    occupancies = list(map_summary["city_occupancy"].values())
    builder.add("map.allowed_city_count", int(map_summary["allowed_city_count"]) / 42.0)
    for occupancy in (0, 1, 2, 3):
        count = sum(int(value) == occupancy for value in occupancies)
        builder.add(f"map.city_occupancy_{occupancy}", count / 42.0)
    connection_costs = [int(value) for value in map_summary["connection_costs"]]
    builder.add("map.connection_min", (min(connection_costs) if connection_costs else 0) / 30.0)
    builder.add("map.connection_mean", (mean(connection_costs) if connection_costs else 0) / 30.0)
    builder.add("map.connection_max", (max(connection_costs) if connection_costs else 0) / 30.0)
    build_costs = sorted(int(target["total_cost"]) for target in map_summary["build_targets"])
    builder.add("map.actor_build_target_count", len(build_costs) / 42.0)
    for slot in range(MAX_BUILD_COST_SLOTS):
        cost = build_costs[slot] if slot < len(build_costs) else 0
        builder.add(f"map.actor_build_cost_{slot}", cost / 100.0)

    return builder.values, tuple(builder.names)


def player_slot_ids(observation: PublicObservation) -> tuple[str, ...]:
    """Return player ids in the exact relative order used by state features."""

    payload = observation.payload
    actor_id = str(payload["actor_id"])
    players = list(payload["players"])
    actor = next(player for player in players if player["player_id"] == actor_id)
    opponents = sorted(
        (player for player in players if player["player_id"] != actor_id),
        key=lambda item: (int(item["turn_order_position"]), str(item["player_id"])),
    )
    return tuple(str(player["player_id"]) for player in (actor, *opponents))


def encode_action_features(
    observation: PublicObservation,
    candidate: CandidateAction,
) -> tuple[list[float], tuple[str, ...]]:
    builder = _FeatureBuilder()
    intent = candidate.intent
    metadata = candidate.metadata
    action_type = intent.intent_type
    for known_action in ACTION_TYPES:
        builder.add(f"action.type.{known_action}", action_type == known_action)

    actor = next(
        player
        for player in observation.payload["players"]
        if player["player_id"] == intent.player_id
    )
    actor_cash = int(actor["elektro"])
    bid = int(intent.payload.get("bid", metadata.get("bid", 0)))
    plant_price = int(
        intent.payload.get(
            "plant_price",
            metadata.get("plant_price", observation.payload.get("active_plant_price", 0)),
        )
    )
    resource = str(intent.payload.get("resource", metadata.get("resource", "")))
    amount = int(intent.payload.get("amount", metadata.get("amount", 0)))
    direct_cost = int(metadata.get("cost", metadata.get("total_cost", bid)))
    builder.add("action.bid", bid / 100.0)
    builder.add("action.plant_price", plant_price / 50.0)
    builder.add("action.amount", amount / 8.0)
    builder.add("action.direct_cost", direct_cost / 200.0)
    builder.add("action.cash_after", max(0, actor_cash - direct_cost) / 200.0)
    for known_resource in RESOURCE_TYPES:
        builder.add(f"action.resource.{known_resource}", resource == known_resource)

    city_ids = list(intent.payload.get("city_ids", metadata.get("city_ids", [])))
    builder.add("action.city_count", len(city_ids) / 10.0)
    builder.add("action.connection_cost", int(metadata.get("connection_cost", 0)) / 100.0)
    builder.add("action.build_cost", int(metadata.get("build_cost", 0)) / 100.0)
    builder.add("action.total_build_cost", int(metadata.get("total_cost", 0)) / 200.0)
    city_id = str(metadata.get("city_id", city_ids[0] if city_ids else ""))
    occupancy = int(
        observation.payload["map_summary"]["city_occupancy"].get(city_id, 0)
    )
    builder.add("action.city_occupancy", occupancy / 3.0)

    powered = int(metadata.get("powered_cities", 0))
    income = int(metadata.get("income", 0))
    builder.add("action.powered_cities", powered / 22.0)
    builder.add("action.income", income / 150.0)
    mix = dict(metadata.get("resource_mix", {}))
    for known_resource in RESOURCE_TYPES:
        builder.add(
            f"action.resource_mix.{known_resource}",
            int(mix.get(known_resource, intent.payload.get(known_resource, 0))) / 8.0,
        )
    builder.add(
        "action.discarded_units",
        (
            int(intent.payload.get("coal", 0))
            + int(intent.payload.get("oil", 0))
        )
        / 8.0,
    )

    plant = _find_public_plant(observation, plant_price)
    _append_plant_features(builder, "action.plant", plant)
    return builder.values, tuple(builder.names)


def _append_player_slot(
    builder: _FeatureBuilder,
    slot: int,
    player: dict[str, Any] | None,
) -> None:
    prefix = f"player_{slot}"
    builder.add(f"{prefix}.present", player is not None)
    builder.add(f"{prefix}.is_actor", bool(player and player["is_actor"]))
    builder.add(
        f"{prefix}.turn_order",
        int(player["turn_order_position"]) / MAX_PLAYERS if player else 0,
    )
    builder.add(f"{prefix}.cash", int(player["elektro"]) / 200.0 if player else 0)
    builder.add(
        f"{prefix}.houses_in_supply",
        int(player["houses_in_supply"]) / 22.0 if player else 0,
    )
    city_count = len(player["network_city_ids"]) if player else 0
    builder.add(f"{prefix}.connected_cities", city_count / 22.0)
    builder.add(
        f"{prefix}.last_powered",
        int(player["last_powered_cities"]) / 22.0 if player else 0,
    )
    builder.add(
        f"{prefix}.last_income",
        int(player["last_income_paid"]) / 150.0 if player else 0,
    )
    resources = player["resources"] if player else {}
    for resource in RESOURCE_TYPES:
        builder.add(f"{prefix}.resource.{resource}", int(resources.get(resource, 0)) / 8.0)
    plants = list(player["power_plants"]) if player else []
    builder.add(
        f"{prefix}.total_output",
        sum(int(plant["output_cities"]) for plant in plants) / 22.0,
    )
    builder.add(
        f"{prefix}.largest_plant",
        max((int(plant["price"]) for plant in plants), default=0) / 50.0,
    )
    _append_plant_slots(builder, f"{prefix}.plant", plants, MAX_PLAYER_PLANTS)


def _append_plant_slots(
    builder: _FeatureBuilder,
    prefix: str,
    plants: list[dict[str, Any]],
    slots: int,
) -> None:
    ordered = sorted(plants, key=lambda item: int(item["price"]))
    for slot in range(slots):
        plant = ordered[slot] if slot < len(ordered) else None
        _append_plant_features(builder, f"{prefix}_{slot}", plant)


def _append_plant_features(
    builder: _FeatureBuilder,
    prefix: str,
    plant: dict[str, Any] | None,
) -> None:
    builder.add(f"{prefix}.present", plant is not None)
    builder.add(f"{prefix}.price", int(plant["price"]) / 50.0 if plant else 0)
    builder.add(
        f"{prefix}.resource_cost",
        int(plant["resource_cost"]) / 3.0 if plant else 0,
    )
    builder.add(
        f"{prefix}.output_cities",
        int(plant["output_cities"]) / 10.0 if plant else 0,
    )
    resource_types = set(plant["resource_types"]) if plant else set()
    for resource in RESOURCE_TYPES:
        builder.add(f"{prefix}.resource.{resource}", resource in resource_types)
    builder.add(f"{prefix}.ecological", bool(plant and plant["is_ecological"]))
    builder.add(f"{prefix}.hybrid", bool(plant and plant["is_hybrid"]))


def _find_public_plant(
    observation: PublicObservation,
    plant_price: int,
) -> dict[str, Any] | None:
    if plant_price <= 0:
        return None
    for plant in [
        *observation.payload["current_market"],
        *observation.payload["future_market"],
    ]:
        if int(plant["price"]) == plant_price:
            return plant
    for player in observation.payload["players"]:
        for plant in player["power_plants"]:
            if int(plant["price"]) == plant_price:
                return plant
    return None


def _public_plant(plant: PowerPlantCard) -> dict[str, Any]:
    return {
        "price": plant.price,
        "resource_types": list(plant.resource_types),
        "resource_cost": plant.resource_cost,
        "output_cities": plant.output_cities,
        "is_hybrid": plant.is_hybrid,
        "is_ecological": plant.is_ecological,
    }


__all__ = [
    "ACTION_FEATURE_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "PublicObservation",
    "build_public_observation",
    "encode_action_features",
    "encode_state_features",
    "player_slot_ids",
]
