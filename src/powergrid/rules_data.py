from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA_ROOT = Path(__file__).resolve().parent / "data"
MAPS_ROOT = DATA_ROOT / "maps"
RULES_ROOT = DATA_ROOT / "rules"


@dataclass(frozen=True)
class RegionDefinition:
    id: str
    label: str
    color: str


@dataclass(frozen=True)
class CityDefinition:
    id: str
    name: str
    region: str


@dataclass(frozen=True)
class ConnectionDefinition:
    city_1: str
    city_2: str
    cost: int


@dataclass(frozen=True)
class MapDefinition:
    id: str
    name: str
    regions: tuple[RegionDefinition, ...]
    cities: tuple[CityDefinition, ...]
    connections: tuple[ConnectionDefinition, ...]
    region_adjacency: dict[str, tuple[str, ...]]
    special_rules: tuple[str, ...]


@dataclass(frozen=True)
class PowerPlantDefinition:
    price: int
    resource_types: tuple[str, ...]
    resource_cost: int
    output_cities: int
    deck_back: str
    is_hybrid: bool
    is_ecological: bool


@dataclass(frozen=True)
class RuleTables:
    starting_money: int
    houses_per_player: int
    resource_supply: dict[str, int]
    resource_market_tracks: dict[str, dict[str, Any]]
    payment_schedule: dict[int, int]
    player_count_rules: dict[int, dict[str, Any]]
    setup: dict[str, Any]


@dataclass(frozen=True)
class ValidationReport:
    maps_loaded: tuple[str, ...]
    power_plant_count: int
    player_counts: tuple[int, ...]
    sample_city: str
    sample_connection: str
    sample_power_plant: str


class DataValidationError(ValueError):
    """Raised when static data fails validation."""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _data_root(override: str | Path | None = None) -> Path:
    if override is None:
        return DATA_ROOT
    return Path(override)


def load_map(map_id: str, data_root: str | Path | None = None) -> MapDefinition:
    root = _data_root(data_root)
    payload = _load_json(root / "maps" / f"{map_id}.json")

    regions = tuple(
        RegionDefinition(
            id=region["id"],
            label=region["label"],
            color=region["color"],
        )
        for region in payload["regions"]
    )
    cities = tuple(
        CityDefinition(
            id=city["id"],
            name=city["name"],
            region=city["region"],
        )
        for city in payload["cities"]
    )
    connections = tuple(
        ConnectionDefinition(
            city_1=connection["city_1"],
            city_2=connection["city_2"],
            cost=int(connection["cost"]),
        )
        for connection in payload["connections"]
    )
    region_by_city = {city.id: city.region for city in cities}
    adjacency: dict[str, set[str]] = {region.id: set() for region in regions}
    for connection in connections:
        region_1 = region_by_city[connection.city_1]
        region_2 = region_by_city[connection.city_2]
        if region_1 != region_2:
            adjacency[region_1].add(region_2)
            adjacency[region_2].add(region_1)

    return MapDefinition(
        id=payload["id"],
        name=payload["name"],
        regions=regions,
        cities=cities,
        connections=connections,
        region_adjacency={
            region_id: tuple(sorted(neighbors))
            for region_id, neighbors in adjacency.items()
        },
        special_rules=tuple(payload.get("special_rules", [])),
    )


def load_power_plants(data_root: str | Path | None = None) -> tuple[PowerPlantDefinition, ...]:
    root = _data_root(data_root)
    payload = _load_json(root / "rules" / "power_plants.json")
    plants = [
        PowerPlantDefinition(
            price=int(item["price"]),
            resource_types=tuple(item["resource_types"]),
            resource_cost=int(item["resource_cost"]),
            output_cities=int(item["output_cities"]),
            deck_back=item["deck_back"],
            is_hybrid=bool(item["is_hybrid"]),
            is_ecological=bool(item["is_ecological"]),
        )
        for item in payload
    ]
    return tuple(sorted(plants, key=lambda plant: plant.price))


def load_rule_tables(data_root: str | Path | None = None) -> RuleTables:
    root = _data_root(data_root)
    payload = _load_json(root / "rules" / "rule_tables.json")
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


def validate_static_data(data_root: str | Path | None = None) -> ValidationReport:
    errors: list[str] = []
    try:
        maps = [load_map("germany", data_root), load_map("usa", data_root)]
        plants = load_power_plants(data_root)
        rules = load_rule_tables(data_root)
    except FileNotFoundError as exc:
        raise DataValidationError(f"missing static data file: {exc.filename}") from exc

    for game_map in maps:
        city_ids = {city.id for city in game_map.cities}
        if len(city_ids) != len(game_map.cities):
            errors.append(f"{game_map.id}: duplicate city ids")
        connection_keys: set[tuple[str, str]] = set()
        for connection in game_map.connections:
            if connection.city_1 == connection.city_2:
                errors.append(f"{game_map.id}: self-loop on {connection.city_1}")
            if connection.city_1 not in city_ids or connection.city_2 not in city_ids:
                errors.append(
                    f"{game_map.id}: invalid connection {connection.city_1}-{connection.city_2}"
                )
            key = tuple(sorted((connection.city_1, connection.city_2)))
            if key in connection_keys:
                errors.append(f"{game_map.id}: duplicate connection {key[0]}-{key[1]}")
            connection_keys.add(key)
        if len(game_map.regions) != 6:
            errors.append(f"{game_map.id}: expected 6 regions, found {len(game_map.regions)}")
        if len(game_map.cities) < 40:
            errors.append(f"{game_map.id}: expected at least 40 cities, found {len(game_map.cities)}")

    if len(plants) != 42:
        errors.append(f"expected 42 power plants, found {len(plants)}")
    plant_prices = [plant.price for plant in plants]
    if plant_prices != sorted(plant_prices):
        errors.append("power plants are not sorted by price")
    if len(set(plant_prices)) != len(plant_prices):
        errors.append("duplicate power plant prices found")
    if set(rules.player_count_rules) != {3, 4, 5, 6}:
        errors.append("player count rules must cover 3-6 players")
    if rules.payment_schedule.get(0) != 10 or rules.payment_schedule.get(20) != 150:
        errors.append("payment schedule endpoints do not match the rulebook")
    if rules.starting_money != 50:
        errors.append("starting money must be 50 Elektro")
    if rules.houses_per_player != 22:
        errors.append("houses per player must be 22")
    if rules.resource_supply != {"coal": 24, "oil": 24, "garbage": 24, "uranium": 12}:
        errors.append("resource supply totals do not match the rulebook")

    if errors:
        raise DataValidationError("\n".join(errors))

    sample_map = maps[0]
    sample_city = sample_map.cities[0]
    sample_connection = sample_map.connections[0]
    sample_plant = plants[0]
    return ValidationReport(
        maps_loaded=tuple(game_map.id for game_map in maps),
        power_plant_count=len(plants),
        player_counts=tuple(sorted(rules.player_count_rules)),
        sample_city=f"{sample_map.name}: {sample_city.name} ({sample_city.region})",
        sample_connection=(
            f"{sample_map.name}: {sample_connection.city_1} - "
            f"{sample_connection.city_2} ({sample_connection.cost})"
        ),
        sample_power_plant=(
            f"Plant {sample_plant.price}: {sample_plant.resource_types or ('eco',)} "
            f"cost {sample_plant.resource_cost} -> {sample_plant.output_cities} cities"
        ),
    )
