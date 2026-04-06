from __future__ import annotations

from dataclasses import replace

from .model import (
    GameConfig,
    GameState,
    PowerPlantCard,
    ResourceStorage,
    advance_phase,
    create_initial_state,
    make_default_seat_configs,
)
from .rules_data import load_power_plants


SCENARIO_NAMES = (
    "opening",
    "resource",
    "build_test",
    "step2",
    "auction_step3",
    "endgame",
)

_RESOURCE_PLANT_ASSIGNMENTS = (
    (5, 10),
    (7, 11),
    (6, 13),
    (12, 14),
    (15, 17),
    (8, 16),
)


def build_game_scenario(name: str, seed: int = 7) -> GameState:
    if name not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {name!r}; expected one of {', '.join(SCENARIO_NAMES)}")
    if name == "opening":
        return _build_opening_state(seed)
    if name == "resource":
        return _build_resource_state(seed)
    if name == "build_test":
        return _build_test_map_build_state(seed)
    if name == "step2":
        return _build_bureaucracy_state(
            seed=seed,
            step=1,
            player_specs={
                "p1": {"cities": 7, "plants": (13,), "storage": {}, "elektro": 40},
                "p2": {"cities": 4, "plants": (18,), "storage": {}, "elektro": 35},
                "p3": {"cities": 2, "plants": (22,), "storage": {}, "elektro": 30},
            },
        )
    if name == "auction_step3":
        return _build_auction_step3_state(seed)
    return _build_bureaucracy_state(
        seed=seed,
        step=2,
        player_specs={
            "p1": {"cities": 17, "plants": (25, 13), "storage": {"coal": 2}, "elektro": 40},
            "p2": {"cities": 15, "plants": (20, 23), "storage": {"coal": 3, "uranium": 1}, "elektro": 20},
            "p3": {"cities": 10, "plants": (18, 22), "storage": {}, "elektro": 50},
        },
    )


def _build_opening_state(seed: int) -> GameState:
    config = GameConfig(
        map_id="germany",
        players=make_default_seat_configs(3),
        seed=seed,
    )
    return advance_phase(create_initial_state(config))


def _build_resource_state(seed: int) -> GameState:
    base_state = create_initial_state(
        GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=seed)
    )
    definitions = _power_plant_definitions_by_price()
    updated_players = []
    for index, player in enumerate(base_state.players):
        plants = tuple(
            PowerPlantCard.from_definition(definitions[price])
            for price in _RESOURCE_PLANT_ASSIGNMENTS[index]
        )
        updated_players.append(replace(player, power_plants=plants))
    return replace(
        base_state,
        players=tuple(updated_players),
        phase="buy_resources",
        round_number=2,
        step=1,
        auction_state=None,
        pending_decision=None,
    )


def _build_test_map_build_state(seed: int) -> GameState:
    base_state = create_initial_state(
        GameConfig(
            map_id="test",
            players=make_default_seat_configs(3),
            seed=seed,
            selected_regions=("alpha", "beta", "gamma"),
        )
    )
    assignments = {
        "p1": ("amber_falls",),
        "p2": ("brass_harbor",),
        "p3": (),
    }
    updated_players = []
    for player in base_state.players:
        owned_cities = assignments[player.player_id]
        updated_players.append(
            replace(
                player,
                elektro=80,
                houses_in_supply=22 - len(owned_cities),
                network_city_ids=owned_cities,
            )
        )
    return replace(
        base_state,
        players=tuple(updated_players),
        phase="build_houses",
        round_number=2,
        step=2,
        auction_state=None,
        pending_decision=None,
    )


def _build_bureaucracy_state(
    *,
    seed: int,
    step: int,
    player_specs: dict[str, dict[str, object]],
) -> GameState:
    base_state = create_initial_state(
        GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=seed)
    )
    definitions = _power_plant_definitions_by_price()
    available_city_ids = [
        city.id for city in base_state.game_map.cities if city.region in base_state.selected_regions
    ]
    updated_players = []
    for player in base_state.players:
        spec = player_specs[player.player_id]
        city_count = int(spec["cities"])
        updated_players.append(
            replace(
                player,
                elektro=int(spec["elektro"]),
                houses_in_supply=22 - city_count,
                network_city_ids=tuple(available_city_ids[:city_count]),
                power_plants=tuple(
                    PowerPlantCard.from_definition(definitions[price])
                    for price in spec["plants"]
                ),
                resource_storage=ResourceStorage.from_dict(spec["storage"]),
            )
        )
    return replace(
        base_state,
        players=tuple(updated_players),
        phase="bureaucracy",
        round_number=2,
        step=step,
        auction_state=None,
        pending_decision=None,
        last_powered_cities={},
        last_income_paid={},
    )


def _build_auction_step3_state(seed: int) -> GameState:
    state = _build_opening_state(seed)
    definitions = _power_plant_definitions_by_price()
    return replace(
        state,
        round_number=2,
        step=2,
        power_plant_draw_stack=(),
        power_plant_bottom_stack=tuple(
            PowerPlantCard.from_definition(definitions[price]) for price in (25, 31, 33)
        ),
    )


def _power_plant_definitions_by_price():
    return {definition.price: definition for definition in load_power_plants()}
