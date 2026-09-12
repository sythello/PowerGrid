from __future__ import annotations

from itertools import combinations

from ...model import GameState, PlayerState, RESOURCE_TYPES


STORAGE_BUCKETS = (
    "coal_dedicated",
    "oil_dedicated",
    "hybrid_shared",
    "garbage",
    "uranium",
)


def resource_planning_metrics(state: GameState, player_id: str) -> dict[str, object]:
    """Return deterministic public resource-capacity and generation summaries."""

    player = next(player for player in state.players if player.player_id == player_id)
    capacities = _storage_capacities(player)
    totals = player.resource_storage.resource_totals()

    coal_overflow = max(0, totals["coal"] - capacities["coal_dedicated"])
    oil_overflow = max(0, totals["oil"] - capacities["oil_dedicated"])
    hybrid_used = coal_overflow + oil_overflow
    free = {
        "coal_dedicated": max(0, capacities["coal_dedicated"] - totals["coal"]),
        "oil_dedicated": max(0, capacities["oil_dedicated"] - totals["oil"]),
        "hybrid_shared": max(0, capacities["hybrid_shared"] - hybrid_used),
        "garbage": max(0, capacities["garbage"] - totals["garbage"]),
        "uranium": max(0, capacities["uranium"] - totals["uranium"]),
    }
    max_additional = {
        "coal": free["coal_dedicated"] + free["hybrid_shared"],
        "oil": free["oil_dedicated"] + free["hybrid_shared"],
        "garbage": free["garbage"],
        "uranium": free["uranium"],
    }
    max_runnable_output = _maximum_runnable_output(player)
    nominal_output = sum(
        plant.output_cities
        for plant in player.power_plants
        if not plant.is_step_3_placeholder
    )
    max_powered = min(player.connected_city_count, max_runnable_output)
    return {
        "free_storage": free,
        "max_additional": max_additional,
        "max_runnable_output": max_runnable_output,
        "fuel_output_shortfall": max(0, nominal_output - max_runnable_output),
        "max_powered_cities": max_powered,
        "power_shortfall": max(0, player.connected_city_count - max_powered),
    }


def _storage_capacities(player: PlayerState) -> dict[str, int]:
    capacities = {bucket: 0 for bucket in STORAGE_BUCKETS}
    for plant in player.power_plants:
        if plant.is_step_3_placeholder or plant.is_ecological:
            continue
        if plant.is_hybrid:
            capacities["hybrid_shared"] += plant.max_storage
            continue
        resource = plant.resource_types[0]
        bucket = (
            resource
            if resource in {"garbage", "uranium"}
            else f"{resource}_dedicated"
        )
        capacities[bucket] += plant.max_storage
    return capacities


def _maximum_runnable_output(player: PlayerState) -> int:
    plants = tuple(
        plant for plant in player.power_plants if not plant.is_step_3_placeholder
    )
    stored = player.resource_storage.resource_totals()
    best_output = 0
    for count in range(len(plants) + 1):
        for selected in combinations(plants, count):
            fixed_usage = {resource: 0 for resource in RESOURCE_TYPES}
            hybrid_usage = 0
            output = 0
            for plant in selected:
                output += plant.output_cities
                if plant.is_ecological:
                    continue
                if plant.is_hybrid:
                    hybrid_usage += plant.resource_cost
                else:
                    fixed_usage[plant.resource_types[0]] += plant.resource_cost
            if any(
                fixed_usage[resource] > stored[resource]
                for resource in ("coal", "oil", "garbage", "uranium")
            ):
                continue
            fossil_remaining = (
                stored["coal"]
                - fixed_usage["coal"]
                + stored["oil"]
                - fixed_usage["oil"]
            )
            if hybrid_usage > fossil_remaining:
                continue
            best_output = max(best_output, output)
    return best_output


__all__ = ["STORAGE_BUCKETS", "resource_planning_metrics"]
