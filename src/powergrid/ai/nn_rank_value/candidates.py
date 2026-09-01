from __future__ import annotations

from dataclasses import dataclass, field
import json

from ...model import (
    GameState,
    ModelValidationError,
    PlantRunPlan,
    choose_plants_to_run,
    compute_powered_cities,
    pay_income,
)
from ...session_types import GameSnapshot, GuiIntent, TurnRequest


@dataclass(frozen=True)
class CandidateAction:
    """One legal, atomic session intent plus public derived metadata."""

    intent: GuiIntent
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> str:
        return json.dumps(self.intent.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "intent": self.intent.to_dict(),
            "metadata": dict(self.metadata),
        }


def generate_candidate_actions(
    request: TurnRequest,
    snapshot: GameSnapshot,
) -> tuple[CandidateAction, ...]:
    """Generate the legal policy-candidate set used by NN rank-value v1.

    The v1 policy deliberately uses minimum-only raises and single-city builds.
    Resource quantities are expanded explicitly. Build and resource phases remain
    sequential because the session lets the same player act repeatedly before finishing.
    """

    state = snapshot.state
    if state.pending_decision is not None:
        candidates = _pending_candidates(request)
    elif request.phase == "auction":
        candidates = _auction_candidates(request, state)
    elif request.phase == "buy_resources":
        candidates = _resource_candidates(request)
    elif request.phase == "build_houses":
        candidates = _build_candidates(request)
    elif request.phase == "bureaucracy":
        candidates = _bureaucracy_candidates(state, request.player_id)
    else:
        raise ModelValidationError(f"unsupported NN candidate phase {request.phase!r}")
    result = _deduplicate_candidates(candidates)
    if not result:
        raise ModelValidationError(
            f"NN candidate generation produced no action for {request.decision_type!r}"
        )
    return result


def find_candidate_for_intent(
    candidates: tuple[CandidateAction, ...],
    intent: GuiIntent,
) -> CandidateAction | None:
    key = json.dumps(intent.to_dict(), sort_keys=True, separators=(",", ":"))
    return next((candidate for candidate in candidates if candidate.key == key), None)


def candidate_from_intent(intent: GuiIntent) -> CandidateAction:
    """Represent a valid teacher intent that is outside the v1 candidate expansion."""

    return CandidateAction(intent=intent, metadata={"teacher_only_candidate": True})


def _pending_candidates(request: TurnRequest) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    for action in request.legal_actions:
        if action.action_type == "discard_power_plant":
            plant_price = int(action.payload["price"])
            candidates.append(
                CandidateAction(
                    GuiIntent.discard_plant(request.player_id, plant_price),
                    {"plant_price": plant_price},
                )
            )
        elif action.action_type == "discard_hybrid_resources":
            coal = int(action.payload.get("coal", 0))
            oil = int(action.payload.get("oil", 0))
            candidates.append(
                CandidateAction(
                    GuiIntent.discard_hybrid_resources(request.player_id, coal=coal, oil=oil),
                    {"coal": coal, "oil": oil},
                )
            )
    return candidates


def _auction_candidates(request: TurnRequest, state: GameState) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    for action in request.legal_actions:
        if action.action_type == "auction_start":
            plant_price = int(action.payload["plant_price"])
            minimum_bid = int(action.payload["min_bid"])
            candidates.append(
                CandidateAction(
                    GuiIntent.auction_start(request.player_id, plant_price, minimum_bid),
                    {
                        "plant_price": plant_price,
                        "bid": minimum_bid,
                        "minimum_bid": minimum_bid,
                        "maximum_bid": int(action.payload["max_bid"]),
                    },
                )
            )
        elif action.action_type == "auction_bid":
            minimum_bid = int(action.payload["min_bid"])
            maximum_bid = int(action.payload["max_bid"])
            # The session request always exposes the bid shape while an auction is
            # active, even when the next minimum raise exceeds this player's cash.
            # In that case passing is the only legal candidate.
            if minimum_bid > maximum_bid:
                continue
            plant_price = int(action.payload["plant_price"])
            candidates.append(
                CandidateAction(
                    GuiIntent.auction_bid(request.player_id, minimum_bid),
                    {
                        "plant_price": plant_price,
                        "bid": minimum_bid,
                        "minimum_bid": minimum_bid,
                        "maximum_bid": maximum_bid,
                    },
                )
            )
        elif action.action_type == "auction_pass":
            active_price = (
                int(state.auction_state.active_plant_price)
                if state.auction_state is not None
                and state.auction_state.active_plant_price is not None
                else 0
            )
            candidates.append(
                CandidateAction(
                    GuiIntent.auction_pass(request.player_id),
                    {"plant_price": active_price},
                )
            )
    return candidates


def _resource_candidates(request: TurnRequest) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    for action in request.legal_actions:
        if action.action_type == "buy_resource":
            resource = str(action.payload["resource"])
            unit_prices = tuple(int(price) for price in action.payload["unit_prices"])
            maximum = int(action.payload["max_affordable_units"])
            for amount in range(1, maximum + 1):
                candidates.append(
                    CandidateAction(
                        GuiIntent.buy_resource(request.player_id, resource, amount),
                        {
                            "resource": resource,
                            "amount": amount,
                            "cost": sum(unit_prices[:amount]),
                            "unit_prices": list(unit_prices[:amount]),
                        },
                    )
                )
        elif action.action_type == "finish_buying":
            candidates.append(CandidateAction(GuiIntent.finish_buying(request.player_id)))
    return candidates


def _build_candidates(request: TurnRequest) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    for action in request.legal_actions:
        if action.action_type == "build_city":
            city_id = str(action.payload["city_id"])
            candidates.append(
                CandidateAction(
                    GuiIntent.commit_build(request.player_id, [city_id]),
                    {
                        "city_ids": [city_id],
                        "city_id": city_id,
                        "connection_cost": int(action.payload["connection_cost"]),
                        "build_cost": int(action.payload["build_cost"]),
                        "total_cost": int(action.payload["total_cost"]),
                    },
                )
            )
        elif action.action_type == "finish_building":
            candidates.append(CandidateAction(GuiIntent.finish_building(request.player_id)))
    return candidates


def _bureaucracy_candidates(state: GameState, player_id: str) -> list[CandidateAction]:
    summaries = _enumerate_generation_plans(state, player_id)
    candidates: list[CandidateAction] = []
    for plans in summaries:
        powered = compute_powered_cities(state, player_id, plans)
        resource_mix = {"coal": 0, "oil": 0, "garbage": 0, "uranium": 0}
        for plan in plans:
            for resource, amount in plan.resource_mix.items():
                resource_mix[resource] += int(amount)
        metadata = {
            "plans": [plan.to_dict() for plan in plans],
            "powered_cities": powered,
            "income": pay_income(state.rules, powered),
            "resource_mix": resource_mix,
        }
        if plans:
            candidates.append(
                CandidateAction(GuiIntent.run_plants(player_id, plans), metadata)
            )
        else:
            candidates.append(CandidateAction(GuiIntent.skip_bureaucracy(player_id), metadata))
    return candidates


def _enumerate_generation_plans(
    state: GameState,
    player_id: str,
) -> tuple[tuple[PlantRunPlan, ...], ...]:
    player = next(player for player in state.players if player.player_id == player_id)
    totals = player.resource_storage.resource_totals()
    plant_choices: list[tuple[PlantRunPlan | None, ...]] = []
    for plant in sorted(player.power_plants, key=lambda item: item.price):
        if plant.is_step_3_placeholder:
            continue
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
        plant_choices.append(tuple(options))

    valid: list[tuple[PlantRunPlan, ...]] = []

    def backtrack(
        index: int,
        remaining: dict[str, int],
        selected: list[PlantRunPlan],
    ) -> None:
        if index >= len(plant_choices):
            plans = tuple(selected)
            try:
                valid.append(choose_plants_to_run(state, player_id, plans))
            except ModelValidationError:
                return
            return
        for option in plant_choices[index]:
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
    unique: dict[str, tuple[PlantRunPlan, ...]] = {}
    for plans in valid:
        key = json.dumps([plan.to_dict() for plan in plans], sort_keys=True)
        unique[key] = plans
    return tuple(unique[key] for key in sorted(unique))


def _deduplicate_candidates(candidates: list[CandidateAction]) -> tuple[CandidateAction, ...]:
    unique: dict[str, CandidateAction] = {}
    for candidate in candidates:
        unique.setdefault(candidate.key, candidate)
    if not unique:
        raise ModelValidationError("NN candidate generator produced no legal actions")
    return tuple(unique.values())


__all__ = [
    "CandidateAction",
    "candidate_from_intent",
    "find_candidate_for_intent",
    "generate_candidate_actions",
]
