from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import Action, GameState, ModelValidationError, PlantRunPlan, WinnerResult


@dataclass(frozen=True)
class GuiIntent:
    intent_type: str
    player_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.intent_type:
            raise ModelValidationError("intent_type must be non-empty")
        if not self.player_id:
            raise ModelValidationError("player_id must be non-empty")
        object.__setattr__(self, "payload", dict(self.payload))

    @classmethod
    def auction_start(cls, player_id: str, plant_price: int, bid: int) -> "GuiIntent":
        return cls(
            intent_type="auction_start",
            player_id=player_id,
            payload={"plant_price": int(plant_price), "bid": int(bid)},
        )

    @classmethod
    def auction_bid(cls, player_id: str, bid: int) -> "GuiIntent":
        return cls(intent_type="auction_bid", player_id=player_id, payload={"bid": int(bid)})

    @classmethod
    def auction_pass(cls, player_id: str) -> "GuiIntent":
        return cls(intent_type="auction_pass", player_id=player_id)

    @classmethod
    def buy_resource(cls, player_id: str, resource: str, amount: int) -> "GuiIntent":
        return cls(
            intent_type="buy_resource",
            player_id=player_id,
            payload={"resource": resource, "amount": int(amount)},
        )

    @classmethod
    def finish_buying(cls, player_id: str) -> "GuiIntent":
        return cls(intent_type="finish_buying", player_id=player_id)

    @classmethod
    def quote_build(cls, player_id: str, city_ids: tuple[str, ...] | list[str]) -> "GuiIntent":
        return cls(intent_type="quote_build", player_id=player_id, payload={"city_ids": list(city_ids)})

    @classmethod
    def commit_build(cls, player_id: str, city_ids: tuple[str, ...] | list[str]) -> "GuiIntent":
        return cls(intent_type="commit_build", player_id=player_id, payload={"city_ids": list(city_ids)})

    @classmethod
    def finish_building(cls, player_id: str) -> "GuiIntent":
        return cls(intent_type="finish_building", player_id=player_id)

    @classmethod
    def run_plants(
        cls,
        player_id: str,
        plans: tuple[PlantRunPlan, ...] | list[PlantRunPlan],
    ) -> "GuiIntent":
        return cls(
            intent_type="run_plants",
            player_id=player_id,
            payload={"plans": [plan.to_dict() for plan in plans]},
        )

    @classmethod
    def skip_bureaucracy(cls, player_id: str) -> "GuiIntent":
        return cls(intent_type="skip_bureaucracy", player_id=player_id)

    @classmethod
    def discard_plant(cls, player_id: str, plant_price: int) -> "GuiIntent":
        return cls(
            intent_type="discard_power_plant",
            player_id=player_id,
            payload={"plant_price": int(plant_price)},
        )

    @classmethod
    def discard_hybrid_resources(cls, player_id: str, coal: int, oil: int) -> "GuiIntent":
        return cls(
            intent_type="discard_hybrid_resources",
            player_id=player_id,
            payload={"coal": int(coal), "oil": int(oil)},
        )


@dataclass(frozen=True)
class TurnRequest:
    player_id: str
    phase: str
    decision_type: str
    prompt: str
    legal_actions: tuple[Action, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.player_id:
            raise ModelValidationError("turn request player_id must be non-empty")
        if not self.phase:
            raise ModelValidationError("turn request phase must be non-empty")
        if not self.decision_type:
            raise ModelValidationError("turn request decision_type must be non-empty")
        if not self.prompt:
            raise ModelValidationError("turn request prompt must be non-empty")
        object.__setattr__(self, "legal_actions", tuple(self.legal_actions))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SessionEvent:
    level: str
    message: str
    player_id: str | None = None
    phase: str | None = None


@dataclass(frozen=True)
class GameSnapshot:
    state: GameState
    active_request: TurnRequest | None
    event_log: tuple[SessionEvent, ...]
    last_round_summary: Any | None = None
    winner_result: WinnerResult | None = None


class SeatAgent:
    controller = "human"

    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        raise NotImplementedError


class HumanSeat(SeatAgent):
    controller = "human"

    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        raise ModelValidationError("human seats require a user-provided intent")
