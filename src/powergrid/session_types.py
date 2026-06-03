from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_type": self.intent_type,
            "player_id": self.player_id,
            "payload": dict(self.payload),
        }

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "phase": self.phase,
            "decision_type": self.decision_type,
            "prompt": self.prompt,
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SessionEvent:
    level: str
    message: str
    player_id: str | None = None
    phase: str | None = None
    event_type: str = "session_event"
    round_number: int | None = None
    step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "player_id": self.player_id,
            "phase": self.phase,
            "event_type": self.event_type,
            "round_number": self.round_number,
            "step": self.step,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class GameLogEntry:
    index: int
    source: str
    event_type: str
    level: str
    message: str
    player_id: str | None = None
    phase: str | None = None
    round_number: int | None = None
    step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    state_snapshot: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(
            self,
            "state_snapshot",
            dict(self.state_snapshot) if self.state_snapshot is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source": self.source,
            "event_type": self.event_type,
            "level": self.level,
            "message": self.message,
            "player_id": self.player_id,
            "phase": self.phase,
            "round_number": self.round_number,
            "step": self.step,
            "payload": dict(self.payload),
            "state_snapshot": dict(self.state_snapshot) if self.state_snapshot is not None else None,
        }


class AnalysisLogWriter:
    def __init__(
        self,
        emit_fn: Callable[..., None] | None = None,
    ) -> None:
        self._emit_fn = emit_fn

    def record(
        self,
        *,
        message: str,
        payload: dict[str, Any] | None = None,
        player_id: str | None = None,
        phase: str | None = None,
        event_type: str = "analysis",
        level: str = "info",
        include_state_snapshot: bool = False,
    ) -> None:
        if self._emit_fn is None:
            return
        self._emit_fn(
            source="ai",
            event_type=event_type,
            level=level,
            message=message,
            player_id=player_id,
            phase=phase,
            payload=dict(payload or {}),
            include_state_snapshot=include_state_snapshot,
        )

    def record_state(
        self,
        *,
        label: str,
        state: dict[str, Any] | None = None,
        player_id: str | None = None,
        phase: str | None = None,
        controller: str | None = None,
        message: str | None = None,
        level: str = "debug",
        include_state_snapshot: bool = False,
    ) -> None:
        payload = {"label": label, "state": dict(state or {})}
        if controller is not None:
            payload["controller"] = controller
        self.record(
            message=message or f"AI state: {label}",
            payload=payload,
            player_id=player_id,
            phase=phase,
            event_type="ai_state",
            level=level,
            include_state_snapshot=include_state_snapshot,
        )


@dataclass(frozen=True)
class GameSnapshot:
    state: GameState
    active_request: TurnRequest | None
    event_log: tuple[SessionEvent, ...]
    last_round_summary: Any | None = None
    winner_result: WinnerResult | None = None
    analysis_log: AnalysisLogWriter | None = None


class SeatAgent:
    controller = "human"

    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        raise NotImplementedError


class HumanSeat(SeatAgent):
    controller = "human"

    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        raise ModelValidationError("human seats require a user-provided intent")
