from __future__ import annotations

from abc import ABC, abstractmethod

from ..session_types import GameSnapshot, GuiIntent, SeatAgent, TurnRequest


class BaseAiController(SeatAgent, ABC):
    controller = "ai_heuristics"

    def log_message(
        self,
        snapshot: GameSnapshot,
        request: TurnRequest,
        *,
        message: str,
        payload: dict[str, object] | None = None,
        event_type: str = "ai_note",
        level: str = "info",
        include_state_snapshot: bool = False,
    ) -> None:
        if snapshot.analysis_log is None:
            return
        enriched_payload = dict(payload or {})
        enriched_payload.setdefault("controller", self.controller)
        snapshot.analysis_log.record(
            message=message,
            payload=enriched_payload,
            player_id=request.player_id,
            phase=request.phase,
            event_type=event_type,
            level=level,
            include_state_snapshot=include_state_snapshot,
        )

    def log_state(
        self,
        snapshot: GameSnapshot,
        request: TurnRequest,
        *,
        label: str,
        state: dict[str, object] | None = None,
        message: str | None = None,
        level: str = "debug",
        include_state_snapshot: bool = False,
    ) -> None:
        if snapshot.analysis_log is None:
            return
        snapshot.analysis_log.record_state(
            label=label,
            state=dict(state or {}),
            player_id=request.player_id,
            phase=request.phase,
            controller=self.controller,
            message=message,
            level=level,
            include_state_snapshot=include_state_snapshot,
        )

    @abstractmethod
    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        raise NotImplementedError
