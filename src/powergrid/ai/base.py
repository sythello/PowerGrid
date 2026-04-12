from __future__ import annotations

from abc import ABC, abstractmethod

from ..session_types import GameSnapshot, GuiIntent, SeatAgent, TurnRequest


class BaseAiController(SeatAgent, ABC):
    controller = "ai"

    @abstractmethod
    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        raise NotImplementedError
