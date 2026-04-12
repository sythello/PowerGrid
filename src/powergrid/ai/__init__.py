from __future__ import annotations

from ..model import ModelValidationError
from .base import BaseAiController
from .deterministic import DeterministicAiController


DeterministicAiSeat = DeterministicAiController

AI_CONTROLLER_REGISTRY: dict[str, type[BaseAiController]] = {
    "ai": DeterministicAiController,
}


def register_ai_controller(controller_name: str, controller_class: type[BaseAiController]) -> None:
    if not controller_name:
        raise ModelValidationError("AI controller name must be non-empty")
    if not issubclass(controller_class, BaseAiController):
        raise ModelValidationError("registered AI controllers must inherit from BaseAiController")
    AI_CONTROLLER_REGISTRY[controller_name] = controller_class


def build_ai_controller(controller_name: str) -> BaseAiController:
    controller_class = AI_CONTROLLER_REGISTRY.get(controller_name)
    if controller_class is None:
        raise ModelValidationError(f"unsupported AI controller {controller_name!r}")
    return controller_class()


__all__ = [
    "AI_CONTROLLER_REGISTRY",
    "BaseAiController",
    "DeterministicAiController",
    "DeterministicAiSeat",
    "build_ai_controller",
    "register_ai_controller",
]
