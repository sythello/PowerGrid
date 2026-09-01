from __future__ import annotations

from ..model import ModelValidationError
from .base import BaseAiController
from .deterministic import DeterministicAiController
from .evaluation import (
    AiControllerPairSummary,
    AiControllerRatingSummary,
    AiEvaluationBucketConfig,
    AiEvaluationGameSummary,
    AiEvaluationReport,
    AiEvaluationStanding,
    build_default_evaluation_lineups,
    derive_final_standings,
    evaluate_ai_bucket,
    select_evaluation_regions,
)
from .nn_rank_value.controller import NnRankValueAiController
from .nn_rl_based.controller import NnRlBasedAiController
from .profiled_deterministic import (
    EfficiencyDeterministicAiController,
    ExpansionDeterministicAiController,
    ReserveDeterministicAiController,
)
from .strategic import StrategicAiController


DeterministicAiSeat = DeterministicAiController

AI_CONTROLLER_REGISTRY: dict[str, type[BaseAiController]] = {
    "ai_nn_rl_based_v1": NnRlBasedAiController,
    "ai_nn_rank_value_v1": NnRankValueAiController,
    "ai_heuristics": StrategicAiController,
    "ai_deterministic_efficiency": EfficiencyDeterministicAiController,
    "ai_deterministic_expansion": ExpansionDeterministicAiController,
    "ai_deterministic_reserve": ReserveDeterministicAiController,
    "ai_deterministic": DeterministicAiController,
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
    "AiControllerPairSummary",
    "AiControllerRatingSummary",
    "AiEvaluationBucketConfig",
    "AiEvaluationGameSummary",
    "AiEvaluationReport",
    "AiEvaluationStanding",
    "BaseAiController",
    "build_default_evaluation_lineups",
    "DeterministicAiController",
    "DeterministicAiSeat",
    "EfficiencyDeterministicAiController",
    "ExpansionDeterministicAiController",
    "NnRankValueAiController",
    "NnRlBasedAiController",
    "ReserveDeterministicAiController",
    "derive_final_standings",
    "evaluate_ai_bucket",
    "select_evaluation_regions",
    "StrategicAiController",
    "build_ai_controller",
    "register_ai_controller",
]
