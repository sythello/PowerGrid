"""Neural rank-value AI components."""

from .candidates import CandidateAction, generate_candidate_actions
from .controller import NnRankValueAiController
from .observation import (
    ACTION_FEATURE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PublicObservation,
    build_public_observation,
    encode_action_features,
    encode_state_features,
)


def __getattr__(name: str):
    if name == "NumpyRankValueNetwork":
        from .model import NumpyRankValueNetwork

        return NumpyRankValueNetwork
    raise AttributeError(name)

__all__ = [
    "ACTION_FEATURE_SCHEMA_VERSION",
    "CandidateAction",
    "NnRankValueAiController",
    "NumpyRankValueNetwork",
    "OBSERVATION_SCHEMA_VERSION",
    "PublicObservation",
    "build_public_observation",
    "encode_action_features",
    "encode_state_features",
    "generate_candidate_actions",
]
