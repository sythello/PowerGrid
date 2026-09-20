from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

import numpy as np

from ...model import ModelValidationError
from ...session import GameSession
from ...session_types import GameSnapshot, GuiIntent, TurnRequest
from .. import build_ai_controller, derive_final_standings
from ..base import BaseAiController
from ..nn_rank_value.candidates import (
    CandidateAction,
    find_candidate_for_intent,
    generate_candidate_actions,
)
from ..nn_rank_value.observation import (
    build_public_observation,
    encode_action_features,
    encode_state_features,
    player_slot_ids,
)
from .model import MAX_PLAYERS, NumpyRlPolicyQNetwork


@dataclass(frozen=True)
class SearchConfig:
    depth: int = 1
    adaptive_depth_2: bool = True
    max_search_nodes: int = 512
    max_boundary_actions: int = 128
    continuation_controller: str = "ai_deterministic"
    leaf_policy: str = "deterministic"
    search_policy_mix: float = 0.5
    search_temperature: float = 0.25

    def __post_init__(self) -> None:
        if self.depth != 1:
            raise ValueError("ai_nn_rl_based_v1 requires base search depth 1")
        if self.max_search_nodes <= 0 or self.max_boundary_actions <= 0:
            raise ValueError("search node/action limits must be positive")
        if self.leaf_policy not in {"deterministic", "checkpoint"}:
            raise ValueError("leaf_policy must be deterministic or checkpoint")
        if not 0.0 <= self.search_policy_mix <= 1.0:
            raise ValueError("search_policy_mix must be between 0 and 1")
        if self.search_temperature <= 0:
            raise ValueError("search_temperature must be positive")


@dataclass(frozen=True)
class SearchResult:
    q_values: tuple[tuple[float, ...], ...]
    player_ids: tuple[str, ...]
    depth_used: int
    nodes_evaluated: int
    depth_2_completed: bool
    policy_advantages: tuple[float, ...] = ()
    policy_confirmed: tuple[bool, ...] = ()


class _SearchBudgetExceeded(RuntimeError):
    pass


@dataclass
class _NodeBudget:
    maximum: int | None
    used: int = 0

    def consume(self) -> None:
        self.used += 1
        if self.maximum is not None and self.used > self.maximum:
            raise _SearchBudgetExceeded


class FullActionSemanticSearcher:
    """Full-width expectimax-style search over semantic Power Grid turns."""

    def __init__(self, model: NumpyRlPolicyQNetwork, config: SearchConfig) -> None:
        self.model = model
        self.config = config
        self._continuation_agents: dict[str, BaseAiController] = {}

    def search(self, session: GameSession) -> SearchResult:
        snapshot = session.snapshot()
        request = snapshot.active_request
        if request is None or snapshot.winner_result is not None:
            raise ModelValidationError("search root requires a non-terminal active request")
        observation = build_public_observation(snapshot.state, request)
        root_player_ids = player_slot_ids(observation)
        candidates = generate_candidate_actions(request, snapshot)

        depth_one_budget = _NodeBudget(maximum=None)
        depth_one = self._action_values(
            session,
            candidates,
            root_player_ids,
            remaining_depth=1,
            budget=depth_one_budget,
        )
        total_nodes = depth_one_budget.used
        if not self.config.adaptive_depth_2:
            return _result(depth_one, root_player_ids, 1, total_nodes, False)

        depth_two_budget = _NodeBudget(maximum=self.config.max_search_nodes)
        try:
            depth_two = self._action_values(
                session,
                candidates,
                root_player_ids,
                remaining_depth=2,
                budget=depth_two_budget,
            )
        except _SearchBudgetExceeded:
            return _result(
                depth_one,
                root_player_ids,
                1,
                total_nodes + depth_two_budget.used,
                False,
            )
        return _result(
            depth_two,
            root_player_ids,
            2,
            total_nodes + depth_two_budget.used,
            True,
        )

    def _action_values(
        self,
        session: GameSession,
        candidates: tuple[CandidateAction, ...],
        parent_player_ids: tuple[str, ...],
        *,
        remaining_depth: int,
        budget: _NodeBudget,
    ) -> np.ndarray:
        rows: list[list[float]] = []
        for candidate in candidates:
            budget.consume()
            child = advance_to_semantic_boundary(
                session,
                candidate,
                continuation_agents=self._agents_for(session.snapshot()),
                max_actions=self.config.max_boundary_actions,
            )
            child_values = self._state_value(
                child,
                remaining_depth=remaining_depth - 1,
                budget=budget,
            )
            rows.append([float(child_values[player_id]) for player_id in parent_player_ids])
        return np.asarray(rows, dtype=np.float32)

    def _state_value(
        self,
        session: GameSession,
        *,
        remaining_depth: int,
        budget: _NodeBudget,
    ) -> dict[str, float]:
        snapshot = session.snapshot()
        if snapshot.winner_result is not None:
            return terminal_rank_values(snapshot)
        request = snapshot.active_request
        if request is None:
            raise ModelValidationError("semantic search reached a non-terminal state without request")
        observation = build_public_observation(snapshot.state, request)
        state_features, state_names = encode_state_features(observation)
        slot_ids = player_slot_ids(observation)
        candidates = generate_candidate_actions(request, snapshot)
        if remaining_depth <= 0:
            action_rows, action_names = _encode_actions(observation, candidates)
            _validate_model_schema(self.model, state_names, action_names)
            predictions = self.model.predict_one(
                np.asarray(state_features, dtype=np.float32),
                np.asarray(action_rows, dtype=np.float32),
            )
            if self.config.leaf_policy == "deterministic":
                teacher_index = self._teacher_index(request, snapshot, candidates)
                probabilities = np.zeros(len(candidates), dtype=np.float32)
                probabilities[teacher_index] = 1.0
            else:
                probabilities = predictions.policy_probabilities
            values = probabilities @ predictions.q_values
            return {player_id: float(values[index]) for index, player_id in enumerate(slot_ids)}

        action_values = self._action_values(
            session,
            candidates,
            slot_ids,
            remaining_depth=remaining_depth,
            budget=budget,
        )
        teacher_index = self._teacher_index(request, snapshot, candidates)
        probabilities = _improved_policy(
            action_values[:, 0],
            teacher_index,
            mix=self.config.search_policy_mix,
            temperature=self.config.search_temperature,
        )
        values = probabilities @ action_values
        return {player_id: float(values[index]) for index, player_id in enumerate(slot_ids)}

    def _teacher_index(
        self,
        request: TurnRequest,
        snapshot: GameSnapshot,
        candidates: tuple[CandidateAction, ...],
    ) -> int:
        agent = self._agents_for(snapshot)[request.player_id]
        chosen = agent.choose_intent(request, snapshot)
        candidate = find_candidate_for_intent(candidates, chosen)
        if candidate is None:
            raise ModelValidationError(
                "deterministic teacher action is outside the runtime candidate set"
            )
        return next(index for index, item in enumerate(candidates) if item.key == candidate.key)

    def _agents_for(
        self, snapshot: GameSnapshot
    ) -> Mapping[str, BaseAiController]:
        for player in snapshot.state.players:
            self._continuation_agents.setdefault(
                player.player_id,
                build_ai_controller(self.config.continuation_controller),
            )
        return self._continuation_agents


class FullActionPairedMcEvaluator:
    """Evaluate every legal root action with a shared frozen continuation policy."""

    def __init__(
        self,
        continuation_agents: Mapping[str, BaseAiController],
        *,
        max_actions: int = 5000,
        confirmation_rollouts: int = 0,
        confirmation_confidence_z: float = 1.645,
    ) -> None:
        if max_actions <= 0:
            raise ValueError("paired-MC max_actions must be positive")
        if confirmation_rollouts < 0:
            raise ValueError("paired-MC confirmation_rollouts may not be negative")
        if confirmation_rollouts == 1:
            raise ValueError("paired-MC confirmation_rollouts must be zero or at least two")
        if confirmation_confidence_z < 0.0:
            raise ValueError("paired-MC confirmation_confidence_z may not be negative")
        self.continuation_agents = continuation_agents
        self.max_actions = max_actions
        self.confirmation_rollouts = confirmation_rollouts
        self.confirmation_confidence_z = confirmation_confidence_z

    def search(
        self,
        session: GameSession,
        *,
        teacher_action_index: int | None = None,
        confirmation_seed: int = 0,
    ) -> SearchResult:
        snapshot = session.snapshot()
        request = snapshot.active_request
        if request is None or snapshot.winner_result is not None:
            raise ModelValidationError(
                "paired-MC root requires a non-terminal active request"
            )
        observation = build_public_observation(snapshot.state, request)
        player_ids = player_slot_ids(observation)
        candidates = generate_candidate_actions(request, snapshot)
        rows: list[list[float]] = []
        for candidate in candidates:
            values = rollout_terminal_values(
                session,
                candidate.intent,
                self.continuation_agents,
                max_actions=self.max_actions,
            )
            rows.append([values[player_id] for player_id in player_ids])
        policy_advantages: tuple[float, ...] = ()
        policy_confirmed: tuple[bool, ...] = ()
        nodes = len(candidates)
        if self.confirmation_rollouts:
            if (
                teacher_action_index is None
                or not 0 <= teacher_action_index < len(candidates)
            ):
                raise ValueError(
                    "paired-MC hidden-state confirmation requires a valid teacher action index"
                )
            base_actor_values = np.asarray(rows, dtype=np.float32)[:, 0]
            teacher_base = float(base_actor_values[teacher_action_index])
            screened = [
                index
                for index, value in enumerate(base_actor_values)
                if index != teacher_action_index and float(value) > teacher_base
            ]
            # Unscreened actions remain zero with a false confirmation mask. This
            # keeps JSONL examples standards-compliant without making them eligible.
            advantages = np.zeros(len(candidates), dtype=np.float32)
            confirmed = np.zeros(len(candidates), dtype=bool)
            advantages[teacher_action_index] = 0.0
            samples_by_candidate = {
                candidate_index: np.empty(
                    self.confirmation_rollouts, dtype=np.float32
                )
                for candidate_index in screened
            }
            if screened:
                for rollout_index in range(self.confirmation_rollouts):
                    resampled = _resample_hidden_plant_order(
                        session,
                        seed=_derived_confirmation_seed(
                            confirmation_seed, rollout_index
                        ),
                    )
                    teacher_values = rollout_terminal_values(
                        resampled,
                        candidates[teacher_action_index].intent,
                        self.continuation_agents,
                        max_actions=self.max_actions,
                    )
                    teacher_actor_value = float(teacher_values[player_ids[0]])
                    for candidate_index in screened:
                        candidate_values = rollout_terminal_values(
                            resampled,
                            candidates[candidate_index].intent,
                            self.continuation_agents,
                            max_actions=self.max_actions,
                        )
                        samples_by_candidate[candidate_index][rollout_index] = (
                            float(candidate_values[player_ids[0]])
                            - teacher_actor_value
                        )
                nodes += self.confirmation_rollouts * (1 + len(screened))
            for candidate_index in screened:
                # The original rollout is deliberately excluded: it selected the
                # candidates and would bias their confirmation estimate upward.
                samples = samples_by_candidate[candidate_index]
                mean_advantage = float(np.mean(samples))
                standard_error = float(np.std(samples, ddof=1)) / np.sqrt(
                    len(samples)
                )
                lower_bound = (
                    mean_advantage
                    - self.confirmation_confidence_z * standard_error
                )
                advantages[candidate_index] = mean_advantage
                confirmed[candidate_index] = lower_bound > 0.0
            policy_advantages = tuple(float(value) for value in advantages)
            policy_confirmed = tuple(bool(value) for value in confirmed)
        return _result(
            np.asarray(rows, dtype=np.float32),
            player_ids,
            1,
            nodes,
            False,
            policy_advantages=policy_advantages,
            policy_confirmed=policy_confirmed,
        )


def _derived_confirmation_seed(seed: int, rollout_index: int) -> int:
    return (int(seed) * 1_000_003 + int(rollout_index) * 97_409) & ((1 << 63) - 1)


def _resample_hidden_plant_order(session: GameSession, *, seed: int) -> GameSession:
    """Resample only plant information omitted from the policy observation."""

    resampled = session.fork()
    rng = random.Random(seed)
    draw_stack = list(resampled._state.power_plant_draw_stack)
    bottom_stack = list(resampled._state.power_plant_bottom_stack)
    rng.shuffle(draw_stack)
    rng.shuffle(bottom_stack)
    resampled._state.power_plant_draw_stack = tuple(draw_stack)
    resampled._state.power_plant_bottom_stack = tuple(bottom_stack)
    return resampled


def rollout_terminal_values(
    session: GameSession,
    first_intent: GuiIntent,
    continuation_agents: Mapping[str, BaseAiController],
    *,
    max_actions: int = 5000,
) -> dict[str, float]:
    """Fork a root, apply one action, then use a frozen policy through terminal."""

    if max_actions <= 0:
        raise ValueError("terminal rollout max_actions must be positive")
    rollout = session.fork()
    result = rollout.submit_intent(first_intent, auto_advance=False)
    _raise_last_error(result, "paired-MC root action")
    actions = 1
    while True:
        snapshot = rollout.advance_until_blocked()
        if snapshot.winner_result is not None:
            return terminal_rank_values(snapshot)
        if actions >= max_actions:
            break
        request = snapshot.active_request
        if request is None:
            raise ModelValidationError(
                "paired-MC rollout stopped without an active request"
            )
        agent = continuation_agents.get(request.player_id)
        if agent is None:
            raise ModelValidationError(
                f"paired-MC continuation policy is missing player {request.player_id!r}"
            )
        intent = agent.choose_intent(request, snapshot)
        result = rollout.submit_intent(intent, auto_advance=False)
        _raise_last_error(result, "paired-MC continuation policy")
        actions += 1
    raise ModelValidationError(
        f"paired-MC rollout exceeded {max_actions} actions"
    )


def advance_to_semantic_boundary(
    session: GameSession,
    candidate: CandidateAction,
    *,
    continuation_agents: Mapping[str, BaseAiController],
    max_actions: int = 128,
) -> GameSession:
    """Apply one candidate and continue to the request's natural semantic boundary."""

    root_snapshot = session.snapshot()
    root_request = root_snapshot.active_request
    if root_request is None:
        raise ModelValidationError("semantic rollout requires an active root request")
    root_pending = root_snapshot.state.pending_decision is not None
    root_auction_active = bool(
        root_snapshot.state.auction_state is not None
        and root_snapshot.state.auction_state.active_plant_price is not None
    )
    auction_resolving = root_auction_active or candidate.intent.intent_type == "auction_start"
    rollout = session.fork()
    result = rollout.submit_intent(candidate.intent, auto_advance=False)
    _raise_last_error(result, "root search candidate")
    actions = 1
    while not _semantic_boundary_reached(
        rollout.snapshot(),
        root_request=root_request,
        root_pending=root_pending,
        auction_resolving=auction_resolving,
    ):
        if actions >= max_actions:
            raise ModelValidationError(
                f"semantic rollout exceeded {max_actions} continuation actions"
            )
        snapshot = rollout.advance_until_blocked()
        if snapshot.winner_result is not None:
            break
        request = snapshot.active_request
        if request is None:
            raise ModelValidationError("semantic rollout stopped without an active request")
        agent = continuation_agents[request.player_id]
        intent = agent.choose_intent(request, snapshot)
        result = rollout.submit_intent(intent, auto_advance=False)
        _raise_last_error(result, "continuation controller")
        actions += 1
    return rollout


def terminal_rank_values(snapshot: GameSnapshot) -> dict[str, float]:
    if snapshot.winner_result is None:
        raise ModelValidationError("terminal rank values require a completed game")
    standings = derive_final_standings(snapshot.state, snapshot.winner_result)
    player_count = len(standings)
    return {
        standing.player_id: float(
            (player_count + 1 - (2 * standing.place)) / (player_count - 1)
        )
        for standing in standings
    }


def pad_player_values(
    player_ids: tuple[str, ...], values: Mapping[str, float]
) -> tuple[list[float], list[bool]]:
    if len(player_ids) > MAX_PLAYERS:
        raise ModelValidationError("player vector exceeds model capacity")
    padded = [float(values[player_id]) for player_id in player_ids]
    mask = [True] * len(padded)
    padded.extend([0.0] * (MAX_PLAYERS - len(padded)))
    mask.extend([False] * (MAX_PLAYERS - len(mask)))
    return padded, mask


def _semantic_boundary_reached(
    snapshot: GameSnapshot,
    *,
    root_request: TurnRequest,
    root_pending: bool,
    auction_resolving: bool,
) -> bool:
    if snapshot.winner_result is not None:
        return True
    request = snapshot.active_request
    if request is None:
        return False
    state = snapshot.state
    if root_pending:
        return state.pending_decision is None
    if root_request.phase == "auction":
        if state.pending_decision is not None:
            return False
        if state.phase != "auction":
            return True
        auction = state.auction_state
        active = bool(auction is not None and auction.active_plant_price is not None)
        if auction_resolving:
            return not active
        return request.player_id != root_request.player_id
    if root_request.phase == "buy_resources":
        if state.pending_decision is not None:
            return False
        if (
            state.phase != root_request.phase
            or request.player_id != root_request.player_id
        ):
            return True
        return request.metadata.get("resource") != root_request.metadata.get("resource")
    if root_request.phase == "build_houses":
        if state.pending_decision is not None:
            return False
        return state.phase != root_request.phase or request.player_id != root_request.player_id
    if root_request.phase == "bureaucracy":
        return state.phase != "bureaucracy"
    raise ModelValidationError(f"unsupported semantic search phase {root_request.phase!r}")


def _encode_actions(
    observation: object, candidates: tuple[CandidateAction, ...]
) -> tuple[list[list[float]], tuple[str, ...]]:
    rows: list[list[float]] = []
    names: tuple[str, ...] | None = None
    for candidate in candidates:
        features, current_names = encode_action_features(observation, candidate)  # type: ignore[arg-type]
        rows.append(features)
        names = names or current_names
        if current_names != names:
            raise ModelValidationError("action feature schema changed within search node")
    return rows, tuple(names or ())


def _validate_model_schema(
    model: NumpyRlPolicyQNetwork,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> None:
    if model.state_dim != len(state_names) or model.state_feature_names != state_names:
        raise ModelValidationError("target checkpoint state feature schema mismatch")
    if model.action_dim != len(action_names) or model.action_feature_names != action_names:
        raise ModelValidationError("target checkpoint action feature schema mismatch")


def _improved_policy(
    actor_q: np.ndarray, teacher_index: int, *, mix: float, temperature: float
) -> np.ndarray:
    shifted = actor_q / temperature
    shifted -= np.max(shifted)
    soft = np.exp(np.clip(shifted, -30.0, 30.0))
    soft /= np.sum(soft)
    result = mix * soft
    result[teacher_index] += 1.0 - mix
    return result.astype(np.float32)


def _raise_last_error(snapshot: GameSnapshot, source: str) -> None:
    if snapshot.event_log and snapshot.event_log[-1].level == "error":
        raise ModelValidationError(f"{source} produced invalid intent: {snapshot.event_log[-1].message}")


def _result(
    q_values: np.ndarray,
    player_ids: tuple[str, ...],
    depth: int,
    nodes: int,
    completed: bool,
    *,
    policy_advantages: tuple[float, ...] = (),
    policy_confirmed: tuple[bool, ...] = (),
) -> SearchResult:
    padded_rows = []
    for row in q_values:
        values = list(float(value) for value in row)
        values.extend([0.0] * (MAX_PLAYERS - len(values)))
        padded_rows.append(tuple(values))
    return SearchResult(
        q_values=tuple(padded_rows),
        player_ids=player_ids,
        depth_used=depth,
        nodes_evaluated=nodes,
        depth_2_completed=completed,
        policy_advantages=policy_advantages,
        policy_confirmed=policy_confirmed,
    )


__all__ = [
    "FullActionPairedMcEvaluator",
    "FullActionSemanticSearcher",
    "SearchConfig",
    "SearchResult",
    "advance_to_semantic_boundary",
    "pad_player_values",
    "rollout_terminal_values",
    "terminal_rank_values",
]
