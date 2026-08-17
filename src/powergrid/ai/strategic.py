from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from statistics import mean
from typing import Any

from ..model import (
    GameState,
    ModelValidationError,
    PlantRunPlan,
    PlayerState,
    PowerPlantCard,
    add_power_plant_to_player,
    apply_builds,
    can_store_resources,
    choose_plants_to_run,
    compute_all_targets_connection_cost,
    compute_powered_cities,
    consume_resources,
    discard_resources_to_fit_storage,
    legal_build_targets,
    legal_resource_purchases,
    list_auctionable_plants,
    pay_income,
    purchase_resources,
    replace_plant_if_needed,
)
from ..session_types import GameSnapshot, GuiIntent, TurnRequest
from .base import BaseAiController


RESOURCE_TYPES = ("coal", "oil", "garbage", "uranium")
MAX_LOGGED_CANDIDATE_ACTIONS = 24
MAX_OPPONENT_THREAT_CACHE_ENTRIES = 50000
_OPPONENT_THREAT_CACHE: dict[tuple[object, ...], "_OpponentThreat"] = {}

PLANT_STATIC_ADJUSTMENTS = {
    5: 1.5,
    6: -1.5,
    7: -2.5,
    10: 1.0,
    11: 1.5,
    13: 0.5,
    15: 2.5,
    17: -0.5,
    18: 0.5,
    20: 3.0,
    21: 1.5,
    23: 1.0,
    25: 3.0,
    27: 1.0,
    29: 3.5,
    31: 1.5,
    34: 1.0,
    35: 1.0,
    37: 1.0,
    42: 0.5,
    44: 3.0,
    46: 1.5,
    50: 3.0,
}


@dataclass(frozen=True)
class _StageWeights:
    connected: float
    powered: float
    income: float
    cash: float
    plants: float
    frontier: float
    resources: float
    order: float
    exposure: float
    overbuild: float
    unused_capacity: float


STAGE_WEIGHTS = {
    "opening": _StageWeights(
        connected=5.0,
        powered=6.0,
        income=2.0,
        cash=0.45,
        plants=3.8,
        frontier=2.2,
        resources=1.0,
        order=3.0,
        exposure=1.8,
        overbuild=4.0,
        unused_capacity=1.2,
    ),
    "step1": _StageWeights(
        connected=6.0,
        powered=6.8,
        income=2.4,
        cash=0.55,
        plants=3.2,
        frontier=2.3,
        resources=1.1,
        order=2.8,
        exposure=2.1,
        overbuild=5.2,
        unused_capacity=1.0,
    ),
    "pre_step2": _StageWeights(
        connected=6.4,
        powered=7.4,
        income=2.8,
        cash=0.6,
        plants=3.3,
        frontier=2.0,
        resources=1.3,
        order=2.4,
        exposure=2.2,
        overbuild=6.2,
        unused_capacity=0.8,
    ),
    "step2": _StageWeights(
        connected=6.2,
        powered=8.2,
        income=3.2,
        cash=0.65,
        plants=3.5,
        frontier=1.6,
        resources=1.4,
        order=2.0,
        exposure=2.4,
        overbuild=4.8,
        unused_capacity=0.7,
    ),
    "sprint": _StageWeights(
        connected=4.0,
        powered=10.5,
        income=4.0,
        cash=0.75,
        plants=3.0,
        frontier=0.9,
        resources=1.0,
        order=1.2,
        exposure=1.6,
        overbuild=2.8,
        unused_capacity=0.4,
    ),
}


@dataclass(frozen=True)
class _GenerationSummary:
    plans: tuple[PlantRunPlan, ...]
    powered: int
    income: int
    spent_units: int
    spent_value: float


@dataclass(frozen=True)
class _BuildPlan:
    city_ids: tuple[str, ...]
    total_cost: int
    score: float


@dataclass(frozen=True)
class _AuctionStartOption:
    action_type: str
    payload: dict[str, int]


@dataclass(frozen=True)
class _OpponentThreat:
    player_id: str
    current_strength: float
    refuel_projected_strength: float
    threat_strength: float
    projected_generation: _GenerationSummary
    refuel_cost: int
    refuel_basket: dict[str, int]
    threat_applied: bool
    refuel_projection_kind: str
    evaluated_refuel_baskets: int


@dataclass(frozen=True)
class _AuctionEconomyProjection:
    score: float
    raw_score: float
    viability_adjustment: float
    evaluation_state: GameState
    plant_price: int | None
    purchase_price: int
    resource_plan: tuple[tuple[str, int], ...]
    resource_cost: int
    build_city_ids: tuple[str, ...]
    build_cost: int
    generation: _GenerationSummary
    cash_after_purchase: int
    cash_after_resources: int
    cash_after_build: int
    cash_after_income: int
    projection_kind: str


@dataclass(frozen=True)
class _AuctionReserveProjection:
    reserve: int
    minimum_bid: int
    hard_cap: int
    required_margin: float
    fallback: _AuctionEconomyProjection
    target_at_minimum: _AuctionEconomyProjection
    target_at_reserve: _AuctionEconomyProjection
    price_samples: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _DecisionTrace:
    current_evaluation: dict[str, object]
    candidate_actions: tuple[dict[str, object], ...]
    selected_action: dict[str, object]
    search_summary: dict[str, object]


class StrategicAiController(BaseAiController):
    controller = "ai_heuristics"

    def choose_intent(self, request: TurnRequest, snapshot: GameSnapshot) -> GuiIntent:
        state = snapshot.state
        if state.pending_decision is not None:
            intent, trace = _choose_pending_intent(state)
            self._log_decision(snapshot, request, intent, trace)
            return intent
        if request.phase == "auction":
            intent, trace = _choose_auction_intent(request, snapshot)
            self._log_decision(snapshot, request, intent, trace)
            return intent
        if request.phase == "buy_resources":
            intent, trace = _choose_resource_intent(state, request.player_id)
            self._log_decision(snapshot, request, intent, trace)
            return intent
        if request.phase == "build_houses":
            intent, trace = _choose_build_intent(state, request.player_id)
            self._log_decision(snapshot, request, intent, trace)
            return intent
        if request.phase == "bureaucracy":
            intent, trace = _choose_bureaucracy_intent(state, request.player_id)
            self._log_decision(snapshot, request, intent, trace)
            return intent
        raise ModelValidationError(f"unsupported request phase {request.phase!r}")

    def _log_decision(
        self,
        snapshot: GameSnapshot,
        request: TurnRequest,
        intent: GuiIntent,
        trace: _DecisionTrace,
    ) -> None:
        candidate_actions = _rank_logged_candidates(trace.candidate_actions)
        self.log_state(
            snapshot,
            request,
            label="heuristic_decision",
            state={
                "schema_version": 5,
                "decision_type": request.decision_type,
                "intent_type": intent.intent_type,
                "intent_payload": dict(intent.payload),
                "current_evaluation": trace.current_evaluation,
                "candidate_action_count": len(trace.candidate_actions),
                "candidate_actions_truncated": len(candidate_actions) < len(trace.candidate_actions),
                "candidate_actions": candidate_actions,
                "selected_action": trace.selected_action,
                "search_summary": trace.search_summary,
            },
            message="Heuristic AI selected an intent.",
        )


def _choose_pending_intent(state: GameState) -> tuple[GuiIntent, _DecisionTrace]:
    decision = state.pending_decision
    assert decision is not None
    current_evaluation = _evaluate_relative_state_detail(state, decision.player_id)
    current_score = _evaluate_relative_state(state, decision.player_id)
    if decision.decision_type == "discard_power_plant":
        best_price = None
        best_score = float("-inf")
        best_state = None
        candidate_actions: list[dict[str, object]] = []
        for action in decision.legal_actions:
            plant_price = int(action.payload["price"])
            candidate = replace_plant_if_needed(_clone_state(state), decision.player_id, plant_price)
            score = _evaluate_relative_state(candidate, decision.player_id)
            candidate_actions.append(
                _candidate_action_trace(
                    intent_type="discard_power_plant",
                    payload={"plant_price": plant_price},
                    decision_score=score,
                    projected_relative_score=score,
                    current_score=current_score,
                )
            )
            if score > best_score or (
                score == best_score and (best_price is None or plant_price < best_price)
            ):
                best_score = score
                best_price = plant_price
                best_state = candidate
        assert best_price is not None
        assert best_state is not None
        intent = GuiIntent.discard_plant(decision.player_id, best_price)
        projected_evaluation = _evaluate_relative_state_detail(best_state, decision.player_id)
        return intent, _DecisionTrace(
            current_evaluation=current_evaluation,
            candidate_actions=tuple(candidate_actions),
            selected_action=_selected_action_trace(
                intent,
                decision_score=best_score,
                projected_evaluation=projected_evaluation,
                current_score=current_score,
                score_terms={"selection_rule": "max_projected_relative_state"},
            ),
            search_summary={
                "decision_family": "pending_discard_power_plant",
                "legal_action_count": len(decision.legal_actions),
            },
        )

    best_choice = None
    best_state = None
    candidate_actions = []
    for action in decision.legal_actions:
        coal = int(action.payload.get("coal", 0))
        oil = int(action.payload.get("oil", 0))
        candidate = discard_resources_to_fit_storage(
            _clone_state(state),
            decision.player_id,
            {"coal": coal, "oil": oil},
        )
        score = _evaluate_relative_state(candidate, decision.player_id)
        candidate_actions.append(
            _candidate_action_trace(
                intent_type="discard_hybrid_resources",
                payload={"coal": coal, "oil": oil},
                decision_score=score,
                projected_relative_score=score,
                current_score=current_score,
                score_terms={"discarded_resource_units": coal + oil},
            )
        )
        signature = (score, -(coal + oil), oil, coal)
        if best_choice is None or signature > best_choice[0]:
            best_choice = (signature, coal, oil)
            best_state = candidate
    assert best_choice is not None
    assert best_state is not None
    intent = GuiIntent.discard_hybrid_resources(
        decision.player_id,
        coal=best_choice[1],
        oil=best_choice[2],
    )
    projected_evaluation = _evaluate_relative_state_detail(best_state, decision.player_id)
    return intent, _DecisionTrace(
        current_evaluation=current_evaluation,
        candidate_actions=tuple(candidate_actions),
        selected_action=_selected_action_trace(
            intent,
            decision_score=float(best_choice[0][0]),
            projected_evaluation=projected_evaluation,
            current_score=current_score,
            score_terms={"selection_rule": "max_projected_state_then_preserve_resources"},
        ),
        search_summary={
            "decision_family": "pending_discard_hybrid_resources",
            "legal_action_count": len(decision.legal_actions),
        },
    )


def _choose_auction_intent(request: TurnRequest, snapshot: GameSnapshot) -> tuple[GuiIntent, _DecisionTrace]:
    state = snapshot.state
    if request.decision_type == "auction_start":
        return _choose_auction_start_intent(state, request.player_id)
    return _choose_auction_bid_intent(state, request.player_id)


def _choose_auction_start_intent(state: GameState, player_id: str) -> tuple[GuiIntent, _DecisionTrace]:
    current_evaluation = _evaluate_relative_state_detail(state, player_id)
    current_score = _evaluate_relative_state(state, player_id)
    start_actions = [
        action
        for action in _legal_auction_start_actions(state, player_id)
        if action.action_type == "auction_start"
    ]
    if not start_actions:
        intent = GuiIntent.auction_pass(player_id)
        return intent, _DecisionTrace(
            current_evaluation=current_evaluation,
            candidate_actions=(),
            selected_action=_selected_action_trace(
                intent,
                decision_score=current_score,
                projected_evaluation=current_evaluation,
                current_score=current_score,
                score_terms={
                    "selection_rule": "no_legal_auction_start_actions",
                    "projected_kind": "current_state_after_pass",
                },
            ),
            search_summary={"decision_family": "auction_start", "legal_action_count": 0},
        )

    player = _get_player(state, player_id)
    best_choice = None
    best_delta = float("-inf")
    best_projected_state = None
    best_projected_kind = None
    best_score_terms: dict[str, Any] | None = None
    candidate_actions: list[dict[str, object]] = []
    for action in start_actions:
        plant_price = int(action.payload["plant_price"])
        min_bid = int(action.payload["min_bid"])
        plant = _get_market_plant(state, plant_price)
        forced_opening_purchase = state.round_number == 1 and not player.power_plants
        reserve_projection = _auction_reserve_projection(
            state,
            player_id,
            plant,
            minimum_bid=min_bid,
        )
        reserve = reserve_projection.reserve
        own_interest = reserve - min_bid
        opponent_reserves = [
            _auction_reserve_level0(state, opponent.player_id, plant)
            for opponent in state.players
            if opponent.player_id != player_id and _can_participate_in_auction(state, opponent.player_id)
        ]
        strongest_opponent = max(opponent_reserves, default=0)
        contest_pressure = max(0, strongest_opponent - reserve)
        market_bonus = _market_roll_bonus(state, plant)
        bait_score = 0.0
        if strongest_opponent > min_bid and reserve + 1 < strongest_opponent:
            bait_score = (strongest_opponent - min_bid) * 0.75 + market_bonus * 0.6
        if own_interest >= 0:
            price_guess = min(reserve, max(min_bid, strongest_opponent))
            economy_projection = _project_post_auction_economy(
                state,
                player_id,
                plant,
                price_guess,
                projection_kind="post_auction_economy_at_expected_price",
            )
            projected_score = economy_projection.score
            fallback_score = reserve_projection.fallback.score
            gain = projected_score - current_score
            fallback_surplus = projected_score - fallback_score
            delta = fallback_surplus + market_bonus - (contest_pressure * 0.35)
            opening_bid = min_bid
            if _is_sprint_state(state, player_id) and reserve - min_bid >= 4:
                opening_bid = min(reserve, min_bid + 2, player.elektro)
            projected_state = economy_projection.evaluation_state
            target_projection = economy_projection
            projected_kind = economy_projection.projection_kind
        else:
            opening_bid = min_bid
            price_guess = min_bid
            fallback_score = reserve_projection.fallback.score
            if forced_opening_purchase:
                economy_projection = reserve_projection.target_at_minimum
                projected_score = economy_projection.score
                fallback_surplus = projected_score - fallback_score
                delta = fallback_surplus + market_bonus
                gain = projected_score - current_score
                projected_kind = economy_projection.projection_kind
                projected_state = economy_projection.evaluation_state
                target_projection = economy_projection
            else:
                delta = bait_score - 1.5
                projected_score = current_score
                fallback_surplus = current_score - fallback_score
                economy_projection = reserve_projection.target_at_minimum
                gain = 0.0
                projected_state = state
                target_projection = economy_projection
                projected_kind = "bait_start_without_expected_purchase"
        score_terms = {
            "auction_value_model": "post_purchase_resource_build_generation_projection",
            "reserve": reserve,
            "hard_cap": reserve_projection.hard_cap,
            "minimum_bid": min_bid,
            "price_guess": price_guess,
            "own_interest": own_interest,
            "strongest_opponent_reserve": strongest_opponent,
            "contest_pressure": contest_pressure,
            "market_bonus": market_bonus,
            "bait_score": bait_score,
            "projected_gain": gain,
            "fallback_score": fallback_score,
            "fallback_surplus": fallback_surplus,
            "projected_kind": projected_kind,
            "plant_purchase_gain": _plant_purchase_gain(state, player_id, plant),
            "reserve_required_margin": reserve_projection.required_margin,
            "target_projection": _auction_projection_to_log(target_projection),
            "fallback_projection": _auction_projection_to_log(reserve_projection.fallback),
            "target_at_minimum_projection": _auction_projection_to_log(
                reserve_projection.target_at_minimum
            ),
            "target_at_reserve_projection": _auction_projection_to_log(
                reserve_projection.target_at_reserve
            ),
            "reserve_price_samples": reserve_projection.price_samples,
        }
        candidate_actions.append(
            _candidate_action_trace(
                intent_type="auction_start",
                payload={"plant_price": plant_price, "bid": opening_bid},
                decision_score=delta,
                projected_relative_score=projected_score,
                current_score=current_score,
                score_terms=score_terms,
            )
        )
        signature = (
            delta,
            own_interest,
            bait_score,
            _plant_purchase_gain(state, player_id, plant),
            -plant.price,
        )
        if best_choice is None or signature > best_choice[0]:
            best_choice = (signature, plant_price, opening_bid)
            best_delta = delta
            best_projected_state = projected_state
            best_projected_kind = projected_kind
            best_score_terms = score_terms

    if state.round_number > 1 and best_delta <= 0:
        intent = GuiIntent.auction_pass(player_id)
        candidate_actions.append(
            _candidate_action_trace(
                intent_type="auction_pass",
                payload={},
                decision_score=0.0,
                projected_relative_score=current_score,
                current_score=current_score,
                score_terms={
                    "selection_rule": "pass_when_best_start_delta_is_not_positive",
                    "projected_kind": "current_state_after_pass",
                },
            )
        )
        return intent, _DecisionTrace(
            current_evaluation=current_evaluation,
            candidate_actions=tuple(candidate_actions),
            selected_action=_selected_action_trace(
                intent,
                decision_score=0.0,
                projected_evaluation=current_evaluation,
                current_score=current_score,
                score_terms={
                    "best_start_delta": best_delta,
                    "projected_kind": "current_state_after_pass",
                },
            ),
            search_summary={
                "decision_family": "auction_start",
                "legal_action_count": len(start_actions),
                "best_start_delta": _score(best_delta),
            },
        )

    assert best_choice is not None
    assert best_projected_state is not None
    assert best_projected_kind is not None
    assert best_score_terms is not None
    intent = GuiIntent.auction_start(
        player_id,
        plant_price=best_choice[1],
        bid=best_choice[2],
    )
    projected_evaluation = _evaluate_relative_state_detail(best_projected_state, player_id)
    selected_score_terms = dict(best_score_terms)
    selected_score_terms["selection_rule"] = "max_auction_start_delta"
    selected_score_terms["projected_kind"] = best_projected_kind
    return intent, _DecisionTrace(
        current_evaluation=current_evaluation,
        candidate_actions=tuple(candidate_actions),
        selected_action=_selected_action_trace(
            intent,
            decision_score=best_delta,
            projected_evaluation=projected_evaluation,
            current_score=current_score,
            score_terms=selected_score_terms,
        ),
        search_summary={
            "decision_family": "auction_start",
            "legal_action_count": len(start_actions),
            "best_start_delta": _score(best_delta),
        },
    )


def _choose_auction_bid_intent(state: GameState, player_id: str) -> tuple[GuiIntent, _DecisionTrace]:
    current_evaluation = _evaluate_relative_state_detail(state, player_id)
    current_score = _evaluate_relative_state(state, player_id)
    auction_state = state.auction_state
    assert auction_state is not None
    plant = _get_market_plant(state, int(auction_state.active_plant_price))
    min_bid = int(auction_state.current_bid) + 1
    reserve_projection = _auction_reserve_projection(
        state,
        player_id,
        plant,
        minimum_bid=min_bid,
        active_auction=True,
    )
    reserve = reserve_projection.reserve
    hard_cap = reserve_projection.hard_cap
    bid_projection = _project_post_auction_economy(
        state,
        player_id,
        plant,
        min_bid,
        projection_kind="post_auction_economy_at_min_bid",
    )
    bid_projected_score = bid_projection.score
    pass_projection = reserve_projection.fallback
    candidate_actions = (
        _candidate_action_trace(
            intent_type="auction_bid",
            payload={"bid": min_bid},
            decision_score=reserve - min_bid,
            projected_relative_score=bid_projected_score,
            current_score=current_score,
            score_terms={
                "auction_value_model": "post_purchase_resource_build_generation_projection",
                "reserve": reserve,
                "hard_cap": hard_cap,
                "minimum_bid": min_bid,
                "projected_kind": bid_projection.projection_kind,
                "plant_price": plant.price,
                "plant_purchase_gain": _plant_purchase_gain(state, player_id, plant),
                "fallback_score": pass_projection.score,
                "fallback_surplus": bid_projection.score - pass_projection.score,
                "reserve_required_margin": reserve_projection.required_margin,
                "target_projection": _auction_projection_to_log(bid_projection),
                "fallback_projection": _auction_projection_to_log(pass_projection),
                "target_at_reserve_projection": _auction_projection_to_log(
                    reserve_projection.target_at_reserve
                ),
                "reserve_price_samples": reserve_projection.price_samples,
            },
        ),
        _candidate_action_trace(
            intent_type="auction_pass",
            payload={},
            decision_score=0.0,
            projected_relative_score=pass_projection.score,
            current_score=current_score,
            score_terms={
                "projected_kind": pass_projection.projection_kind,
                "fallback_projection": _auction_projection_to_log(pass_projection),
            },
        ),
    )
    if min_bid > reserve or min_bid > hard_cap:
        intent = GuiIntent.auction_pass(player_id)
        return intent, _DecisionTrace(
            current_evaluation=current_evaluation,
            candidate_actions=candidate_actions,
            selected_action=_selected_action_trace(
                intent,
                decision_score=0.0,
                projected_evaluation=_evaluate_relative_state_detail(pass_projection.evaluation_state, player_id),
                current_score=current_score,
                score_terms={
                    "selection_rule": "pass_when_min_bid_exceeds_reserve_or_hard_cap",
                    "reserve": reserve,
                    "hard_cap": hard_cap,
                    "minimum_bid": min_bid,
                    "projected_kind": pass_projection.projection_kind,
                    "fallback_projection": _auction_projection_to_log(pass_projection),
                },
            ),
            search_summary={"decision_family": "auction_bid", "legal_action_count": 2},
        )
    intent = GuiIntent.auction_bid(player_id, min_bid)
    return intent, _DecisionTrace(
        current_evaluation=current_evaluation,
        candidate_actions=candidate_actions,
        selected_action=_selected_action_trace(
            intent,
            decision_score=reserve - min_bid,
            projected_evaluation=_evaluate_relative_state_detail(bid_projection.evaluation_state, player_id),
            current_score=current_score,
            score_terms={
                "selection_rule": "bid_minimum_while_within_reserve_and_hard_cap",
                "reserve": reserve,
                "hard_cap": hard_cap,
                "minimum_bid": min_bid,
                "projected_kind": bid_projection.projection_kind,
                "auction_value_model": "post_purchase_resource_build_generation_projection",
                "plant_price": plant.price,
                "plant_purchase_gain": _plant_purchase_gain(state, player_id, plant),
                "fallback_score": pass_projection.score,
                "fallback_surplus": bid_projection.score - pass_projection.score,
                "reserve_required_margin": reserve_projection.required_margin,
                "target_projection": _auction_projection_to_log(bid_projection),
                "fallback_projection": _auction_projection_to_log(pass_projection),
                "target_at_reserve_projection": _auction_projection_to_log(
                    reserve_projection.target_at_reserve
                ),
                "reserve_price_samples": reserve_projection.price_samples,
            },
        ),
        search_summary={"decision_family": "auction_bid", "legal_action_count": 2},
    )


def _choose_resource_intent(state: GameState, player_id: str) -> tuple[GuiIntent, _DecisionTrace]:
    current_evaluation = _evaluate_relative_state_detail(state, player_id)
    current_score = _evaluate_relative_state(state, player_id)
    depth = 3
    actions = legal_resource_purchases(state, player_id)
    if len(actions) > 8:
        depth = 2
    score, move, candidate_actions = _search_resource_purchase_decision(state, player_id, depth=depth)
    if move is None:
        intent = GuiIntent.finish_buying(player_id)
        return intent, _DecisionTrace(
            current_evaluation=current_evaluation,
            candidate_actions=tuple(candidate_actions),
            selected_action=_selected_action_trace(
                intent,
                decision_score=score,
                projected_evaluation=current_evaluation,
                current_score=current_score,
                score_terms={"selection_rule": "finish_when_no_purchase_improves_stop_score"},
            ),
            search_summary={
                "decision_family": "resource_purchase",
                "search_depth": depth,
                "legal_action_count": len(actions),
                "first_level_candidate_count": len(candidate_actions),
                "stop_score": _score(_resource_finish_score(state, player_id)),
                "best_score": _score(score),
            },
        )
    next_state = purchase_resources(_clone_state(state), player_id, {move[0]: move[1]})
    intent = GuiIntent.buy_resource(player_id, resource=move[0], amount=move[1])
    return intent, _DecisionTrace(
        current_evaluation=current_evaluation,
        candidate_actions=tuple(candidate_actions),
        selected_action=_selected_action_trace(
            intent,
            decision_score=score,
            projected_evaluation=_evaluate_relative_state_detail(next_state, player_id),
            current_score=current_score,
            score_terms={
                "selection_rule": "max_recursive_resource_purchase_score",
                "immediate_projected_relative_score": _evaluate_relative_state(next_state, player_id),
            },
        ),
        search_summary={
            "decision_family": "resource_purchase",
            "search_depth": depth,
            "legal_action_count": len(actions),
            "first_level_candidate_count": len(candidate_actions),
            "stop_score": _score(_resource_finish_score(state, player_id)),
            "best_score": _score(score),
        },
    )


def _choose_build_intent(state: GameState, player_id: str) -> tuple[GuiIntent, _DecisionTrace]:
    current_evaluation = _evaluate_relative_state_detail(state, player_id)
    current_score = _evaluate_relative_state(state, player_id)
    plan, candidate_actions, search_summary = _search_best_build_plan(state, player_id)
    if plan is None:
        intent = GuiIntent.finish_building(player_id)
        return intent, _DecisionTrace(
            current_evaluation=current_evaluation,
            candidate_actions=tuple(candidate_actions),
            selected_action=_selected_action_trace(
                intent,
                decision_score=current_score,
                projected_evaluation=current_evaluation,
                current_score=current_score,
                score_terms={"selection_rule": "finish_when_no_profitable_plan_found"},
            ),
            search_summary=search_summary,
        )
    if plan.score <= current_score + 0.25:
        intent = GuiIntent.finish_building(player_id)
        return intent, _DecisionTrace(
            current_evaluation=current_evaluation,
            candidate_actions=tuple(candidate_actions),
            selected_action=_selected_action_trace(
                intent,
                decision_score=current_score,
                projected_evaluation=current_evaluation,
                current_score=current_score,
                score_terms={
                    "selection_rule": "finish_when_best_plan_is_marginal",
                    "best_plan_score": plan.score,
                    "minimum_commit_score": current_score + 0.25,
                },
            ),
            search_summary=search_summary,
        )
    next_state = apply_builds(_clone_state(state), player_id, plan.city_ids)
    intent = GuiIntent.commit_build(player_id, list(plan.city_ids))
    return intent, _DecisionTrace(
        current_evaluation=current_evaluation,
        candidate_actions=tuple(candidate_actions),
        selected_action=_selected_action_trace(
            intent,
            decision_score=plan.score,
            projected_evaluation=_evaluate_relative_state_detail(next_state, player_id),
            current_score=current_score,
            score_terms={
                "selection_rule": "commit_best_build_plan",
                "total_cost": plan.total_cost,
                "city_count": len(plan.city_ids),
            },
        ),
        search_summary=search_summary,
    )


def _choose_bureaucracy_intent(state: GameState, player_id: str) -> tuple[GuiIntent, _DecisionTrace]:
    current_evaluation = _evaluate_relative_state_detail(state, player_id)
    current_score = _evaluate_relative_state(state, player_id)
    player = _get_player(state, player_id)
    plans = _enumerate_generation_summaries(state, player_id)
    if not plans:
        intent = GuiIntent.skip_bureaucracy(player_id)
        return intent, _DecisionTrace(
            current_evaluation=current_evaluation,
            candidate_actions=(),
            selected_action=_selected_action_trace(
                intent,
                decision_score=current_score,
                projected_evaluation=current_evaluation,
                current_score=current_score,
                score_terms={"selection_rule": "skip_when_no_generation_plans"},
            ),
            search_summary={"decision_family": "bureaucracy", "candidate_plan_count": 0},
        )

    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    final_round = any(candidate.connected_city_count >= end_threshold for candidate in state.players)
    candidate_actions = []
    scored_options = []
    for summary in plans:
        simulated = _simulate_bureaucracy_summary(state, player_id, summary)
        projected_score = _evaluate_relative_state(simulated, player_id)
        winner_tuple = (
            summary.powered,
            player.elektro + summary.income,
            player.connected_city_count,
        )
        decision_score = (
            (winner_tuple[0] * 10000.0) + (winner_tuple[1] * 100.0) + winner_tuple[2]
            if final_round
            else projected_score
        )
        intent_type = "run_plants" if summary.plans else "skip_bureaucracy"
        payload = {"plans": [plan.to_dict() for plan in summary.plans]} if summary.plans else {}
        candidate_actions.append(
            _candidate_action_trace(
                intent_type=intent_type,
                payload=payload,
                decision_score=decision_score,
                projected_relative_score=projected_score,
                current_score=current_score,
                score_terms={
                    "powered": summary.powered,
                    "income": summary.income,
                    "spent_units": summary.spent_units,
                    "spent_value": summary.spent_value,
                    "winner_tuple": winner_tuple,
                    "final_round": final_round,
                },
            )
        )
        scored_options.append((summary, decision_score, projected_score, winner_tuple, simulated))
    if final_round:
        best = max(
            plans,
            key=lambda summary: (
                summary.powered,
                player.elektro + summary.income,
                player.connected_city_count,
                -summary.spent_value,
                tuple(plan.plant_price for plan in summary.plans),
            ),
        )
    else:
        best = max(
            plans,
            key=lambda summary: _evaluate_bureaucracy_summary(state, player_id, summary),
        )
    if not best.plans:
        intent = GuiIntent.skip_bureaucracy(player_id)
        projected = next(option for option in scored_options if option[0] == best)
        return intent, _DecisionTrace(
            current_evaluation=current_evaluation,
            candidate_actions=tuple(candidate_actions),
            selected_action=_selected_action_trace(
                intent,
                decision_score=projected[1],
                projected_evaluation=_evaluate_relative_state_detail(projected[4], player_id),
                current_score=current_score,
                score_terms={
                    "selection_rule": "selected_empty_generation_plan",
                    "final_round": final_round,
                    "winner_tuple": projected[3],
                },
            ),
            search_summary={
                "decision_family": "bureaucracy",
                "candidate_plan_count": len(plans),
                "final_round": final_round,
            },
        )
    intent = GuiIntent.run_plants(player_id, best.plans)
    projected = next(option for option in scored_options if option[0] == best)
    return intent, _DecisionTrace(
        current_evaluation=current_evaluation,
        candidate_actions=tuple(candidate_actions),
        selected_action=_selected_action_trace(
            intent,
            decision_score=projected[1],
            projected_evaluation=_evaluate_relative_state_detail(projected[4], player_id),
            current_score=current_score,
            score_terms={
                "selection_rule": "max_winner_tuple" if final_round else "max_projected_relative_state",
                "final_round": final_round,
                "winner_tuple": projected[3],
            },
        ),
        search_summary={
            "decision_family": "bureaucracy",
            "candidate_plan_count": len(plans),
            "final_round": final_round,
        },
    )


def _legal_auction_start_actions(state: GameState, player_id: str) -> tuple[_AuctionStartOption, ...]:
    auction_state = state.auction_state
    assert auction_state is not None
    player = _get_player(state, player_id)
    actions: list[_AuctionStartOption] = []
    for plant in list_auctionable_plants(state):
        minimum_bid = 1 if auction_state.discount_token_plant_price == plant.price else plant.price
        if minimum_bid <= player.elektro:
            actions.append(
                _AuctionStartOption(
                    action_type="auction_start",
                    payload={
                        "plant_price": plant.price,
                        "min_bid": minimum_bid,
                        "max_bid": player.elektro,
                    },
                )
            )
    return tuple(actions)


def _search_resource_purchase_decision(
    state: GameState,
    player_id: str,
    *,
    depth: int,
) -> tuple[float, tuple[str, int] | None, list[dict[str, object]]]:
    current_score = _evaluate_relative_state(state, player_id)
    best_score = _resource_finish_score(state, player_id)
    best_move = None
    candidate_actions = [
        _candidate_action_trace(
            intent_type="finish_buying",
            payload={},
            decision_score=best_score,
            projected_relative_score=current_score,
            current_score=current_score,
            score_terms=_resource_finish_score_terms(state, player_id),
        )
    ]
    if depth <= 0:
        return best_score, None, candidate_actions

    for action in legal_resource_purchases(state, player_id):
        resource = str(action.payload["resource"])
        for amount in _candidate_resource_amounts(state, player_id, action):
            try:
                next_state = purchase_resources(
                    _clone_state(state),
                    player_id,
                    {resource: amount},
                )
            except ModelValidationError:
                continue
            next_score, _ = _search_resource_purchase(next_state, player_id, depth=depth - 1)
            immediate_projected_score = _evaluate_relative_state(next_state, player_id)
            candidate_actions.append(
                _candidate_action_trace(
                    intent_type="buy_resource",
                    payload={"resource": resource, "amount": amount},
                    decision_score=next_score,
                    projected_relative_score=immediate_projected_score,
                    current_score=current_score,
                    score_terms={
                        "recursive_depth_remaining": depth - 1,
                        "resource": resource,
                        "amount": amount,
                        "recursive_score": next_score,
                        "immediate_projected_relative_score": immediate_projected_score,
                    },
                )
            )
            signature = (next_score, amount, -RESOURCE_TYPES.index(resource))
            if best_move is None or signature > (best_score, best_move[1], -RESOURCE_TYPES.index(best_move[0])):
                if next_score > best_score + 0.15:
                    best_score = next_score
                    best_move = (resource, amount)
    return best_score, best_move, candidate_actions


def _search_resource_purchase(
    state: GameState,
    player_id: str,
    *,
    depth: int,
) -> tuple[float, tuple[str, int] | None]:
    best_score = _resource_finish_score(state, player_id)
    best_move = None
    if depth <= 0:
        return best_score, None

    for action in legal_resource_purchases(state, player_id):
        resource = str(action.payload["resource"])
        for amount in _candidate_resource_amounts(state, player_id, action):
            try:
                next_state = purchase_resources(
                    _clone_state(state),
                    player_id,
                    {resource: amount},
                )
            except ModelValidationError:
                continue
            next_score, _ = _search_resource_purchase(next_state, player_id, depth=depth - 1)
            signature = (next_score, amount, -RESOURCE_TYPES.index(resource))
            if best_move is None or signature > (best_score, best_move[1], -RESOURCE_TYPES.index(best_move[0])):
                if next_score > best_score + 0.15:
                    best_score = next_score
                    best_move = (resource, amount)
    return best_score, best_move


def _candidate_resource_amounts(state: GameState, player_id: str, action) -> tuple[int, ...]:
    resource = str(action.payload["resource"])
    max_affordable = int(action.payload["max_affordable_units"])
    max_units = min(int(action.payload["max_units"]), max_affordable)
    deficits = _resource_need_by_type(state, player_id)
    market_prices = tuple(int(price) for price in action.payload["unit_prices"])
    scarce = _resource_pressure(state)[resource] >= 0.7
    cheap_run = 0
    for price in market_prices:
        if price <= 2:
            cheap_run += 1
        else:
            break
    amounts = {
        1,
        max_units,
        min(max_units, max(1, deficits.get(resource, 0))),
        min(max_units, 2),
        min(max_units, cheap_run) if cheap_run else 1,
    }
    if scarce:
        amounts.add(min(max_units, max(1, deficits.get(resource, 0) + 1)))
    return tuple(sorted(amount for amount in amounts if 1 <= amount <= max_units))


def _resource_finish_score(state: GameState, player_id: str) -> float:
    score = _evaluate_relative_state(state, player_id)
    score += _quick_build_potential(state, player_id) * 0.6
    deficits = _resource_need_by_type(state, player_id)
    shortfall_penalty = 0.0
    for resource, amount in deficits.items():
        if amount <= 0:
            continue
        shortfall_penalty += amount * (_resource_unit_price(state, resource) + _resource_pressure(state)[resource] * 2.0)
    score -= shortfall_penalty * 0.35
    return score


def _search_best_build_plan(
    state: GameState,
    player_id: str,
) -> tuple[_BuildPlan | None, list[dict[str, object]], dict[str, object]]:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return None, [], {
            "decision_family": "build",
            "legal_action_count": 0,
            "candidate_city_count": 0,
            "evaluated_plan_count": 0,
        }

    player = _get_player(state, player_id)
    powered_now = _best_generation_summary(state, player_id).powered
    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    max_depth = 4
    if _is_sprint_state(state, player_id):
        max_depth = 5
    else:
        max_depth = max(1, min(4, max(1, powered_now - player.connected_city_count + 2)))
    max_depth = min(max_depth, len(actions), player.houses_in_supply)

    ranked = sorted(
        actions,
        key=lambda action: (
            _quick_city_score(state, player_id, str(action.payload["city_id"]), int(action.payload["total_cost"])),
            -int(action.payload["total_cost"]),
            str(action.payload["city_id"]),
        ),
        reverse=True,
    )
    candidate_ids = [str(action.payload["city_id"]) for action in ranked[: min(10, len(ranked))]]
    current_score = _evaluate_relative_state(state, player_id)
    current_player = _get_player(state, player_id)
    current_connected = current_player.connected_city_count
    current_powered = _best_generation_summary(state, player_id).powered
    seen: set[frozenset[str]] = set()
    beam: list[tuple[str, ...]] = [()]
    best_plan: _BuildPlan | None = None
    candidate_actions: list[dict[str, object]] = []

    for depth_index in range(max_depth):
        expansions: list[tuple[float, float, tuple[str, ...], int]] = []
        for base in beam:
            used = set(base)
            for city_id in candidate_ids:
                if city_id in used:
                    continue
                proposal = tuple((*base, city_id))
                key = frozenset(proposal)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    next_state = apply_builds(_clone_state(state), player_id, proposal)
                except ModelValidationError:
                    continue
                next_score = _evaluate_relative_state(next_state, player_id)
                next_player = _get_player(next_state, player_id)
                next_powered = _best_generation_summary(next_state, player_id).powered
                total_cost = player.elektro - _get_player(next_state, player_id).elektro
                trigger_bonus = _build_trigger_adjustment(next_state, player_id)
                build_bonus = (
                    (next_player.connected_city_count - current_connected) * 14.0
                    + max(0, next_powered - current_powered) * 4.0
                    - (total_cost * 0.08)
                )
                score = next_score + trigger_bonus + build_bonus
                candidate_actions.append(
                    _candidate_action_trace(
                        intent_type="commit_build",
                        payload={"city_ids": list(proposal)},
                        decision_score=score,
                        projected_relative_score=next_score,
                        current_score=current_score,
                        score_terms={
                            "depth": depth_index + 1,
                            "total_cost": total_cost,
                            "trigger_bonus": trigger_bonus,
                            "build_bonus": build_bonus,
                            "connected_delta": next_player.connected_city_count - current_connected,
                            "powered_delta": max(0, next_powered - current_powered),
                        },
                    )
                )
                expansions.append((score, -total_cost, proposal, total_cost))
                if score > current_score + 0.75:
                    plan = _BuildPlan(proposal, total_cost, score)
                    if best_plan is None or (plan.score, -plan.total_cost, plan.city_ids) > (
                        best_plan.score,
                        -best_plan.total_cost,
                        best_plan.city_ids,
                    ):
                        best_plan = plan
        if not expansions:
            break
        expansions.sort(reverse=True)
        beam = [proposal for _, _, proposal, _ in expansions[:6]]
        if best_plan is not None and len(best_plan.city_ids) >= end_threshold:
            break
    return best_plan, candidate_actions, {
        "decision_family": "build",
        "legal_action_count": len(actions),
        "candidate_city_count": len(candidate_ids),
        "evaluated_plan_count": len(candidate_actions),
        "max_depth": max_depth,
        "beam_width": 6,
        "best_plan_score": _score(best_plan.score) if best_plan is not None else None,
        "best_plan_city_count": len(best_plan.city_ids) if best_plan is not None else 0,
    }


def _quick_city_score(state: GameState, player_id: str, city_id: str, total_cost: int) -> float:
    savings = _future_connection_savings(state, player_id, city_id)
    pressure = _city_contest_pressure(state, player_id, city_id)
    stage = STAGE_WEIGHTS[_stage_name(state)]
    return savings * 0.18 - total_cost * 0.42 - pressure * 1.4 + stage.frontier * 3.0


def _future_connection_savings(state: GameState, player_id: str, city_id: str) -> float:
    player = _get_player(state, player_id)
    current = compute_all_targets_connection_cost(state, player_id)
    updated = compute_all_targets_connection_cost(
        state,
        player_id,
        source_city_ids=tuple((*player.network_city_ids, city_id)),
    )
    savings = 0
    deltas = []
    for target, current_cost in current.items():
        if target == city_id or target in player.network_city_ids:
            continue
        next_cost = updated.get(target, current_cost)
        if next_cost < current_cost:
            deltas.append(current_cost - next_cost)
    for delta in sorted(deltas, reverse=True)[:5]:
        savings += delta
    return float(savings)


def _city_contest_pressure(state: GameState, player_id: str, city_id: str) -> int:
    player_costs = compute_all_targets_connection_cost(state, player_id)
    own_cost = player_costs.get(city_id, 999)
    pressure = 0
    for opponent in state.players:
        if opponent.player_id == player_id or city_id in opponent.network_city_ids:
            continue
        opponent_costs = compute_all_targets_connection_cost(state, opponent.player_id)
        candidate_cost = opponent_costs.get(city_id)
        if candidate_cost is None:
            continue
        if candidate_cost <= own_cost + 10 and opponent.elektro >= 10:
            pressure += 1
    return pressure


def _build_trigger_adjustment(state: GameState, player_id: str) -> float:
    player = _get_player(state, player_id)
    step_2_threshold = state.rules.player_count_rules[len(state.players)]["step_2_cities"]
    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    if player.connected_city_count >= end_threshold:
        own_power = _best_generation_summary(state, player_id)
        own_tuple = (
            own_power.powered,
            player.elektro + own_power.income,
            player.connected_city_count,
        )
        opponent_best = max(
            (
                _best_generation_summary(state, opponent.player_id).powered,
                opponent.elektro + _best_generation_summary(state, opponent.player_id).income,
                opponent.connected_city_count,
            )
            for opponent in state.players
            if opponent.player_id != player_id
        )
        return 80.0 if own_tuple > opponent_best else -120.0
    if state.step == 1 and player.connected_city_count >= step_2_threshold:
        own_powered = _best_generation_summary(state, player_id).powered
        rival_power = max(
            _best_generation_summary(state, opponent.player_id).powered
            for opponent in state.players
            if opponent.player_id != player_id
        )
        return 18.0 if own_powered >= rival_power else -26.0
    return 0.0


def _evaluate_bureaucracy_summary(
    state: GameState,
    player_id: str,
    summary: _GenerationSummary,
) -> float:
    simulated = _simulate_bureaucracy_summary(state, player_id, summary)
    return _evaluate_relative_state(simulated, player_id)


def _simulate_bureaucracy_summary(
    state: GameState,
    player_id: str,
    summary: _GenerationSummary,
) -> GameState:
    simulated = _clone_state(state)
    simulated = consume_resources(simulated, player_id, summary.plans)
    player = _get_player(simulated, player_id)
    return _replace_player_on_state(
        simulated,
        replace(player, elektro=player.elektro + summary.income),
    )


def _best_generation_summary(state: GameState, player_id: str) -> _GenerationSummary:
    options = _enumerate_generation_summaries(state, player_id)
    if not options:
        return _GenerationSummary((), 0, pay_income(state.rules, 0), 0, 0.0)
    return max(
        options,
        key=lambda summary: (
            summary.powered,
            summary.income,
            -summary.spent_value,
            -summary.spent_units,
            tuple(plan.plant_price for plan in summary.plans),
        ),
    )


def _enumerate_generation_summaries(state: GameState, player_id: str) -> tuple[_GenerationSummary, ...]:
    player = _get_player(state, player_id)
    totals = player.resource_storage.resource_totals()
    plants = [plant for plant in sorted(player.power_plants, key=lambda item: item.price) if not plant.is_step_3_placeholder]
    if not plants:
        return (_GenerationSummary((), 0, pay_income(state.rules, 0), 0, 0.0),)

    choices: list[tuple[PowerPlantCard, tuple[PlantRunPlan | None, ...]]] = []
    for plant in plants:
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
        choices.append((plant, tuple(options)))

    summaries: list[_GenerationSummary] = []

    def backtrack(index: int, remaining: dict[str, int], selected: list[PlantRunPlan]) -> None:
        if index >= len(choices):
            plans = tuple(selected)
            try:
                validated = choose_plants_to_run(state, player_id, plans)
            except ModelValidationError:
                return
            powered = compute_powered_cities(state, player_id, validated)
            spent_units = sum(sum(plan.resource_mix.values()) for plan in validated)
            spent_value = 0.0
            for plan in validated:
                for resource, amount in plan.resource_mix.items():
                    spent_value += amount * (_resource_unit_price(state, resource) + _resource_pressure(state)[resource] * 0.8)
            summaries.append(
                _GenerationSummary(
                    plans=validated,
                    powered=powered,
                    income=pay_income(state.rules, powered),
                    spent_units=spent_units,
                    spent_value=spent_value,
                )
            )
            return

        _, options = choices[index]
        for option in options:
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
    return tuple(summaries)


def _evaluate_relative_state(state: GameState, player_id: str) -> float:
    own = _evaluate_player_strength(state, player_id)
    opponent_threats = _opponent_threats(state, player_id)
    opponent_strengths = [threat.threat_strength for threat in opponent_threats]
    if not opponent_strengths:
        return own
    return own - (0.65 * max(opponent_strengths)) - (0.2 * mean(opponent_strengths))


def _evaluate_relative_state_detail(state: GameState, player_id: str) -> dict[str, object]:
    own_score = _evaluate_player_strength(state, player_id)
    opponent_details = [
        _evaluate_player_strength_detail(state, opponent.player_id)
        for opponent in state.players
        if opponent.player_id != player_id
    ]
    opponent_threats = _opponent_threats(state, player_id)
    opponent_scores = [threat.threat_strength for threat in opponent_threats]
    max_opponent = max(opponent_scores, default=0.0)
    average_opponent = mean(opponent_scores) if opponent_scores else 0.0
    relative_score = own_score - (0.65 * max_opponent) - (0.2 * average_opponent)
    opponent_adjustment = (-0.65 * max_opponent) - (0.2 * average_opponent)
    return {
        "player_id": player_id,
        "stage": _stage_name(state),
        "round_number": state.round_number,
        "step": state.step,
        "phase": state.phase,
        "relative_score": _score(relative_score),
        "opponent_strength_model": "max(current_strength, best_affordable_partial_refuel_projected_strength)",
        "formula": {
            "own_strength_weight": 1.0,
            "max_opponent_strength_weight": -0.65,
            "average_opponent_strength_weight": -0.2,
        },
        "own_strength": _score(own_score),
        "max_opponent_strength": _score(max_opponent),
        "average_opponent_strength": _score(average_opponent),
        "opponent_adjustment": _score(opponent_adjustment),
        "scoreboard": _relative_scoreboard(
            own_score=own_score,
            opponent_threats=opponent_threats,
            max_opponent=max_opponent,
            average_opponent=average_opponent,
            opponent_adjustment=opponent_adjustment,
            relative_score=relative_score,
        ),
        "resource_pressure": _jsonable(_resource_pressure(state)),
        "own": _evaluate_player_strength_detail(state, player_id),
        "opponents": opponent_details,
        "opponent_threats": [_opponent_threat_to_dict(threat) for threat in opponent_threats],
    }


def _evaluate_player_strength(state: GameState, player_id: str) -> float:
    return _evaluate_player_strength_with_overrides(state, player_id)


def _evaluate_player_strength_with_overrides(
    state: GameState,
    player_id: str,
    *,
    generation: _GenerationSummary | None = None,
    cash: int | None = None,
) -> float:
    components = _player_strength_components(
        state,
        player_id,
        generation_override=generation,
        cash_override=cash,
    )
    return float(components["total_score"])


def _player_strength_components(
    state: GameState,
    player_id: str,
    *,
    generation_override: _GenerationSummary | None = None,
    cash_override: int | None = None,
) -> dict[str, object]:
    stage_name = _stage_name(state)
    weights = STAGE_WEIGHTS[stage_name]
    player = _get_player(state, player_id)
    generation = generation_override or _best_generation_summary(state, player_id)
    cash = player.elektro if cash_override is None else int(cash_override)
    portfolio = _portfolio_score(state, player.power_plants)
    frontier = _frontier_score(state, player_id)
    resource_value = _stored_resource_value(state, player_id)
    exposure = _resource_exposure_penalty(state, player_id)
    order_value = len(state.players) - 1 - state.player_order.index(player_id)
    total_output = sum(plant.output_cities for plant in player.power_plants)
    overbuild = max(0, player.connected_city_count - total_output)
    unused_capacity = max(0, total_output - player.connected_city_count)
    trigger_score = _trigger_timing_score(state, player_id, generation, player)

    weighted_terms = {
        "connected": weights.connected * player.connected_city_count,
        "powered": weights.powered * generation.powered,
        "income": weights.income * generation.income,
        "cash": weights.cash * cash,
        "plants": weights.plants * portfolio,
        "frontier": weights.frontier * frontier,
        "resources": weights.resources * resource_value,
        "order": weights.order * order_value,
        "resource_exposure_penalty": -weights.exposure * exposure,
        "overbuild_penalty": -weights.overbuild * overbuild,
        "unused_capacity_penalty": -weights.unused_capacity * unused_capacity,
        "trigger_timing": trigger_score,
    }
    total_score = sum(weighted_terms.values())
    return {
        "player_id": player_id,
        "stage": stage_name,
        "total_score": total_score,
        "weights": weights,
        "weighted_terms": weighted_terms,
        "metrics": {
            "connected_cities": player.connected_city_count,
            "best_generation": generation,
            "cash": cash,
            "plant_portfolio_score": portfolio,
            "frontier_score": frontier,
            "stored_resource_value": resource_value,
            "turn_order_value": order_value,
            "resource_exposure": exposure,
            "overbuild": overbuild,
            "unused_capacity": unused_capacity,
            "total_output": total_output,
            "trigger_timing_score": trigger_score,
            "power_plants": [plant.price for plant in player.power_plants if not plant.is_step_3_placeholder],
            "resource_totals": dict(player.resource_storage.resource_totals()),
        },
    }


def _evaluate_player_strength_detail(state: GameState, player_id: str) -> dict[str, object]:
    components = _player_strength_components(state, player_id)
    metrics = components["metrics"]
    assert isinstance(metrics, dict)
    weights = components["weights"]
    assert isinstance(weights, _StageWeights)
    weighted_terms = components["weighted_terms"]
    assert isinstance(weighted_terms, dict)
    generation = metrics["best_generation"]
    assert isinstance(generation, _GenerationSummary)
    return {
        "player_id": player_id,
        "stage": str(components["stage"]),
        "total_score": _score(float(components["total_score"])),
        "weights": _stage_weights_to_dict(weights),
        "metrics": {
            "connected_cities": metrics["connected_cities"],
            "best_generation": _generation_summary_to_dict(generation),
            "cash": metrics["cash"],
            "plant_portfolio_score": _score(float(metrics["plant_portfolio_score"])),
            "frontier_score": _score(float(metrics["frontier_score"])),
            "stored_resource_value": _score(float(metrics["stored_resource_value"])),
            "turn_order_value": metrics["turn_order_value"],
            "resource_exposure": _score(float(metrics["resource_exposure"])),
            "overbuild": metrics["overbuild"],
            "unused_capacity": metrics["unused_capacity"],
            "total_output": metrics["total_output"],
            "trigger_timing_score": _score(float(metrics["trigger_timing_score"])),
            "power_plants": list(metrics["power_plants"]),
            "resource_totals": dict(metrics["resource_totals"]),
        },
        "weighted_terms": _jsonable(weighted_terms),
        "positive_terms_total": _score(sum(float(value) for value in weighted_terms.values() if float(value) > 0)),
        "penalty_terms_total": _score(sum(float(value) for value in weighted_terms.values() if float(value) < 0)),
    }


def _opponent_threats(state: GameState, player_id: str) -> tuple[_OpponentThreat, ...]:
    return tuple(
        _opponent_refuel_threat(state, opponent.player_id)
        for opponent in state.players
        if opponent.player_id != player_id
    )


def _opponent_refuel_threat(state: GameState, player_id: str) -> _OpponentThreat:
    cache_key = _opponent_threat_cache_key(state, player_id)
    cached = _OPPONENT_THREAT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    current_strength = _evaluate_player_strength(state, player_id)
    current_generation = _best_generation_summary(state, player_id)
    best_option = _best_affordable_refuel_threat_option(state, player_id)
    if best_option is None:
        threat = _OpponentThreat(
            player_id=player_id,
            current_strength=current_strength,
            refuel_projected_strength=current_strength,
            threat_strength=current_strength,
            projected_generation=current_generation,
            refuel_cost=0,
            refuel_basket={},
            threat_applied=False,
            refuel_projection_kind="no_affordable_refuel_improvement",
            evaluated_refuel_baskets=0,
        )
        _cache_opponent_threat(cache_key, threat)
        return threat

    projected_strength, projected_generation, refuel_cost, refuel_basket, evaluated_count = best_option
    threat_applied = projected_strength > current_strength
    threat = _OpponentThreat(
        player_id=player_id,
        current_strength=current_strength,
        refuel_projected_strength=projected_strength,
        threat_strength=max(current_strength, projected_strength),
        projected_generation=projected_generation if threat_applied else current_generation,
        refuel_cost=refuel_cost if threat_applied else 0,
        refuel_basket=dict(refuel_basket) if threat_applied else {},
        threat_applied=threat_applied,
        refuel_projection_kind="best_affordable_partial_refuel" if threat_applied else "no_strength_gain",
        evaluated_refuel_baskets=evaluated_count,
    )
    _cache_opponent_threat(cache_key, threat)
    return threat


def _cache_opponent_threat(cache_key: tuple[object, ...], threat: _OpponentThreat) -> None:
    if len(_OPPONENT_THREAT_CACHE) >= MAX_OPPONENT_THREAT_CACHE_ENTRIES:
        _OPPONENT_THREAT_CACHE.clear()
    _OPPONENT_THREAT_CACHE[cache_key] = threat


def _opponent_threat_cache_key(state: GameState, player_id: str) -> tuple[object, ...]:
    player_fingerprints = []
    for player in state.players:
        player_fingerprints.append(
            (
                player.player_id,
                player.elektro,
                player.connected_city_count,
                player.houses_in_supply,
                tuple(sorted(player.network_city_ids)),
                tuple(plant.price for plant in player.power_plants),
                tuple(sorted(player.resource_storage.resource_totals().items())),
            )
        )
    market_fingerprint = tuple(
        (resource, state.resource_market.available_unit_prices(resource))
        for resource in RESOURCE_TYPES
    )
    return (
        state.game_map.id,
        tuple(sorted(state.selected_regions)),
        len(state.players),
        state.round_number,
        state.step,
        state.phase,
        tuple(state.player_order),
        player_id,
        tuple(player_fingerprints),
        market_fingerprint,
    )


def _best_affordable_refuel_threat_option(
    state: GameState,
    player_id: str,
) -> tuple[float, _GenerationSummary, int, dict[str, int], int] | None:
    caps, fossil_limit = _refuel_projection_caps(state, player_id)
    if not any(amount > 0 for amount in caps.values()):
        return None
    current_generation = _best_generation_summary(state, player_id)
    player = _get_player(state, player_id)
    best: tuple[tuple[float, int, int, int, int, tuple[int, ...]], float, _GenerationSummary, int, dict[str, int]] | None = None
    evaluated_count = 0
    for basket in _candidate_refuel_baskets(state, player_id, caps, fossil_limit=fossil_limit):
        cost = sum(_resource_purchase_cost(state, resource, amount) for resource, amount in basket.items())
        if cost > player.elektro:
            continue
        if not can_store_resources(player, basket):
            continue
        try:
            refueled_state = purchase_resources(state, player_id, basket)
        except ModelValidationError:
            continue
        evaluated_count += 1
        summary = _best_generation_summary(refueled_state, player_id)
        if summary.powered <= current_generation.powered and summary.income <= current_generation.income:
            continue
        refueled_player = _get_player(refueled_state, player_id)
        strength = _evaluate_player_strength_with_overrides(
            refueled_state,
            player_id,
            generation=summary,
            cash=refueled_player.elektro,
        )
        signature = (
            strength,
            summary.powered,
            summary.income,
            -cost,
            -sum(basket.values()),
            tuple(basket.get(resource, 0) for resource in RESOURCE_TYPES),
        )
        if best is None or signature > best[0]:
            best = (signature, strength, summary, cost, dict(basket))
    if best is None:
        return None
    _, strength, summary, cost, basket = best
    return strength, summary, cost, basket, evaluated_count


def _refuel_projection_caps(
    state: GameState,
    player_id: str,
) -> tuple[dict[str, int], int]:
    player = _get_player(state, player_id)
    stored = player.resource_storage.resource_totals()
    fixed_need = {resource: 0 for resource in RESOURCE_TYPES}
    hybrid_cost = 0
    for plant in player.power_plants:
        if plant.is_ecological or plant.is_step_3_placeholder:
            continue
        if plant.is_hybrid:
            hybrid_cost += plant.resource_cost
        else:
            fixed_need[plant.resource_types[0]] += plant.resource_cost

    fixed_deficit = {
        resource: max(0, fixed_need[resource] - stored[resource])
        for resource in RESOURCE_TYPES
    }
    stored_after_fixed_fossil = max(0, stored["coal"] - fixed_need["coal"]) + max(
        0,
        stored["oil"] - fixed_need["oil"],
    )
    hybrid_shortfall = max(0, hybrid_cost - stored_after_fixed_fossil)
    caps = dict(fixed_deficit)
    caps["coal"] += hybrid_shortfall
    caps["oil"] += hybrid_shortfall
    fossil_limit = fixed_deficit["coal"] + fixed_deficit["oil"] + hybrid_shortfall
    for resource in RESOURCE_TYPES:
        caps[resource] = min(caps[resource], len(state.resource_market.available_unit_prices(resource)))
    return caps, fossil_limit


def _candidate_refuel_baskets(
    state: GameState,
    player_id: str,
    caps: dict[str, int],
    *,
    fossil_limit: int,
) -> tuple[dict[str, int], ...]:
    player = _get_player(state, player_id)
    stored = player.resource_storage.resource_totals()
    plant_options: list[tuple[dict[str, int], ...]] = []
    for plant in sorted(player.power_plants, key=lambda item: item.price):
        if plant.is_step_3_placeholder:
            continue
        options: list[dict[str, int]] = [{}]
        if plant.is_ecological:
            options.append({})
        elif plant.is_hybrid:
            for coal in range(plant.resource_cost + 1):
                oil = plant.resource_cost - coal
                options.append(
                    {
                        resource: amount
                        for resource, amount in {"coal": coal, "oil": oil}.items()
                        if amount > 0
                    }
                )
        else:
            options.append({plant.resource_types[0]: plant.resource_cost})
        plant_options.append(tuple(options))

    seen: set[tuple[tuple[str, int], ...]] = set()
    baskets: list[dict[str, int]] = []
    for choices in _cartesian_product(plant_options):
        desired = {resource: 0 for resource in RESOURCE_TYPES}
        for mix in choices:
            for resource, amount in mix.items():
                desired[resource] += amount
        basket = {
            resource: max(0, desired[resource] - stored[resource])
            for resource in RESOURCE_TYPES
        }
        basket = {resource: amount for resource, amount in basket.items() if amount > 0}
        if not basket:
            continue
        if any(amount > int(caps.get(resource, 0)) for resource, amount in basket.items()):
            continue
        if basket.get("coal", 0) + basket.get("oil", 0) > fossil_limit:
            continue
        key = tuple(sorted(basket.items()))
        if key in seen:
            continue
        seen.add(key)
        baskets.append(basket)
    return tuple(baskets)


def _cartesian_product(groups: list[tuple[dict[str, int], ...]]) -> tuple[tuple[dict[str, int], ...], ...]:
    if not groups:
        return ((),)
    result: list[tuple[dict[str, int], ...]] = [()]
    for group in groups:
        result = [(*prefix, option) for prefix in result for option in group]
    return tuple(result)


def _opponent_threat_to_dict(threat: _OpponentThreat) -> dict[str, object]:
    return {
        "player_id": threat.player_id,
        "current_strength": _score(threat.current_strength),
        "refuel_projected_strength": _score(threat.refuel_projected_strength),
        "threat_strength": _score(threat.threat_strength),
        "threat_applied": threat.threat_applied,
        "refuel_cost": threat.refuel_cost,
        "refuel_basket": dict(threat.refuel_basket),
        "refuel_projection_kind": threat.refuel_projection_kind,
        "evaluated_refuel_baskets": threat.evaluated_refuel_baskets,
        "projected_generation": _generation_summary_to_dict(threat.projected_generation),
    }


def _relative_scoreboard(
    *,
    own_score: float,
    opponent_threats: tuple[_OpponentThreat, ...],
    max_opponent: float,
    average_opponent: float,
    opponent_adjustment: float,
    relative_score: float,
) -> dict[str, object]:
    return {
        "own_score": _score(own_score),
        "opponent_scores": [
            {
                "player_id": threat.player_id,
                "current_score": _score(threat.current_strength),
                "refuel_projected_score": _score(threat.refuel_projected_strength),
                "score_for_relative": _score(threat.threat_strength),
                "threat_applied": threat.threat_applied,
            }
            for threat in opponent_threats
        ],
        "max_opponent_score": _score(max_opponent),
        "average_opponent_score": _score(average_opponent),
        "opponent_adjustment": _score(opponent_adjustment),
        "relative_score": _score(relative_score),
    }


def _frontier_score(state: GameState, player_id: str) -> float:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return 0.0
    cheapest = [int(action.payload["total_cost"]) for action in sorted(actions, key=lambda item: int(item.payload["total_cost"]))[:4]]
    return (len(actions) * 0.65) + max(0.0, 18.0 - mean(cheapest))


def _trigger_timing_score(
    state: GameState,
    player_id: str,
    generation: _GenerationSummary,
    player: PlayerState,
) -> float:
    step_2_threshold = state.rules.player_count_rules[len(state.players)]["step_2_cities"]
    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    if player.connected_city_count >= end_threshold:
        own_tuple = (
            generation.powered,
            player.elektro + generation.income,
            player.connected_city_count,
        )
        opponent_tuples = [
            (
                _best_generation_summary(state, opponent.player_id).powered,
                opponent.elektro + _best_generation_summary(state, opponent.player_id).income,
                opponent.connected_city_count,
            )
            for opponent in state.players
            if opponent.player_id != player_id
        ]
        if own_tuple > max(opponent_tuples):
            return 120.0
        return -160.0
    if state.step == 1 and player.connected_city_count >= step_2_threshold:
        best_opponent_power = max(
            _best_generation_summary(state, opponent.player_id).powered
            for opponent in state.players
            if opponent.player_id != player_id
        )
        return 18.0 if generation.powered >= best_opponent_power else -20.0
    return 0.0


def _stage_name(state: GameState) -> str:
    step_2_threshold = state.rules.player_count_rules[len(state.players)]["step_2_cities"]
    end_threshold = state.rules.player_count_rules[len(state.players)]["end_game_cities"]
    highest_connected = max(player.connected_city_count for player in state.players)
    if state.round_number <= 1:
        return "opening"
    if state.step == 3 or highest_connected >= end_threshold - 2:
        return "sprint"
    if state.step == 2:
        return "step2"
    if highest_connected >= step_2_threshold - 1:
        return "pre_step2"
    return "step1"


def _is_sprint_state(state: GameState, player_id: str) -> bool:
    return _stage_name(state) == "sprint"


def _stored_resource_value(state: GameState, player_id: str) -> float:
    player = _get_player(state, player_id)
    totals = player.resource_storage.resource_totals()
    pressure = _resource_pressure(state)
    useful = {resource: 0 for resource in RESOURCE_TYPES}
    for plant in player.power_plants:
        if plant.is_step_3_placeholder or plant.is_ecological:
            continue
        if plant.is_hybrid:
            useful["coal"] += plant.max_storage
            useful["oil"] += plant.max_storage
        else:
            useful[plant.resource_types[0]] += plant.max_storage
    value = 0.0
    for resource, amount in totals.items():
        if amount <= 0:
            continue
        utility = 1.0 if useful[resource] > 0 else 0.2
        value += amount * ((_resource_unit_price(state, resource) * 0.55) + (pressure[resource] * 1.8)) * utility
    return value


def _resource_exposure_penalty(state: GameState, player_id: str) -> float:
    demand = _estimated_round_resource_demand(state, player_id)
    total = sum(demand.values())
    if total <= 0:
        return 0.0
    pressure = _resource_pressure(state)
    penalty = 0.0
    for resource, amount in demand.items():
        if amount <= 0:
            continue
        share = amount / total
        if share <= 0.5:
            continue
        penalty += (share - 0.5) * amount * (1.0 + pressure[resource])
    return penalty


def _resource_pressure(state: GameState) -> dict[str, float]:
    refill_key = f"step_{state.step}"
    refill_table = state.rules.player_count_rules[len(state.players)]["resource_refill"][refill_key]
    pressure: dict[str, float] = {}
    demand_by_resource = {resource: 0.0 for resource in RESOURCE_TYPES}
    for player in state.players:
        player_demand = _estimated_round_resource_demand(state, player.player_id)
        for resource, amount in player_demand.items():
            demand_by_resource[resource] += amount
    for resource in RESOURCE_TYPES:
        available = len(state.resource_market.available_unit_prices(resource))
        refill = int(refill_table[resource])
        pressure[resource] = demand_by_resource[resource] / max(1.0, available + refill)
    return pressure


def _estimated_round_resource_demand(state: GameState, player_id: str) -> dict[str, float]:
    player = _get_player(state, player_id)
    demand = {resource: 0.0 for resource in RESOURCE_TYPES}
    for plant in player.power_plants:
        if plant.is_step_3_placeholder or plant.is_ecological:
            continue
        if plant.is_hybrid:
            coal_price = _resource_unit_price(state, "coal")
            oil_price = _resource_unit_price(state, "oil")
            preferred = "coal" if coal_price <= oil_price else "oil"
            secondary = "oil" if preferred == "coal" else "coal"
            demand[preferred] += plant.resource_cost * 0.7
            demand[secondary] += plant.resource_cost * 0.3
            continue
        demand[plant.resource_types[0]] += plant.resource_cost
    return demand


def _portfolio_score(state: GameState, plants: tuple[PowerPlantCard, ...]) -> float:
    if not plants:
        return 0.0
    values = [_plant_value(state, plant) for plant in plants if not plant.is_step_3_placeholder]
    total_output = sum(plant.output_cities for plant in plants if not plant.is_step_3_placeholder)
    resource_counts = {resource: 0 for resource in RESOURCE_TYPES}
    hybrids = 0
    ecological = 0
    for plant in plants:
        if plant.is_step_3_placeholder:
            continue
        if plant.is_ecological:
            ecological += 1
        elif plant.is_hybrid:
            hybrids += 1
            resource_counts["coal"] += 1
            resource_counts["oil"] += 1
        else:
            resource_counts[plant.resource_types[0]] += 1
    diversity = sum(1 for count in resource_counts.values() if count > 0)
    concentration = max(resource_counts.values(), default=0)
    return (
        sum(values)
        + (total_output * 1.7)
        + (diversity * 1.5)
        + (hybrids * 1.2)
        + (ecological * 1.8)
        - max(0, concentration - 2) * 2.0
    )


def _plant_value(state: GameState, plant: PowerPlantCard) -> float:
    if plant.is_step_3_placeholder:
        return 0.0
    if plant.is_ecological:
        run_cost = 0.0
        pressure_penalty = 0.0
        efficiency = plant.output_cities * 2.5
        flexibility = 3.5
    elif plant.is_hybrid:
        run_cost = _hybrid_run_cost(state, plant.resource_cost)
        pressure = (_resource_pressure(state)["coal"] + _resource_pressure(state)["oil"]) / 2.0
        pressure_penalty = pressure * plant.resource_cost * 0.9
        efficiency = (plant.output_cities / max(1, plant.resource_cost)) * 4.5
        flexibility = 2.8
    else:
        resource = plant.resource_types[0]
        run_cost = _resource_purchase_cost(state, resource, plant.resource_cost)
        pressure_penalty = _resource_pressure(state)[resource] * plant.resource_cost * 1.1
        efficiency = (plant.output_cities / max(1, plant.resource_cost)) * 4.2
        flexibility = 0.0
    return (
        plant.output_cities * 5.5
        + efficiency
        + (plant.max_storage * 0.2)
        + flexibility
        + PLANT_STATIC_ADJUSTMENTS.get(plant.price, 0.0)
        - (run_cost * 0.8)
        - pressure_penalty
    )


def _auction_reserve_projection(
    state: GameState,
    player_id: str,
    plant: PowerPlantCard,
    *,
    minimum_bid: int | None = None,
    active_auction: bool = False,
) -> _AuctionReserveProjection:
    player = _get_player(state, player_id)
    min_bid = minimum_bid if minimum_bid is not None else _auction_minimum_bid(state, plant)
    hard_cap = player.elektro
    if hard_cap < min_bid:
        fallback = _best_auction_fallback_projection(
            state,
            player_id,
            excluded_plant_price=plant.price,
            active_auction=active_auction,
        )
        target = _project_post_auction_economy(
            state,
            player_id,
            plant,
            min_bid,
            projection_kind="post_auction_economy_at_min_bid",
        )
        return _AuctionReserveProjection(
            reserve=0,
            minimum_bid=min_bid,
            hard_cap=hard_cap,
            required_margin=_auction_required_margin(state, player_id),
            fallback=fallback,
            target_at_minimum=target,
            target_at_reserve=target,
            price_samples=(
                _auction_price_sample(
                    min_bid,
                    target,
                    fallback,
                    label="minimum_bid_unaffordable",
                ),
            ),
        )

    fallback = _best_auction_fallback_projection(
        state,
        player_id,
        excluded_plant_price=plant.price,
        active_auction=active_auction,
    )
    required_margin = _auction_required_margin(state, player_id)
    target_at_minimum = _project_post_auction_economy(
        state,
        player_id,
        plant,
        min_bid,
        projection_kind="post_auction_economy_at_min_bid",
    )
    projection_cache: dict[int, _AuctionEconomyProjection] = {min_bid: target_at_minimum}

    def projection_at(price: int) -> _AuctionEconomyProjection:
        cached = projection_cache.get(price)
        if cached is not None:
            return cached
        projection = _project_post_auction_economy(
            state,
            player_id,
            plant,
            price,
            projection_kind="post_auction_economy_at_scanned_price",
        )
        projection_cache[price] = projection
        return projection

    reserve = 0
    target_at_reserve = target_at_minimum
    failed_sample: dict[str, object] | None = None
    evaluated_prices = 0
    scan_stride = _auction_reserve_scan_stride(min_bid, hard_cap)
    failed_price: int | None = None
    price = min_bid
    while price <= hard_cap:
        projection = projection_at(price)
        evaluated_prices += 1
        surplus = projection.score - fallback.score
        if surplus >= required_margin:
            reserve = price
            target_at_reserve = projection
            price += scan_stride
            continue
        failed_price = price
        failed_sample = _auction_price_sample(
            price,
            projection,
            fallback,
            label="first_below_required_margin",
        )
        break

    if scan_stride > 1:
        if failed_price is not None and reserve + 1 < failed_price:
            for price in range(max(min_bid, reserve + 1), failed_price):
                projection = projection_at(price)
                evaluated_prices += 1
                surplus = projection.score - fallback.score
                if surplus >= required_margin:
                    reserve = price
                    target_at_reserve = projection
                    continue
                failed_sample = _auction_price_sample(
                    price,
                    projection,
                    fallback,
                    label="first_below_required_margin",
                )
                break
        elif failed_price is None and reserve < hard_cap:
            for price in range(reserve + 1, hard_cap + 1):
                projection = projection_at(price)
                evaluated_prices += 1
                surplus = projection.score - fallback.score
                if surplus >= required_margin:
                    reserve = price
                    target_at_reserve = projection
                    continue
                failed_sample = _auction_price_sample(
                    price,
                    projection,
                    fallback,
                    label="first_below_required_margin",
                )
                break

    samples = [
        _auction_price_sample(
            min_bid,
            target_at_minimum,
            fallback,
            label="minimum_bid",
        )
    ]
    if reserve >= min_bid:
        samples.append(
            _auction_price_sample(
                reserve,
                target_at_reserve,
                fallback,
                label="reserve",
            )
        )
    if failed_sample is not None:
        samples.append(failed_sample)
    if hard_cap not in {min_bid, reserve} and failed_sample is None:
        hard_cap_projection = target_at_reserve if hard_cap == reserve else projection_at(hard_cap)
        samples.append(
            _auction_price_sample(
                hard_cap,
                hard_cap_projection,
                fallback,
                label="hard_cap",
            )
        )
    samples.append(
        {
            "label": "scan_summary",
            "evaluated_price_count": evaluated_prices,
            "scan_stride": scan_stride,
        }
    )
    return _AuctionReserveProjection(
        reserve=reserve,
        minimum_bid=min_bid,
        hard_cap=hard_cap,
        required_margin=required_margin,
        fallback=fallback,
        target_at_minimum=target_at_minimum,
        target_at_reserve=target_at_reserve,
        price_samples=tuple(samples),
    )


def _auction_reserve(state: GameState, player_id: str, plant: PowerPlantCard) -> int:
    return _auction_reserve_projection(state, player_id, plant).reserve


def _auction_reserve_level0(state: GameState, player_id: str, plant: PowerPlantCard) -> int:
    player = _get_player(state, player_id)
    min_bid = _auction_minimum_bid(state, plant)
    hard_cap = _auction_hard_cap(state, player_id)
    if hard_cap < min_bid:
        return 0
    gain = _plant_purchase_gain(state, player_id, plant)
    stage = _stage_name(state)
    reserve = min_bid + int(max(0.0, gain * (1.6 if stage == "opening" else 1.25)))
    if state.round_number == 1 and not player.power_plants:
        reserve = max(reserve, min_bid)
    if _is_sprint_state(state, player_id) and plant.output_cities >= 4:
        reserve += 2
    return max(0, min(reserve, player.elektro, hard_cap))


def _auction_minimum_bid(state: GameState, plant: PowerPlantCard) -> int:
    if state.auction_state and state.auction_state.discount_token_plant_price == plant.price:
        return 1
    return plant.price


def _auction_required_margin(state: GameState, player_id: str) -> float:
    player = _get_player(state, player_id)
    if state.round_number == 1 and not player.power_plants:
        return 0.0
    if _is_sprint_state(state, player_id):
        return 0.25
    return 0.75


def _auction_reserve_scan_stride(min_bid: int, hard_cap: int) -> int:
    span = hard_cap - min_bid
    if span <= 12:
        return 1
    if span <= 60:
        return 4
    return 8


def _best_auction_fallback_projection(
    state: GameState,
    player_id: str,
    *,
    excluded_plant_price: int | None,
    active_auction: bool,
) -> _AuctionEconomyProjection:
    candidates: list[_AuctionEconomyProjection] = []
    player = _get_player(state, player_id)
    if state.round_number > 1 or player.power_plants:
        candidates.append(
            _project_post_auction_economy(
                state,
                player_id,
                None,
                0,
                projection_kind="auction_pass_fallback",
            )
        )

    fallback_plants = [
        plant
        for plant in list_auctionable_plants(state)
        if plant.price != excluded_plant_price and not plant.is_step_3_placeholder
    ]
    fallback_plants.sort(
        key=lambda plant: (
            _plant_purchase_gain(state, player_id, plant) - (_auction_minimum_bid(state, plant) * 0.2),
            -plant.price,
        ),
        reverse=True,
    )
    for plant in fallback_plants[:6]:
        candidates.append(
            _project_post_auction_economy(
                state,
                player_id,
                plant,
                _auction_minimum_bid(state, plant),
                projection_kind="auction_fallback_purchase_at_min_bid",
            )
        )

    if not candidates:
        candidates.append(
            _project_post_auction_economy(
                state,
                player_id,
                None,
                0,
                projection_kind=(
                    "auction_active_pass_no_visible_fallback"
                    if active_auction
                    else "auction_pass_fallback"
                ),
            )
        )
    return max(
        candidates,
        key=lambda projection: (
            projection.score,
            projection.generation.powered,
            projection.cash_after_income,
            -(projection.purchase_price),
        ),
    )


def _project_post_auction_economy(
    state: GameState,
    player_id: str,
    plant: PowerPlantCard | None,
    price_paid: int,
    *,
    projection_kind: str,
) -> _AuctionEconomyProjection:
    if plant is None:
        purchased_state = state
        plant_price = None
        purchase_price = 0
    else:
        purchased_state = _simulate_hypothetical_purchase(state, player_id, plant, price_paid)
        plant_price = plant.price
        purchase_price = price_paid
    cash_after_purchase = _get_player(purchased_state, player_id).elektro
    return _best_fast_resource_build_generation_projection(
        purchased_state,
        player_id,
        plant_price=plant_price,
        purchase_price=purchase_price,
        cash_after_purchase=cash_after_purchase,
        projection_kind=projection_kind,
    )


def _best_fast_resource_build_generation_projection(
    state: GameState,
    player_id: str,
    *,
    plant_price: int | None,
    purchase_price: int,
    cash_after_purchase: int,
    projection_kind: str,
) -> _AuctionEconomyProjection:
    resource_baskets = [dict()]
    caps, fossil_limit = _refuel_projection_caps(state, player_id)
    if any(amount > 0 for amount in caps.values()):
        resource_baskets.extend(
            _candidate_refuel_baskets(state, player_id, caps, fossil_limit=fossil_limit)
        )

    best_projection: _AuctionEconomyProjection | None = None
    seen: set[tuple[tuple[str, int], ...]] = set()
    for basket in resource_baskets:
        key = tuple(sorted((resource, amount) for resource, amount in basket.items() if amount > 0))
        if key in seen:
            continue
        seen.add(key)
        player = _get_player(state, player_id)
        resource_cost = sum(
            _resource_purchase_cost(state, resource, amount)
            for resource, amount in basket.items()
        )
        if resource_cost > player.elektro:
            continue
        if basket and not can_store_resources(player, basket):
            continue
        try:
            resource_state = purchase_resources(state, player_id, basket) if basket else state
        except ModelValidationError:
            continue
        cash_after_resources = _get_player(resource_state, player_id).elektro
        build_state, build_city_ids, build_cost = _apply_fast_auction_build_plan(
            resource_state,
            player_id,
        )
        cash_after_build = _get_player(build_state, player_id).elektro
        generation = _best_generation_summary(build_state, player_id)
        final_state = _simulate_bureaucracy_summary(build_state, player_id, generation)
        cash_after_income = _get_player(final_state, player_id).elektro
        raw_score = _evaluate_relative_state(final_state, player_id)
        viability_adjustment = _auction_economy_viability_adjustment(
            build_state,
            player_id,
            plant_price=plant_price,
            generation=generation,
        )
        score = raw_score + viability_adjustment
        projection = _AuctionEconomyProjection(
            score=score,
            raw_score=raw_score,
            viability_adjustment=viability_adjustment,
            evaluation_state=final_state,
            plant_price=plant_price,
            purchase_price=purchase_price,
            resource_plan=key,
            resource_cost=resource_cost,
            build_city_ids=tuple(build_city_ids),
            build_cost=build_cost,
            generation=generation,
            cash_after_purchase=cash_after_purchase,
            cash_after_resources=cash_after_resources,
            cash_after_build=cash_after_build,
            cash_after_income=cash_after_income,
            projection_kind=projection_kind,
        )
        signature = (
            projection.score,
            projection.generation.powered,
            projection.generation.income,
            projection.cash_after_income,
            -projection.resource_cost,
            -projection.build_cost,
        )
        if best_projection is None or signature > (
            best_projection.score,
            best_projection.generation.powered,
            best_projection.generation.income,
            best_projection.cash_after_income,
            -best_projection.resource_cost,
            -best_projection.build_cost,
        ):
            best_projection = projection
    assert best_projection is not None
    return best_projection


def _apply_fast_auction_build_plan(state: GameState, player_id: str) -> tuple[GameState, tuple[str, ...], int]:
    simulated = state
    built_city_ids: list[str] = []
    total_cost = 0
    player = _get_player(simulated, player_id)
    total_output = sum(
        plant.output_cities
        for plant in player.power_plants
        if not plant.is_step_3_placeholder
    )
    if total_output <= player.connected_city_count or player.houses_in_supply <= 0:
        return simulated, (), 0

    growth_limit = _auction_projection_build_growth_limit(state, player_id)
    target_city_count = min(
        total_output,
        player.connected_city_count + player.houses_in_supply,
        player.connected_city_count + growth_limit,
    )
    while _get_player(simulated, player_id).connected_city_count < target_city_count:
        actions = legal_build_targets(simulated, player_id)
        if not actions:
            break
        ranked_actions = sorted(
            actions,
            key=lambda action: (
                int(action.payload["total_cost"]),
                str(action.payload["city_id"]),
            ),
        )
        chosen_state = None
        chosen_city_id = ""
        for action in ranked_actions[:8]:
            city_id = str(action.payload["city_id"])
            try:
                candidate_state = apply_builds(simulated, player_id, (city_id,))
            except ModelValidationError:
                continue
            chosen_state = candidate_state
            chosen_city_id = city_id
            break
        if chosen_state is None:
            break
        before_cash = _get_player(simulated, player_id).elektro
        after_cash = _get_player(chosen_state, player_id).elektro
        total_cost += before_cash - after_cash
        built_city_ids.append(chosen_city_id)
        simulated = chosen_state
    return simulated, tuple(built_city_ids), total_cost


def _auction_projection_build_growth_limit(state: GameState, player_id: str) -> int:
    player = _get_player(state, player_id)
    if state.round_number <= 1 and not player.network_city_ids:
        return 1
    if _is_sprint_state(state, player_id):
        return 4
    if state.step >= 2:
        return 3
    return 2


def _auction_economy_viability_adjustment(
    state_after_build: GameState,
    player_id: str,
    *,
    plant_price: int | None,
    generation: _GenerationSummary,
) -> float:
    adjustment = 0.0
    player = _get_player(state_after_build, player_id)
    if plant_price is not None:
        purchased_plant = next(
            (plant for plant in player.power_plants if plant.price == plant_price),
            None,
        )
        if purchased_plant is not None and not purchased_plant.is_step_3_placeholder:
            runs_purchased_plant = any(plan.plant_price == plant_price for plan in generation.plans)
            if not runs_purchased_plant and purchased_plant.output_cities > 0:
                adjustment -= 16.0 + (purchased_plant.output_cities * 8.0)
    if state_after_build.round_number <= 1:
        if player.connected_city_count == 0:
            adjustment -= 80.0
        if generation.powered == 0:
            adjustment -= 50.0
    elif player.connected_city_count > 0 and generation.powered == 0:
        adjustment -= 14.0
    return adjustment


def _auction_price_sample(
    price: int,
    projection: _AuctionEconomyProjection,
    fallback: _AuctionEconomyProjection,
    *,
    label: str,
) -> dict[str, object]:
    return {
        "label": label,
        "price": price,
        "score": _score(projection.score),
        "raw_score": _score(projection.raw_score),
        "viability_adjustment": _score(projection.viability_adjustment),
        "surplus_vs_fallback": _score(projection.score - fallback.score),
        "cash_after_purchase": projection.cash_after_purchase,
        "cash_after_resources": projection.cash_after_resources,
        "cash_after_build": projection.cash_after_build,
        "cash_after_income": projection.cash_after_income,
        "resource_cost": projection.resource_cost,
        "build_cost": projection.build_cost,
        "powered": projection.generation.powered,
        "income": projection.generation.income,
    }


def _auction_projection_to_log(projection: _AuctionEconomyProjection) -> dict[str, object]:
    return {
        "projection_kind": projection.projection_kind,
        "score": _score(projection.score),
        "raw_score": _score(projection.raw_score),
        "viability_adjustment": _score(projection.viability_adjustment),
        "plant_price": projection.plant_price,
        "purchase_price": projection.purchase_price,
        "resource_plan": dict(projection.resource_plan),
        "resource_cost": projection.resource_cost,
        "build_city_ids": list(projection.build_city_ids),
        "build_cost": projection.build_cost,
        "generation": _generation_summary_to_dict(projection.generation),
        "cash_after_purchase": projection.cash_after_purchase,
        "cash_after_resources": projection.cash_after_resources,
        "cash_after_build": projection.cash_after_build,
        "cash_after_income": projection.cash_after_income,
    }


def _auction_hard_cap(state: GameState, player_id: str) -> int:
    player = _get_player(state, player_id)
    build_floor = _recommended_build_cash_floor(state, player_id)
    resource_floor = _resource_purchase_floor(state, player_id)
    floor = build_floor + resource_floor
    if _is_sprint_state(state, player_id):
        floor = max(0, floor - 4)
    return max(0, player.elektro - floor)


def _recommended_build_cash_floor(state: GameState, player_id: str) -> int:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return 0
    player = _get_player(state, player_id)
    powered = _best_generation_summary(state, player_id).powered
    desired = 1 if player.connected_city_count >= powered else 2
    ranked = sorted(actions, key=lambda action: int(action.payload["total_cost"]))[:5]
    best_cost = int(ranked[0].payload["total_cost"])
    if desired <= 1:
        return best_cost
    city_ids = [str(action.payload["city_id"]) for action in ranked]
    for size in range(2, min(desired, len(city_ids)) + 1):
        for combo in combinations(city_ids, size):
            try:
                next_state = apply_builds(_clone_state(state), player_id, combo)
            except ModelValidationError:
                continue
            total_cost = player.elektro - _get_player(next_state, player_id).elektro
            if total_cost > best_cost:
                best_cost = total_cost
    return best_cost


def _quick_build_potential(state: GameState, player_id: str) -> float:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return 0.0
    ranked = sorted(int(action.payload["total_cost"]) for action in actions)[:3]
    return max(0.0, 20.0 - mean(ranked))


def _resource_purchase_floor(state: GameState, player_id: str) -> int:
    deficits = _resource_need_by_type(state, player_id)
    cost = 0
    for resource, amount in deficits.items():
        if amount <= 0:
            continue
        cost += _resource_purchase_cost(state, resource, amount)
    return cost


def _plant_purchase_gain(state: GameState, player_id: str, plant: PowerPlantCard) -> float:
    player = _get_player(state, player_id)
    before = _portfolio_score(state, player.power_plants)
    after_plants = _best_portfolio_after_purchase(state, player, plant)
    after = _portfolio_score(state, after_plants)
    before_output = sum(card.output_cities for card in player.power_plants if not card.is_step_3_placeholder)
    after_output = sum(card.output_cities for card in after_plants if not card.is_step_3_placeholder)
    capacity_bonus = max(0, after_output - before_output) * 1.8
    return after - before + capacity_bonus


def _best_portfolio_after_purchase(
    state: GameState,
    player: PlayerState,
    plant: PowerPlantCard,
) -> tuple[PowerPlantCard, ...]:
    candidates = tuple(sorted((*player.power_plants, plant), key=lambda card: card.price))
    max_plants = state.rules.player_count_rules[len(state.players)]["max_power_plants"]
    if len(candidates) <= max_plants:
        return candidates
    best = max(
        combinations(candidates, max_plants),
        key=lambda combo: (
            _portfolio_score(state, tuple(combo)),
            sum(card.output_cities for card in combo),
            tuple(card.price for card in combo),
        ),
    )
    return tuple(sorted(best, key=lambda card: card.price))


def _simulate_hypothetical_purchase(
    state: GameState,
    player_id: str,
    plant: PowerPlantCard,
    price_paid: int,
) -> GameState:
    simulated = _clone_state(state)
    player = _get_player(simulated, player_id)
    simulated = _replace_player_on_state(
        simulated,
        replace(player, elektro=max(0, player.elektro - price_paid)),
    )
    simulated = add_power_plant_to_player(simulated, player_id, plant.price)
    if simulated.pending_decision is not None and simulated.pending_decision.decision_type == "discard_power_plant":
        best_state = None
        best_score = float("-inf")
        for action in simulated.pending_decision.legal_actions:
            discard_price = int(action.payload["price"])
            candidate = replace_plant_if_needed(_clone_state(simulated), player_id, discard_price)
            score = _evaluate_relative_state(candidate, player_id)
            if score > best_score:
                best_score = score
                best_state = candidate
        assert best_state is not None
        simulated = best_state
    return simulated


def _market_roll_bonus(state: GameState, plant: PowerPlantCard) -> float:
    future_values = [_plant_value(state, candidate) for candidate in state.future_market if candidate.price != plant.price]
    if not future_values:
        return 0.0
    return mean(future_values) * 0.08


def _resource_need_by_type(state: GameState, player_id: str) -> dict[str, int]:
    player = _get_player(state, player_id)
    stored = player.resource_storage.resource_totals()
    needed = {resource: 0 for resource in RESOURCE_TYPES}
    hybrid_cost = 0
    for plant in player.power_plants:
        if plant.is_ecological or plant.is_step_3_placeholder:
            continue
        if plant.is_hybrid:
            hybrid_cost += plant.resource_cost
            continue
        needed[plant.resource_types[0]] += plant.resource_cost
    deficits = {
        resource: max(0, needed[resource] - stored[resource])
        for resource in RESOURCE_TYPES
    }
    hybrid_remaining = max(0, hybrid_cost - (stored["coal"] + stored["oil"]))
    if hybrid_remaining > 0:
        coal_price = _resource_unit_price(state, "coal")
        oil_price = _resource_unit_price(state, "oil")
        preferred = "coal" if coal_price <= oil_price else "oil"
        deficits[preferred] += hybrid_remaining
    return deficits


def _resource_purchase_cost(state: GameState, resource: str, amount: int) -> int:
    prices = state.resource_market.available_unit_prices(resource)
    if amount <= 0:
        return 0
    if len(prices) >= amount:
        return sum(prices[:amount])
    if not prices:
        return amount * 8
    return sum(prices) + ((amount - len(prices)) * max(prices[-1], 8))


def _resource_unit_price(state: GameState, resource: str) -> int:
    prices = state.resource_market.available_unit_prices(resource)
    if not prices:
        return 8
    return int(prices[0])


def _hybrid_run_cost(state: GameState, amount: int) -> float:
    best = None
    coal_prices = state.resource_market.available_unit_prices("coal")
    oil_prices = state.resource_market.available_unit_prices("oil")
    for coal in range(amount + 1):
        oil = amount - coal
        if len(coal_prices) < coal or len(oil_prices) < oil:
            continue
        cost = sum(coal_prices[:coal]) + sum(oil_prices[:oil])
        if best is None or cost < best:
            best = cost
    if best is None:
        return float(amount * 8)
    return float(best)


def _candidate_action_trace(
    *,
    intent_type: str,
    payload: dict[str, Any],
    decision_score: float,
    projected_relative_score: float,
    current_score: float,
    score_terms: dict[str, Any] | None = None,
) -> dict[str, object]:
    terms = _jsonable(score_terms or {})
    assert isinstance(terms, dict)
    projected_kind = terms.get("projected_kind")
    trace = {
        "intent_type": intent_type,
        "intent_payload": _jsonable(payload),
        "decision_score": _score(decision_score),
        "projected_relative_score": _score(projected_relative_score),
        "projected_score_delta": _score(projected_relative_score - current_score),
        "projection_horizon": _projection_horizon(intent_type, projected_kind),
        "score_terms": terms,
    }
    if isinstance(projected_kind, str):
        trace["projected_kind"] = projected_kind
    return trace


def _selected_action_trace(
    intent: GuiIntent,
    *,
    decision_score: float,
    projected_evaluation: dict[str, object],
    current_score: float,
    score_terms: dict[str, Any] | None = None,
) -> dict[str, object]:
    projected_relative_score = float(projected_evaluation["relative_score"])
    trace = _candidate_action_trace(
        intent_type=intent.intent_type,
        payload=dict(intent.payload),
        decision_score=decision_score,
        projected_relative_score=projected_relative_score,
        current_score=current_score,
        score_terms=score_terms,
    )
    trace["projected_evaluation"] = projected_evaluation
    return trace


def _projection_horizon(intent_type: str, projected_kind: object) -> str:
    if projected_kind in {
        "hypothetical_purchase_at_min_bid",
        "hypothetical_purchase_at_price_guess",
    }:
        return "terminal_if_won"
    if projected_kind in {
        "post_auction_economy_at_expected_price",
        "post_auction_economy_at_min_bid",
        "post_auction_economy_at_scanned_price",
        "post_auction_economy_at_hard_cap",
    }:
        return "post_auction_economy"
    if projected_kind in {
        "auction_fallback_purchase_at_min_bid",
        "auction_pass_fallback",
        "auction_active_pass_no_visible_fallback",
    }:
        return "auction_fallback"
    if projected_kind in {
        "current_state_after_pass",
        "bait_start_without_expected_purchase",
    }:
        return "immediate_state"
    if intent_type in {"auction_start", "auction_bid"}:
        return "auction_projection"
    return "immediate_state"


def _rank_logged_candidates(candidates: tuple[dict[str, object], ...]) -> list[dict[str, object]]:
    def sort_score(candidate: dict[str, object]) -> float:
        value = candidate.get("decision_score", candidate.get("projected_relative_score"))
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (sort_score(item[1]), -item[0]),
        reverse=True,
    )
    logged = []
    for rank, (_, candidate) in enumerate(ranked[:MAX_LOGGED_CANDIDATE_ACTIONS], start=1):
        payload = dict(candidate)
        payload["rank"] = rank
        logged.append(payload)
    return logged


def _resource_finish_score_terms(state: GameState, player_id: str) -> dict[str, object]:
    base_score = _evaluate_relative_state(state, player_id)
    build_potential = _quick_build_potential(state, player_id)
    deficits = _resource_need_by_type(state, player_id)
    pressure = _resource_pressure(state)
    shortfall_details = {}
    shortfall_penalty = 0.0
    for resource, amount in deficits.items():
        if amount <= 0:
            continue
        unit_price = _resource_unit_price(state, resource)
        resource_penalty = amount * (unit_price + pressure[resource] * 2.0)
        shortfall_penalty += resource_penalty
        shortfall_details[resource] = {
            "amount": amount,
            "unit_price": unit_price,
            "pressure": pressure[resource],
            "raw_penalty": resource_penalty,
        }
    return {
        "base_relative_score": base_score,
        "quick_build_potential": build_potential,
        "quick_build_bonus": build_potential * 0.6,
        "deficits": deficits,
        "shortfall_details": shortfall_details,
        "shortfall_penalty": shortfall_penalty,
        "weighted_shortfall_penalty": shortfall_penalty * 0.35,
    }


def _stage_weights_to_dict(weights: _StageWeights) -> dict[str, float]:
    return {
        "connected": weights.connected,
        "powered": weights.powered,
        "income": weights.income,
        "cash": weights.cash,
        "plants": weights.plants,
        "frontier": weights.frontier,
        "resources": weights.resources,
        "order": weights.order,
        "exposure": weights.exposure,
        "overbuild": weights.overbuild,
        "unused_capacity": weights.unused_capacity,
    }


def _generation_summary_to_dict(summary: _GenerationSummary) -> dict[str, object]:
    return {
        "plans": [plan.to_dict() for plan in summary.plans],
        "powered": summary.powered,
        "income": summary.income,
        "spent_units": summary.spent_units,
        "spent_value": _score(summary.spent_value),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return _score(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _score(value: float) -> float:
    return round(float(value), 4)


def _can_participate_in_auction(state: GameState, player_id: str) -> bool:
    auction_state = state.auction_state
    assert auction_state is not None
    if player_id in auction_state.players_with_plants:
        return False
    if player_id in auction_state.players_passed_phase:
        return False
    return True


def _get_market_plant(state: GameState, plant_price: int) -> PowerPlantCard:
    for plant in (*state.current_market, *state.future_market):
        if plant.price == plant_price:
            return plant
    raise ModelValidationError(f"power plant {plant_price} is not visible in the market")


def _get_player(state: GameState, player_id: str) -> PlayerState:
    for player in state.players:
        if player.player_id == player_id:
            return player
    raise ModelValidationError(f"unknown player {player_id!r}")


def _replace_player_on_state(state: GameState, updated_player: PlayerState) -> GameState:
    players = tuple(
        updated_player if player.player_id == updated_player.player_id else player
        for player in state.players
    )
    return replace(state, players=players)


def _clone_state(state: GameState) -> GameState:
    return GameState.from_dict(state.to_dict())
