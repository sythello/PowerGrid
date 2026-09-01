from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .ai import BaseAiController, DeterministicAiController, build_ai_controller
from .model import (
    Action,
    BureaucracySummary,
    GameConfig,
    GameState,
    ModelValidationError,
    PowerPlantCard,
    PlantRunPlan,
    ResourceMarket,
    ResourceStorage,
    AuctionState,
    DecisionRequest,
    PlayerState,
    advance_phase,
    apply_builds,
    build_city,
    choose_plants_to_run,
    compute_powered_cities,
    discard_resources_to_fit_storage,
    legal_build_targets,
    legal_resource_purchases,
    list_auctionable_plants,
    make_default_seat_configs,
    pass_auction,
    pay_income,
    purchase_resources,
    raise_bid,
    replace_plant_if_needed,
    resolve_bureaucracy,
    start_auction,
    WinnerResult,
)
from .rules_data import load_power_plants
from .scenarios import build_game_scenario
from .session_types import (
    AnalysisLogWriter,
    GameLogEntry,
    GameSnapshot,
    GuiIntent,
    HumanSeat,
    SeatAgent,
    SessionEvent,
    TurnRequest,
)


class GameSession:
    def __init__(
        self,
        state: GameState,
        seat_agents: dict[str, SeatAgent],
    ) -> None:
        expected = {player.player_id for player in state.players}
        if set(seat_agents) != expected:
            raise ModelValidationError("seat agents must match the active game state's player ids exactly")
        for player in state.players:
            agent = seat_agents[player.player_id]
            if player.controller != "human" and not isinstance(agent, BaseAiController):
                raise ModelValidationError(
                    f"AI-controlled seat {player.player_id} must use a BaseAiController instance"
                )
        self._state = state
        self._seat_agents = dict(seat_agents)
        self._event_log: list[SessionEvent] = []
        self._game_log: list[GameLogEntry] = []
        self._static_log_data = _build_static_log_data(state)
        self._last_round_summary = None
        self._round_summaries: list[BureaucracySummary] = []
        self._winner_result: WinnerResult | None = None
        self._phase_marker: tuple[int, str, int] | None = None
        self._active_index = 0
        self._bureaucracy_choices: dict[str, tuple[PlantRunPlan, ...]] = {}
        self._sync_phase_cursor()
        self._append_game_log(
            source="session",
            event_type="session_start",
            level="info",
            message="Game session created.",
            payload={
                "map_id": self._state.game_map.id,
                "player_count": len(self._state.players),
                "resolved_selected_regions": list(self._state.selected_regions),
            },
            state=self._state,
            include_state_snapshot=True,
        )

    @classmethod
    def new_game(
        cls,
        config: GameConfig,
        seat_agents: dict[str, SeatAgent] | None = None,
    ) -> "GameSession":
        from .model import initialize_game

        initial_state = advance_phase(initialize_game(config, controllers=None))
        return cls(initial_state, seat_agents or default_seat_agents(config))

    @classmethod
    def from_scenario(
        cls,
        name: str,
        *,
        seed: int = 7,
        seat_agents: dict[str, SeatAgent] | None = None,
    ) -> "GameSession":
        state = build_game_scenario(name, seed=seed)
        return cls(state, seat_agents or default_seat_agents(state.config))

    def snapshot(self) -> GameSnapshot:
        return GameSnapshot(
            state=self._state,
            active_request=self.current_request(),
            event_log=tuple(self._event_log),
            last_round_summary=self._last_round_summary,
            winner_result=self._winner_result,
            analysis_log=self._make_analysis_log_writer(),
        )

    def fork(
        self,
        *,
        seat_agents: dict[str, SeatAgent] | None = None,
        include_logs: bool = False,
    ) -> "GameSession":
        """Create an isolated in-memory copy for AI rollouts.

        The phase cursor and pending bureaucracy choices live in the session rather
        than ``GameState`` and therefore must be copied together with the rules
        state. Logs are omitted by default to keep counterfactual rollouts cheap.
        """

        cloned = GameSession(
            GameState.from_dict(self._state.to_dict()),
            seat_agents or dict(self._seat_agents),
        )
        cloned._event_log = deepcopy(self._event_log) if include_logs else []
        cloned._game_log = deepcopy(self._game_log) if include_logs else []
        cloned._static_log_data = deepcopy(self._static_log_data)
        cloned._last_round_summary = deepcopy(self._last_round_summary)
        cloned._round_summaries = deepcopy(self._round_summaries)
        cloned._winner_result = deepcopy(self._winner_result)
        cloned._phase_marker = self._phase_marker
        cloned._active_index = self._active_index
        cloned._bureaucracy_choices = deepcopy(self._bureaucracy_choices)
        return cloned

    def current_request(self) -> TurnRequest | None:
        self._sync_phase_cursor()
        if self._winner_result is not None:
            return None
        if self._state.pending_decision is not None:
            return _build_pending_request(self._state)
        if self._state.phase == "auction":
            return _build_auction_request(self._state)
        if self._state.phase == "buy_resources":
            player_id = self._current_ordered_player(reverse=True)
            return TurnRequest(
                player_id=player_id,
                phase=self._state.phase,
                decision_type="buy_resources",
                prompt=f"Resource buying for {player_id}.",
                legal_actions=(
                    *legal_resource_purchases(self._state, player_id),
                    Action("finish_buying", player_id, {}),
                ),
                metadata={"phase": self._state.phase},
            )
        if self._state.phase == "build_houses":
            player_id = self._current_ordered_player(reverse=True)
            return TurnRequest(
                player_id=player_id,
                phase=self._state.phase,
                decision_type="build_houses",
                prompt=f"Build phase for {player_id}.",
                legal_actions=(
                    *legal_build_targets(self._state, player_id),
                    Action("finish_building", player_id, {}),
                ),
                metadata={"phase": self._state.phase},
            )
        if self._state.phase == "bureaucracy":
            player_id = self._current_ordered_player(reverse=False)
            player = _get_player(self._state, player_id)
            plant_actions = tuple(
                Action(
                    action_type="run_plant",
                    player_id=player_id,
                    payload={
                        "plant_price": plant.price,
                        "resource_types": list(plant.resource_types),
                        "resource_cost": plant.resource_cost,
                        "output_cities": plant.output_cities,
                        "is_hybrid": plant.is_hybrid,
                        "is_ecological": plant.is_ecological,
                    },
                )
                for plant in sorted(player.power_plants, key=lambda item: item.price)
                if not plant.is_step_3_placeholder
            )
            return TurnRequest(
                player_id=player_id,
                phase=self._state.phase,
                decision_type="bureaucracy",
                prompt=f"Bureaucracy selection for {player_id}.",
                legal_actions=(*plant_actions, Action("skip_bureaucracy", player_id, {})),
                metadata={"phase": self._state.phase},
            )
        return None

    def advance_until_blocked(self) -> GameSnapshot:
        while True:
            snapshot, action_applied = self.advance_one_ai_action()
            if not action_applied:
                return snapshot

    def advance_one_ai_action(self) -> tuple[GameSnapshot, bool]:
        while True:
            self._sync_phase_cursor()
            if self._state.phase in {"setup", "determine_order"}:
                before_state = self._state
                self._state = advance_phase(self._state)
                self._append_game_log(
                    source="session",
                    event_type="phase_transition",
                    level="info",
                    message=f"Automatically advanced from {before_state.phase} to {self._state.phase}.",
                    payload={
                        "from_phase": before_state.phase,
                        "to_phase": self._state.phase,
                        "from_round_number": before_state.round_number,
                        "to_round_number": self._state.round_number,
                        "from_step": before_state.step,
                        "to_step": self._state.step,
                    },
                    state=self._state,
                    include_state_snapshot=True,
                )
                self._sync_phase_cursor(force_reset=True)
                continue
            if self._winner_result is not None:
                return self.snapshot(), False
            request = self.current_request()
            if request is None:
                return self.snapshot(), False
            seat = self._seat_agents[request.player_id]
            if isinstance(seat, HumanSeat):
                return self.snapshot(), False
            intent = seat.choose_intent(request, self.snapshot())
            if not self._apply_and_log(intent, auto_generated=True):
                return self.snapshot(), False
            return self.snapshot(), True

    def submit_intent(self, intent: GuiIntent, *, auto_advance: bool = True) -> GameSnapshot:
        self._apply_and_log(intent, auto_generated=False)
        if not auto_advance:
            return self.snapshot()
        return self.advance_until_blocked()

    def _apply_and_log(self, intent: GuiIntent, *, auto_generated: bool) -> bool:
        before_state = self._state
        before_index = self._active_index
        before_summary = self._last_round_summary
        before_winner = self._winner_result
        before_choices = dict(self._bureaucracy_choices)
        try:
            self._apply_intent(intent)
        except (ModelValidationError, ValueError) as exc:
            self._state = before_state
            self._active_index = before_index
            self._last_round_summary = before_summary
            self._winner_result = before_winner
            self._bureaucracy_choices = before_choices
            self._append_session_event(
                level="error",
                message=str(exc),
                player_id=intent.player_id,
                phase=before_state.phase,
                state=before_state,
                event_type="intent_error",
                payload={
                    "auto_generated": auto_generated,
                    "intent": _serialize_intent(intent),
                },
                include_state_snapshot=True,
            )
            return False
        self._append_session_event(
            level="info",
            message=_describe_intent(before_state, self._state, intent, auto_generated=auto_generated),
            player_id=intent.player_id,
            phase=before_state.phase,
            state=self._state,
            event_type="intent_applied",
            payload={
                "auto_generated": auto_generated,
                "intent": _serialize_intent(intent),
            },
            include_state_snapshot=True,
        )
        return True

    def _apply_intent(self, intent: GuiIntent) -> None:
        request = self.current_request()
        if request is None:
            raise ModelValidationError("the game session is not waiting on a player action")
        if intent.player_id != request.player_id:
            raise ModelValidationError(
                f"intent belongs to {intent.player_id}, but the active player is {request.player_id}"
            )
        if self._state.pending_decision is not None:
            self._apply_pending_intent(intent)
            return
        if self._state.phase == "auction":
            self._apply_auction_intent(intent)
            return
        if self._state.phase == "buy_resources":
            self._apply_resource_intent(intent)
            return
        if self._state.phase == "build_houses":
            self._apply_build_intent(intent)
            return
        if self._state.phase == "bureaucracy":
            self._apply_bureaucracy_intent(intent)
            return
        raise ModelValidationError(f"unsupported phase {self._state.phase!r}")

    def _apply_pending_intent(self, intent: GuiIntent) -> None:
        assert self._state.pending_decision is not None
        if self._state.pending_decision.decision_type == "discard_power_plant":
            if intent.intent_type != "discard_power_plant":
                raise ModelValidationError("expected a discard_power_plant intent")
            self._state = replace_plant_if_needed(
                self._state,
                intent.player_id,
                int(intent.payload["plant_price"]),
            )
            return
        if self._state.pending_decision.decision_type == "discard_hybrid_resources":
            if intent.intent_type != "discard_hybrid_resources":
                raise ModelValidationError("expected a discard_hybrid_resources intent")
            self._state = discard_resources_to_fit_storage(
                self._state,
                intent.player_id,
                {
                    "coal": int(intent.payload.get("coal", 0)),
                    "oil": int(intent.payload.get("oil", 0)),
                },
            )
            return
        raise ModelValidationError(
            f"unsupported pending decision {self._state.pending_decision.decision_type!r}"
        )

    def _apply_auction_intent(self, intent: GuiIntent) -> None:
        if intent.intent_type == "auction_start":
            self._state = start_auction(
                self._state,
                intent.player_id,
                int(intent.payload["plant_price"]),
                int(intent.payload["bid"]),
            )
            return
        if intent.intent_type == "auction_bid":
            self._state = raise_bid(self._state, intent.player_id, int(intent.payload["bid"]))
            return
        if intent.intent_type == "auction_pass":
            self._state = pass_auction(self._state, intent.player_id)
            return
        raise ModelValidationError("unsupported auction intent")

    def _apply_resource_intent(self, intent: GuiIntent) -> None:
        if intent.intent_type == "buy_resource":
            self._state = purchase_resources(
                self._state,
                intent.player_id,
                {str(intent.payload["resource"]): int(intent.payload["amount"])},
            )
            return
        if intent.intent_type != "finish_buying":
            raise ModelValidationError("unsupported resource-phase intent")
        self._active_index += 1
        if self._active_index >= len(self._state.player_order):
            self._state = advance_phase(self._state)
            self._sync_phase_cursor(force_reset=True)

    def _apply_build_intent(self, intent: GuiIntent) -> None:
        city_ids = tuple(str(city_id) for city_id in intent.payload.get("city_ids", []))
        if intent.intent_type == "quote_build":
            quoted = GameState.from_dict(self._state.to_dict())
            player_before = _get_player(quoted, intent.player_id)
            quoted = apply_builds(quoted, intent.player_id, city_ids)
            player_after = _get_player(quoted, intent.player_id)
            self._append_session_event(
                level="info",
                message=(
                    f"Quote for {intent.player_id}: cost={player_before.elektro - player_after.elektro} "
                    f"cities={', '.join(city_ids)}"
                ),
                player_id=intent.player_id,
                phase=self._state.phase,
                state=self._state,
                event_type="build_quote",
                payload={
                    "city_ids": list(city_ids),
                    "quoted_cost": player_before.elektro - player_after.elektro,
                },
            )
            return
        if intent.intent_type == "commit_build":
            if len(city_ids) == 1:
                self._state = build_city(self._state, intent.player_id, city_ids[0])
            else:
                self._state = apply_builds(self._state, intent.player_id, city_ids)
            return
        if intent.intent_type != "finish_building":
            raise ModelValidationError("unsupported build-phase intent")
        self._active_index += 1
        if self._active_index >= len(self._state.player_order):
            self._state = advance_phase(self._state)
            self._sync_phase_cursor(force_reset=True)

    def _apply_bureaucracy_intent(self, intent: GuiIntent) -> None:
        if intent.intent_type == "skip_bureaucracy":
            self._bureaucracy_choices[intent.player_id] = ()
        elif intent.intent_type == "run_plants":
            plans = tuple(
                PlantRunPlan.from_dict(payload)
                for payload in intent.payload.get("plans", [])
            )
            validated = choose_plants_to_run(self._state, intent.player_id, plans)
            self._bureaucracy_choices[intent.player_id] = validated
            powered = compute_powered_cities(self._state, intent.player_id, validated)
            self._append_session_event(
                level="info",
                message=(
                    f"{intent.player_id} will power {powered} cities and receive "
                    f"{pay_income(self._state.rules, powered)} Elektro."
                ),
                player_id=intent.player_id,
                phase=self._state.phase,
                state=self._state,
                event_type="bureaucracy_plan",
                payload={
                    "plans": [plan.to_dict() for plan in validated],
                    "powered_cities": powered,
                    "income": pay_income(self._state.rules, powered),
                },
            )
        else:
            raise ModelValidationError("unsupported bureaucracy intent")
        self._active_index += 1
        if self._active_index < len(self._state.player_order):
            return
        self._state, self._last_round_summary = resolve_bureaucracy(
            self._state,
            generation_choices=dict(self._bureaucracy_choices),
        )
        self._round_summaries.append(self._last_round_summary)
        self._winner_result = self._last_round_summary.winner_result
        self._append_game_log(
            source="session",
            event_type="round_summary",
            level="info",
            message=(
                f"Resolved bureaucracy for round {self._state.round_number}."
                if self._winner_result is None
                else "Resolved final bureaucracy and winner."
            ),
            payload=self._last_round_summary.to_dict(),
            state=self._state,
            include_state_snapshot=True,
        )
        self._sync_phase_cursor(force_reset=True)

    def _sync_phase_cursor(self, *, force_reset: bool = False) -> None:
        marker = (self._state.round_number, self._state.phase, self._state.step)
        if force_reset or marker != self._phase_marker:
            self._phase_marker = marker
            self._active_index = 0
            self._bureaucracy_choices = {}

    def _current_ordered_player(self, *, reverse: bool) -> str:
        ordered = tuple(reversed(self._state.player_order)) if reverse else tuple(self._state.player_order)
        if not ordered:
            raise ModelValidationError("player order may not be empty")
        bounded_index = min(self._active_index, len(ordered) - 1)
        return ordered[bounded_index]

    def game_log_entries(self) -> tuple[GameLogEntry, ...]:
        return tuple(self._game_log)

    def game_log_payload(self) -> dict[str, object]:
        active_request = self.current_request()
        return {
            "format_version": 2,
            "state_snapshot_format": "compact_v1",
            "config": self._state.config.to_dict(),
            "static_data": dict(self._static_log_data),
            "final_state": _serialize_compact_state_snapshot(self._state),
            "current_state": _serialize_compact_state_snapshot(self._state),
            "active_request": (
                _serialize_turn_request_compact(active_request) if active_request is not None else None
            ),
            "winner_result": self._winner_result.to_dict() if self._winner_result is not None else None,
            "last_round_summary": (
                self._last_round_summary.to_dict() if self._last_round_summary is not None else None
            ),
            "round_summaries": [summary.to_dict() for summary in self._round_summaries],
            "event_log": [event.to_dict() for event in self._event_log],
            "game_log": [entry.to_dict() for entry in self._game_log],
        }

    def dump_game_log(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.game_log_payload(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return output_path

    def _make_analysis_log_writer(self) -> AnalysisLogWriter:
        return AnalysisLogWriter(self._record_ai_analysis_log)

    def _record_ai_analysis_log(
        self,
        *,
        source: str,
        event_type: str,
        level: str,
        message: str,
        player_id: str | None,
        phase: str | None,
        payload: dict[str, object],
        include_state_snapshot: bool,
    ) -> None:
        self._append_game_log(
            source=source,
            event_type=event_type,
            level=level,
            message=message,
                player_id=player_id,
                phase=phase,
                payload=payload,
                state=self._state,
                include_state_snapshot=include_state_snapshot,
        )

    def _append_session_event(
        self,
        *,
        level: str,
        message: str,
        player_id: str | None = None,
        phase: str | None = None,
        state: GameState | None = None,
        event_type: str = "session_event",
        payload: dict[str, object] | None = None,
        include_state_snapshot: bool = False,
    ) -> None:
        reference_state = state or self._state
        event = SessionEvent(
            level=level,
            message=message,
            player_id=player_id,
            phase=phase if phase is not None else reference_state.phase,
            event_type=event_type,
            round_number=reference_state.round_number,
            step=reference_state.step,
            payload=dict(payload or {}),
        )
        self._event_log.append(event)
        self._append_game_log(
            source="session",
            event_type=event.event_type,
            level=event.level,
            message=event.message,
                player_id=event.player_id,
                phase=event.phase,
                payload=event.payload,
                state=reference_state,
                include_state_snapshot=include_state_snapshot,
        )

    def _append_game_log(
        self,
        *,
        source: str,
        event_type: str,
        level: str,
        message: str,
        player_id: str | None = None,
        phase: str | None = None,
        payload: dict[str, object] | None = None,
        state: GameState | None = None,
        include_state_snapshot: bool = False,
    ) -> None:
        reference_state = state or self._state
        self._game_log.append(
            GameLogEntry(
                index=len(self._game_log),
                source=source,
                event_type=event_type,
                level=level,
                message=message,
                player_id=player_id,
                phase=phase if phase is not None else reference_state.phase,
                round_number=reference_state.round_number,
                step=reference_state.step,
                payload=dict(payload or {}),
                state_snapshot=(
                    _serialize_compact_state_snapshot(reference_state)
                    if include_state_snapshot
                    else None
                ),
            )
        )


def default_seat_agents(config: GameConfig) -> dict[str, SeatAgent]:
    agents: dict[str, SeatAgent] = {}
    for seat in config.players:
        agents[seat.player_id] = (
            HumanSeat() if seat.controller == "human" else build_ai_controller(seat.controller)
        )
    return agents


DeterministicAiSeat = DeterministicAiController


def default_game_config(
    *,
    player_count: int = 3,
    ai_players: int = 0,
    map_id: str = "germany",
    seed: int = 7,
) -> GameConfig:
    return GameConfig(
        map_id=map_id,
        players=make_default_seat_configs(player_count, ai_players=ai_players),
        seed=seed,
    )


def _build_pending_request(state: GameState) -> TurnRequest:
    assert state.pending_decision is not None
    prompt = (
        f"{state.pending_decision.player_id} must discard coal/oil resources to fit the remaining plants."
        if state.pending_decision.decision_type == "discard_hybrid_resources"
        else f"{state.pending_decision.player_id} must discard a power plant."
    )
    return TurnRequest(
        player_id=state.pending_decision.player_id,
        phase=state.phase,
        decision_type=state.pending_decision.decision_type,
        prompt=prompt,
        legal_actions=state.pending_decision.legal_actions,
        metadata=dict(state.pending_decision.metadata),
    )


def _build_auction_request(state: GameState) -> TurnRequest:
    auction_state = state.auction_state
    if auction_state is None:
        raise ModelValidationError("auction state is missing")
    if auction_state.has_active_auction:
        player_id = str(auction_state.next_bidder_id)
        player = _get_player(state, player_id)
        return TurnRequest(
            player_id=player_id,
            phase=state.phase,
            decision_type="auction_bid",
            prompt=(
                f"Active bidder: {player_id}. "
                f"Plant {auction_state.active_plant_price}, current bid {auction_state.current_bid}."
            ),
            legal_actions=(
                Action(
                    action_type="auction_bid",
                    player_id=player_id,
                    payload={
                        "plant_price": int(auction_state.active_plant_price),
                        "min_bid": int(auction_state.current_bid) + 1,
                        "max_bid": player.elektro,
                    },
                ),
                Action("auction_pass", player_id, {}),
            ),
            metadata={"phase": state.phase},
        )
    player_id = str(auction_state.current_chooser_id)
    player = _get_player(state, player_id)
    actions = []
    for plant in list_auctionable_plants(state):
        minimum_bid = (
            1
            if auction_state.discount_token_plant_price == plant.price
            else plant.price
        )
        if minimum_bid <= player.elektro:
            actions.append(
                Action(
                    action_type="auction_start",
                    player_id=player_id,
                    payload={
                        "plant_price": plant.price,
                        "min_bid": minimum_bid,
                        "max_bid": player.elektro,
                    },
                )
            )
    if state.round_number > 1:
        actions.append(Action("auction_pass", player_id, {}))
    return TurnRequest(
        player_id=player_id,
        phase=state.phase,
        decision_type="auction_start",
        prompt=f"Chooser: {player_id}.",
        legal_actions=tuple(actions),
        metadata={"phase": state.phase},
    )


def _get_player(state: GameState, player_id: str):
    for player in state.players:
        if player.player_id == player_id:
            return player
    raise ModelValidationError(f"unknown player {player_id!r}")


def _describe_intent(
    before_state: GameState,
    after_state: GameState,
    intent: GuiIntent,
    *,
    auto_generated: bool,
) -> str:
    def actor_message(body: str) -> str:
        return f"AI {body}" if auto_generated else body[:1].upper() + body[1:]

    if intent.intent_type == "auction_start":
        return actor_message(
            f"opened bidding for plant {intent.payload['plant_price']} "
            f"at {intent.payload['bid']} Elektro."
        )
    if intent.intent_type == "auction_bid":
        return actor_message(
            f"raised the bid to {intent.payload['bid']} Elektro"
            + _describe_active_auction_suffix(before_state)
        )
    if intent.intent_type == "auction_pass":
        return actor_message(_describe_auction_pass(before_state))
    if intent.intent_type == "buy_resource":
        resource = str(intent.payload["resource"])
        amount = int(intent.payload["amount"])
        total_cost = before_state.resource_market.quote_purchase_cost(resource, amount)
        return actor_message(f"bought {amount} {resource} for {total_cost} Elektro.")
    if intent.intent_type == "finish_buying":
        return actor_message("finished buying resources.")
    if intent.intent_type == "quote_build":
        city_names = _city_labels(before_state, intent.payload.get("city_ids", []))
        return f"Requested a build quote for {', '.join(city_names)}."
    if intent.intent_type == "commit_build":
        city_names = _city_labels(before_state, intent.payload.get("city_ids", []))
        before_player = _get_player(before_state, intent.player_id)
        after_player = _get_player(after_state, intent.player_id)
        total_cost = before_player.elektro - after_player.elektro
        return actor_message(
            f"built in {', '.join(city_names)} for {total_cost} Elektro."
        )
    if intent.intent_type == "finish_building":
        return actor_message("finished building.")
    if intent.intent_type == "run_plants":
        plans = tuple(
            PlantRunPlan.from_dict(payload)
            for payload in intent.payload.get("plans", [])
        )
        if not plans:
            return actor_message("selected no plants to run.")
        powered = compute_powered_cities(before_state, intent.player_id, plans)
        income = pay_income(before_state.rules, powered)
        prices = ", ".join(str(plan.plant_price) for plan in plans)
        return actor_message(
            f"selected plants {prices} to power {powered} cities "
            f"for {income} Elektro."
        )
    if intent.intent_type == "skip_bureaucracy":
        return actor_message("skipped plant operation selection.")
    if intent.intent_type == "discard_power_plant":
        return actor_message(f"discarded plant {intent.payload['plant_price']}.")
    if intent.intent_type == "discard_hybrid_resources":
        return actor_message(
            f"discarded {intent.payload.get('coal', 0)} coal and "
            f"{intent.payload.get('oil', 0)} oil to fit storage."
        )
    return actor_message(f"took action {intent.intent_type}.")


def _describe_active_auction_suffix(state: GameState) -> str:
    auction_state = state.auction_state
    if auction_state is None or auction_state.active_plant_price is None:
        return "."
    return f" for plant {auction_state.active_plant_price}."


def _describe_auction_pass(state: GameState) -> str:
    auction_state = state.auction_state
    if auction_state is None or not auction_state.has_active_auction:
        return "passed on starting an auction this round."
    return (
        f"passed on bidding for plant {auction_state.active_plant_price} "
        f"at {auction_state.current_bid} Elektro."
    )


def _city_labels(state: GameState, city_ids) -> list[str]:
    labels = []
    names = {city.id: city.name for city in state.game_map.cities}
    for city_id in city_ids:
        city_key = str(city_id)
        labels.append(names.get(city_key, city_key))
    return labels


def _serialize_intent(intent: GuiIntent) -> dict[str, object]:
    return intent.to_dict()


def _build_static_log_data(state: GameState) -> dict[str, object]:
    return {
        "game_map": _serialize_game_map_for_log(state),
        "rules": _serialize_rules_for_log(state),
        "resolved_selected_regions": list(state.selected_regions),
        "players": [
            {
                "player_id": player.player_id,
                "name": player.name,
                "controller": player.controller,
                "color": player.color,
            }
            for player in state.players
        ],
        "power_plant_catalog": _serialize_power_plant_catalog(),
    }


def _serialize_power_plant_catalog() -> dict[str, dict[str, object]]:
    catalog = {
        str(definition.price): {
            "price": definition.price,
            "resource_types": list(definition.resource_types),
            "resource_cost": definition.resource_cost,
            "output_cities": definition.output_cities,
            "deck_back": definition.deck_back,
            "is_hybrid": definition.is_hybrid,
            "is_ecological": definition.is_ecological,
        }
        for definition in load_power_plants()
    }
    placeholder = PowerPlantCard.step_3_placeholder()
    catalog[str(placeholder.price)] = placeholder.to_dict()
    return catalog


def _serialize_game_map_for_log(state: GameState) -> dict[str, object]:
    return {
        "id": state.game_map.id,
        "name": state.game_map.name,
        "regions": [
            {"id": region.id, "label": region.label, "color": region.color}
            for region in state.game_map.regions
        ],
        "cities": [
            {"id": city.id, "name": city.name, "region": city.region}
            for city in state.game_map.cities
        ],
        "connections": [
            {"city_1": connection.city_1, "city_2": connection.city_2, "cost": connection.cost}
            for connection in state.game_map.connections
        ],
        "region_adjacency": {
            region: list(neighbors) for region, neighbors in state.game_map.region_adjacency.items()
        },
        "special_rules": list(state.game_map.special_rules),
    }


def _serialize_rules_for_log(state: GameState) -> dict[str, object]:
    return {
        "starting_money": state.rules.starting_money,
        "houses_per_player": state.rules.houses_per_player,
        "resource_supply": dict(state.rules.resource_supply),
        "resource_market_tracks": state.rules.resource_market_tracks,
        "payment_schedule": {
            str(key): value for key, value in state.rules.payment_schedule.items()
        },
        "player_count_rules": {
            str(key): value for key, value in state.rules.player_count_rules.items()
        },
        "setup": state.rules.setup,
    }


def _serialize_compact_state_snapshot(state: GameState) -> dict[str, object]:
    return {
        "players": [_serialize_player_state_compact(player) for player in state.players],
        "player_order": list(state.player_order),
        "resource_market": _serialize_resource_market_compact(state.resource_market),
        "current_market_prices": [plant.price for plant in state.current_market],
        "future_market_prices": [plant.price for plant in state.future_market],
        "power_plant_draw_stack_prices": [plant.price for plant in state.power_plant_draw_stack],
        "power_plant_bottom_stack_prices": [plant.price for plant in state.power_plant_bottom_stack],
        "step_3_card_pending": state.step_3_card_pending,
        "auction_step_3_pending": state.auction_step_3_pending,
        "round_number": state.round_number,
        "step": state.step,
        "phase": state.phase,
        "auction_state": _serialize_auction_state_compact(state.auction_state),
        "pending_decision": _serialize_decision_request_compact(state.pending_decision),
        "last_powered_cities": dict(state.last_powered_cities),
        "last_income_paid": dict(state.last_income_paid),
    }


def _serialize_player_state_compact(player: PlayerState) -> dict[str, object]:
    return {
        "player_id": player.player_id,
        "elektro": player.elektro,
        "houses_in_supply": player.houses_in_supply,
        "network_city_ids": list(player.network_city_ids),
        "power_plant_prices": [plant.price for plant in player.power_plants],
        "resource_storage": _serialize_resource_storage_compact(player.resource_storage),
        "turn_order_position": player.turn_order_position,
    }


def _serialize_resource_storage_compact(storage: ResourceStorage) -> dict[str, int]:
    values = storage.to_dict()
    return {name: amount for name, amount in values.items() if amount}


def _serialize_resource_market_compact(resource_market: ResourceMarket) -> dict[str, object]:
    return {
        "market": {
            resource: {
                str(price): amount
                for price, amount in price_bands.items()
                if amount
            }
            for resource, price_bands in resource_market.market.items()
        },
        "supply": dict(resource_market.supply),
    }


def _serialize_auction_state_compact(auction_state: AuctionState | None) -> dict[str, object] | None:
    if auction_state is None:
        return None
    return auction_state.to_dict()


def _serialize_decision_request_compact(
    decision: DecisionRequest | None,
) -> dict[str, object] | None:
    if decision is None:
        return None
    return {
        "player_id": decision.player_id,
        "decision_type": decision.decision_type,
        "prompt": decision.prompt,
        "legal_actions": [
            _serialize_action_compact(action)
            for action in decision.legal_actions
        ],
        "metadata": dict(decision.metadata),
    }


def _serialize_turn_request_compact(request: TurnRequest) -> dict[str, object]:
    return {
        "player_id": request.player_id,
        "phase": request.phase,
        "decision_type": request.decision_type,
        "prompt": request.prompt,
        "legal_actions": [_serialize_action_compact(action) for action in request.legal_actions],
        "metadata": dict(request.metadata),
    }


def _serialize_action_compact(action: Action) -> dict[str, object]:
    payload = dict(action.payload)
    if action.action_type == "run_plant" and "plant_price" in payload:
        payload = {"plant_price": int(payload["plant_price"])}
    return {
        "action_type": action.action_type,
        "player_id": action.player_id,
        "payload": payload,
    }
