from __future__ import annotations

from .ai import BaseAiController, DeterministicAiController, build_ai_controller
from .model import (
    Action,
    GameConfig,
    GameState,
    ModelValidationError,
    PlantRunPlan,
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
from .scenarios import build_game_scenario
from .session_types import (
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
            if player.controller == "ai" and not isinstance(agent, BaseAiController):
                raise ModelValidationError(
                    f"AI-controlled seat {player.player_id} must use a BaseAiController instance"
                )
        self._state = state
        self._seat_agents = dict(seat_agents)
        self._event_log: list[SessionEvent] = []
        self._last_round_summary = None
        self._winner_result: WinnerResult | None = None
        self._phase_marker: tuple[int, str, int] | None = None
        self._active_index = 0
        self._bureaucracy_choices: dict[str, tuple[PlantRunPlan, ...]] = {}
        self._sync_phase_cursor()

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
        )

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
            self._sync_phase_cursor()
            if self._state.phase in {"setup", "determine_order"}:
                self._state = advance_phase(self._state)
                self._sync_phase_cursor(force_reset=True)
                continue
            if self._winner_result is not None:
                return self.snapshot()
            request = self.current_request()
            if request is None:
                return self.snapshot()
            seat = self._seat_agents[request.player_id]
            if isinstance(seat, HumanSeat):
                return self.snapshot()
            intent = seat.choose_intent(request, self.snapshot())
            if not self._apply_and_log(intent, auto_generated=True):
                return self.snapshot()

    def submit_intent(self, intent: GuiIntent) -> GameSnapshot:
        self._apply_and_log(intent, auto_generated=False)
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
            self._event_log.append(
                SessionEvent(
                    level="error",
                    message=str(exc),
                    player_id=intent.player_id,
                    phase=before_state.phase,
                )
            )
            return False
        descriptor = "AI accepted" if auto_generated else "Accepted"
        self._event_log.append(
            SessionEvent(
                level="info",
                message=f"{descriptor}: {intent.intent_type}",
                player_id=intent.player_id,
                phase=self._state.phase,
            )
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
            self._event_log.append(
                SessionEvent(
                    level="info",
                    message=(
                        f"Quote for {intent.player_id}: cost={player_before.elektro - player_after.elektro} "
                        f"cities={', '.join(city_ids)}"
                    ),
                    player_id=intent.player_id,
                    phase=self._state.phase,
                )
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
            self._event_log.append(
                SessionEvent(
                    level="info",
                    message=(
                        f"{intent.player_id} will power {powered} cities and receive "
                        f"{pay_income(self._state.rules, powered)} Elektro."
                    ),
                    player_id=intent.player_id,
                    phase=self._state.phase,
                )
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
        self._winner_result = self._last_round_summary.winner_result
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
