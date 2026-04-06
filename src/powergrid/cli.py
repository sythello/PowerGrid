from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .model import (
    BureaucracySummary,
    DecisionRequest,
    GameState,
    ModelValidationError,
    PlantRunPlan,
    PowerPlantCard,
    advance_phase,
    apply_builds,
    build_city,
    choose_plants_to_run,
    compute_powered_cities,
    legal_build_targets,
    legal_resource_purchases,
    list_auctionable_plants,
    pass_auction,
    pay_income,
    purchase_resources,
    raise_bid,
    replace_plant_if_needed,
    resolve_bureaucracy,
    start_auction,
    WinnerResult,
)


OutputFn = Callable[[str], None]
InputFn = Callable[[str], str]
StopCondition = Callable[[GameState], bool]


@dataclass(frozen=True)
class PhaseTraceEntry:
    round_number: int
    phase: str
    step: int


@dataclass
class GameRunResult:
    final_state: GameState
    phase_history: tuple[PhaseTraceEntry, ...]
    round_summaries: tuple[BureaucracySummary, ...]
    winner_result: WinnerResult | None = None
    quit_requested: bool = False


class CommandController:
    interactive = False

    def choose_command(self, request: DecisionRequest) -> str:
        raise NotImplementedError


@dataclass
class CLIController(CommandController):
    player_id: str
    input_fn: InputFn = input
    output_fn: OutputFn = print
    interactive = True

    def choose_command(self, request: DecisionRequest) -> str:
        return prompt_decision(request, input_fn=self.input_fn, output_fn=self.output_fn)


@dataclass
class ScriptedController(CommandController):
    player_id: str
    commands: list[str] = field(default_factory=list)
    fallback_command: Callable[[DecisionRequest], str] | None = None
    interactive = False

    def choose_command(self, request: DecisionRequest) -> str:
        if self.commands:
            return self.commands.pop(0)
        if self.fallback_command is not None:
            return self.fallback_command(request)
        raise ModelValidationError(
            f"scripted controller for {self.player_id} has no command for {request.decision_type}"
        )


def prompt_decision(
    request: DecisionRequest,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> str:
    output_fn(request.prompt)
    return input_fn("> ").strip()


def render_game_state(state: GameState, *, active_player_id: str | None = None) -> str:
    lines = [
        f"Round {state.round_number} | Phase={state.phase} | Step={state.step}",
        "Turn order: " + ", ".join(state.player_order),
        "Selected regions: " + (", ".join(state.selected_regions) if state.selected_regions else "(all)"),
    ]
    if active_player_id is not None:
        lines.append(f"Active player: {active_player_id}")
    lines.append("Current market: " + _format_market(state.current_market))
    lines.append(
        "Future market: " + (_format_market(state.future_market) if state.future_market else "(none)")
    )
    lines.append(
        "Resource market: "
        + ", ".join(
            f"{resource}={state.resource_market.total_in_market(resource)}"
            for resource in state.resource_market.market
        )
    )
    lines.append(
        "Supply: "
        + ", ".join(
            f"{resource}={state.resource_market.supply[resource]}"
            for resource in state.resource_market.supply
        )
    )
    lines.append(
        "Deck: "
        f"draw={len(state.power_plant_draw_stack)} "
        f"bottom={len(state.power_plant_bottom_stack)} "
        f"step3_pending={state.step_3_card_pending} "
        f"auction_step3_pending={state.auction_step_3_pending}"
    )
    if state.auction_state is not None:
        lines.append(
            "Auction: "
            f"chooser={state.auction_state.current_chooser_id or '(none)'} "
            f"discount={state.auction_state.discount_token_plant_price or '(off)'} "
            f"buyers={','.join(state.auction_state.players_with_plants) or '(none)'} "
            f"phase_passes={','.join(state.auction_state.players_passed_phase) or '(none)'}"
        )
        if state.auction_state.has_active_auction:
            lines.append(
                "Active auction: "
                f"plant={state.auction_state.active_plant_price} "
                f"bid={state.auction_state.current_bid} "
                f"leader={state.auction_state.highest_bidder_id} "
                f"next={state.auction_state.next_bidder_id}"
            )
    if state.last_powered_cities:
        lines.append(
            "Last powered: " + ", ".join(
                f"{player_id}={value}" for player_id, value in sorted(state.last_powered_cities.items())
            )
        )
    if state.last_income_paid:
        lines.append(
            "Last income: " + ", ".join(
                f"{player_id}={value}" for player_id, value in sorted(state.last_income_paid.items())
            )
        )
    lines.append("Players:")
    for player in sorted(state.players, key=lambda item: item.turn_order_position):
        lines.append(
            "  "
            f"{player.turn_order_position}. {player.player_id} {player.name} "
            f"elektro={player.elektro} houses={player.houses_in_supply} "
            f"cities={player.connected_city_count} "
            f"plants=[{_format_player_plants(player.power_plants)}] "
            f"storage=[{_format_storage(player)}]"
        )
    if state.pending_decision is not None:
        lines.append(
            f"Pending decision: {state.pending_decision.decision_type} for {state.pending_decision.player_id}"
        )
    return "\n".join(lines)


def render_round_summary(summary: BureaucracySummary, state: GameState) -> str:
    lines = [
        "Round Summary",
        "Powered cities: "
        + ", ".join(f"{player_id}={value}" for player_id, value in summary.powered_cities.items()),
        "Income paid: "
        + ", ".join(f"{player_id}={value}" for player_id, value in summary.income_paid.items()),
        f"Triggered Step 2: {summary.triggered_step_2}",
        f"Triggered Step 3: {summary.triggered_step_3}",
        f"Refill step used: {summary.refill_step_used}",
        f"Game end triggered: {summary.game_end_triggered}",
    ]
    if summary.winner_result is not None:
        lines.append("Winner: " + ", ".join(summary.winner_result.winner_ids))
    lines.append(
        f"Next state: round={state.round_number} phase={state.phase} step={state.step}"
    )
    return "\n".join(lines)


def run_game(
    state: GameState,
    controllers: dict[str, CommandController],
    *,
    output_fn: OutputFn | None = print,
    render_state: bool = True,
    stop_condition: StopCondition | None = None,
) -> GameRunResult:
    _validate_controllers(state, controllers)
    emit = output_fn if output_fn is not None else (lambda _: None)
    phase_history: list[PhaseTraceEntry] = [_phase_trace_entry(state)]
    round_summaries: list[BureaucracySummary] = []
    winner_result: WinnerResult | None = None
    quit_requested = False

    if render_state:
        _emit_phase_start(emit, state)
        emit(render_game_state(state))

    while True:
        if stop_condition is not None and stop_condition(state):
            break
        if state.phase in {"setup", "determine_order"}:
            state = advance_phase(state)
            _append_phase_history(phase_history, state)
            if render_state:
                _emit_phase_start(emit, state)
                emit(render_game_state(state))
            continue
        if state.phase == "auction":
            state, quit_requested = _run_auction_phase(
                state,
                controllers,
                output_fn=emit,
                render_state=render_state,
            )
            if quit_requested:
                break
            _append_phase_history(phase_history, state)
            if render_state:
                _emit_phase_start(emit, state)
                emit(render_game_state(state))
            continue
        if state.phase == "buy_resources":
            state, quit_requested = _run_resource_phase(
                state,
                controllers,
                output_fn=emit,
                render_state=render_state,
            )
            if quit_requested:
                break
            state = advance_phase(state)
            _append_phase_history(phase_history, state)
            if render_state:
                _emit_phase_start(emit, state)
                emit(render_game_state(state))
            continue
        if state.phase == "build_houses":
            state, quit_requested = _run_build_phase(
                state,
                controllers,
                output_fn=emit,
                render_state=render_state,
            )
            if quit_requested:
                break
            state = advance_phase(state)
            _append_phase_history(phase_history, state)
            if render_state:
                _emit_phase_start(emit, state)
                emit(render_game_state(state))
            continue
        if state.phase == "bureaucracy":
            state, summary, quit_requested = _run_bureaucracy_phase(
                state,
                controllers,
                output_fn=emit,
            )
            if quit_requested:
                break
            round_summaries.append(summary)
            _append_phase_history(phase_history, state)
            if render_state:
                emit(render_round_summary(summary, state))
                if not summary.game_end_triggered:
                    _emit_phase_start(emit, state)
                emit(render_game_state(state))
            if summary.game_end_triggered:
                winner_result = summary.winner_result
                break
            continue
        raise ModelValidationError(f"unsupported game phase {state.phase!r}")

    return GameRunResult(
        final_state=state,
        phase_history=tuple(phase_history),
        round_summaries=tuple(round_summaries),
        winner_result=winner_result,
        quit_requested=quit_requested,
    )


def _run_auction_phase(
    state: GameState,
    controllers: dict[str, CommandController],
    *,
    output_fn: OutputFn,
    render_state: bool,
) -> tuple[GameState, bool]:
    while state.phase == "auction":
        request = _build_auction_request(state)
        controller = controllers[request.player_id]
        command = controller.choose_command(request).strip()
        if not command:
            if getattr(controller, "interactive", False):
                continue
            raise ModelValidationError(f"{request.player_id} returned an empty auction command")
        lowered = command.lower()
        if lowered == "quit":
            return state, True
        if lowered == "status":
            output_fn(render_game_state(state, active_player_id=request.player_id))
            continue
        if lowered == "help":
            output_fn(_auction_help_text(state))
            continue
        if lowered == "options":
            output_fn(_auction_options_text(state))
            continue
        prior_state = state
        try:
            state = _apply_auction_command(state, request.player_id, command)
        except (ModelValidationError, ValueError) as exc:
            if getattr(controller, "interactive", False):
                output_fn(f"Rejected: {exc}")
                continue
            raise ModelValidationError(f"{request.player_id} auction command {command!r} rejected: {exc}") from exc
        output_fn("Accepted.")
        if render_state:
            output_fn(render_game_state(state, active_player_id=request.player_id))
        if _should_insert_auction_separator(prior_state, state, lowered):
            output_fn("")
    return state, False


def _run_resource_phase(
    state: GameState,
    controllers: dict[str, CommandController],
    *,
    output_fn: OutputFn,
    render_state: bool,
) -> tuple[GameState, bool]:
    buy_order = tuple(reversed(state.player_order))
    active_index = 0
    while active_index < len(buy_order):
        player_id = buy_order[active_index]
        request = DecisionRequest(
            player_id=player_id,
            decision_type="buy_resources",
            prompt=(
                f"Resource buying for {player_id}. "
                "Use: options, buy <resource> <amount>, done, status, help, quit"
            ),
            metadata={"phase": state.phase},
        )
        controller = controllers[player_id]
        command = controller.choose_command(request).strip()
        if not command:
            if getattr(controller, "interactive", False):
                continue
            raise ModelValidationError(f"{player_id} returned an empty resource-buying command")
        lowered = command.lower()
        if lowered == "quit":
            return state, True
        if lowered == "status":
            output_fn(render_game_state(state, active_player_id=player_id))
            continue
        if lowered == "help":
            output_fn(_resource_help_text())
            continue
        if lowered == "options":
            output_fn(_resource_options_text(state, player_id))
            continue
        if lowered == "done":
            active_index += 1
            output_fn(f"{player_id} finished resource buying.")
            output_fn("")
            continue
        try:
            state = _apply_resource_command(state, player_id, command)
        except (ModelValidationError, ValueError) as exc:
            if getattr(controller, "interactive", False):
                output_fn(f"Rejected: {exc}")
                continue
            raise ModelValidationError(
                f"{player_id} resource command {command!r} rejected: {exc}"
            ) from exc
        output_fn("Accepted.")
        if render_state:
            output_fn(render_game_state(state, active_player_id=player_id))
    return state, False


def _run_build_phase(
    state: GameState,
    controllers: dict[str, CommandController],
    *,
    output_fn: OutputFn,
    render_state: bool,
) -> tuple[GameState, bool]:
    build_order = tuple(reversed(state.player_order))
    active_index = 0
    while active_index < len(build_order):
        player_id = build_order[active_index]
        request = DecisionRequest(
            player_id=player_id,
            decision_type="build_houses",
            prompt=(
                f"Build phase for {player_id}. "
                "Use: options, quote <city_id> [city_id ...], build <city_id> [city_id ...], "
                "done, status, help, quit"
            ),
            metadata={"phase": state.phase},
        )
        controller = controllers[player_id]
        command = controller.choose_command(request).strip()
        if not command:
            if getattr(controller, "interactive", False):
                continue
            raise ModelValidationError(f"{player_id} returned an empty build command")
        lowered = command.lower()
        if lowered == "quit":
            return state, True
        if lowered == "status":
            output_fn(render_game_state(state, active_player_id=player_id))
            continue
        if lowered == "help":
            output_fn(_build_help_text())
            continue
        if lowered == "options":
            output_fn(_build_options_text(state, player_id))
            continue
        if lowered.startswith("quote "):
            try:
                output_fn(_quote_build_text(state, player_id, command.split()[1:]))
            except (ModelValidationError, ValueError) as exc:
                if getattr(controller, "interactive", False):
                    output_fn(f"Rejected: {exc}")
                    continue
                raise ModelValidationError(f"{player_id} build quote {command!r} rejected: {exc}") from exc
            continue
        if lowered == "done":
            active_index += 1
            output_fn(f"{player_id} finished building.")
            output_fn("")
            continue
        try:
            state = _apply_build_command(state, player_id, command)
        except (ModelValidationError, ValueError) as exc:
            if getattr(controller, "interactive", False):
                output_fn(f"Rejected: {exc}")
                continue
            raise ModelValidationError(f"{player_id} build command {command!r} rejected: {exc}") from exc
        output_fn("Accepted.")
        if render_state:
            output_fn(render_game_state(state, active_player_id=player_id))
    return state, False


def _run_bureaucracy_phase(
    state: GameState,
    controllers: dict[str, CommandController],
    *,
    output_fn: OutputFn,
) -> tuple[GameState, BureaucracySummary, bool]:
    choices: dict[str, tuple[PlantRunPlan, ...]] = {}
    for player_id in state.player_order:
        while True:
            request = DecisionRequest(
                player_id=player_id,
                decision_type="bureaucracy",
                prompt=(
                    f"Bureaucracy for {player_id}. "
                    "Use: options, run <plant_price>[:resource=amount,...] ..., skip, status, help, quit"
                ),
                metadata={"phase": state.phase},
            )
            controller = controllers[player_id]
            command = controller.choose_command(request).strip()
            if not command:
                if getattr(controller, "interactive", False):
                    continue
                raise ModelValidationError(f"{player_id} returned an empty bureaucracy command")
            lowered = command.lower()
            if lowered == "quit":
                return state, BureaucracySummary({}, {}, False, False, state.step, False), True
            if lowered == "status":
                output_fn(render_game_state(state, active_player_id=player_id))
                continue
            if lowered == "help":
                output_fn(_bureaucracy_help_text())
                continue
            if lowered == "options":
                output_fn(_bureaucracy_options_text(state, player_id))
                continue
            if lowered == "skip":
                choices[player_id] = ()
                output_fn(
                    f"{player_id} will power 0 cities and receive {pay_income(state.rules, 0)} Elektro."
                )
                break
            try:
                plans = _parse_run_command(command)
                chosen_plans = choose_plants_to_run(state, player_id, plans)
                projected_powered = compute_powered_cities(state, player_id, chosen_plans)
            except (ModelValidationError, ValueError) as exc:
                if getattr(controller, "interactive", False):
                    output_fn(f"Rejected: {exc}")
                    continue
                raise ModelValidationError(
                    f"{player_id} bureaucracy command {command!r} rejected: {exc}"
                ) from exc
            choices[player_id] = chosen_plans
            output_fn(
                f"{player_id} will power {projected_powered} cities and receive "
                f"{pay_income(state.rules, projected_powered)} Elektro."
            )
            break
    resolved_state, summary = resolve_bureaucracy(state, generation_choices=choices)
    return resolved_state, summary, False


def _apply_auction_command(state: GameState, player_id: str, command: str) -> GameState:
    tokens = command.split()
    if state.pending_decision is not None:
        if tokens[0].lower() != "discard" or len(tokens) != 2:
            raise ValueError("expected: discard <plant_price>")
        return replace_plant_if_needed(state, state.pending_decision.player_id, int(tokens[1]))
    auction_state = state.auction_state
    if auction_state is None:
        raise ModelValidationError("auction state is missing")
    if auction_state.has_active_auction:
        if tokens[0].lower() == "bid" and len(tokens) == 2:
            return raise_bid(state, player_id, int(tokens[1]))
        if tokens[0].lower() == "pass" and len(tokens) == 1:
            return pass_auction(state, player_id)
        raise ValueError("expected: bid <amount> or pass")
    if tokens[0].lower() == "start" and len(tokens) == 3:
        return start_auction(state, player_id, int(tokens[1]), int(tokens[2]))
    if tokens[0].lower() == "pass" and len(tokens) == 1:
        return pass_auction(state, player_id)
    raise ValueError("expected: start <plant_price> <bid> or pass")


def _apply_resource_command(state: GameState, player_id: str, command: str) -> GameState:
    tokens = command.split()
    if len(tokens) != 3 or tokens[0].lower() != "buy":
        raise ValueError("expected: buy <resource> <amount>")
    return purchase_resources(state, player_id, {tokens[1].lower(): int(tokens[2])})


def _apply_build_command(state: GameState, player_id: str, command: str) -> GameState:
    tokens = command.split()
    if len(tokens) < 2 or tokens[0].lower() != "build":
        raise ValueError("expected: build <city_id> [city_id ...]")
    city_ids = tuple(tokens[1:])
    if len(city_ids) == 1:
        return build_city(state, player_id, city_ids[0])
    return apply_builds(state, player_id, city_ids)


def _parse_run_command(command: str) -> tuple[PlantRunPlan, ...]:
    tokens = command.split()
    if len(tokens) < 2 or tokens[0].lower() != "run":
        raise ValueError("expected: run <plant_price>[:resource=amount,...] ...")
    plans: list[PlantRunPlan] = []
    for token in tokens[1:]:
        if ":" not in token:
            plans.append(PlantRunPlan(plant_price=int(token), resource_mix={}))
            continue
        plant_text, mix_text = token.split(":", 1)
        resource_mix: dict[str, int] = {}
        for entry in mix_text.split(","):
            resource, amount = entry.split("=", 1)
            resource_mix[resource.strip()] = int(amount)
        plans.append(PlantRunPlan(plant_price=int(plant_text), resource_mix=resource_mix))
    return tuple(plans)


def _build_auction_request(state: GameState) -> DecisionRequest:
    if state.pending_decision is not None:
        return DecisionRequest(
            player_id=state.pending_decision.player_id,
            decision_type="discard_power_plant",
            prompt=(
                f"{state.pending_decision.player_id} must discard a power plant. "
                "Use: options, discard <plant_price>, status, help, quit"
            ),
            legal_actions=state.pending_decision.legal_actions,
            metadata={"phase": state.phase},
        )
    auction_state = state.auction_state
    if auction_state is None:
        raise ModelValidationError("auction state is missing")
    if auction_state.has_active_auction:
        return DecisionRequest(
            player_id=str(auction_state.next_bidder_id),
            decision_type="auction_bid",
            prompt=(
                f"Active bidder: {auction_state.next_bidder_id}. "
                f"Plant {auction_state.active_plant_price}, current bid {auction_state.current_bid}. "
                "Use: options, bid <amount>, pass, status, help, quit"
            ),
            metadata={"phase": state.phase},
        )
    return DecisionRequest(
        player_id=str(auction_state.current_chooser_id),
        decision_type="auction_start",
        prompt=(
            f"Chooser: {auction_state.current_chooser_id}. "
            "Use: options, start <plant_price> <bid>, pass, status, help, quit"
        ),
        metadata={"phase": state.phase},
    )


def _auction_options_text(state: GameState) -> str:
    if state.pending_decision is not None:
        prices = ", ".join(str(action.payload["price"]) for action in state.pending_decision.legal_actions)
        return "Legal discard choices: " + prices
    auction_state = state.auction_state
    if auction_state is None:
        return "Auction state missing."
    if auction_state.has_active_auction:
        bidder_id = str(auction_state.next_bidder_id)
        player = _get_player(state, bidder_id)
        minimum_raise = int(auction_state.current_bid) + 1
        return (
            f"Legal options for {bidder_id}: pass, "
            f"bid <amount> where {minimum_raise} <= amount <= {player.elektro}"
        )
    chooser_id = str(auction_state.current_chooser_id)
    player = _get_player(state, chooser_id)
    options = []
    for plant in list_auctionable_plants(state):
        minimum_bid = 1 if auction_state.discount_token_plant_price == plant.price else plant.price
        if minimum_bid <= player.elektro:
            options.append(f"start {plant.price} <bid> where {minimum_bid} <= bid <= {player.elektro}")
    if state.round_number > 1:
        options.append("pass")
    return "Legal options for {player_id}: {options}".format(
        player_id=chooser_id,
        options=", ".join(options) if options else "(none)",
    )


def _resource_options_text(state: GameState, player_id: str) -> str:
    actions = legal_resource_purchases(state, player_id)
    if not actions:
        return f"No legal resource purchases for {player_id}."
    lines = [f"Legal resource purchases for {player_id}:"]
    for action in actions:
        payload = action.payload
        lines.append(
            f"  {payload['resource']}: "
            f"max_units={payload['max_units']} "
            f"max_affordable={payload['max_affordable_units']} "
            f"unit_prices={payload['unit_prices']}"
        )
    return "\n".join(lines)


def _build_options_text(state: GameState, player_id: str) -> str:
    actions = legal_build_targets(state, player_id)
    if not actions:
        return f"No legal single-city builds for {player_id}."
    lines = [f"Legal single-city builds for {player_id}:"]
    for action in sorted(actions, key=lambda item: (item.payload["total_cost"], item.payload["city_id"])):
        payload = action.payload
        lines.append(
            f"  {payload['city_id']} ({payload['city_name']}): "
            f"connection={payload['connection_cost']} "
            f"build={payload['build_cost']} total={payload['total_cost']}"
        )
    return "\n".join(lines)


def _bureaucracy_options_text(state: GameState, player_id: str) -> str:
    player = _get_player(state, player_id)
    if not player.power_plants:
        return f"{player_id} owns no power plants. Use `skip`."
    lines = [f"Generation options for {player_id}:"]
    for plant in sorted(player.power_plants, key=lambda item: item.price):
        if plant.is_step_3_placeholder:
            continue
        if plant.is_ecological:
            cost_text = "free"
            example = f"run {plant.price}"
        elif plant.is_hybrid:
            cost_text = f"{plant.resource_cost} coal/oil in any mix"
            example = f"run {plant.price}:coal={plant.resource_cost}"
        else:
            resource = plant.resource_types[0]
            cost_text = f"{plant.resource_cost} {resource}"
            example = f"run {plant.price}"
        lines.append(
            f"  {plant.price}: powers {plant.output_cities}, consumes {cost_text}, example `{example}`"
        )
    lines.append(
        "Available storage: "
        + ", ".join(
            f"{resource}={player.resource_storage.total(resource)}"
            for resource in ("coal", "oil", "garbage", "uranium")
        )
    )
    return "\n".join(lines)


def _quote_build_text(state: GameState, player_id: str, city_ids: list[str]) -> str:
    if not city_ids:
        raise ValueError("expected at least one city id after quote")
    simulated = GameState.from_dict(state.to_dict())
    player_before = _get_player(simulated, player_id)
    simulated = apply_builds(simulated, player_id, tuple(city_ids))
    player_after = _get_player(simulated, player_id)
    return (
        f"Quote for {player_id}: cities={', '.join(city_ids)} "
        f"cost={player_before.elektro - player_after.elektro} "
        f"new_network={', '.join(player_after.network_city_ids)}"
    )


def _auction_help_text(state: GameState) -> str:
    if state.pending_decision is not None:
        return "Commands: options, discard <plant_price>, status, help, quit"
    if state.auction_state is not None and state.auction_state.has_active_auction:
        return "Commands: options, bid <amount>, pass, status, help, quit"
    return "Commands: options, start <plant_price> <bid>, pass, status, help, quit"


def _resource_help_text() -> str:
    return "Commands: options, buy <resource> <amount>, done, status, help, quit"


def _build_help_text() -> str:
    return "Commands: options, quote <city_id> [city_id ...], build <city_id> [city_id ...], done, status, help, quit"


def _bureaucracy_help_text() -> str:
    return "Commands: options, run <plant_price>[:resource=amount,...] ..., skip, status, help, quit"


def _append_phase_history(phase_history: list[PhaseTraceEntry], state: GameState) -> None:
    entry = _phase_trace_entry(state)
    if phase_history[-1] != entry:
        phase_history.append(entry)


def _phase_trace_entry(state: GameState) -> PhaseTraceEntry:
    return PhaseTraceEntry(
        round_number=state.round_number,
        phase=state.phase,
        step=state.step,
    )


def _emit_phase_start(output_fn: OutputFn, state: GameState) -> None:
    output_fn("")
    output_fn(_phase_header_text(state))


def _phase_header_text(state: GameState) -> str:
    phase_label = state.phase.replace("_", " ").title()
    return f"========== Round {state.round_number} | {phase_label} | Step {state.step} =========="


def _should_insert_auction_separator(
    prior_state: GameState,
    current_state: GameState,
    lowered_command: str,
) -> bool:
    prior_buyers = set(prior_state.auction_state.players_with_plants) if prior_state.auction_state is not None else set()
    current_buyers = (
        set(current_state.auction_state.players_with_plants)
        if current_state.auction_state is not None
        else set()
    )
    if len(current_buyers) > len(prior_buyers):
        return True
    prior_auction = prior_state.auction_state
    if (
        lowered_command == "pass"
        and prior_auction is not None
        and not prior_auction.has_active_auction
    ):
        return True
    return False


def _validate_controllers(state: GameState, controllers: dict[str, CommandController]) -> None:
    expected = {player.player_id for player in state.players}
    if set(controllers) != expected:
        raise ModelValidationError("controllers must match the active game state's player ids exactly")


def _get_player(state: GameState, player_id: str):
    for player in state.players:
        if player.player_id == player_id:
            return player
    raise ModelValidationError(f"unknown player {player_id!r}")


def _format_market(plants: tuple[PowerPlantCard, ...]) -> str:
    return ", ".join(_format_plant(plant) for plant in plants) if plants else "(none)"


def _format_player_plants(plants: tuple[PowerPlantCard, ...]) -> str:
    if not plants:
        return "-"
    return "; ".join(_format_plant(plant) for plant in sorted(plants, key=lambda item: item.price))


def _format_plant(plant: PowerPlantCard) -> str:
    return "STEP3" if plant.is_step_3_placeholder else str(plant.price)


def _format_storage(player) -> str:
    storage = player.resource_storage
    return ", ".join(
        (
            f"coal={storage.coal}",
            f"oil={storage.oil}",
            f"hybrid_coal={storage.hybrid_coal}",
            f"hybrid_oil={storage.hybrid_oil}",
            f"garbage={storage.garbage}",
            f"uranium={storage.uranium}",
        )
    )
