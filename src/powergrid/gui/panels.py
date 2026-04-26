from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..model import (
    GameState,
    ModelValidationError,
    PlantRunPlan,
    RESOURCE_TYPES,
    apply_builds,
    choose_plants_to_run,
)
from ..session import GameSnapshot, GuiIntent
from .components import SnapshotRenderable


IntentCallback = Callable[[GuiIntent], None]
LocalChangeCallback = Callable[[], None]


class BasePhasePanel(SnapshotRenderable):
    def __init__(
        self,
        master,
        on_intent: IntentCallback,
        on_local_change: LocalChangeCallback | None = None,
    ) -> None:
        super().__init__(master, padding=(10, 8))
        self._on_intent = on_intent
        self._on_local_change = on_local_change
        self._snapshot: GameSnapshot | None = None

    def render(self, snapshot: GameSnapshot) -> None:
        self._snapshot = snapshot

    def _notify_local_change(self) -> None:
        if self._on_local_change is not None:
            self._on_local_change()

    def board_interaction_state(self) -> dict[str, object]:
        return {}

    def market_interaction_state(self) -> dict[str, object]:
        return {}

    def handle_city_click(self, city_id: str) -> bool:
        return False

    def handle_resource_click(self, resource: str) -> bool:
        return False

    def handle_market_plant_click(self, plant_price: int) -> bool:
        return False


class AuctionPanel(BasePhasePanel):
    def __init__(
        self,
        master,
        on_intent: IntentCallback,
        on_local_change: LocalChangeCallback | None = None,
    ) -> None:
        super().__init__(master, on_intent, on_local_change)
        self.player_id = ""
        self.help_var = tk.StringVar()
        self.selection_var = tk.StringVar(value="Click a plant in the current market")
        self.range_var = tk.StringVar(value="")
        self.bid_var = tk.IntVar(value=0)
        self._decision_type = ""
        self._selected_plant_price: int | None = None
        self._active_plant_price: int | None = None
        self._start_actions_by_price: dict[int, object] = {}
        self._bid_bounds: tuple[int, int] | None = None
        self._pass_allowed = False

        ttk.Label(self, text="Auction", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self, textvariable=self.help_var, wraplength=420).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 10)
        )
        ttk.Label(self, text="Plant").grid(row=2, column=0, sticky="w")
        ttk.Label(self, textvariable=self.selection_var).grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Label(self, text="Bid").grid(row=3, column=0, sticky="w")
        self.bid_spin = ttk.Spinbox(self, from_=0, to=0, textvariable=self.bid_var, width=10, state="disabled")
        self.bid_spin.grid(row=3, column=1, sticky="w")
        ttk.Label(self, textvariable=self.range_var).grid(row=3, column=2, sticky="w", padx=(8, 0))
        self.confirm_button = ttk.Button(self, text="Start Auction", command=self._submit_confirm)
        self.confirm_button.grid(row=4, column=0, pady=(10, 0), sticky="w")
        self.pass_button = ttk.Button(self, text="Pass", command=self._submit_pass)
        self.pass_button.grid(row=4, column=1, pady=(10, 0), sticky="w")

    def render(self, snapshot: GameSnapshot) -> None:
        super().render(snapshot)
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        self._decision_type = request.decision_type if request is not None else ""
        self._start_actions_by_price = {}
        self._active_plant_price = None
        self._pass_allowed = False
        self.confirm_button.configure(state="disabled")
        self.pass_button.configure(state="disabled")
        if request is None:
            self.help_var.set("No auction action is pending.")
            self.selection_var.set("No active auction")
            self._configure_bid_bounds(None)
            return

        self.help_var.set(request.prompt)
        self._pass_allowed = any(action.action_type == "auction_pass" for action in request.legal_actions)
        if request.decision_type == "auction_start":
            self.help_var.set(request.prompt + " Click a current-market plant to prepare the auction.")
            self._start_actions_by_price = {
                int(action.payload["plant_price"]): action
                for action in request.legal_actions
                if action.action_type == "auction_start"
            }
            if self._selected_plant_price not in self._start_actions_by_price:
                self._selected_plant_price = None
            if self._selected_plant_price is None:
                self.selection_var.set("Click a plant in the current market")
                self._configure_bid_bounds(None)
            else:
                action = self._start_actions_by_price[self._selected_plant_price]
                self.selection_var.set(f"Plant {self._selected_plant_price}")
                self._configure_bid_bounds(
                    (int(action.payload["min_bid"]), int(action.payload["max_bid"]))
                )
                self.confirm_button.configure(state="normal", text="Start Auction")
            self.pass_button.configure(state=("normal" if self._pass_allowed else "disabled"))
            return

        bid_action = next(
            (action for action in request.legal_actions if action.action_type == "auction_bid"),
            None,
        )
        self._active_plant_price = int(bid_action.payload["plant_price"]) if bid_action is not None else None
        if self._active_plant_price is not None:
            self.selection_var.set(f"Plant {self._active_plant_price}")
        else:
            self.selection_var.set("No active plant")
        if bid_action is None:
            self._configure_bid_bounds(None)
        else:
            self._configure_bid_bounds((int(bid_action.payload["min_bid"]), int(bid_action.payload["max_bid"])))
            self.confirm_button.configure(state="normal", text="Submit Bid")
        self.pass_button.configure(state=("normal" if self._pass_allowed else "disabled"))

    def market_interaction_state(self) -> dict[str, object]:
        if self._decision_type == "auction_start":
            return {
                "selected_plant_price": self._selected_plant_price,
                "clickable_plant_prices": tuple(sorted(self._start_actions_by_price)),
            }
        if self._decision_type == "auction_bid":
            return {
                "selected_plant_price": self._active_plant_price,
                "clickable_plant_prices": (),
            }
        return {}

    def handle_market_plant_click(self, plant_price: int) -> bool:
        if self._decision_type != "auction_start" or plant_price not in self._start_actions_by_price:
            return False
        self._selected_plant_price = int(plant_price)
        action = self._start_actions_by_price[self._selected_plant_price]
        self.selection_var.set(f"Plant {self._selected_plant_price}")
        self._configure_bid_bounds(
            (int(action.payload["min_bid"]), int(action.payload["max_bid"])),
            force_minimum=True,
        )
        self.confirm_button.configure(state="normal", text="Start Auction")
        self.bid_spin.focus_set()
        return True

    def _configure_bid_bounds(
        self,
        bounds: tuple[int, int] | None,
        *,
        force_minimum: bool = False,
    ) -> None:
        self._bid_bounds = bounds
        if bounds is None:
            self.bid_spin.configure(state="disabled", from_=0, to=0)
            self.range_var.set("")
            self.bid_var.set(0)
            return
        minimum, maximum = bounds
        self.bid_spin.configure(state="readonly", from_=minimum, to=maximum)
        current = int(self.bid_var.get())
        if force_minimum or current < minimum or current > maximum:
            self.bid_var.set(minimum)
        self.range_var.set(f"Legal range: {minimum} - {maximum}")

    def _submit_confirm(self) -> None:
        if self._decision_type == "auction_start":
            if self._selected_plant_price is None:
                return
            self._on_intent(
                GuiIntent.auction_start(
                    self.player_id,
                    plant_price=self._selected_plant_price,
                    bid=int(self.bid_var.get()),
                )
            )
            return
        self._on_intent(GuiIntent.auction_bid(self.player_id, int(self.bid_var.get())))

    def _submit_pass(self) -> None:
        self._on_intent(GuiIntent.auction_pass(self.player_id))


class ResourcePanel(BasePhasePanel):
    def __init__(
        self,
        master,
        on_intent: IntentCallback,
        on_local_change: LocalChangeCallback | None = None,
    ) -> None:
        super().__init__(master, on_intent, on_local_change)
        self.player_id = ""
        self.help_var = tk.StringVar()
        self._buy_actions_by_resource: dict[str, object] = {}
        self.grid_columnconfigure(0, weight=1)
        ttk.Label(self, text="Buy Resources", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self, textvariable=self.help_var, wraplength=420, justify="left").grid(
            row=1, column=0, sticky="w", pady=(6, 10)
        )
        self.button_row = ttk.Frame(self)
        self.button_row.grid(row=2, column=0, sticky="ew")
        self.resource_buttons: dict[str, ttk.Button] = {}
        for index, resource in enumerate(RESOURCE_TYPES):
            self.button_row.grid_columnconfigure(index, weight=1, uniform="resource")
            button = ttk.Button(
                self.button_row,
                text=resource.title(),
                command=lambda resource=resource: self.handle_resource_click(resource),
                width=14,
            )
            button.grid(row=0, column=index, sticky="ew", padx=(0, 8 if index < len(RESOURCE_TYPES) - 1 else 0))
            self.resource_buttons[resource] = button
        ttk.Button(self, text="Done", command=self._submit_done).grid(row=3, column=0, pady=(10, 0), sticky="w")

    def render(self, snapshot: GameSnapshot) -> None:
        super().render(snapshot)
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        self._buy_actions_by_resource = {}
        if request is None:
            self.help_var.set("No resource action is pending.")
            for button in self.resource_buttons.values():
                button.configure(state="disabled")
            return
        for action in request.legal_actions:
            if action.action_type == "buy_resource":
                self._buy_actions_by_resource[str(action.payload["resource"])] = action
        self.help_var.set(request.prompt + " Click a resource token on the board or a resource button below.")
        for resource in RESOURCE_TYPES:
            button = self.resource_buttons[resource]
            action = self._buy_actions_by_resource.get(resource)
            if action is None:
                button.configure(
                    text=f"{resource.title()}\nUnavailable",
                    state="disabled",
                )
                continue
            unit_prices = list(action.payload["unit_prices"])
            first_price = unit_prices[0] if unit_prices else "-"
            button.configure(
                text=(
                    f"{resource.title()}\n"
                    f"${first_price} | max {action.payload['max_affordable_units']}"
                ),
                state="normal",
            )

    def board_interaction_state(self) -> dict[str, object]:
        return {
            "resource_phase_active": self._snapshot is not None and self._snapshot.active_request is not None,
            "buyable_resources": tuple(sorted(self._buy_actions_by_resource)),
        }

    def handle_resource_click(self, resource: str) -> bool:
        if resource not in self._buy_actions_by_resource:
            return False
        self._on_intent(GuiIntent.buy_resource(self.player_id, resource=resource, amount=1))
        return True

    def _submit_done(self) -> None:
        self._on_intent(GuiIntent.finish_buying(self.player_id))


class BuildPanel(BasePhasePanel):
    def __init__(
        self,
        master,
        on_intent: IntentCallback,
        on_local_change: LocalChangeCallback | None = None,
    ) -> None:
        super().__init__(master, on_intent, on_local_change)
        self.player_id = ""
        self.help_var = tk.StringVar()
        self.quote_var = tk.StringVar(value="No quote yet. Click a legal city on the board.")
        self._city_actions_by_id: dict[str, object] = {}
        self._selected_city_ids: list[str] = []
        self._quote_valid = False
        ttk.Label(self, text="Build Houses", font=("Helvetica", 12, "bold")).pack(anchor="w")
        ttk.Label(self, textvariable=self.help_var, wraplength=420, justify="left").pack(anchor="w", pady=(6, 8))
        ttk.Label(self, textvariable=self.quote_var, wraplength=420, justify="left").pack(anchor="w", pady=(0, 8))
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=(10, 0))
        self.submit_button = ttk.Button(buttons, text="Submit Build", command=self._submit_build)
        self.submit_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="Cancel Quote", command=self._cancel_quote)
        self.cancel_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Done", command=self._submit_done).pack(side="left", padx=(8, 0))

    def render(self, snapshot: GameSnapshot) -> None:
        super().render(snapshot)
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        self._city_actions_by_id = {}
        if request is None:
            self.help_var.set("No build action is pending.")
            self._selected_city_ids = []
            self.quote_var.set("No quote yet.")
            self._sync_button_state()
            return
        self.help_var.set(request.prompt + " Click cities on the board to build a quote.")
        for action in request.legal_actions:
            if action.action_type == "build_city":
                self._city_actions_by_id[str(action.payload["city_id"])] = action
        self._selected_city_ids = [city_id for city_id in self._selected_city_ids if city_id in self._city_actions_by_id]
        self._refresh_quote_summary()

    def board_interaction_state(self) -> dict[str, object]:
        return {
            "clickable_city_ids": tuple(sorted(self._city_actions_by_id)),
            "selected_city_ids": tuple(self._selected_city_ids),
        }

    def handle_city_click(self, city_id: str) -> bool:
        if city_id in self._selected_city_ids:
            self._selected_city_ids = [item for item in self._selected_city_ids if item != city_id]
            self._refresh_quote_summary()
            return True
        if city_id not in self._city_actions_by_id or self._snapshot is None:
            return False
        tentative = [*self._selected_city_ids, city_id]
        valid, message = self._validate_quote_selection(tentative)
        if not valid:
            self.quote_var.set(message)
            self._quote_valid = False
            self._sync_button_state()
            return True
        self._selected_city_ids = tentative
        self._refresh_quote_summary()
        return True

    def _validate_quote_selection(self, city_ids: list[str]) -> tuple[bool, str]:
        if self._snapshot is None:
            return False, "No build snapshot is available."
        if not city_ids:
            return True, ""
        try:
            quoted_state = apply_builds(GameState.from_dict(self._snapshot.state.to_dict()), self.player_id, city_ids)
        except ModelValidationError as exc:
            return False, f"Quote invalid: {exc}"
        before_player = next(player for player in self._snapshot.state.players if player.player_id == self.player_id)
        after_player = next(player for player in quoted_state.players if player.player_id == self.player_id)
        total_cost = before_player.elektro - after_player.elektro
        city_labels = []
        for city_id in city_ids:
            action = self._city_actions_by_id.get(city_id)
            if action is None:
                city_labels.append(city_id)
            else:
                city_labels.append(str(action.payload["city_name"]))
        return True, f"Quoted: {', '.join(city_labels)}\nTotal cost: {total_cost} Elektro"

    def _refresh_quote_summary(self) -> None:
        if not self._selected_city_ids:
            self._quote_valid = False
            self.quote_var.set("No quote yet. Click a legal city on the board.")
            self._sync_button_state()
            return
        self._quote_valid, message = self._validate_quote_selection(self._selected_city_ids)
        self.quote_var.set(message)
        self._sync_button_state()

    def _sync_button_state(self) -> None:
        state = "normal" if self._quote_valid and self._selected_city_ids else "disabled"
        self.submit_button.configure(state=state)
        self.cancel_button.configure(state=("normal" if self._selected_city_ids else "disabled"))

    def _submit_build(self) -> None:
        if not self._quote_valid or not self._selected_city_ids:
            return
        self._on_intent(GuiIntent.commit_build(self.player_id, tuple(self._selected_city_ids)))

    def _cancel_quote(self) -> None:
        self._selected_city_ids = []
        self._refresh_quote_summary()
        self._notify_local_change()

    def _submit_done(self) -> None:
        self._on_intent(GuiIntent.finish_building(self.player_id))


class PendingDecisionPanel(BasePhasePanel):
    def __init__(
        self,
        master,
        on_intent: IntentCallback,
        on_local_change: LocalChangeCallback | None = None,
    ) -> None:
        super().__init__(master, on_intent, on_local_change)
        self.player_id = ""
        self.help_var = tk.StringVar()
        ttk.Label(self, text="Pending Decision", font=("Helvetica", 12, "bold")).pack(anchor="w")
        ttk.Label(self, textvariable=self.help_var, wraplength=420).pack(anchor="w", pady=(6, 10))
        self.option_box = ttk.Frame(self)
        self.option_box.pack(fill="both", expand=True)

    def render(self, snapshot: GameSnapshot) -> None:
        super().render(snapshot)
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        self.help_var.set(request.prompt if request is not None else "No pending decision.")
        for child in self.option_box.winfo_children():
            child.destroy()
        if request is None:
            return
        if request.decision_type == "discard_power_plant":
            for action in request.legal_actions:
                ttk.Button(
                    self.option_box,
                    text=f"Discard plant {action.payload['price']}",
                    command=lambda price=action.payload["price"]: self._on_intent(
                        GuiIntent.discard_plant(self.player_id, int(price))
                    ),
                ).pack(anchor="w", pady=(0, 6))
            return
        for action in request.legal_actions:
            coal = int(action.payload.get("coal", 0))
            oil = int(action.payload.get("oil", 0))
            ttk.Button(
                self.option_box,
                text=f"Discard coal={coal} oil={oil}",
                command=lambda coal=coal, oil=oil: self._on_intent(
                    GuiIntent.discard_hybrid_resources(self.player_id, coal=coal, oil=oil)
                ),
            ).pack(anchor="w", pady=(0, 6))


class BureaucracyPanel(BasePhasePanel):
    def __init__(
        self,
        master,
        on_intent: IntentCallback,
        on_local_change: LocalChangeCallback | None = None,
    ) -> None:
        super().__init__(master, on_intent, on_local_change)
        self.player_id = ""
        self.help_var = tk.StringVar()
        ttk.Label(self, text="Bureaucracy", font=("Helvetica", 12, "bold")).pack(anchor="w")
        ttk.Label(self, textvariable=self.help_var, wraplength=420, justify="left").pack(anchor="w", pady=(6, 10))
        self.rows = ttk.Frame(self)
        self.rows.pack(fill="both", expand=True)
        button_row = ttk.Frame(self)
        button_row.pack(fill="x", pady=(10, 0))
        ttk.Button(button_row, text="Submit Runs", command=self._submit_runs).pack(side="left")
        ttk.Button(button_row, text="Skip", command=self._submit_skip).pack(side="left", padx=(8, 0))
        self._plant_controls: dict[int, dict[str, object]] = {}
        self._player_resource_totals = {resource: 0 for resource in RESOURCE_TYPES}

    def render(self, snapshot: GameSnapshot) -> None:
        super().render(snapshot)
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        self.help_var.set(
            request.prompt + " Click a plant button to toggle it on or off."
            if request is not None
            else "No bureaucracy action is pending."
        )
        previous_state = {
            plant_price: {
                "selected": bool(control["selected"].get()),
                "coal": int(control["coal"].get()),
            }
            for plant_price, control in self._plant_controls.items()
        }
        for child in self.rows.winfo_children():
            child.destroy()
        self._plant_controls = {}
        if request is None:
            return
        player = next(player for player in snapshot.state.players if player.player_id == self.player_id)
        self._player_resource_totals = player.resource_storage.resource_totals()
        for row_index, plant in enumerate(sorted(player.power_plants, key=lambda item: item.price)):
            if plant.is_step_3_placeholder:
                continue
            previous = previous_state.get(plant.price, {})
            selected_var = tk.BooleanVar(value=bool(previous.get("selected", False)))
            toggle = ttk.Checkbutton(
                self.rows,
                text=_format_plant_toggle_label(plant),
                variable=selected_var,
                style="Toolbutton",
                command=lambda plant_price=plant.price: self._handle_toggle_change(plant_price),
            )
            toggle.grid(row=row_index, column=0, sticky="w")
            coal_var = tk.IntVar(value=0)
            oil_var = tk.StringVar(value="0")
            note_var = tk.StringVar(value="")
            coal_spin = None
            if plant.is_hybrid:
                max_coal = min(plant.resource_cost, self._player_resource_totals["coal"])
                default_coal = int(previous.get("coal", max_coal))
                default_coal = max(0, min(default_coal, plant.resource_cost))
                coal_var.set(default_coal)
                oil_var.set(str(plant.resource_cost - default_coal))
                ttk.Label(self.rows, text="Coal").grid(row=row_index, column=1, sticky="w", padx=(10, 0))
                coal_spin = ttk.Spinbox(
                    self.rows,
                    from_=0,
                    to=plant.resource_cost,
                    textvariable=coal_var,
                    width=4,
                    state="readonly",
                    command=lambda plant_price=plant.price: self._handle_hybrid_mix_change(plant_price),
                )
                coal_spin.grid(row=row_index, column=2, sticky="w")
                ttk.Label(self.rows, text="Oil").grid(row=row_index, column=3, sticky="w", padx=(10, 0))
                ttk.Label(self.rows, textvariable=oil_var).grid(row=row_index, column=4, sticky="w")
                ttk.Label(self.rows, textvariable=note_var).grid(row=row_index, column=5, sticky="w", padx=(10, 0))
            else:
                ttk.Label(self.rows, textvariable=note_var).grid(
                    row=row_index, column=1, columnspan=3, sticky="w", padx=(10, 0)
                )
            self._plant_controls[plant.price] = {
                "plant": plant,
                "selected": selected_var,
                "coal": coal_var,
                "oil": oil_var,
                "coal_spin": coal_spin,
                "toggle": toggle,
                "note": note_var,
            }
        self._reconcile_plant_controls()

    def _handle_toggle_change(self, plant_price: int) -> None:
        self._reconcile_plant_controls(changed_plant_price=plant_price)

    def _handle_hybrid_mix_change(self, plant_price: int) -> None:
        self._reconcile_plant_controls(changed_plant_price=plant_price)

    def _reconcile_plant_controls(self, changed_plant_price: int | None = None) -> None:
        for _ in range(max(1, len(self._plant_controls))):
            changed = False
            for plant_price, control in sorted(self._plant_controls.items()):
                plant = control["plant"]
                selected = bool(control["selected"].get())
                toggle = control["toggle"]
                coal_spin = control["coal_spin"]
                note_var = control["note"]

                if plant.is_ecological:
                    toggle.configure(state="normal")
                    note_var.set("No fuel needed")
                    if coal_spin is not None:
                        coal_spin.configure(state="disabled")
                    continue

                if plant.is_hybrid:
                    legal_range = self._hybrid_legal_range(plant_price)
                    if legal_range is None:
                        if selected:
                            control["selected"].set(False)
                            selected = False
                            changed = True
                        toggle.configure(state="disabled")
                        note_var.set("Not enough coal/oil")
                        control["coal"].set(0)
                        control["oil"].set("0")
                        if coal_spin is not None:
                            coal_spin.configure(state="disabled", from_=0, to=0)
                        continue
                    minimum_coal, maximum_coal = legal_range
                    current_coal = max(0, min(int(control["coal"].get()), plant.resource_cost))
                    clamped_coal = min(max(current_coal, minimum_coal), maximum_coal)
                    if clamped_coal != current_coal:
                        control["coal"].set(clamped_coal)
                        changed = True
                    control["oil"].set(str(plant.resource_cost - clamped_coal))
                    toggle.configure(state="normal")
                    note_var.set(
                        f"Coal {minimum_coal}-{maximum_coal} | Oil {plant.resource_cost - maximum_coal}-{plant.resource_cost - minimum_coal}"
                    )
                    if coal_spin is not None:
                        coal_spin.configure(
                            state="readonly" if selected else "disabled",
                            from_=minimum_coal,
                            to=maximum_coal,
                        )
                    continue

                resource = plant.resource_types[0]
                remaining = self._player_resource_totals[resource] - self._resource_usage(exclude_price=plant_price)[resource]
                can_run = remaining >= plant.resource_cost
                if not can_run and selected:
                    control["selected"].set(False)
                    changed = True
                toggle.configure(state=("normal" if can_run else "disabled"))
                note_var.set(
                    f"Need {plant.resource_cost} {resource}"
                    if can_run
                    else f"Need {plant.resource_cost} {resource} | unavailable"
                )

            if not changed:
                break

        if changed_plant_price is not None and self._snapshot is not None:
            try:
                choose_plants_to_run(self._snapshot.state, self.player_id, self._current_run_selection())
            except ModelValidationError:
                control = self._plant_controls.get(changed_plant_price)
                if control is not None:
                    control["selected"].set(False)
                    if control["coal_spin"] is not None:
                        control["coal"].set(0)
                        control["oil"].set("0")
                self._reconcile_plant_controls()

    def _resource_usage(self, exclude_price: int | None = None) -> dict[str, int]:
        usage = {resource: 0 for resource in RESOURCE_TYPES}
        for plant_price, control in self._plant_controls.items():
            if plant_price == exclude_price or not bool(control["selected"].get()):
                continue
            plant = control["plant"]
            if plant.is_ecological:
                continue
            if plant.is_hybrid:
                usage["coal"] += int(control["coal"].get())
                usage["oil"] += int(control["oil"].get())
                continue
            usage[plant.resource_types[0]] += plant.resource_cost
        return usage

    def _hybrid_legal_range(self, plant_price: int) -> tuple[int, int] | None:
        control = self._plant_controls[plant_price]
        plant = control["plant"]
        usage = self._resource_usage(exclude_price=plant_price)
        coal_left = self._player_resource_totals["coal"] - usage["coal"]
        oil_left = self._player_resource_totals["oil"] - usage["oil"]
        if coal_left < 0 or oil_left < 0 or coal_left + oil_left < plant.resource_cost:
            return None
        minimum_coal = max(0, plant.resource_cost - oil_left)
        maximum_coal = min(plant.resource_cost, coal_left)
        if minimum_coal > maximum_coal:
            return None
        return minimum_coal, maximum_coal

    def _current_run_selection(self) -> list[PlantRunPlan]:
        selection: list[PlantRunPlan] = []
        for plant_price, control in sorted(self._plant_controls.items()):
            if not bool(control["selected"].get()):
                continue
            plant = control["plant"]
            mix = {}
            if plant.is_hybrid:
                coal = int(control["coal"].get())
                oil = int(control["oil"].get())
                if coal:
                    mix["coal"] = coal
                if oil:
                    mix["oil"] = oil
            selection.append(PlantRunPlan(plant_price=plant_price, resource_mix=mix))
        return selection

    def _submit_runs(self) -> None:
        if self._snapshot is None:
            return
        plans = self._current_run_selection()
        try:
            choose_plants_to_run(self._snapshot.state, self.player_id, plans)
        except ModelValidationError as exc:
            self.help_var.set(f"Cannot submit runs: {exc}")
            return
        self._on_intent(GuiIntent.run_plants(self.player_id, plans))

    def _submit_skip(self) -> None:
        self._on_intent(GuiIntent.skip_bureaucracy(self.player_id))


def _format_plant_toggle_label(plant) -> str:
    if plant.is_ecological:
        resource_text = "eco"
    elif plant.is_hybrid:
        resource_text = f"hybrid {plant.resource_cost}"
    else:
        resource_text = f"{plant.resource_types[0]} {plant.resource_cost}"
    return f"Plant {plant.price} | {resource_text} | powers {plant.output_cities}"
