from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..model import PlantRunPlan
from ..session import GameSnapshot, GuiIntent
from .components import SnapshotRenderable


IntentCallback = Callable[[GuiIntent], None]


class BasePhasePanel(SnapshotRenderable):
    def __init__(self, master, on_intent: IntentCallback) -> None:
        super().__init__(master, padding=(10, 8))
        self._on_intent = on_intent


class AuctionPanel(BasePhasePanel):
    def __init__(self, master, on_intent: IntentCallback) -> None:
        super().__init__(master, on_intent)
        self.player_id = ""
        self.help_var = tk.StringVar()
        self.plant_var = tk.StringVar()
        self.bid_var = tk.StringVar()
        ttk.Label(self, text="Auction", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self, textvariable=self.help_var, wraplength=420).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 10)
        )
        ttk.Label(self, text="Plant").grid(row=2, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.plant_var, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(self, text="Bid").grid(row=3, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.bid_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Button(self, text="Start Auction", command=self._submit_start).grid(
            row=4, column=0, pady=(10, 0), sticky="w"
        )
        ttk.Button(self, text="Raise Bid", command=self._submit_bid).grid(
            row=4, column=1, pady=(10, 0), sticky="w"
        )
        ttk.Button(self, text="Pass", command=self._submit_pass).grid(
            row=4, column=2, pady=(10, 0), sticky="w"
        )

    def render(self, snapshot: GameSnapshot) -> None:
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        if request is None:
            self.help_var.set("No auction action is pending.")
            return
        self.help_var.set(request.prompt)
        if request.legal_actions:
            first = request.legal_actions[0]
            if "plant_price" in first.payload and not self.plant_var.get():
                self.plant_var.set(str(first.payload["plant_price"]))
            if "min_bid" in first.payload and not self.bid_var.get():
                self.bid_var.set(str(first.payload["min_bid"]))

    def _submit_start(self) -> None:
        self._on_intent(
            GuiIntent.auction_start(
                self.player_id,
                plant_price=int(self.plant_var.get()),
                bid=int(self.bid_var.get()),
            )
        )

    def _submit_bid(self) -> None:
        self._on_intent(GuiIntent.auction_bid(self.player_id, int(self.bid_var.get())))

    def _submit_pass(self) -> None:
        self._on_intent(GuiIntent.auction_pass(self.player_id))


class ResourcePanel(BasePhasePanel):
    def __init__(self, master, on_intent: IntentCallback) -> None:
        super().__init__(master, on_intent)
        self.player_id = ""
        self.help_var = tk.StringVar()
        self.resource_var = tk.StringVar(value="coal")
        self.amount_var = tk.StringVar(value="1")
        ttk.Label(self, text="Buy Resources", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self, textvariable=self.help_var, wraplength=420).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 10)
        )
        ttk.Label(self, text="Resource").grid(row=2, column=0, sticky="w")
        self.resource_box = ttk.Combobox(
            self,
            textvariable=self.resource_var,
            state="readonly",
            values=("coal", "oil", "garbage", "uranium"),
            width=12,
        )
        self.resource_box.grid(row=2, column=1, sticky="w")
        ttk.Label(self, text="Amount").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(self, from_=1, to=10, textvariable=self.amount_var, width=8).grid(
            row=3, column=1, sticky="w"
        )
        ttk.Button(self, text="Buy", command=self._submit_buy).grid(row=4, column=0, pady=(10, 0), sticky="w")
        ttk.Button(self, text="Done", command=self._submit_done).grid(row=4, column=1, pady=(10, 0), sticky="w")

    def render(self, snapshot: GameSnapshot) -> None:
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        if request is None:
            self.help_var.set("No resource action is pending.")
            return
        lines = [request.prompt]
        buy_actions = [action for action in request.legal_actions if action.action_type == "buy_resource"]
        for action in buy_actions:
            lines.append(
                f"{action.payload['resource']}: max={action.payload['max_units']} "
                f"affordable={action.payload['max_affordable_units']} "
                f"prices={action.payload['unit_prices']}"
            )
        self.help_var.set("\n".join(lines))
        if buy_actions:
            self.resource_var.set(str(buy_actions[0].payload["resource"]))

    def _submit_buy(self) -> None:
        self._on_intent(
            GuiIntent.buy_resource(
                self.player_id,
                resource=self.resource_var.get(),
                amount=int(self.amount_var.get()),
            )
        )

    def _submit_done(self) -> None:
        self._on_intent(GuiIntent.finish_buying(self.player_id))


class BuildPanel(BasePhasePanel):
    def __init__(self, master, on_intent: IntentCallback) -> None:
        super().__init__(master, on_intent)
        self.player_id = ""
        ttk.Label(self, text="Build Houses", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.help_var = tk.StringVar()
        ttk.Label(self, textvariable=self.help_var, wraplength=420).pack(anchor="w", pady=(6, 8))
        self.city_list = tk.Listbox(self, selectmode="extended", height=12, exportselection=False)
        self.city_list.pack(fill="both", expand=True)
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Quote", command=self._submit_quote).pack(side="left")
        ttk.Button(buttons, text="Build", command=self._submit_build).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Done", command=self._submit_done).pack(side="left", padx=(8, 0))

    def render(self, snapshot: GameSnapshot) -> None:
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        self.city_list.delete(0, "end")
        if request is None:
            self.help_var.set("No build action is pending.")
            return
        self.help_var.set(request.prompt)
        for action in sorted(
            (item for item in request.legal_actions if item.action_type == "build_city"),
            key=lambda item: (item.payload["total_cost"], item.payload["city_id"]),
        ):
            self.city_list.insert(
                "end",
                f"{action.payload['city_id']} | {action.payload['city_name']} | total={action.payload['total_cost']}",
            )

    def _selected_city_ids(self) -> list[str]:
        values = []
        for index in self.city_list.curselection():
            label = self.city_list.get(index)
            values.append(label.split("|", 1)[0].strip())
        return values

    def _submit_quote(self) -> None:
        self._on_intent(GuiIntent.quote_build(self.player_id, self._selected_city_ids()))

    def _submit_build(self) -> None:
        self._on_intent(GuiIntent.commit_build(self.player_id, self._selected_city_ids()))

    def _submit_done(self) -> None:
        self._on_intent(GuiIntent.finish_building(self.player_id))


class PendingDecisionPanel(BasePhasePanel):
    def __init__(self, master, on_intent: IntentCallback) -> None:
        super().__init__(master, on_intent)
        self.player_id = ""
        self.mode = ""
        self.help_var = tk.StringVar()
        ttk.Label(self, text="Pending Decision", font=("Helvetica", 12, "bold")).pack(anchor="w")
        ttk.Label(self, textvariable=self.help_var, wraplength=420).pack(anchor="w", pady=(6, 10))
        self.option_box = ttk.Frame(self)
        self.option_box.pack(fill="both", expand=True)

    def render(self, snapshot: GameSnapshot) -> None:
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        self.mode = request.decision_type if request is not None else ""
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
    def __init__(self, master, on_intent: IntentCallback) -> None:
        super().__init__(master, on_intent)
        self.player_id = ""
        self.help_var = tk.StringVar()
        ttk.Label(self, text="Bureaucracy", font=("Helvetica", 12, "bold")).pack(anchor="w")
        ttk.Label(self, textvariable=self.help_var, wraplength=420).pack(anchor="w", pady=(6, 10))
        self.rows = ttk.Frame(self)
        self.rows.pack(fill="both", expand=True)
        button_row = ttk.Frame(self)
        button_row.pack(fill="x", pady=(10, 0))
        ttk.Button(button_row, text="Submit Runs", command=self._submit_runs).pack(side="left")
        ttk.Button(button_row, text="Skip", command=self._submit_skip).pack(side="left", padx=(8, 0))
        self._plant_controls: dict[int, dict[str, tk.Variable]] = {}

    def render(self, snapshot: GameSnapshot) -> None:
        request = snapshot.active_request
        self.player_id = request.player_id if request is not None else ""
        self.help_var.set(request.prompt if request is not None else "No bureaucracy action is pending.")
        for child in self.rows.winfo_children():
            child.destroy()
        self._plant_controls = {}
        if request is None:
            return
        player = next(player for player in snapshot.state.players if player.player_id == self.player_id)
        for row_index, plant in enumerate(sorted(player.power_plants, key=lambda item: item.price)):
            if plant.is_step_3_placeholder:
                continue
            selected_var = tk.BooleanVar(value=False)
            coal_var = tk.StringVar(value=str(plant.resource_cost if plant.is_hybrid else 0))
            oil_var = tk.StringVar(value="0")
            ttk.Checkbutton(
                self.rows,
                text=f"Plant {plant.price} powers {plant.output_cities}",
                variable=selected_var,
            ).grid(row=row_index, column=0, sticky="w")
            if plant.is_hybrid:
                ttk.Label(self.rows, text="Coal").grid(row=row_index, column=1, sticky="w")
                ttk.Entry(self.rows, textvariable=coal_var, width=4).grid(row=row_index, column=2, sticky="w")
                ttk.Label(self.rows, text="Oil").grid(row=row_index, column=3, sticky="w")
                ttk.Entry(self.rows, textvariable=oil_var, width=4).grid(row=row_index, column=4, sticky="w")
            self._plant_controls[plant.price] = {
                "selected": selected_var,
                "coal": coal_var,
                "oil": oil_var,
            }

    def _submit_runs(self) -> None:
        plans = []
        for plant_price, vars_by_name in sorted(self._plant_controls.items()):
            if not bool(vars_by_name["selected"].get()):
                continue
            coal = int(str(vars_by_name["coal"].get() or "0"))
            oil = int(str(vars_by_name["oil"].get() or "0"))
            mix = {}
            if coal:
                mix["coal"] = coal
            if oil:
                mix["oil"] = oil
            plans.append(PlantRunPlan(plant_price=plant_price, resource_mix=mix))
        self._on_intent(GuiIntent.run_plants(self.player_id, plans))

    def _submit_skip(self) -> None:
        self._on_intent(GuiIntent.skip_bureaucracy(self.player_id))
