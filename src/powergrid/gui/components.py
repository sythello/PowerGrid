from __future__ import annotations

import json
import math
import tkinter as tk
from tkinter import ttk

from ..session import GameSnapshot
from .board_view import PLAYER_COLOR_MAP, draw_power_plant_card


class SnapshotRenderable(ttk.Frame):
    def render(self, snapshot: GameSnapshot) -> None:
        raise NotImplementedError


class HeaderView(SnapshotRenderable):
    def __init__(self, master) -> None:
        super().__init__(master, padding=(10, 8))
        self.status_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, font=("Helvetica", 16, "bold")).pack(
            anchor="w"
        )
        ttk.Label(self, textvariable=self.summary_var).pack(anchor="w", pady=(4, 0))

    def render(self, snapshot: GameSnapshot) -> None:
        state = snapshot.state
        active = snapshot.active_request.player_id if snapshot.active_request is not None else "-"
        self.status_var.set(
            f"Round {state.round_number} | {state.phase.replace('_', ' ').title()} | Step {state.step}"
        )
        if snapshot.winner_result is not None:
            self.summary_var.set("Winner: " + ", ".join(snapshot.winner_result.winner_ids))
            return
        self.summary_var.set(f"Active player: {active}")


class PlayerRail(SnapshotRenderable):
    def __init__(self, master) -> None:
        super().__init__(master, padding=(10, 8))
        ttk.Label(self, text="Players", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(self, width=270, background="#f3ede1", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.scrollbar.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.cards = tk.Frame(self.canvas, bg="#f3ede1")
        self._cards_window = self.canvas.create_window((0, 0), window=self.cards, anchor="nw")
        self.cards.bind("<Configure>", self._sync_scrollregion)
        self.canvas.bind("<Configure>", self._sync_window_width)

    def render(self, snapshot: GameSnapshot) -> None:
        active = snapshot.active_request.player_id if snapshot.active_request is not None else None
        for child in self.cards.winfo_children():
            child.destroy()
        for player in sorted(snapshot.state.players, key=lambda item: item.turn_order_position):
            self._render_player_card(player, is_active=(player.player_id == active))
        self.canvas.update_idletasks()
        self._sync_scrollregion()

    def _render_player_card(self, player, *, is_active: bool) -> None:
        border_color = "#d97706" if is_active else PLAYER_COLOR_MAP.get(player.color, "#475569")
        card = tk.Frame(
            self.cards,
            bg="#fffaf0",
            highlightbackground=border_color,
            highlightcolor=border_color,
            highlightthickness=3 if is_active else 2,
            bd=0,
        )
        card.pack(fill="x", pady=(0, 10))

        header = tk.Frame(card, bg="#fffaf0")
        header.pack(fill="x", padx=8, pady=(8, 6))
        title_text = f"{player.turn_order_position}. {player.name}"
        tk.Label(
            header,
            text=title_text,
            bg="#fffaf0",
            fg="#111827",
            font=("Helvetica", 11, "bold"),
            anchor="w",
        ).pack(side="left")
        if is_active:
            tk.Label(
                header,
                text="ACTIVE",
                bg="#fbbf24",
                fg="#111827",
                font=("Helvetica", 8, "bold"),
                padx=6,
                pady=2,
            ).pack(side="right")

        tk.Label(
            card,
            text=f"{player.player_id} [{player.controller}]",
            bg="#fffaf0",
            fg="#475569",
            font=("Helvetica", 9),
            anchor="w",
        ).pack(fill="x", padx=8)
        tk.Label(
            card,
            text=f"Elektro ${player.elektro}    Cities {player.connected_city_count}",
            bg="#fffaf0",
            fg="#111827",
            font=("Helvetica", 10, "bold"),
            anchor="w",
        ).pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(
            card,
            text=(
                f"Coal {player.resource_storage.total('coal')}   "
                f"Oil {player.resource_storage.total('oil')}"
            ),
            bg="#fffaf0",
            fg="#1f2937",
            font=("Helvetica", 9),
            anchor="w",
        ).pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(
            card,
            text=(
                f"Garbage {player.resource_storage.total('garbage')}   "
                f"Uranium {player.resource_storage.total('uranium')}"
            ),
            bg="#fffaf0",
            fg="#1f2937",
            font=("Helvetica", 9),
            anchor="w",
        ).pack(fill="x", padx=8)

        plants = sorted(player.power_plants, key=lambda item: item.price)
        tk.Label(
            card,
            text="Power Plants",
            bg="#fffaf0",
            fg="#334155",
            font=("Helvetica", 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 4))
        self._render_power_plants(card, plants)

    def _render_power_plants(self, parent, plants) -> None:
        columns = 4
        size = 50
        gap = 8
        rows = max(1, math.ceil(max(1, len(plants)) / columns))
        canvas_height = rows * (size + gap) + 8
        canvas_width = columns * (size + gap) + 8
        canvas = tk.Canvas(parent, height=canvas_height, width=canvas_width, bg="#fffaf0", highlightthickness=0)
        canvas.pack(fill="x", padx=8, pady=(0, 8))
        if not plants:
            canvas.create_text(
                canvas_width / 2,
                canvas_height / 2,
                text="No plants",
                fill="#64748b",
                font=("Helvetica", 9, "italic"),
            )
            return
        for index, plant in enumerate(plants):
            row = index // columns
            column = index % columns
            x = 4 + column * (size + gap)
            y = 4 + row * (size + gap)
            draw_power_plant_card(canvas, x, y, plant, size=size)
        canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))

    def _sync_scrollregion(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_window_width(self, event) -> None:
        self.canvas.itemconfigure(self._cards_window, width=event.width)


class MarketPanel(SnapshotRenderable):
    def __init__(self, master) -> None:
        super().__init__(master, padding=(10, 8))
        self.body = tk.Text(self, width=42, height=16, wrap="word")
        self.body.pack(fill="both", expand=True)
        self.body.configure(state="disabled")

    def render(self, snapshot: GameSnapshot) -> None:
        state = snapshot.state
        lines = [
            "Current market: " + _format_market(state.current_market),
            "Future market: " + (_format_market(state.future_market) if state.future_market else "(none)"),
            "",
            "Resource market:",
        ]
        for resource in ("coal", "oil", "garbage", "uranium"):
            unit_prices = state.resource_market.available_unit_prices(resource)
            preview = ", ".join(str(price) for price in unit_prices[:10]) or "(empty)"
            lines.append(f"- {resource}: {preview}")
        lines.extend(
            [
                "",
                "Supply:",
                ", ".join(
                    f"{resource}={state.resource_market.supply[resource]}"
                    for resource in ("coal", "oil", "garbage", "uranium")
                ),
            ]
        )
        self._set_text("\n".join(lines))

    def _set_text(self, text: str) -> None:
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        self.body.insert("1.0", text)
        self.body.configure(state="disabled")


class EventLogView(SnapshotRenderable):
    def __init__(self, master) -> None:
        super().__init__(master, padding=(10, 8))
        self.body = tk.Text(self, width=42, height=16, wrap="word")
        self.body.pack(fill="both", expand=True)
        self.body.configure(state="disabled")

    def render(self, snapshot: GameSnapshot) -> None:
        if not snapshot.event_log:
            self._set_text("No session events yet.")
            return
        lines = []
        for event in snapshot.event_log[-20:]:
            actor = f" player={event.player_id}" if event.player_id else ""
            phase = f" phase={event.phase}" if event.phase else ""
            lines.append(f"[{event.level}]{phase}{actor} {event.message}")
        self._set_text("\n".join(lines))

    def _set_text(self, text: str) -> None:
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        self.body.insert("1.0", text)
        self.body.configure(state="disabled")


class InspectorView(SnapshotRenderable):
    def __init__(self, master) -> None:
        super().__init__(master, padding=(10, 8))
        self.body = tk.Text(self, width=50, height=16, wrap="none")
        self.body.pack(fill="both", expand=True)
        self.body.configure(state="disabled")

    def render(self, snapshot: GameSnapshot) -> None:
        payload = {
            "active_request": (
                {
                    "player_id": snapshot.active_request.player_id,
                    "phase": snapshot.active_request.phase,
                    "decision_type": snapshot.active_request.decision_type,
                    "metadata": snapshot.active_request.metadata,
                    "legal_actions": [action.to_dict() for action in snapshot.active_request.legal_actions],
                }
                if snapshot.active_request is not None
                else None
            ),
            "last_round_summary": (
                snapshot.last_round_summary.to_dict()
                if snapshot.last_round_summary is not None
                else None
            ),
            "state": snapshot.state.to_dict(),
        }
        self._set_text(json.dumps(payload, indent=2, sort_keys=True))

    def _set_text(self, text: str) -> None:
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        self.body.insert("1.0", text)
        self.body.configure(state="disabled")


def _format_market(plants) -> str:
    if not plants:
        return "(none)"
    return ", ".join("STEP3" if plant.is_step_3_placeholder else str(plant.price) for plant in plants)
