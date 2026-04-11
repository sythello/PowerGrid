from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk

from ..session import GameSnapshot


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
        ttk.Label(self, text="Players", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.body = tk.Text(self, width=36, height=20, wrap="word")
        self.body.pack(fill="both", expand=True, pady=(8, 0))
        self.body.configure(state="disabled")

    def render(self, snapshot: GameSnapshot) -> None:
        lines = []
        active = snapshot.active_request.player_id if snapshot.active_request is not None else None
        for player in sorted(snapshot.state.players, key=lambda item: item.turn_order_position):
            prefix = "> " if player.player_id == active else "  "
            lines.append(
                prefix
                + f"{player.turn_order_position}. {player.player_id} {player.name} [{player.controller}]"
            )
            lines.append(
                f"   Elektro={player.elektro} Cities={player.connected_city_count} Houses={player.houses_in_supply}"
            )
            plants = ", ".join(str(plant.price) for plant in player.power_plants) or "-"
            storage = ", ".join(
                f"{resource}={player.resource_storage.total(resource)}"
                for resource in ("coal", "oil", "garbage", "uranium")
            )
            lines.append(f"   Plants={plants}")
            lines.append(f"   Storage={storage}")
            lines.append("")
        self._set_text("\n".join(lines).rstrip())

    def _set_text(self, text: str) -> None:
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        self.body.insert("1.0", text)
        self.body.configure(state="disabled")


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
