from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..model import GameConfig, SeatConfig
from ..scenarios import SCENARIO_NAMES
from ..session import GameSession, GameSnapshot, GuiIntent
from .board_view import BoardView, PowerPlantMarketView
from .components import EventLogView, HeaderView, PlayerRail
from .panels import AuctionPanel, BuildPanel, BureaucracyPanel, PendingDecisionPanel, ResourcePanel


class LauncherFrame(ttk.Frame):
    def __init__(self, master, on_new_game, on_scenario) -> None:
        super().__init__(master, padding=16)
        self._on_new_game = on_new_game
        self._on_scenario = on_scenario
        self.player_count_var = tk.IntVar(value=3)
        self.seed_var = tk.IntVar(value=7)
        self.map_var = tk.StringVar(value="germany")
        self.scenario_var = tk.StringVar(value=SCENARIO_NAMES[0])
        self.seat_type_vars = [tk.StringVar(value="human") for _ in range(6)]
        ttk.Label(self, text="PowerGrid GUI", font=("Helvetica", 18, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )

        scenario_frame = ttk.LabelFrame(self, text="Load Scenario", padding=12)
        scenario_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12), pady=(16, 0))
        ttk.Label(scenario_frame, text="Scenario").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            scenario_frame,
            textvariable=self.scenario_var,
            values=SCENARIO_NAMES,
            state="readonly",
            width=20,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Button(scenario_frame, text="Load", command=self._load_scenario).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

        game_frame = ttk.LabelFrame(self, text="New Game", padding=12)
        game_frame.grid(row=1, column=1, sticky="nsew", pady=(16, 0))
        ttk.Label(game_frame, text="Map").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            game_frame,
            textvariable=self.map_var,
            values=("germany", "usa", "test"),
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(game_frame, text="Players").grid(row=1, column=0, sticky="w", pady=(8, 0))
        player_box = ttk.Spinbox(
            game_frame,
            from_=3,
            to=6,
            textvariable=self.player_count_var,
            width=6,
            command=self._render_seat_controls,
        )
        player_box.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(game_frame, text="Seed").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(game_frame, from_=0, to=9999, textvariable=self.seed_var, width=8).grid(
            row=2, column=1, sticky="w", pady=(8, 0)
        )
        self.seat_frame = ttk.Frame(game_frame)
        self.seat_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        ttk.Button(game_frame, text="Start Game", command=self._start_new_game).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        self._render_seat_controls()

    def render(self) -> None:
        self._render_seat_controls()

    def _render_seat_controls(self) -> None:
        for child in self.seat_frame.winfo_children():
            child.destroy()
        count = int(self.player_count_var.get())
        for index in range(count):
            ttk.Label(self.seat_frame, text=f"Seat {index + 1}").grid(row=index, column=0, sticky="w")
            ttk.Combobox(
                self.seat_frame,
                textvariable=self.seat_type_vars[index],
                values=("human", "ai"),
                state="readonly",
                width=10,
            ).grid(row=index, column=1, sticky="w", padx=(8, 0))

    def _load_scenario(self) -> None:
        self._on_scenario(self.scenario_var.get(), int(self.seed_var.get()))

    def _start_new_game(self) -> None:
        count = int(self.player_count_var.get())
        seats = tuple(
            SeatConfig(
                player_id=f"p{index + 1}",
                name=f"Player {index + 1}",
                controller=self.seat_type_vars[index].get(),
            )
            for index in range(count)
        )
        self._on_new_game(
            GameConfig(
                map_id=self.map_var.get(),
                players=seats,
                seed=int(self.seed_var.get()),
            )
        )


class GameShell(ttk.Frame):
    def __init__(self, master, on_intent, *, board_render_mode: str = "drawn") -> None:
        super().__init__(master, padding=8)
        self._last_snapshot: GameSnapshot | None = None
        self._current_panel_key = "auction"
        self.header = HeaderView(self)
        self.header.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.player_rail = PlayerRail(self)
        self.player_rail.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.workspace = ttk.Frame(self)
        self.workspace.grid(row=1, column=1, sticky="nsew", padx=(0, 8))
        self.board_view = BoardView(
            self.workspace,
            board_render_mode=board_render_mode,
            on_city_click=self._handle_city_click,
            on_resource_click=self._handle_resource_click,
        )
        self.board_view.grid(row=0, column=0, sticky="nsew")
        self.phase_controls = ttk.Frame(self.workspace, padding=(0, 8, 0, 0))
        self.phase_controls.grid(row=1, column=0, sticky="ew")
        self.right_column = ttk.Frame(self)
        self.right_column.grid(row=1, column=2, sticky="nsew")
        self.market_panel = PowerPlantMarketView(self.right_column, on_plant_click=self._handle_market_plant_click)
        self.market_panel.grid(row=0, column=0, sticky="nsew")
        self.log_panel = EventLogView(self.right_column)
        self.log_panel.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(0, weight=1)
        self.right_column.grid_columnconfigure(0, weight=1)
        self.right_column.grid_rowconfigure(0, weight=0)
        self.right_column.grid_rowconfigure(1, weight=1)
        self.panels = {
            "auction": AuctionPanel(self.phase_controls, on_intent, self._rerender_last_snapshot),
            "buy_resources": ResourcePanel(self.phase_controls, on_intent, self._rerender_last_snapshot),
            "build_houses": BuildPanel(self.phase_controls, on_intent, self._rerender_last_snapshot),
            "bureaucracy": BureaucracyPanel(self.phase_controls, on_intent, self._rerender_last_snapshot),
            "pending": PendingDecisionPanel(self.phase_controls, on_intent, self._rerender_last_snapshot),
        }
        for panel in self.panels.values():
            panel.grid(row=0, column=0, sticky="nsew")

    def render(self, snapshot: GameSnapshot) -> None:
        self._last_snapshot = snapshot
        self.header.render(snapshot)
        self.player_rail.render(snapshot)
        for panel in self.panels.values():
            panel.grid_remove()
        if snapshot.state.pending_decision is not None:
            panel_key = "pending"
        else:
            panel_key = snapshot.state.phase if snapshot.state.phase in self.panels else "auction"
        self._current_panel_key = panel_key
        panel = self.panels[panel_key]
        panel.grid()
        panel.render(snapshot)
        self.board_view.render(snapshot, interaction_state=panel.board_interaction_state())
        self.market_panel.render(snapshot, interaction_state=panel.market_interaction_state())
        self.log_panel.render(snapshot)

    def _handle_city_click(self, city_id: str) -> None:
        panel = self.panels[self._current_panel_key]
        if panel.handle_city_click(city_id) and self._last_snapshot is not None:
            self.render(self._last_snapshot)

    def _handle_resource_click(self, resource: str) -> None:
        panel = self.panels[self._current_panel_key]
        panel.handle_resource_click(resource)

    def _handle_market_plant_click(self, plant_price: int) -> None:
        panel = self.panels[self._current_panel_key]
        if panel.handle_market_plant_click(plant_price) and self._last_snapshot is not None:
            self.render(self._last_snapshot)

    def _rerender_last_snapshot(self) -> None:
        if self._last_snapshot is not None:
            self.render(self._last_snapshot)


class PowerGridApp(ttk.Frame):
    def __init__(self, master, *, board_render_mode: str = "drawn") -> None:
        super().__init__(master, padding=0)
        self.master = master
        self.board_render_mode = board_render_mode
        self.session: GameSession | None = None
        self.launcher = LauncherFrame(self, self.start_new_game, self.load_scenario)
        self.shell = GameShell(self, self.dispatch_intent, board_render_mode=board_render_mode)
        self.launcher.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.shell.grid(row=0, column=0, sticky="nsew")
        self.show_launcher()

    def show_launcher(self) -> None:
        self.shell.grid_remove()
        self.launcher.grid()
        self.launcher.render()

    def start_new_game(self, config: GameConfig) -> None:
        self.session = GameSession.new_game(config)
        self._render_session()

    def load_scenario(self, scenario_name: str, seed: int = 7) -> None:
        self.session = GameSession.from_scenario(scenario_name, seed=seed)
        self._render_session()

    def dispatch_intent(self, intent: GuiIntent) -> None:
        if self.session is None:
            return
        self.session.submit_intent(intent)
        self._render_session()

    def _render_session(self) -> None:
        assert self.session is not None
        snapshot = self.session.advance_until_blocked()
        self.launcher.grid_remove()
        self.shell.grid()
        self.shell.render(snapshot)


def create_root() -> tk.Tk:
    root = tk.Tk()
    root.title("PowerGrid GUI")
    root.geometry("1400x900")
    return root


def launch_app(*, board_render_mode: str = "drawn") -> PowerGridApp:
    root = create_root()
    app = PowerGridApp(root, board_render_mode=board_render_mode)
    app.pack(fill="both", expand=True)
    return app
