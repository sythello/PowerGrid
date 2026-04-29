from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from ..board_layout import load_board_layout, resolve_board_art_path
from ..model import GameState, PowerPlantCard, RESOURCE_TYPES
from ..session_types import GameSnapshot


DRAWN_BOARD_SIZES = {
    "germany": (1424, 2000),
    "usa": (1800, 1120),
    "test": (1200, 780),
}
PLAYER_COLOR_MAP = {
    "black": "#111827",
    "purple": "#7e22ce",
    "green": "#15803d",
    "blue": "#2563eb",
    "yellow": "#ca8a04",
    "red": "#dc2626",
}
RESOURCE_COLOR_MAP = {
    "coal": "#7c4a21",
    "oil": "#111111",
    "garbage": "#facc15",
    "uranium": "#dc2626",
}
RESOURCE_LABEL_MAP = {
    "coal": "coal",
    "oil": "oil",
    "garbage": "garbage",
    "uranium": "uranium",
}
RESOURCE_MARKET_PRICE_COLUMNS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)
RESOURCE_MARKET_VALID_PRICES = {
    "coal": set(range(1, 9)),
    "oil": set(range(1, 9)),
    "garbage": set(range(1, 9)),
    "uranium": set(RESOURCE_MARKET_PRICE_COLUMNS),
}
BOARD_BG = "#d8cfbe"
BOARD_PANEL = "#ece4d3"
PIPE_FILL = "#d1d5db"
PIPE_BADGE_FILL = "#4b5563"
PIPE_BADGE_OUTLINE = "#d4a017"
CITY_FILL = "#4b5563"
CITY_OUTLINE = "#1f2937"
CITY_SLOT_FILL = "#d1d5db"
CITY_NAME_FILL = "#f8fafc"
CITY_NAME_TEXT = "#111827"
INACTIVE_CITY_FILL = "#9ca3af"
INACTIVE_CITY_SLOT = "#e5e7eb"
BOARD_TEXT = "#1f2937"
RESOURCE_MARKET_PANEL_FILL = "#efe8db"
RESOURCE_MARKET_HEADER_FILL = "#ddd0b8"
RESOURCE_MARKET_CELL_FILL = "#f8f4ea"
RESOURCE_MARKET_DISABLED_FILL = "#ddd3c0"
RESOURCE_MARKET_BORDER = "#c4b79f"


class BoardView(ttk.Frame):
    def __init__(
        self,
        master,
        *,
        board_render_mode: str = "drawn",
        layout_path: str | Path | None = None,
        on_city_click=None,
        on_resource_click=None,
    ) -> None:
        super().__init__(master, padding=(0, 0))
        self._layout_path = layout_path
        self._board_render_mode = board_render_mode
        self._on_city_click = on_city_click
        self._on_resource_click = on_resource_click
        self._images: dict[Path, tk.PhotoImage] = {}
        self._board_size = DRAWN_BOARD_SIZES["germany"]
        self._interaction_state: dict[str, object] = {}
        self.summary_var = tk.StringVar()
        ttk.Label(self, textvariable=self.summary_var, anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 0)
        )
        self.canvas = tk.Canvas(self, background="#0f172a", highlightthickness=0)
        self.y_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.x_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.y_scroll.set, xscrollcommand=self.x_scroll.set)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.y_scroll.grid(row=1, column=1, sticky="ns")
        self.x_scroll.grid(row=2, column=0, sticky="ew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def set_render_mode(self, board_render_mode: str) -> None:
        self._board_render_mode = board_render_mode

    def render(self, snapshot: GameSnapshot, interaction_state: dict[str, object] | None = None) -> None:
        state = snapshot.state
        self._interaction_state = dict(interaction_state or {})
        try:
            map_layout = load_board_layout(state.game_map.id, self._layout_path)
        except ValueError:
            map_layout = {"cities": {}, "board_art": {}, "resource_market": None, "city_defaults": {}}
        board_image = self._load_board_image(map_layout)
        width, height = self._board_dimensions(state.game_map.id, map_layout, board_image)
        self._board_size = (width, height)
        self.canvas.delete("all")
        if board_image is not None:
            self.canvas.create_image(0, 0, anchor="nw", image=board_image)
            self.canvas.image = board_image
        else:
            self.canvas.image = None
            self._draw_background(state, width, height)

        positions = self._city_positions(state, map_layout, width, height)
        self._draw_connections(state, positions)
        self._draw_cities(state, map_layout, positions)
        self._draw_houses(state, map_layout, positions)
        self._draw_unplayed_city_clouds(state, positions)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        missing = len(state.game_map.cities) - len(positions)
        image_status = (
            f"asset mode using {resolve_board_art_path(map_layout.get('board_art', {}).get('image_path'), self._layout_path)}"
            if board_image is not None
            else f"{self._board_render_mode} mode"
        )
        self.summary_var.set(
            f"Board: {state.game_map.name} | {image_status} | mapped cities: {len(positions)}/{len(state.game_map.cities)}"
            + (f" | missing coordinates: {missing}" if missing else "")
        )

    def _load_board_image(self, map_layout: dict) -> tk.PhotoImage | None:
        if self._board_render_mode != "asset":
            return None
        board_art = map_layout.get("board_art", {})
        path = resolve_board_art_path(board_art.get("image_path"), self._layout_path)
        if path is None or not path.exists():
            return None
        cached = self._images.get(path)
        if cached is not None:
            return cached
        try:
            image = tk.PhotoImage(file=str(path))
        except tk.TclError:
            return None
        self._images[path] = image
        return image

    def _board_dimensions(
        self,
        map_id: str,
        map_layout: dict,
        board_image: tk.PhotoImage | None,
    ) -> tuple[int, int]:
        if board_image is not None:
            return board_image.width(), board_image.height()
        natural_size = map_layout.get("board_art", {}).get("natural_size", {})
        width = natural_size.get("width")
        height = natural_size.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height
        return DRAWN_BOARD_SIZES.get(map_id, (1600, 1000))

    def _draw_background(self, state: GameState, width: int, height: int) -> None:
        self.canvas.create_rectangle(0, 0, width, height, fill=BOARD_BG, outline="")
        self.canvas.create_rectangle(24, 24, width - 24, height - 24, fill=BOARD_PANEL, outline="#c4b79f", width=3)
        self.canvas.create_text(
            48,
            42,
            text=f"{state.game_map.name} Grid",
            anchor="w",
            fill=BOARD_TEXT,
            font=("Helvetica", 24, "bold"),
        )
        self.canvas.create_text(
            48,
            74,
            text=f"Round {state.round_number}  Phase {state.phase.replace('_', ' ')}  Step {state.step}",
            anchor="w",
            fill=BOARD_TEXT,
            font=("Helvetica", 12),
        )
        if state.selected_regions:
            self.canvas.create_text(
                width - 48,
                50,
                text="Regions: " + ", ".join(state.selected_regions),
                anchor="e",
                fill=BOARD_TEXT,
                font=("Helvetica", 12),
            )
        if state.game_map.id != "test":
            self.canvas.create_text(
                width / 2,
                height - 48,
                text="Map silhouette is schematic until city coordinates are filled in board_layout_placeholders.json",
                fill="#6b7280",
                font=("Helvetica", 12, "italic"),
            )

    def _city_positions(self, state: GameState, map_layout: dict, width: int, height: int) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        for city in state.game_map.cities:
            city_payload = map_layout.get("cities", {}).get(city.id, {})
            anchor = city_payload.get("anchor", {})
            anchor_x = anchor.get("x")
            anchor_y = anchor.get("y")
            if anchor_x is None or anchor_y is None:
                continue
            positions[city.id] = (float(anchor_x) * width, float(anchor_y) * height)
        return positions

    def _draw_connections(self, state: GameState, positions: dict[str, tuple[float, float]]) -> None:
        for connection in state.game_map.connections:
            start = positions.get(connection.city_1)
            end = positions.get(connection.city_2)
            if start is None or end is None:
                continue
            self._draw_pipe(start, end, connection.cost)

    def _draw_pipe(self, start: tuple[float, float], end: tuple[float, float], cost: int) -> None:
        start_x, start_y = start
        end_x, end_y = end
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.hypot(dx, dy)
        if length == 0:
            return
        nx = -dy / length
        ny = dx / length
        half_width = 5
        points = [
            start_x + nx * half_width,
            start_y + ny * half_width,
            end_x + nx * half_width,
            end_y + ny * half_width,
            end_x - nx * half_width,
            end_y - ny * half_width,
            start_x - nx * half_width,
            start_y - ny * half_width,
        ]
        self.canvas.create_polygon(points, fill=PIPE_FILL, outline="")
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        self.canvas.create_oval(
            mid_x - 16,
            mid_y - 16,
            mid_x + 16,
            mid_y + 16,
            fill=PIPE_BADGE_FILL,
            outline=PIPE_BADGE_OUTLINE,
            width=2,
        )
        _draw_outlined_text(
            self.canvas,
            mid_x,
            mid_y,
            text=str(cost),
            fill="#ffffff",
            outline="#000000",
            font=("Helvetica", 11, "bold"),
        )

    def _draw_resource_market(self, state: GameState, map_layout: dict, width: int, height: int) -> None:
        has_layout_slots = self._has_concrete_resource_market_slots(map_layout)
        buyable_resources = set(self._interaction_state.get("buyable_resources", ()))
        resource_phase_active = bool(self._interaction_state.get("resource_phase_active"))
        if not has_layout_slots:
            self._draw_resource_market_table(
                width,
                height,
                buyable_resources=buyable_resources,
                resource_phase_active=resource_phase_active,
            )
        for row_index, resource in enumerate(RESOURCE_TYPES):
            for price in sorted(state.resource_market.market[resource]):
                amount = state.resource_market.market[resource][price]
                if has_layout_slots:
                    for index in range(amount):
                        point = self._resource_slot_point(
                            map_layout,
                            width,
                            height,
                            resource,
                            price,
                            index,
                            row_index,
                        )
                        tags = ()
                        if (
                            resource_phase_active
                            and resource in buyable_resources
                            and self._on_resource_click is not None
                        ):
                            tags = (f"resource:{resource}:{price}:{index}",)
                        item_ids = self._draw_resource_token(resource, point[0], point[1], scale=0.7, tags=tags)
                        if tags:
                            self._bind_items_click(item_ids, lambda _event, resource=resource: self._on_resource_click(resource))
                    continue
                for point in self._resource_table_cell_points(resource, price, amount, row_index, width, height):
                    tags = ()
                    if (
                        resource_phase_active
                        and resource in buyable_resources
                        and self._on_resource_click is not None
                    ):
                        tags = (f"resource:{resource}:{price}:{point[0]:.1f}:{point[1]:.1f}",)
                    item_ids = self._draw_resource_token(resource, point[0], point[1], scale=0.58, tags=tags)
                    if tags:
                        self._bind_items_click(item_ids, lambda _event, resource=resource: self._on_resource_click(resource))

    def _has_concrete_resource_market_slots(self, map_layout: dict) -> bool:
        market_layout = map_layout.get("resource_market")
        if not isinstance(market_layout, dict):
            return False
        found_any = False
        for resource_slots in market_layout.values():
            if not isinstance(resource_slots, dict):
                continue
            for slots in resource_slots.values():
                for slot in slots or ():
                    found_any = True
                    if slot.get("x") is None or slot.get("y") is None:
                        return False
        return found_any

    def _draw_resource_market_table(
        self,
        width: int,
        height: int,
        *,
        buyable_resources: set[str],
        resource_phase_active: bool,
    ) -> None:
        geometry = _resource_market_table_geometry(width, height)
        panel_x = geometry["panel_x"]
        panel_y = geometry["panel_y"]
        panel_width = geometry["panel_width"]
        panel_height = geometry["panel_height"]
        grid_x = geometry["grid_x"]
        grid_y = geometry["grid_y"]
        label_width = geometry["label_width"]
        header_height = geometry["header_height"]
        cell_width = geometry["cell_width"]
        cell_height = geometry["cell_height"]

        self.canvas.create_rectangle(
            panel_x,
            panel_y,
            panel_x + panel_width,
            panel_y + panel_height,
            fill=RESOURCE_MARKET_PANEL_FILL,
            outline=RESOURCE_MARKET_BORDER,
            width=2,
        )
        self.canvas.create_text(
            panel_x + 14,
            panel_y + 16,
            text="Resource Market",
            anchor="w",
            fill=BOARD_TEXT,
            font=("Helvetica", 12, "bold"),
        )

        self.canvas.create_rectangle(
            grid_x,
            grid_y,
            grid_x + label_width,
            grid_y + header_height,
            fill=RESOURCE_MARKET_HEADER_FILL,
            outline=RESOURCE_MARKET_BORDER,
            width=1,
        )
        self.canvas.create_text(
            grid_x + label_width / 2,
            grid_y + header_height / 2,
            text="Type",
            fill=BOARD_TEXT,
            font=("Helvetica", 9, "bold"),
        )
        for price_index, price in enumerate(RESOURCE_MARKET_PRICE_COLUMNS):
            cell_x = grid_x + label_width + price_index * cell_width
            self.canvas.create_rectangle(
                cell_x,
                grid_y,
                cell_x + cell_width,
                grid_y + header_height,
                fill=RESOURCE_MARKET_HEADER_FILL,
                outline=RESOURCE_MARKET_BORDER,
                width=1,
            )
            self.canvas.create_text(
                cell_x + cell_width / 2,
                grid_y + header_height / 2,
                text=str(price),
                fill=BOARD_TEXT,
                font=("Helvetica", 8, "bold"),
            )

        for row_index, resource in enumerate(RESOURCE_TYPES):
            row_y = grid_y + header_height + row_index * cell_height
            is_buyable = not resource_phase_active or resource in buyable_resources
            label_fill = RESOURCE_MARKET_HEADER_FILL if is_buyable else RESOURCE_MARKET_DISABLED_FILL
            label_text = BOARD_TEXT if is_buyable else "#6b7280"
            self.canvas.create_rectangle(
                grid_x,
                row_y,
                grid_x + label_width,
                row_y + cell_height,
                fill=label_fill,
                outline=RESOURCE_MARKET_BORDER,
                width=1,
            )
            self.canvas.create_text(
                grid_x + label_width / 2,
                row_y + cell_height / 2,
                text=resource.title(),
                fill=label_text,
                font=("Helvetica", 9, "bold"),
            )
            for price_index, price in enumerate(RESOURCE_MARKET_PRICE_COLUMNS):
                cell_x = grid_x + label_width + price_index * cell_width
                fill = (
                    RESOURCE_MARKET_CELL_FILL
                    if price in RESOURCE_MARKET_VALID_PRICES[resource]
                    else RESOURCE_MARKET_DISABLED_FILL
                )
                if resource_phase_active and resource not in buyable_resources and price in RESOURCE_MARKET_VALID_PRICES[resource]:
                    fill = RESOURCE_MARKET_DISABLED_FILL
                self.canvas.create_rectangle(
                    cell_x,
                    row_y,
                    cell_x + cell_width,
                    row_y + cell_height,
                    fill=fill,
                    outline=RESOURCE_MARKET_BORDER,
                    width=1,
                )

    def _resource_slot_point(
        self,
        map_layout: dict,
        width: int,
        height: int,
        resource: str,
        price: int,
        index: int,
        row_index: int,
    ) -> tuple[float, float]:
        market_layout = map_layout.get("resource_market")
        slots = None
        if isinstance(market_layout, dict):
            slots = market_layout.get(resource, {}).get(str(price)) or market_layout.get(resource, {}).get(price)
        if slots:
            concrete = [
                (float(slot["x"]) * width, float(slot["y"]) * height)
                for slot in slots
                if slot.get("x") is not None and slot.get("y") is not None
            ]
            if index < len(concrete):
                return concrete[index]
        return _fallback_resource_slot(resource, price, index, row_index, width, height)

    def _resource_table_cell_points(
        self,
        resource: str,
        price: int,
        amount: int,
        row_index: int,
        width: int,
        height: int,
    ) -> list[tuple[float, float]]:
        if amount <= 0:
            return []
        geometry = _resource_market_table_geometry(width, height)
        price_index = RESOURCE_MARKET_PRICE_COLUMNS.index(price)
        center_x = (
            geometry["grid_x"]
            + geometry["label_width"]
            + price_index * geometry["cell_width"]
            + geometry["cell_width"] / 2
        )
        center_y = (
            geometry["grid_y"]
            + geometry["header_height"]
            + row_index * geometry["cell_height"]
            + geometry["cell_height"] / 2
        )
        if resource == "uranium" or amount == 1:
            offsets = [(0.0, 0.0)]
        elif amount == 2:
            offsets = [(-6.0, 0.0), (6.0, 0.0)]
        else:
            offsets = [(-7.0, 6.0), (0.0, -6.0), (7.0, 6.0)]
        return [(center_x + offset_x, center_y + offset_y) for offset_x, offset_y in offsets[:amount]]

    def _draw_resource_token(
        self,
        resource: str,
        x: float,
        y: float,
        *,
        scale: float = 1.0,
        tags: tuple[str, ...] = (),
    ) -> list[int]:
        return _draw_resource_token_on_canvas(self.canvas, resource, x, y, scale=scale, tags=tags)

    def _draw_cities(
        self,
        state: GameState,
        map_layout: dict,
        positions: dict[str, tuple[float, float]],
    ) -> None:
        city_defaults = map_layout.get("city_defaults", {})
        selected_city_ids = set(self._interaction_state.get("selected_city_ids", ()))
        clickable_city_ids = set(self._interaction_state.get("clickable_city_ids", ()))
        for city in state.game_map.cities:
            point = positions.get(city.id)
            if point is None:
                continue
            city_payload = map_layout.get("cities", {}).get(city.id, {})
            is_active_region = not state.selected_regions or city.region in state.selected_regions
            item_ids = self._draw_city(
                point[0],
                point[1],
                city.name,
                is_active_region,
                city_payload,
                city_defaults,
                is_selected=(city.id in selected_city_ids),
                is_clickable=(city.id in clickable_city_ids),
            )
            if city.id in clickable_city_ids and self._on_city_click is not None:
                self._bind_items_click(item_ids, lambda _event, city_id=city.id: self._on_city_click(city_id))

    def _draw_city(
        self,
        x: float,
        y: float,
        name: str,
        is_active_region: bool,
        city_payload: dict,
        city_defaults: dict,
        *,
        is_selected: bool,
        is_clickable: bool,
    ) -> list[int]:
        radius = 34
        fill = CITY_FILL if is_active_region else INACTIVE_CITY_FILL
        slot_fill = CITY_SLOT_FILL if is_active_region else INACTIVE_CITY_SLOT
        outline = "#d97706" if is_selected else ("#2563eb" if is_clickable else CITY_OUTLINE)
        outline_width = 3 if is_selected else (2.5 if is_clickable else 2)
        item_ids = [
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=fill,
                outline=outline,
                width=outline_width,
            )
        ]
        item_ids.append(
            self.canvas.create_arc(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            start=35,
            extent=110,
            fill=slot_fill,
            outline=outline,
            style="pieslice",
        ))
        item_ids.append(self.canvas.create_arc(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            start=150,
            extent=85,
            fill=slot_fill,
            outline=outline,
            style="pieslice",
        ))
        item_ids.append(self.canvas.create_arc(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            start=305,
            extent=85,
            fill=slot_fill,
            outline=outline,
            style="pieslice",
        ))
        inner = radius * 0.56
        item_ids.append(self.canvas.create_oval(x - inner, y - inner, x + inner, y + inner, fill=fill, outline=""))
        item_ids.append(
            self.canvas.create_text(x, y - radius * 0.43, text="10", fill=BOARD_TEXT, font=("Helvetica", 8, "bold"))
        )
        item_ids.append(self.canvas.create_text(
            x - radius * 0.42,
            y + radius * 0.28,
            text="15",
            fill=BOARD_TEXT,
            font=("Helvetica", 8, "bold"),
        ))
        item_ids.append(self.canvas.create_text(
            x + radius * 0.42,
            y + radius * 0.28,
            text="20",
            fill=BOARD_TEXT,
            font=("Helvetica", 8, "bold"),
        ))
        label_x, label_y = self._city_label_center(x, y, radius, city_payload, city_defaults)
        half_width = max(34, min(70, len(name) * 4.5))
        item_ids.append(self.canvas.create_rectangle(
            label_x - half_width,
            label_y - 11,
            label_x + half_width,
            label_y + 11,
            fill=CITY_NAME_FILL,
            outline=outline if is_selected else CITY_OUTLINE,
            width=1,
        ))
        item_ids.append(
            self.canvas.create_text(label_x, label_y, text=name, fill=CITY_NAME_TEXT, font=("Helvetica", 9, "bold"))
        )
        return item_ids

    def _city_label_center(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        city_payload: dict,
        city_defaults: dict,
    ) -> tuple[float, float]:
        return resolve_city_label_center(
            center_x,
            center_y,
            self._board_size,
            city_payload,
            city_defaults,
            fallback_radius=radius,
        )

    def _draw_houses(
        self,
        state: GameState,
        map_layout: dict,
        positions: dict[str, tuple[float, float]],
    ) -> None:
        occupancy: dict[str, list[str]] = {}
        players_by_id = {player.player_id: player for player in state.players}
        for player in sorted(state.players, key=lambda item: item.turn_order_position):
            for city_id in player.network_city_ids:
                occupancy.setdefault(city_id, []).append(player.player_id)
        for city_id, player_ids in occupancy.items():
            point = positions.get(city_id)
            if point is None:
                continue
            city_payload = map_layout.get("cities", {}).get(city_id, {})
            slot_points = self._city_slot_points(point[0], point[1], city_payload)
            for index, player_id in enumerate(player_ids[: len(slot_points)]):
                player = players_by_id[player_id]
                slot_x, slot_y = slot_points[index]
                _draw_house_icon(self.canvas, slot_x, slot_y, PLAYER_COLOR_MAP.get(player.color, player.color))

    def _city_slot_points(
        self,
        center_x: float,
        center_y: float,
        city_payload: dict,
    ) -> list[tuple[float, float]]:
        slot_points: list[tuple[float, float]] = []
        for slot in city_payload.get("house_slots", ()):
            if slot.get("x") is not None and slot.get("y") is not None:
                slot_points.append((float(slot["x"]), float(slot["y"])))
        if slot_points:
            board_width, board_height = self._board_size
            return [(x * board_width, y * board_height) for x, y in slot_points]
        return [
            (center_x, center_y - 18),
            (center_x - 18, center_y + 16),
            (center_x + 18, center_y + 16),
        ]

    def _draw_unplayed_city_clouds(
        self,
        state: GameState,
        positions: dict[str, tuple[float, float]],
    ) -> None:
        if not state.selected_regions:
            return
        selected_regions = set(state.selected_regions)
        for city in state.game_map.cities:
            if city.region in selected_regions:
                continue
            point = positions.get(city.id)
            if point is None:
                continue
            self._draw_cloud_mask(point[0], point[1])

    def _draw_cloud_mask(self, center_x: float, center_y: float) -> None:
        offsets = (
            (-34, 10, 28, 22),
            (-10, -12, 26, 24),
            (18, 8, 30, 22),
            (-4, 20, 40, 18),
        )
        bounds = [center_x - 52, center_y - 28, center_x + 56, center_y + 44]
        item_ids: list[int] = []
        for offset_x, offset_y, radius_x, radius_y in offsets:
            left = center_x + offset_x - radius_x
            top = center_y + offset_y - radius_y
            right = center_x + offset_x + radius_x
            bottom = center_y + offset_y + radius_y
            bounds[0] = min(bounds[0], left)
            bounds[1] = min(bounds[1], top)
            bounds[2] = max(bounds[2], right)
            bounds[3] = max(bounds[3], bottom)
            item_ids.append(
                self.canvas.create_oval(
                    left,
                    top,
                    right,
                    bottom,
                    fill="#ffffff",
                    outline="#111111",
                    width=2,
                )
            )
        item_ids.append(
            self.canvas.create_rectangle(
                bounds[0] + 8,
                center_y + 8,
                bounds[2] - 8,
                bounds[3] - 4,
                fill="#ffffff",
                outline="#111111",
                width=2,
            )
        )
        for item_id in item_ids[:-1]:
            self.canvas.tag_raise(item_id)
        self.canvas.tag_raise(item_ids[-1])

    def _bind_items_click(self, item_ids: list[int], callback) -> None:
        _bind_canvas_items_click(self.canvas, item_ids, callback)


class PowerPlantMarketView(ttk.Frame):
    def __init__(self, master, *, on_plant_click=None) -> None:
        super().__init__(master, padding=(8, 8))
        self._on_plant_click = on_plant_click
        ttk.Label(self, text="Power Plant Market", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=360, height=330, background="#f4efe6", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(8, 0))

    def render(self, snapshot: GameSnapshot, interaction_state: dict[str, object] | None = None) -> None:
        state = snapshot.state
        auction_state = state.auction_state
        discount_price = auction_state.discount_token_plant_price if auction_state is not None else None
        interaction_state = dict(interaction_state or {})
        selected_price = interaction_state.get("selected_plant_price")
        clickable_prices = set(interaction_state.get("clickable_plant_prices", ()))
        self.canvas.delete("all")
        self.canvas.create_text(18, 18, text="Current", anchor="w", fill=BOARD_TEXT, font=("Helvetica", 10, "bold"))
        self.canvas.create_text(18, 156, text="Future", anchor="w", fill=BOARD_TEXT, font=("Helvetica", 10, "bold"))
        for index, plant in enumerate(state.current_market):
            x = 18 + index * 84
            y = 28
            item_ids = draw_power_plant_card(
                self.canvas,
                x,
                y,
                plant,
                discount_token=(discount_price == plant.price),
                selected=(selected_price == plant.price),
                clickable=(plant.price in clickable_prices),
            )
            if plant.price in clickable_prices and self._on_plant_click is not None:
                for item_id in item_ids:
                    self.canvas.tag_bind(
                        item_id,
                        "<Button-1>",
                        lambda _event, price=plant.price: self._on_plant_click(price),
                    )
        for index, plant in enumerate(state.future_market):
            x = 18 + index * 84
            y = 166
            draw_power_plant_card(self.canvas, x, y, plant)

        draw_x = 18
        draw_y = 270
        self.canvas.create_text(draw_x, draw_y - 10, text="Deck", anchor="w", fill=BOARD_TEXT, font=("Helvetica", 10, "bold"))
        _draw_deck_back_card(
            self.canvas,
            draw_x,
            draw_y,
            state.power_plant_draw_stack[0] if state.power_plant_draw_stack else None,
            len(state.power_plant_draw_stack),
        )
        discount_text = f"discount token: plant {discount_price}" if discount_price is not None else "discount token: off market"
        self.canvas.create_text(132, 300, text=discount_text, anchor="w", fill=BOARD_TEXT, font=("Helvetica", 10))
        bottom_count = len(state.power_plant_bottom_stack)
        self.canvas.create_text(
            132,
            280,
            text=f"draw stack: {len(state.power_plant_draw_stack)}  bottom stack: {bottom_count}",
            anchor="w",
            fill=BOARD_TEXT,
            font=("Helvetica", 10),
        )


class ResourceMarketView(ttk.Frame):
    def __init__(self, master, *, on_resource_click=None) -> None:
        super().__init__(master, padding=(8, 8))
        self._on_resource_click = on_resource_click
        ttk.Label(self, text="Resource Market", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=360, height=228, background="#f4efe6", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(8, 0))

    def render(self, snapshot: GameSnapshot, interaction_state: dict[str, object] | None = None) -> None:
        interaction_state = dict(interaction_state or {})
        buyable_resources = set(interaction_state.get("buyable_resources", ()))
        resource_phase_active = bool(interaction_state.get("resource_phase_active"))
        width = max(336, self.canvas.winfo_width())
        height = max(214, self.canvas.winfo_height())
        self.canvas.delete("all")
        _draw_resource_market_sidebar(
            self.canvas,
            snapshot.state,
            width,
            height,
            buyable_resources=buyable_resources,
            resource_phase_active=resource_phase_active,
            on_resource_click=self._on_resource_click,
        )
        self.canvas.configure(scrollregion=(0, 0, width, height))


def draw_power_plant_card(
    canvas: tk.Canvas,
    x: float,
    y: float,
    plant: PowerPlantCard,
    *,
    size: int = 72,
    discount_token: bool = False,
    selected: bool = False,
    clickable: bool = False,
) -> list[int]:
    fill = "#fffaf0" if not plant.is_step_3_placeholder else "#d6d3d1"
    outline = "#d97706" if selected else ("#2563eb" if clickable else "#334155")
    width = 3 if selected else (2.5 if clickable else 2)
    item_ids: list[int] = []
    item_ids.append(canvas.create_rectangle(x, y, x + size, y + size, fill=fill, outline=outline, width=width))
    price_text = "STEP3" if plant.is_step_3_placeholder else str(plant.price)
    item_ids.append(canvas.create_text(x + 8, y + 8, text=price_text, anchor="nw", fill=BOARD_TEXT, font=("Helvetica", 12, "bold")))
    if plant.is_step_3_placeholder:
        item_ids.append(
            canvas.create_text(x + size / 2, y + size / 2, text="Deck\nShift", fill=BOARD_TEXT, font=("Helvetica", 10, "bold"))
        )
    else:
        resource_text = "eco" if plant.is_ecological else "/".join(RESOURCE_LABEL_MAP[item] for item in plant.resource_types)
        item_ids.append(
            canvas.create_text(
                x + size / 2,
                y + size / 2 - 4,
                text=resource_text,
                fill=BOARD_TEXT,
                font=("Helvetica", 9, "bold"),
                width=size - 12,
            )
        )
        item_ids.append(
            canvas.create_text(
                x + 8,
                y + size - 10,
                text=f"r {plant.resource_cost}",
                anchor="sw",
                fill=BOARD_TEXT,
                font=("Helvetica", 10, "bold"),
            )
        )
        item_ids.append(
            canvas.create_text(
                x + size - 8,
                y + size - 10,
                text=f"h {plant.output_cities}",
                anchor="se",
                fill=BOARD_TEXT,
                font=("Helvetica", 10, "bold"),
            )
        )
    if discount_token:
        item_ids.append(canvas.create_oval(x + size - 18, y + 6, x + size - 4, y + 20, fill="#d4a017", outline="#7c5d0d", width=2))
    return item_ids


def _draw_deck_back_card(
    canvas: tk.Canvas,
    x: float,
    y: float,
    top_card: PowerPlantCard | None,
    deck_count: int,
) -> None:
    canvas.create_rectangle(x, y, x + 72, y + 72, fill="#374151", outline="#111827", width=2)
    label = "empty" if top_card is None else top_card.deck_back
    canvas.create_text(x + 36, y + 28, text="deck", fill="#f8fafc", font=("Helvetica", 11, "bold"))
    canvas.create_text(x + 36, y + 46, text=label, fill="#f8fafc", font=("Helvetica", 10))
    canvas.create_text(x + 36, y + 61, text=f"{deck_count} left", fill="#f8fafc", font=("Helvetica", 9))


def _draw_resource_market_sidebar(
    canvas: tk.Canvas,
    state: GameState,
    width: int,
    height: int,
    *,
    buyable_resources: set[str],
    resource_phase_active: bool,
    on_resource_click,
) -> None:
    geometry = _resource_market_sidebar_geometry(width, height)
    panel_x = geometry["panel_x"]
    panel_y = geometry["panel_y"]
    panel_width = geometry["panel_width"]
    panel_height = geometry["panel_height"]
    grid_x = geometry["grid_x"]
    grid_y = geometry["grid_y"]
    label_width = geometry["label_width"]
    header_height = geometry["header_height"]
    cell_width = geometry["cell_width"]
    cell_height = geometry["cell_height"]

    canvas.create_rectangle(
        panel_x,
        panel_y,
        panel_x + panel_width,
        panel_y + panel_height,
        fill=RESOURCE_MARKET_PANEL_FILL,
        outline=RESOURCE_MARKET_BORDER,
        width=2,
    )
    canvas.create_text(
        panel_x + 12,
        panel_y + 14,
        text="Type",
        anchor="w",
        fill=BOARD_TEXT,
        font=("Helvetica", 10, "bold"),
    )
    for price_index, price in enumerate(RESOURCE_MARKET_PRICE_COLUMNS):
        cell_x = grid_x + label_width + price_index * cell_width
        canvas.create_rectangle(
            cell_x,
            grid_y,
            cell_x + cell_width,
            grid_y + header_height,
            fill=RESOURCE_MARKET_HEADER_FILL,
            outline=RESOURCE_MARKET_BORDER,
            width=1,
        )
        canvas.create_text(
            cell_x + cell_width / 2,
            grid_y + header_height / 2,
            text=str(price),
            fill=BOARD_TEXT,
            font=("Helvetica", 7, "bold"),
        )
    canvas.create_rectangle(
        grid_x,
        grid_y,
        grid_x + label_width,
        grid_y + header_height,
        fill=RESOURCE_MARKET_HEADER_FILL,
        outline=RESOURCE_MARKET_BORDER,
        width=1,
    )
    canvas.create_text(
        grid_x + label_width / 2,
        grid_y + header_height / 2,
        text="Resource",
        fill=BOARD_TEXT,
        font=("Helvetica", 8, "bold"),
    )

    for row_index, resource in enumerate(RESOURCE_TYPES):
        row_y = grid_y + header_height + row_index * cell_height
        is_buyable = not resource_phase_active or resource in buyable_resources
        label_fill = RESOURCE_MARKET_HEADER_FILL if is_buyable else RESOURCE_MARKET_DISABLED_FILL
        label_text = BOARD_TEXT if is_buyable else "#6b7280"
        canvas.create_rectangle(
            grid_x,
            row_y,
            grid_x + label_width,
            row_y + cell_height,
            fill=label_fill,
            outline=RESOURCE_MARKET_BORDER,
            width=1,
        )
        canvas.create_text(
            grid_x + label_width / 2,
            row_y + cell_height / 2,
            text=resource.title(),
            fill=label_text,
            font=("Helvetica", 8, "bold"),
        )
        for price_index, price in enumerate(RESOURCE_MARKET_PRICE_COLUMNS):
            cell_x = grid_x + label_width + price_index * cell_width
            fill = (
                RESOURCE_MARKET_CELL_FILL
                if price in RESOURCE_MARKET_VALID_PRICES[resource]
                else RESOURCE_MARKET_DISABLED_FILL
            )
            if resource_phase_active and resource not in buyable_resources and price in RESOURCE_MARKET_VALID_PRICES[resource]:
                fill = RESOURCE_MARKET_DISABLED_FILL
            canvas.create_rectangle(
                cell_x,
                row_y,
                cell_x + cell_width,
                row_y + cell_height,
                fill=fill,
                outline=RESOURCE_MARKET_BORDER,
                width=1,
            )
            amount = state.resource_market.market[resource].get(price, 0)
            if amount <= 0:
                continue
            for point in _sidebar_resource_cell_points(geometry, resource, price, amount, row_index):
                tags = ()
                if resource_phase_active and resource in buyable_resources and on_resource_click is not None:
                    tags = (f"resource:{resource}:{price}:{point[0]:.1f}:{point[1]:.1f}",)
                item_ids = _draw_resource_token_on_canvas(canvas, resource, point[0], point[1], scale=0.48, tags=tags)
                if tags:
                    _bind_canvas_items_click(
                        canvas,
                        item_ids,
                        lambda _event, resource=resource: on_resource_click(resource),
                    )


def _draw_resource_token_on_canvas(
    canvas: tk.Canvas,
    resource: str,
    x: float,
    y: float,
    *,
    scale: float = 1.0,
    tags: tuple[str, ...] = (),
) -> list[int]:
    color = RESOURCE_COLOR_MAP[resource]
    if resource == "coal":
        points = _regular_polygon_points(x, y, 10 * scale, 6, rotation=math.pi / 6)
        return [canvas.create_polygon(points, fill=color, outline="#1f2937", tags=tags)]
    if resource == "oil":
        points = [
            x,
            y - 12 * scale,
            x + 8 * scale,
            y,
            x + 5 * scale,
            y + 10 * scale,
            x,
            y + 14 * scale,
            x - 5 * scale,
            y + 10 * scale,
            x - 8 * scale,
            y,
        ]
        return [canvas.create_polygon(points, fill=color, outline="#374151", smooth=True, tags=tags)]
    if resource == "garbage":
        return [canvas.create_rectangle(
            x - 5 * scale,
            y - 11 * scale,
            x + 5 * scale,
            y + 11 * scale,
            fill=color,
            outline="#374151",
            tags=tags,
        )]
    return [canvas.create_oval(
        x - 9 * scale,
        y - 9 * scale,
        x + 9 * scale,
        y + 9 * scale,
        fill=color,
        outline="#7f1d1d",
        tags=tags,
    )]


def _bind_canvas_items_click(canvas: tk.Canvas, item_ids: list[int], callback) -> None:
    for item_id in item_ids:
        canvas.tag_bind(item_id, "<Button-1>", callback)


def _draw_house_icon(canvas: tk.Canvas, x: float, y: float, fill: str) -> None:
    points = [
        x,
        y - 10,
        x + 10,
        y - 1,
        x + 7,
        y - 1,
        x + 7,
        y + 10,
        x - 7,
        y + 10,
        x - 7,
        y - 1,
        x - 10,
        y - 1,
    ]
    canvas.create_polygon(points, fill=fill, outline="#f8fafc", width=1.5)
    canvas.create_rectangle(x - 2, y + 2, x + 2, y + 10, fill="#f8fafc", outline="")


def _draw_outlined_text(
    canvas: tk.Canvas,
    x: float,
    y: float,
    *,
    text: str,
    fill: str,
    outline: str,
    font: tuple[str, int, str] | tuple[str, int],
) -> None:
    for offset_x, offset_y in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        canvas.create_text(x + offset_x, y + offset_y, text=text, fill=outline, font=font)
    canvas.create_text(x, y, text=text, fill=fill, font=font)


def resolve_city_label_center(
    center_x: float,
    center_y: float,
    board_size: tuple[int, int],
    city_payload: dict,
    city_defaults: dict,
    *,
    fallback_radius: float = 34,
) -> tuple[float, float]:
    offset_payload = city_payload.get("label_offset") or city_defaults.get("label_offset") or {}
    offset_x = offset_payload.get("x")
    offset_y = offset_payload.get("y")
    board_width, board_height = board_size
    if isinstance(offset_x, (int, float)) and isinstance(offset_y, (int, float)):
        return (
            center_x + float(offset_x) * board_width,
            center_y + float(offset_y) * board_height,
        )
    return center_x, center_y + fallback_radius * 0.84


def _regular_polygon_points(
    center_x: float,
    center_y: float,
    radius: float,
    sides: int,
    *,
    rotation: float = 0.0,
) -> list[float]:
    points: list[float] = []
    for index in range(sides):
        angle = rotation + (math.tau * index / sides)
        points.extend((center_x + math.cos(angle) * radius, center_y + math.sin(angle) * radius))
    return points


def _fallback_resource_slot(
    resource: str,
    price: int,
    index: int,
    row_index: int,
    width: int,
    height: int,
) -> tuple[float, float]:
    geometry = _resource_market_table_geometry(width, height)
    price_index = RESOURCE_MARKET_PRICE_COLUMNS.index(price)
    group_x = (
        geometry["grid_x"]
        + geometry["label_width"]
        + price_index * geometry["cell_width"]
        + geometry["cell_width"] / 2
    )
    row_y = (
        geometry["grid_y"]
        + geometry["header_height"]
        + row_index * geometry["cell_height"]
        + geometry["cell_height"] / 2
    )
    if resource == "uranium":
        return group_x, row_y
    offsets = ((0, 0), (-6, 0), (6, 0))
    offset_x, offset_y = offsets[min(index, len(offsets) - 1)]
    return group_x + offset_x, row_y + offset_y


def _resource_market_table_geometry(width: int, height: int) -> dict[str, float]:
    padding = 12
    title_height = 22
    title_gap = 10
    label_width = 82
    header_height = 26
    cell_width = 24
    cell_height = 44
    grid_width = label_width + len(RESOURCE_MARKET_PRICE_COLUMNS) * cell_width
    panel_width = grid_width + padding * 2
    panel_height = padding + title_height + title_gap + header_height + len(RESOURCE_TYPES) * cell_height + padding
    panel_x = max(24, width - panel_width - 24)
    panel_y = max(96, height - panel_height - 24)
    grid_x = panel_x + padding
    grid_y = panel_y + padding + title_height + title_gap
    return {
        "panel_x": panel_x,
        "panel_y": panel_y,
        "panel_width": panel_width,
        "panel_height": panel_height,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "label_width": label_width,
        "header_height": header_height,
        "cell_width": cell_width,
        "cell_height": cell_height,
    }


def _resource_market_sidebar_geometry(width: int, height: int) -> dict[str, float]:
    padding = 10
    label_width = 72
    header_height = 24
    cell_width = max(18, min(22, (width - padding * 2 - label_width) / len(RESOURCE_MARKET_PRICE_COLUMNS)))
    cell_height = 36
    grid_width = label_width + len(RESOURCE_MARKET_PRICE_COLUMNS) * cell_width
    panel_width = grid_width + padding * 2
    panel_height = header_height + len(RESOURCE_TYPES) * cell_height + padding * 2
    panel_x = max(0, (width - panel_width) / 2)
    panel_y = max(0, (height - panel_height) / 2)
    grid_x = panel_x + padding
    grid_y = panel_y + padding
    return {
        "panel_x": panel_x,
        "panel_y": panel_y,
        "panel_width": panel_width,
        "panel_height": panel_height,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "label_width": label_width,
        "header_height": header_height,
        "cell_width": cell_width,
        "cell_height": cell_height,
    }


def _sidebar_resource_cell_points(
    geometry: dict[str, float],
    resource: str,
    price: int,
    amount: int,
    row_index: int,
) -> list[tuple[float, float]]:
    if amount <= 0:
        return []
    price_index = RESOURCE_MARKET_PRICE_COLUMNS.index(price)
    center_x = (
        geometry["grid_x"]
        + geometry["label_width"]
        + price_index * geometry["cell_width"]
        + geometry["cell_width"] / 2
    )
    center_y = (
        geometry["grid_y"]
        + geometry["header_height"]
        + row_index * geometry["cell_height"]
        + geometry["cell_height"] / 2
    )
    horizontal = geometry["cell_width"] * 0.22
    vertical = geometry["cell_height"] * 0.16
    if resource == "uranium" or amount == 1:
        offsets = [(0.0, 0.0)]
    elif amount == 2:
        offsets = [(-horizontal, 0.0), (horizontal, 0.0)]
    else:
        offsets = [(-horizontal, vertical), (0.0, -vertical), (horizontal, vertical)]
    return [(center_x + offset_x, center_y + offset_y) for offset_x, offset_y in offsets[:amount]]
