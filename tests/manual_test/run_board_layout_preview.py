from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from powergrid.board_layout import DEFAULT_BOARD_LAYOUT_PATH, load_board_layout
from powergrid.gui.board_view import BoardView, PowerPlantMarketView, resolve_city_label_center
from powergrid.model import GameConfig, ResourceStorage, make_default_seat_configs
from powergrid.session import GameSession
from powergrid.session_types import GameSnapshot, SessionEvent


DEFAULT_GERMANY_REFERENCE = Path(
    "/Users/mac/Desktop/syt/Boardgame-assets/Power Grid/Power Grid/assets/data/pics/game/deut.jpg"
)


class ReferenceCalibrationView(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master, padding=(0, 0))
        self.summary_var = tk.StringVar(value="Reference: not loaded")
        ttk.Label(self, textvariable=self.summary_var, anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 0)
        )
        self.canvas = tk.Canvas(self, background="#111827", highlightthickness=0)
        self.y_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.x_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.y_scroll.set, xscrollcommand=self.x_scroll.set)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.y_scroll.grid(row=1, column=1, sticky="ns")
        self.x_scroll.grid(row=2, column=0, sticky="ew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._cached_images: dict[Path, tuple[object, object]] = {}
        self._tk_image = None

    def render(
        self,
        snapshot: GameSnapshot,
        map_layout: dict,
        reference_path: Path,
        *,
        show_city_ids: bool = False,
    ) -> None:
        image_size, tk_image = self._load_reference_image(reference_path)
        width, height = image_size
        self._tk_image = tk_image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=tk_image)

        city_defaults = map_layout.get("city_defaults", {})
        mapped = 0
        total = len(snapshot.state.game_map.cities)
        for city in snapshot.state.game_map.cities:
            city_payload = map_layout.get("cities", {}).get(city.id, {})
            anchor = city_payload.get("anchor", {})
            anchor_x = anchor.get("x")
            anchor_y = anchor.get("y")
            if anchor_x is None or anchor_y is None:
                continue
            mapped += 1
            center_x = float(anchor_x) * width
            center_y = float(anchor_y) * height
            label_x, label_y = resolve_city_label_center(
                center_x,
                center_y,
                (width, height),
                city_payload,
                city_defaults,
            )
            half_width = max(34, min(70, len(city.name) * 4.5))
            self._draw_crosshair(center_x, center_y)
            self.canvas.create_rectangle(
                label_x - half_width,
                label_y - 11,
                label_x + half_width,
                label_y + 11,
                outline="#22c55e",
                width=2,
            )
            if show_city_ids:
                self.canvas.create_text(
                    center_x,
                    center_y - 26,
                    text=city.id,
                    fill="#fde047",
                    font=("Helvetica", 8, "bold"),
                )
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.summary_var.set(
            f"Reference: {reference_path.name} | mapped cities: {mapped}/{total} | crosshair = anchor, green box = label"
        )

    def _load_reference_image(self, reference_path: Path) -> tuple[tuple[int, int], object]:
        cached = self._cached_images.get(reference_path)
        if cached is not None:
            return cached
        try:
            from PIL import Image, ImageTk
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Pillow is required to load JPEG calibration references. Install it or use a PNG reference."
            ) from exc
        image = Image.open(reference_path).convert("RGBA")
        tk_image = ImageTk.PhotoImage(image)
        payload = ((image.width, image.height), tk_image)
        self._cached_images[reference_path] = payload
        return payload

    def _draw_crosshair(self, center_x: float, center_y: float) -> None:
        self.canvas.create_oval(
            center_x - 4,
            center_y - 4,
            center_x + 4,
            center_y + 4,
            fill="#ef4444",
            outline="#111827",
            width=1,
        )
        self.canvas.create_line(center_x - 14, center_y, center_x + 14, center_y, fill="#ef4444", width=2)
        self.canvas.create_line(center_x, center_y - 14, center_x, center_y + 14, fill="#ef4444", width=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual test for the drawn Power Grid board, Germany calibration view, and market preview."
    )
    parser.add_argument("--map", dest="map_id", default="test", choices=("test", "germany", "usa"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layout", default=str(DEFAULT_BOARD_LAYOUT_PATH))
    parser.add_argument("--board-render-mode", choices=("drawn", "asset"), default="drawn")
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Override the reference image path. Defaults to deut.jpg for --map germany.",
    )
    parser.add_argument(
        "--show-city-ids",
        action="store_true",
        help="Draw each city id above the anchor on the reference image.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Render once and exit without entering the interactive main loop.",
    )
    args = parser.parse_args()

    snapshot = build_preview_snapshot(args.map_id, args.seed)
    map_layout = load_board_layout(args.map_id, args.layout)
    reference_path = _resolve_reference_path(args.map_id, args.reference_image)

    root = tk.Tk()
    root.title(f"Power Grid Board Preview - {args.map_id}")
    root.geometry("2260x1060")
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    summary = tk.Label(
        root,
        justify="left",
        anchor="w",
        text=_build_summary_text(args.map_id, args.board_render_mode, args.layout, reference_path),
    )
    summary.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 0))

    board_view = BoardView(root, board_render_mode=args.board_render_mode, layout_path=args.layout)
    board_view.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)

    reference_view = None
    if reference_path is not None:
        reference_view = ReferenceCalibrationView(root)
        reference_view.grid(row=1, column=1, sticky="nsew", padx=4, pady=8)
    else:
        root.grid_columnconfigure(1, weight=0)

    market_view = PowerPlantMarketView(root)
    market_view.grid(row=1, column=2, sticky="ns", padx=(4, 8), pady=8)

    board_view.render(snapshot)
    if reference_view is not None:
        reference_view.render(snapshot, map_layout, reference_path, show_city_ids=args.show_city_ids)
    market_view.render(snapshot)

    root.update_idletasks()
    root.update()
    if args.smoke_test:
        root.destroy()
        return
    root.mainloop()


def build_preview_snapshot(map_id: str, seed: int) -> GameSnapshot:
    selected_regions = ("alpha", "beta", "gamma") if map_id == "test" else ()
    session = GameSession.new_game(
        GameConfig(
            map_id=map_id,
            players=make_default_seat_configs(3),
            seed=seed,
            selected_regions=selected_regions,
        )
    )
    snapshot = session.advance_until_blocked()
    plants = (*snapshot.state.current_market, *snapshot.state.future_market)
    sample_city_sets = _preview_city_sets(snapshot.state.game_map.id, [city.id for city in snapshot.state.game_map.cities])
    updated_players = []
    ordered_players = sorted(snapshot.state.players, key=lambda item: item.turn_order_position)
    for index, player in enumerate(ordered_players):
        player_plants = tuple(plants[index * 2 : index * 2 + 2])
        updated_players.append(
            replace(
                player,
                elektro=62 - index * 7,
                houses_in_supply=22 - len(sample_city_sets[index]),
                network_city_ids=sample_city_sets[index],
                power_plants=player_plants,
                resource_storage=_sample_storage(player_plants),
            )
        )
    preview_events = [
        SessionEvent(level="info", message="Preview state seeded for board sanity checks."),
    ]
    if map_id == "germany":
        preview_events.append(
            SessionEvent(
                level="info",
                message="Germany calibration mode: compare the drawn board against the JPEG reference on the right.",
            )
        )
    else:
        preview_events.append(
            SessionEvent(level="info", message="Fill board_layout_placeholders.json to refine actual city positions.")
        )
    preview_state = replace(
        snapshot.state,
        players=tuple(updated_players),
        phase="build_houses",
        pending_decision=None,
    )
    return GameSnapshot(
        state=preview_state,
        active_request=None,
        event_log=tuple(preview_events),
        last_round_summary=None,
        winner_result=None,
    )


def _resolve_reference_path(map_id: str, raw_reference_path: str | None) -> Path | None:
    if raw_reference_path:
        return Path(raw_reference_path).expanduser().resolve()
    if map_id == "germany":
        return DEFAULT_GERMANY_REFERENCE
    return None


def _build_summary_text(
    map_id: str,
    board_render_mode: str,
    layout_path: str,
    reference_path: Path | None,
) -> str:
    lines = [
        f"Map: {map_id}",
        f"Render mode: {board_render_mode}",
        f"Layout file: {layout_path}",
        "This preview uses the same board and card drawing code as the main GUI.",
    ]
    if reference_path is not None:
        lines.append(f"Reference image: {reference_path}")
        lines.append("Calibration aids: red crosshair = city anchor, green rectangle = city label position.")
    return "\n".join(lines)


def _preview_city_sets(map_id: str, city_ids: list[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if map_id == "test":
        return (
            ("amber_falls",),
            ("amber_falls", "brass_harbor"),
            ("amber_falls", "cinder_grove"),
        )
    if len(city_ids) < 3:
        first = tuple(city_ids[:1])
        second = tuple(city_ids[:2])
        third = tuple(city_ids[:3])
        return first, second, third
    return (
        (city_ids[0],),
        (city_ids[0], city_ids[1]),
        (city_ids[0], city_ids[2]),
    )


def _sample_storage(plants) -> ResourceStorage:
    storage = {
        "coal": 0,
        "oil": 0,
        "hybrid_coal": 0,
        "hybrid_oil": 0,
        "garbage": 0,
        "uranium": 0,
    }
    for plant in plants:
        if plant.is_ecological or plant.is_step_3_placeholder:
            continue
        if plant.is_hybrid:
            storage["hybrid_coal"] += min(1, plant.max_storage)
            continue
        resource = plant.resource_types[0]
        storage[resource] += min(1, plant.max_storage)
    return ResourceStorage(**storage)


if __name__ == "__main__":
    main()
