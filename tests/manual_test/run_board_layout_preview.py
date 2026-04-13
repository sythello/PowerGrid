from __future__ import annotations

import argparse
from dataclasses import replace
import tkinter as tk

from powergrid.board_layout import DEFAULT_BOARD_LAYOUT_PATH
from powergrid.gui.board_view import BoardView, PowerPlantMarketView
from powergrid.model import GameConfig, ResourceStorage, make_default_seat_configs
from powergrid.session import GameSession
from powergrid.session_types import GameSnapshot, SessionEvent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual test for the drawn Power Grid board and market preview."
    )
    parser.add_argument("--map", dest="map_id", default="test", choices=("test", "germany", "usa"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layout", default=str(DEFAULT_BOARD_LAYOUT_PATH))
    parser.add_argument("--board-render-mode", choices=("drawn", "asset"), default="drawn")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Render once and exit without entering the interactive main loop.",
    )
    args = parser.parse_args()

    snapshot = build_preview_snapshot(args.map_id, args.seed)

    root = tk.Tk()
    root.title(f"Power Grid Board Preview - {args.map_id}")
    root.geometry("1680x960")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)

    summary = tk.Label(
        root,
        justify="left",
        anchor="w",
        text=(
            f"Map: {args.map_id}\n"
            f"Render mode: {args.board_render_mode}\n"
            f"Layout file: {args.layout}\n"
            "This preview uses the same board and card drawing code as the main GUI."
        ),
    )
    summary.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))

    board_view = BoardView(root, board_render_mode=args.board_render_mode, layout_path=args.layout)
    board_view.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
    market_view = PowerPlantMarketView(root)
    market_view.grid(row=1, column=1, sticky="ns", padx=(4, 8), pady=8)

    board_view.render(snapshot)
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
    preview_state = replace(
        snapshot.state,
        players=tuple(updated_players),
        phase="build_houses",
        pending_decision=None,
    )
    return GameSnapshot(
        state=preview_state,
        active_request=None,
        event_log=(
            SessionEvent(level="info", message="Preview state seeded for board sanity checks."),
            SessionEvent(level="info", message="Fill board_layout_placeholders.json to refine actual city positions."),
        ),
        last_round_summary=None,
        winner_result=None,
    )


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
