from __future__ import annotations

import argparse

from powergrid.cli import CLIController, run_game
from powergrid.model import GameConfig, advance_phase, initialize_game, make_default_seat_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Power Grid in the terminal.")
    parser.add_argument("--map", dest="map_id", default="germany")
    parser.add_argument("--players", type=int, choices=range(3, 7), default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--regions",
        help="Optional comma-separated region ids. If omitted, a valid contiguous area is auto-selected.",
    )
    args = parser.parse_args()

    selected_regions = ()
    if args.regions:
        selected_regions = tuple(
            region_id.strip() for region_id in args.regions.split(",") if region_id.strip()
        )

    config = GameConfig(
        map_id=args.map_id,
        players=make_default_seat_configs(args.players),
        seed=args.seed,
        selected_regions=selected_regions,
    )
    controllers = {
        seat.player_id: CLIController(player_id=seat.player_id)
        for seat in config.players
    }
    state = advance_phase(initialize_game(config, controllers))
    result = run_game(state, controllers, render_state=True)

    if result.quit_requested:
        print("Game stopped before completion.")
        return
    if result.winner_result is not None:
        print("Final winner: " + ", ".join(result.winner_result.winner_ids))
    else:
        print(
            f"Stopped at round={result.final_state.round_number} "
            f"phase={result.final_state.phase} step={result.final_state.step}"
        )


if __name__ == "__main__":
    main()
