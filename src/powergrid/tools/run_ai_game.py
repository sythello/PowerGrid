from __future__ import annotations

import argparse
import json
from pathlib import Path

from powergrid.ai import derive_final_standings
from powergrid.model import GameConfig, ModelValidationError, SeatConfig
from powergrid.session import GameSession


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a full AI-vs-AI Power Grid game and dump the structured game log."
    )
    parser.add_argument("--map", dest="map_id", default="germany")
    parser.add_argument("--players", type=int, choices=range(3, 7), default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--regions",
        help="Optional comma-separated region ids. If omitted, a valid contiguous area is auto-selected.",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["ai_deterministic", "ai_heuristics", "ai_deterministic"],
        help=(
            "Controller ids for seats in player order. "
            "Provide either one controller to repeat for all seats or one per seat."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional JSON log output path. Defaults to artifacts/game_logs/{map}_{players}p_seed{seed}.json",
    )
    parser.add_argument(
        "--strategy-output",
        help="Optional JSON output path containing only AI strategy/analysis log entries.",
    )
    args = parser.parse_args(argv)

    controller_names = _resolve_controllers(args.controllers, args.players)
    selected_regions = ()
    if args.regions:
        selected_regions = tuple(
            region_id.strip() for region_id in args.regions.split(",") if region_id.strip()
        )

    config = GameConfig(
        map_id=args.map_id,
        players=tuple(
            SeatConfig(
                player_id=f"p{index + 1}",
                name=f"Player {index + 1}",
                controller=controller_name,
            )
            for index, controller_name in enumerate(controller_names)
        ),
        seed=args.seed,
        selected_regions=selected_regions,
    )

    session = GameSession.new_game(config)
    snapshot = session.advance_until_blocked()
    output_path = Path(args.output) if args.output else _default_output_path(config)
    session.dump_game_log(output_path)
    strategy_output_path = Path(args.strategy_output) if args.strategy_output else None
    if strategy_output_path is not None:
        _dump_strategy_log(session.game_log_payload(), strategy_output_path, game_log_path=output_path)

    print("AI Game Completed")
    print(
        f"Game: map={snapshot.state.game_map.id} players={len(snapshot.state.players)} "
        f"seed={config.seed}"
    )
    print("Controllers:")
    for index, controller_name in enumerate(controller_names, start=1):
        print(f"{index}. p{index}: {controller_name}")
    print("Resolved regions: " + ", ".join(snapshot.state.selected_regions))

    if snapshot.winner_result is not None:
        print("Winner: " + ", ".join(snapshot.winner_result.winner_ids))
        print("Standings:")
        for standing in derive_final_standings(snapshot.state, snapshot.winner_result):
            print(
                f"{standing.place}. {standing.player_id} "
                f"controller={standing.controller_name} "
                f"powered={standing.powered_cities} "
                f"money={standing.money} "
                f"connected={standing.connected_cities}"
            )
    else:
        print(
            f"Game stopped early at round={snapshot.state.round_number} "
            f"phase={snapshot.state.phase} step={snapshot.state.step}"
        )
    print(f"Wrote game log to {output_path}")
    if strategy_output_path is not None:
        print(f"Wrote strategy log to {strategy_output_path}")
    return 0


def _resolve_controllers(controller_names: list[str], player_count: int) -> tuple[str, ...]:
    if len(controller_names) == 1:
        return tuple(controller_names[0] for _ in range(player_count))
    if len(controller_names) != player_count:
        raise ModelValidationError(
            "controllers must contain either one controller id or exactly one id per player"
        )
    return tuple(controller_names)


def _default_output_path(config: GameConfig) -> Path:
    return Path("artifacts") / "game_logs" / f"{config.map_id}_{len(config.players)}p_seed{config.seed}.json"


def _dump_strategy_log(game_log_payload: dict[str, object], path: Path, *, game_log_path: Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    game_log = game_log_payload.get("game_log", [])
    strategy_entries = [
        entry
        for entry in game_log
        if isinstance(entry, dict) and entry.get("source") == "ai"
    ]
    payload = {
        "format_version": 1,
        "source_game_log": str(game_log_path),
        "config": game_log_payload.get("config"),
        "winner_result": game_log_payload.get("winner_result"),
        "strategy_log": strategy_entries,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
