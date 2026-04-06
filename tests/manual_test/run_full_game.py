from __future__ import annotations

import argparse

from powergrid.cli import CLIController, run_game
from powergrid.scenarios import SCENARIO_NAMES, build_game_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive manual test for the full Power Grid gameplay loop."
    )
    parser.add_argument("--scenario", choices=SCENARIO_NAMES)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    scenario = args.scenario or prompt_scenario()
    state = build_game_scenario(scenario, seed=args.seed)
    controllers = {
        player.player_id: CLIController(player_id=player.player_id)
        for player in state.players
    }

    print("Full Game Manual Test")
    print(f"Scenario: {scenario}")
    print(f"Map: {state.game_map.name} ({state.game_map.id})")
    print(f"Seed: {args.seed}")
    print()
    result = run_game(state, controllers, render_state=True)

    if result.quit_requested:
        print("Manual full-game test stopped.")
        return
    if result.winner_result is not None:
        print("Winner: " + ", ".join(result.winner_result.winner_ids))
    else:
        print(
            f"Stopped at round={result.final_state.round_number} "
            f"phase={result.final_state.phase} step={result.final_state.step}"
        )


def prompt_scenario() -> str:
    print("Available scenarios: " + ", ".join(SCENARIO_NAMES))
    while True:
        raw = input("Scenario [opening]: ").strip()
        if not raw:
            return "opening"
        if raw in SCENARIO_NAMES:
            return raw
        print("Please choose one of: " + ", ".join(SCENARIO_NAMES))


if __name__ == "__main__":
    main()
