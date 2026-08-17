# PowerGrid

PowerGrid is a Python implementation of the Power Grid board game with:

- a shared rules engine
- a frontend-neutral session layer
- a Tkinter GUI
- a terminal CLI
- multiple rule-based AI controllers
- AI-vs-AI evaluation and structured game-log dumping

The repository is organized for both gameplay and engine development. Most commands assume you are in the repo root and run them with `PYTHONPATH=src`.

## Quick Start

This repo currently runs as plain Python modules rather than an installed package. The runtime is mostly standard-library only; the GUI uses `tkinter`.

Validate the bundled maps/rules data:

```bash
PYTHONPATH=src python -m powergrid.tools.validate_static_data
```

Run the main GUI:

```bash
PYTHONPATH=src python -m powergrid.tools.play_tkinter_gui
```

Run a full terminal game:

```bash
PYTHONPATH=src python -m powergrid.tools.play_cli_game --map germany --players 3 --seed 7
```

Run the automated test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -q
```

On macOS, GUI-related test runs may print LaunchServices warnings. They have been harmless in current local runs.

## Main Ways To Use The Repo

### 1. Play a full game in the GUI

```bash
PYTHONPATH=src python -m powergrid.tools.play_tkinter_gui
```

Useful options:

- `--scenario opening`
- `--seed 7`
- `--board-render-mode drawn`
- `--smoke-test`

GUI launcher behavior:

- choose `Map`: `germany`, `usa`, or `test`
- choose player count: `3` to `6`
- choose `Seed`
- choose each seat as `human` or `ai`
- when a seat is `ai`, choose an AI version:
  - `ai_heuristics`: stronger rule-based AI
  - `ai_deterministic`: simpler, more predictable baseline AI

During AI turns, the GUI advances AI actions with a pause between actions and exposes a `Pause AI` / `Resume AI` control.

### 2. Play a full game in the terminal

```bash
PYTHONPATH=src python -m powergrid.tools.play_cli_game --map germany --players 4 --seed 11
```

Useful options:

- `--regions black,blue,magenta`
- `--allow-debug-commands`

This path uses the older CLI game loop directly over the rules engine, while the GUI uses the newer session layer.

### 3. Run AI-vs-AI games and dump logs

```bash
PYTHONPATH=src python -m powergrid.tools.run_ai_game \
  --controllers ai_deterministic ai_heuristics ai_deterministic \
  --seed 7
```

Useful options:

- `--map germany`
- `--players 3`
- `--regions black,blue,magenta`
- `--output artifacts/game_logs/germany_3p_seed7.json`
- `--strategy-output artifacts/strategy_logs/germany_3p_seed7.json`

Behavior:

- runs the full game to completion through `GameSession`
- prints winner and final standings
- writes a structured JSON game log
- optionally writes a strategy-only JSON log containing AI analysis entries
  - heuristic auction entries include fallback-aware reserve samples and post-auction economy projections

The default log output directory is `artifacts/game_logs/`, which is ignored by git.

Analyze strategy logs:

```bash
PYTHONPATH=src python -m powergrid.tools.analyze_strategy_logs \
  --strategy-dir artifacts/strategy_logs/heuristic_v005 \
  --game-dir artifacts/game_logs/heuristic_v005
```

### 4. Evaluate AI ratings

```bash
PYTHONPATH=src python -m powergrid.tools.evaluate_ai_ratings --games-per-lineup 20 --seed-start 1
```

Current scope:

- Elo evaluation is currently implemented for `germany` with `3` players
- the default evaluated controllers are `ai_deterministic` and `ai_heuristics`

Output:

- prints a compact leaderboard
- writes JSON under `artifacts/ai_ratings/`

## Configuration Model

The runtime configuration objects are:

- `GameConfig` in [src/powergrid/model.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/model.py)
- `SeatConfig` in [src/powergrid/model.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/model.py)

Important fields:

- `map_id`
  - real game maps: `germany`, `usa`
  - development/support map: `test`
- `players`
  - tuple of `SeatConfig`
  - supported player counts: `3` to `6`
- `seed`
  - drives shuffled order, deck prep, and other seeded setup behavior
- `selected_regions`
  - optional explicit region ids
  - if omitted, the engine auto-selects a valid contiguous set for the player count

Seat controller names:

- `human`
- `ai_deterministic`
- `ai_heuristics`
- `ai`
  - generic alias
  - currently resolves to the deterministic AI

Non-obvious defaults:

- helper-generated AI seats from `make_default_seat_configs(..., ai_players=...)` use `ai_deterministic`
- the GUI AI version picker defaults to `ai_heuristics`

## Structured Game Logs

The session layer supports compact structured logs for later analysis.

Relevant APIs:

- `GameSession.game_log_entries()`
- `GameSession.game_log_payload()`
- `GameSession.dump_game_log(path)`

Current log format:

- top-level `static_data`
  - map
  - rules
  - seat metadata
  - power-plant catalog
- compact per-entry `state_snapshot`
  - dynamic state only
  - avoids repeating full config, map, and rule payloads every turn

AI controllers can also write custom structured state into the log through `GameSnapshot.analysis_log`. See [src/powergrid/ai/agent.md](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/agent.md) for details.

## Useful Tools

### Inspect a seeded initial state

```bash
PYTHONPATH=src python -m powergrid.tools.show_initial_state --players 3 --seed 7
```

### Run small scripted auction scenarios

```bash
PYTHONPATH=src python -m powergrid.tools.run_auction_scenario --scenario first-round
```

### Explore manual interactive scripts

See [tests/manual_test/README.md](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/manual_test/README.md).

These cover:

- session-layer manual driving
- GUI manual checks
- board layout preview
- phase-specific scenario drivers
- full-game manual scenarios

## Repository Layout

Main code lives under [src/powergrid](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid):

- `model.py`: core rules engine and immutable-ish state transforms
- `session.py`: frontend-neutral game session, AI integration, event/game logs
- `session_types.py`: requests, intents, snapshots, log entry types
- `cli.py`: terminal game loop and rendering helpers
- `gui/`: Tkinter application shell and panels
- `ai/`: AI registry, controllers, logging hooks, Elo evaluation
- `rules_data.py`: JSON loading and static-data validation
- `scenarios.py`: deterministic scenario builders for testing and manual flows
- `tools/`: user-facing entrypoints
- `data/`: maps, rules, and GUI layout placeholders

Tests live under [tests](/Users/mac/Desktop/syt/Projects/PowerGrid/tests).

## Programmatic Usage

The preferred programmable runtime surface is `GameSession`.

Example:

```python
from powergrid.model import GameConfig, SeatConfig
from powergrid.session import GameSession

config = GameConfig(
    map_id="germany",
    players=(
        SeatConfig("p1", "Player 1", controller="human"),
        SeatConfig("p2", "Bot", controller="ai_heuristics"),
        SeatConfig("p3", "Player 3", controller="human"),
    ),
    seed=7,
)

session = GameSession.new_game(config)
snapshot = session.snapshot()
```

## Notes For Contributors

- prefer the session layer for new UI, AI, or logging work
- keep rules legality inside the model/rules helpers rather than re-implementing it in higher layers
- if you change AI behavior or AI integration, also update [src/powergrid/ai/agent.md](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/agent.md)
- if you change repo entrypoints, configuration behavior, or developer workflows, also update the root docs
