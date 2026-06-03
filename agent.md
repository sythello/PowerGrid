# PowerGrid Agent Guide

This file is the root maintenance guide for future coding agents working in this repository.

It complements:

- [README.md](/Users/mac/Desktop/syt/Projects/PowerGrid/README.md): human-facing usage guide
- [src/powergrid/ai/agent.md](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/agent.md): AI-subsystem-specific guide
- [tests/manual_test/README.md](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/manual_test/README.md): manual interactive test guide

## Documentation Rule

Treat this file and the root [README.md](/Users/mac/Desktop/syt/Projects/PowerGrid/README.md) as part of the repo’s public maintenance surface.

When changing any of the following, update the relevant docs in the same change:

- repo entrypoints or runnable commands
- root-level workflows for humans
- top-level architecture or directory responsibilities
- session/controller integration behavior
- logging, evaluation, or artifact output conventions
- configuration defaults or controller naming
- recommended test commands

Also update:

- [src/powergrid/ai/agent.md](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/agent.md) for AI-layer changes
- [tests/manual_test/README.md](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/manual_test/README.md) when manual scripts or usage change materially

## Big Picture

This repo implements Power Grid across several layers:

1. Static data
   - JSON maps, rules, and GUI layout placeholders
2. Rules engine
   - game state types and legality/apply functions
3. Session layer
   - frontend-neutral requests/intents, AI integration, structured logging
4. Frontends
   - Tkinter GUI
   - terminal CLI
5. AI
   - rule-based controllers
   - Elo evaluation tooling
6. Tests and manual scripts

The most important architectural boundary is:

- `model.py` owns rule legality and state transitions
- `session.py` owns turn orchestration, frontends/AI integration, and logs
- higher layers should prefer reusing model/session helpers instead of recreating rules or state logic

## Repository Map

### Root

- [README.md](/Users/mac/Desktop/syt/Projects/PowerGrid/README.md): human guide
- [agent.md](/Users/mac/Desktop/syt/Projects/PowerGrid/agent.md): this file
- [.gitignore](/Users/mac/Desktop/syt/Projects/PowerGrid/.gitignore)
- [tools/build_stage1_data.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tools/build_stage1_data.py)
  - one-off static-data generation helper
  - not part of normal runtime

### Package: `src/powergrid`

- [model.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/model.py)
  - largest and most important rules module
  - owns `GameConfig`, `GameState`, `PlayerState`, `AuctionState`, `DecisionRequest`, `WinnerResult`, and many pure-ish rule helpers
- [rules_data.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/rules_data.py)
  - loads maps/rules from JSON
  - validates static data
- [scenarios.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/scenarios.py)
  - builds deterministic scenario states for testing/manual flows
- [session_types.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/session_types.py)
  - `GuiIntent`, `TurnRequest`, `GameSnapshot`, logging dataclasses, `SeatAgent`
- [session.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/session.py)
  - preferred integration surface for GUI, AI, and structured logs
  - owns `GameSession`, default seat-agent resolution, compact game-log dumping
- [cli.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/cli.py)
  - older direct CLI loop over the rules engine
  - still useful and tested
- [board_layout.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/board_layout.py)
  - GUI board layout loading and art-path resolution

### Package: `src/powergrid/gui`

- [app.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/app.py)
  - top-level GUI app, launcher, AI stepping/pause behavior
- [board_view.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/board_view.py)
  - board and market rendering
- [components.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/components.py)
  - header, player rail, event log, shared widgets
- [panels.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/panels.py)
  - phase-specific interaction panels

### Package: `src/powergrid/ai`

- [__init__.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/__init__.py)
  - controller registry and construction
- [base.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/base.py)
  - abstract AI controller and analysis-log helpers
- [deterministic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/deterministic.py)
  - simple baseline AI
- [strategic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/strategic.py)
  - stronger heuristic AI
- [evaluation.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/evaluation.py)
  - offline Elo evaluation system
- [agent.md](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/agent.md)
  - subsystem-specific deep guide

### Package: `src/powergrid/tools`

User-facing entrypoints:

- [play_tkinter_gui.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/tools/play_tkinter_gui.py)
- [play_cli_game.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/tools/play_cli_game.py)
- [run_ai_game.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/tools/run_ai_game.py)
- [evaluate_ai_ratings.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/tools/evaluate_ai_ratings.py)
- [show_initial_state.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/tools/show_initial_state.py)
- [run_auction_scenario.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/tools/run_auction_scenario.py)
- [validate_static_data.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/tools/validate_static_data.py)

### Static Data

Under [src/powergrid/data](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/data):

- `maps/`
  - `germany.json`
  - `usa.json`
  - `test.json`
- `rules/`
  - `power_plants.json`
  - `rule_tables.json`
- `gui_layouts/`
  - board layout placeholders and image notes

### Tests

Automated tests live under [tests](/Users/mac/Desktop/syt/Projects/PowerGrid/tests):

- `test_model.py`: rules engine
- `test_session.py`: session layer and logs
- `test_ai.py`: AI behavior
- `test_ai_evaluation.py`: Elo evaluation
- `test_gui.py`: GUI app behavior
- `test_cli.py`: terminal loop behavior
- `test_static_data.py`: JSON data validation
- `test_board_layout.py`: board layout utilities
- `test_run_ai_game_tool.py`: AI-vs-AI runner tool

Interactive scripts live under [tests/manual_test](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/manual_test).

## Must-Know Runtime Facts

### 1. Use `PYTHONPATH=src`

There is no root packaging config like `pyproject.toml` in this repo right now. Local commands should typically look like:

```bash
PYTHONPATH=src python -m powergrid.tools.play_tkinter_gui
```

### 2. `GameSession` is the preferred orchestration seam

For most new work, especially UI, AI, and logging:

- prefer `GameSession`
- prefer `GuiIntent` / `TurnRequest` / `GameSnapshot`
- avoid building new control flows directly on the older CLI loop unless that is the explicit target

The GUI and AI infrastructure are built around the session layer, not the legacy CLI loop.

### 3. Keep rule legality in `model.py`

If the change affects whether something is legal, how costs are computed, or how phases resolve:

- start in `model.py`
- use existing helpers like `purchase_resources(...)`, `apply_builds(...)`, `start_auction(...)`, `resolve_bureaucracy(...)`
- do not copy rule logic into GUI or AI code

### 4. AI names and defaults are not all the same

Current controller names:

- `human`
- `ai_deterministic`
- `ai_heuristics`
- `ai`

Current meanings:

- generic alias `ai` resolves to `ai_deterministic`
- helper-generated AI seats from `make_default_seat_configs(..., ai_players=...)` use `ai_deterministic`
- the GUI AI picker defaults to `ai_heuristics`

If any of these change, update both docs and tests.

### 5. Structured game logs are compact and self-contained

`GameSession.dump_game_log(path)` writes:

- top-level static metadata once
- compact dynamic per-entry snapshots

Important current format traits:

- `format_version = 2`
- `state_snapshot_format = "compact_v1"`
- static map/rules/player metadata and plant catalog live under `static_data`
- per-entry `state_snapshot` omits repeated config/map/rules payloads

AI controllers can append custom analysis state through `GameSnapshot.analysis_log`.

### 6. The `test` map is development-oriented

The real full-play maps are `germany` and `usa`. The `test` map exists to support focused GUI/layout/build scenarios and development workflows.

## Current Human Entry Points

Use these when validating user-facing behavior:

- GUI:
  - `PYTHONPATH=src python -m powergrid.tools.play_tkinter_gui`
- CLI:
  - `PYTHONPATH=src python -m powergrid.tools.play_cli_game`
- AI-vs-AI with log dump:
  - `PYTHONPATH=src python -m powergrid.tools.run_ai_game --controllers ai_deterministic ai_heuristics ai_deterministic`
- AI Elo evaluation:
  - `PYTHONPATH=src python -m powergrid.tools.evaluate_ai_ratings --games-per-lineup 20 --seed-start 1`
- Static data validation:
  - `PYTHONPATH=src python -m powergrid.tools.validate_static_data`
- Manual scripts:
  - see [tests/manual_test/README.md](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/manual_test/README.md)

## Development Workflow By Change Type

### Rules or phase behavior

Start with:

- [src/powergrid/model.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/model.py)
- [tests/test_model.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_model.py)
- often also:
  - [tests/test_session.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_session.py)
  - [tests/test_cli.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_cli.py)

### Session, intents, logs, or AI integration

Start with:

- [src/powergrid/session.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/session.py)
- [src/powergrid/session_types.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/session_types.py)
- [tests/test_session.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_session.py)

If the change is visible in the GUI, also check:

- [tests/test_gui.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_gui.py)

### GUI behavior

Start with:

- [src/powergrid/gui/app.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/app.py)
- [src/powergrid/gui/panels.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/panels.py)
- [src/powergrid/gui/components.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/components.py)
- [src/powergrid/gui/board_view.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/board_view.py)

Validate with:

- [tests/test_gui.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_gui.py)
- manual GUI runs if the change is visual

### AI behavior

Start with:

- [src/powergrid/ai/agent.md](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/agent.md)
- [src/powergrid/ai/base.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/base.py)
- [src/powergrid/ai/deterministic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/deterministic.py)
- [src/powergrid/ai/strategic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/strategic.py)
- [src/powergrid/ai/evaluation.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/evaluation.py)

Validate with:

- [tests/test_ai.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_ai.py)
- [tests/test_ai_evaluation.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_ai_evaluation.py)
- [tests/test_session.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_session.py) if logging or session coupling changes

### Static data or map/rule payloads

Start with:

- [src/powergrid/data](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/data)
- [src/powergrid/rules_data.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/rules_data.py)
- [src/powergrid/tools/validate_static_data.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/tools/validate_static_data.py)
- maybe [tools/build_stage1_data.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tools/build_stage1_data.py) if regenerating from reference sources

Validate with:

- [tests/test_static_data.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_static_data.py)
- map-specific tests or manual runs as needed

## Recommended Validation Commands

Broad suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -q
```

Targeted AI/session suite:

```bash
PYTHONPATH=src python -m unittest tests.test_session tests.test_ai tests.test_ai_evaluation -q
```

Targeted GUI/model suite:

```bash
PYTHONPATH=src python -m unittest tests.test_model tests.test_gui -q
```

If you add a new tool entrypoint, add or update an automated test for it where practical.

## Non-Obvious Behaviors Worth Remembering

- `GameSession.new_game(...)` advances one phase from initialized setup into the first actionable state
- automatic session progression still advances `setup` and `determine_order` without human input
- GUI AI stepping uses a delay between actions and exposes pause/resume
- `run_ai_game.py` writes logs under `artifacts/game_logs/`
- `evaluate_ai_ratings.py` writes reports under `artifacts/ai_ratings/`
- `artifacts/game_logs/` is gitignored

## Safe Change Patterns

- Reuse `state.to_dict()` / `GameState.from_dict(...)` cloning only where full cloning is actually needed
- Prefer deterministic tie-breakers in AI and evaluation code
- Keep analysis logs compact and JSON-serializable
- If you add new persistent log fields, think about redundancy and dump size, not just convenience
- Do not break existing controller ids casually; aliases may be used by tests, saved configs, or tools

## Before You Finish A Change

Quick checklist:

- update code
- update tests
- update root docs if entrypoints/workflows/architecture changed
- update [src/powergrid/ai/agent.md](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/agent.md) if AI-related behavior or contracts changed
- update [tests/manual_test/README.md](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/manual_test/README.md) if manual scripts or usage materially changed
- run the smallest meaningful validation suite, then broader coverage if the change touched shared layers
