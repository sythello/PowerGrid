# PowerGrid

PowerGrid is a Python implementation of the Power Grid board game with:

- a shared rules engine
- a frontend-neutral session layer
- a Tkinter GUI
- a terminal CLI
- deterministic, heuristic-search, and trainable neural AI controllers
- AI-vs-AI evaluation and structured game-log dumping

The repository is organized for both gameplay and engine development. Most commands assume you are in the repo root and run them with `PYTHONPATH=src`.

## Quick Start

This repo currently runs as plain Python modules rather than an installed package. The runtime is mostly standard-library only; the GUI uses `tkinter`. The optional trainable AI/data pipeline requires NumPy and PyArrow, listed in `requirements-ml.txt`.

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
    - `ai_deterministic_efficiency`: no-search plant/fuel-efficiency data policy
    - `ai_deterministic_expansion`: no-search output/network-growth data policy
    - `ai_deterministic_reserve`: no-search cash-preservation data policy
    - `ai_nn_rank_value_v1`: feature-engineered NumPy MLP baseline using the bundled bootstrap checkpoint
    - `ai_nn_rl_based_v1`: Germany/3-player Policy + multi-player vector-Q controller (requires a trained checkpoint)

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

默认情况下，每个 game seed 会从当前地图/人数的全部合法连续区域组合中可复现地随机
采样；两种换座 lineup 在相同 seed 下使用同一区域组合。需要复现固定区域实验时可传：

```bash
PYTHONPATH=src python -m powergrid.tools.evaluate_ai_ratings \
  --games-per-lineup 20 --seed-start 1 \
  --regions black,blue,magenta
```

`--region-sampling-seed` 可改变随机区域调度而不改变游戏 seed。

Current scope:

- Elo evaluation is currently implemented for `germany` with `3` players
- the default evaluated controllers are `ai_deterministic` and `ai_heuristics`
- `ai_nn_rank_value_v1` can be supplied explicitly with `--controllers`, two controller types at a time
- JSON output records every game's selected regions and aggregate region coverage/counts

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
- `ai_deterministic_efficiency`
- `ai_deterministic_expansion`
- `ai_deterministic_reserve`
- `ai_nn_rank_value_v1`
  - trainable state/action rank-value baseline
  - loads the bundled checkpoint unless `POWERGRID_NN_RANK_VALUE_CHECKPOINT` is set
- `ai_nn_rl_based_v1`
  - offline-RL/search-distilled Policy + vector-Q AI for Germany/3-player games
  - loads `src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz` unless `POWERGRID_NN_RL_BASED_CHECKPOINT` is set
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

### Train and validate the neural rank-value AI

Install its optional NumPy/Parquet dependencies:

```bash
python -m pip install -r requirements-ml.txt
```

Generate terminal-rank data and train a checkpoint:

```bash
PYTHONPATH=src python -m powergrid.tools.generate_nn_rank_value_dataset \
  --output artifacts/nn_rank_value/train \
  --games 100 \
  --controllers ai_deterministic \
    ai_deterministic_efficiency \
    ai_deterministic_expansion \
    ai_deterministic_reserve \
  --region-set blue,magenta,black \
  --region-set black,yellow,green \
  --target-shard-size-mib 512 \
  --workers 4

PYTHONPATH=src python -m powergrid.tools.train_nn_rank_value \
  --dataset artifacts/nn_rank_value/train \
  --output artifacts/nn_rank_value/model.npz
```

The generated directory contains Zstandard-compressed Parquet shards, a checksummed
`manifest.json`, deterministic game-level train/validation/test splits, and exactly
three whole-game JSONL examples. Generated neural artifacts remain ignored by Git.

Run the component validations:

```bash
PYTHONPATH=src python -m powergrid.tools.validate_nn_observation
PYTHONPATH=src python -m powergrid.tools.validate_nn_candidates
PYTHONPATH=src python -m powergrid.tools.validate_nn_dataset
PYTHONPATH=src python -m powergrid.tools.validate_nn_model
PYTHONPATH=src python -m powergrid.tools.validate_nn_training
PYTHONPATH=src python -m powergrid.tools.validate_nn_controller
```

The complete feature, label, architecture, checkpoint, training, and validation
specification is in [docs/ai_nn_rank_value_v1.md](docs/ai_nn_rank_value_v1.md). The
bundled checkpoint is a functional bootstrap artifact; it is not yet a claim of
strength over `ai_heuristics`.

The three fast behavior policies, held-out strength calibration, heuristic search
speed controls, and generation benchmarks are documented in
[docs/profiled_deterministic_ai.md](docs/profiled_deterministic_ai.md).

### Train and validate the neural RL Policy/Q AI

`ai_nn_rl_based_v1` behavior-clones `ai_deterministic`, anchors vector Q with terminal
rank, then distills full-action finite-depth semantic search into a Policy-only online
controller. Bootstrap, search-iteration, validation, and evaluation commands are in
[docs/ai_nn_rl_based_v1.md](docs/ai_nn_rl_based_v1.md).
Dataset generation defaults to a deterministic seed-based cycle over all 13 legal
Germany/3-player region sets.

```bash
PYTHONPATH=src python -m unittest tests.test_nn_rl_based -v
PYTHONPATH=src python -m powergrid.tools.validate_nn_rl_based --section all
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
- `ai/`: AI registry, rule-based and neural controllers, training data/model code, logging hooks, Elo evaluation
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
