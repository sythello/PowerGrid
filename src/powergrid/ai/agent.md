# AI Agent Guide

This directory contains the rule-based AI controllers for Power Grid.

## Documentation Rule

Treat this file as part of the AI implementation surface.

Whenever an agent changes any of the following, it should also update this document in the same change:

- controller names, aliases, or defaults
- registry wiring
- session/controller integration behavior
- AI logging or analysis interfaces
- major AI decision logic
- heuristic evaluation dimensions
- search depth or search structure
- GUI-exposed AI options
- recommended development workflow or test strategy

Minimum expectation for follow-up updates:

- update the architecture or behavior sections that changed
- update the defaults/naming section if aliases or defaults changed
- update the development guidance if the preferred extension path changed
- update the validation section if the recommended test command changed

If a code change is too small to deserve a documentation edit, the agent should still quickly verify that this file remains accurate.

## Current Architecture

The AI layer is intentionally small at the public seam:

- `BaseAiController` in [base.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/base.py)
  - Abstract seat agent interface.
  - Every AI must implement `choose_intent(request, snapshot) -> GuiIntent`.
- Registry in [__init__.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/__init__.py)
  - `ai_nn_rl_based_v1` -> `NnRlBasedAiController`
  - `ai_nn_rank_value_v1` -> `NnRankValueAiController`
  - `ai_heuristics` -> `StrategicAiController`
  - `ai_deterministic_efficiency` -> `EfficiencyDeterministicAiController`
  - `ai_deterministic_expansion` -> `ExpansionDeterministicAiController`
  - `ai_deterministic_reserve` -> `ReserveDeterministicAiController`
  - `ai_deterministic` -> `DeterministicAiController`
  - `ai` -> `DeterministicAiController` as the generic/default alias
- Session integration in [../session.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/session.py)
  - `GameSession.default_seat_agents()` resolves non-human controllers through `build_ai_controller(...)`
  - `advance_one_ai_action()` and `advance_until_blocked()` call `choose_intent(...)`
  - The AI layer does not mutate session state directly; it only returns intents
  - `GameSnapshot.analysis_log` exposes an AI-safe structured logging hook for diagnostics
  - `GameSession.fork()` copies state plus session-owned cursors for isolated rollout labels

This means the core contract is:

1. Session constructs a `TurnRequest` plus full `GameSnapshot`
2. AI chooses one legal `GuiIntent`
3. Session validates and applies that intent through model/rules helpers

## Files

- [base.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/base.py): abstract controller base
- [deterministic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/deterministic.py): simple baseline AI
- [profiled_deterministic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/profiled_deterministic.py): three no-lookahead, strength-calibrated data-generation policies
- [strategic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/strategic.py): stronger heuristic AI
- [nn_rank_value/](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rank_value): public observation, candidates, dataset generation, NumPy MLP/training, and neural controller
- [nn_rl_based/](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/nn_rl_based): listwise Policy/vector-Q model, full-action semantic search, decision-grouped dataset/training, and Policy-only controller
- [evaluation.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/evaluation.py): offline AI rating/evaluation subsystem
- [__init__.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/__init__.py): registry and controller construction

The neural controller's full feature/label/model/command specification is in
[../../../docs/ai_nn_rank_value_v1.md](/Users/mac/Desktop/syt/Projects/PowerGrid/docs/ai_nn_rank_value_v1.md).
The offline-RL Policy/vector-Q controller is specified in
[../../../docs/ai_nn_rl_based_v1.md](/Users/mac/Desktop/syt/Projects/PowerGrid/docs/ai_nn_rl_based_v1.md).
Its exhaustive 513-state/42-action feature dictionary is in
[../../../docs/ai_nn_rank_value_v1_feature_dictionary.md](/Users/mac/Desktop/syt/Projects/PowerGrid/docs/ai_nn_rank_value_v1_feature_dictionary.md).
The profiled deterministic design, calibration, heuristic speed levers, and generation
benchmarks are in [../../../docs/profiled_deterministic_ai.md](/Users/mac/Desktop/syt/Projects/PowerGrid/docs/profiled_deterministic_ai.md).

## AI Logging And Game Logs

The game engine now has two separate logging layers:

- `event_log`
  - lightweight user-facing session transcript
  - used by the GUI and simple status views
- structured `game_log`
  - analysis-oriented JSON log maintained by `GameSession`
  - compact format: static map/rules/player metadata is stored once at the top level, while per-entry `state_snapshot` payloads only contain dynamic state
  - available through:
    - `GameSession.game_log_entries()`
    - `GameSession.game_log_payload()`
    - `GameSession.dump_game_log(path)`

Current structured game-log behavior:

- session start is logged
- automatic phase transitions like setup/determine-order advancement are logged
- applied intents and intent errors are logged with structured payloads
- bureaucracy summaries are logged
- AI custom diagnostic entries can be appended during `choose_intent(...)`
- full AI-vs-AI runs can be dumped from the CLI with:
  - `PYTHONPATH=src python -m powergrid.tools.run_ai_game --controllers ai_deterministic ai_heuristics ai_deterministic`
- strategy-only logs can also be split out with:
  - `PYTHONPATH=src python -m powergrid.tools.run_ai_game --strategy-output artifacts/strategy_logs/example.json`
- strategy logs can be analyzed with:
  - `PYTHONPATH=src python -m powergrid.tools.analyze_strategy_logs --strategy-dir artifacts/strategy_logs/example --game-dir artifacts/game_logs/example`

AI-facing interface:

- `GameSnapshot.analysis_log`
  - exposes:
    - `record(...)`
    - `record_state(...)`
- `BaseAiController`
  - exposes convenience helpers:
    - `log_message(...)`
    - `log_state(...)`

Recommended usage inside controllers:

- log compact, decision-relevant state only
- prefer JSON-serializable payloads
- use `log_state(...)` for search summaries, candidate rankings, reserve calculations, chosen plans, or other custom reasoning state
- do not log huge cloned states unless they are genuinely needed for later analysis

Current shipped AIs:

- deterministic controllers emit one compact structured AI decision log entry per chosen intent
- profiled deterministic controllers emit `profiled_deterministic_decision` entries
  with controller/profile angle and the chosen intent
- heuristic controllers emit one detailed structured AI reasoning log entry per chosen intent
  - label: `heuristic_decision`
  - includes current relative-state evaluation with weighted subterms
  - includes a `scoreboard` with own score and each opponent score used in the relative formula
  - includes ranked candidate actions with decision scores and projected relative scores
  - includes the selected action, its decision score, and a detailed projected evaluation
  - includes a search summary for the phase-specific search path
- neural rank-value controllers emit one batched candidate-ranking entry per intent
  - label: `nn_rank_value_decision`
  - includes checkpoint metadata, all candidate intents, win/rank predictions, combined scores, and selection
- neural RL controllers emit one Policy-ranked candidate entry per intent
  - label: `nn_rl_based_decision`
  - includes checkpoint/player-slot metadata, Policy probabilities, actor Q, all-player Q vectors, and selection

## Current Controller Behavior

### `DeterministicAiController`

Purpose:

- Straightforward, predictable baseline bot
- Generic/default AI alias for helper-generated AI seats
- Good reference implementation for simple controller logic

Implemented behavior:

- Pending decisions
  - Discards the lowest plant price
  - Chooses the lexicographically earliest legal hybrid discard tuple
- Auction
  - Starts the cheapest available legal plant
  - Opens at minimum bid
  - During bidding, continues only while `min_bid <= min(max_bid, plant_price + 2)`
- Resources
  - Computes immediate deficits only
  - Buys the cheapest needed resource
  - Stops when there are no useful deficit-filling purchases
- Build
  - Commits the single cheapest legal city
- Bureaucracy
  - Exhaustively enumerates legal plant-run combinations
  - Maximizes powered cities, then minimizes resource units spent, then tie-breaks by plant prices

### `StrategicAiController`

Purpose:

- Stronger rule-based AI using state evaluation and bounded search
- Explicit advanced AI option exposed as `ai_heuristics`

Implemented behavior:

- Pending decisions
  - Simulates legal discard options
  - Keeps the portfolio/resource mix with the best evaluated future state
- Auction
  - Uses fallback-aware reserve prices via `_auction_reserve_projection(...)`
  - Values a plant by a fast post-auction economy projection:
    - buy the plant at a candidate price
    - choose a feasible refuel basket
    - greedily build cities affordable with remaining cash
    - simulate generation and income
    - compare the resulting score against later-buy/pass fallback projections
  - Keeps the older `_auction_reserve_level0(...)` as a cheap opponent interest estimate for contest pressure
  - Can pass on overpriced bids
  - In round 1, avoids passing when forced to buy
  - Avoids bait starts in round 1 when the AI has no plant yet
  - Supports “bait” style starts in later rounds when opponents value a plant more than the AI does
- Resources
  - Uses bounded recursive search with `_search_resource_purchase(...)`
  - Evaluates stop-vs-buy tradeoffs rather than only immediate deficits
  - Considers scarcity, cheap blocks, build potential, and resource pressure
- Build
  - Uses beam-search-like exploration in `_search_best_build_plan(...)`
  - Scores expansion by state evaluation, trigger timing, frontier value, connection savings, and contest pressure
  - Can commit multi-city builds
  - Stops building when the best plan is only marginally better than the current state
- Bureaucracy
  - Enumerates legal run plans
  - In normal rounds, chooses plans that maximize evaluated future state
  - In final-round conditions, maximizes the real winner tuple:
    - powered cities
    - money after income
    - connected cities
- Evaluation framework
  - Stage-aware weights in `STAGE_WEIGHTS`
  - Stage detection via `_stage_name(...)`
  - Relative state evaluation uses opponent threat strength:
    - `max(current_strength, best_affordable_partial_refuel_projected_strength)`
    - the refuel projection is opponent-only and searches feasible partial resource baskets constrained by current market availability, cash, and storage legality
  - Player strength includes:
    - connected cities
    - powered cities
    - income
    - cash
    - plant portfolio quality
    - frontier/build potential
    - stored resource value
    - turn-order value
    - resource exposure penalty
    - overbuild / unused-capacity penalties
    - trigger timing score
- Strategy logging
  - Records `schema_version=5` inside each `heuristic_decision` state payload
  - Records current evaluation details before selecting
  - Records `scoreboard` with own score, opponent scores, opponent adjustment, and relative score
  - Records `opponent_threats` with current strength, best affordable partial-refuel projected strength, projected generation, and the selected refuel basket
  - Records selected-action `projected_kind` / `projection_horizon` so analysis can separate immediate, fallback, and post-auction-economy projections
  - Records auction reserve details including fallback score, target score, price samples, cash after purchase/resources/build/income, resource plan, build plan, generation, raw score, and viability adjustment
  - Records candidate action scores from auction, resource, build, pending, and bureaucracy searches
  - Records selected-action projected evaluation details for cross-turn strategy analysis

### Profiled deterministic controllers

Purpose:

- Fast and reproducible behavior diversity for NN dataset generation
- No state cloning, recursive resource search, auction economy rollout, or build beam search
- Held-out strength calibrated against `ai_deterministic`

Profiles:

- `ai_deterministic_efficiency`
  - plant/output/run-cost value, cheap-fuel buffer, total-build-cost priority
- `ai_deterministic_expansion`
  - output and network growth, largest resource deficit, connection-cost priority
- `ai_deterministic_reserve`
  - purchase/run-cost and ecological preference, one-unit resource buying, cash reserve

All three reuse the original deterministic pending-decision and bureaucracy legality
logic. Dataset generation rotates the original plus these three profiles by default;
`ai_heuristics` must be requested explicitly when its slower search behavior is wanted.

### `NnRankValueAiController`

Purpose:

- First trainable AI baseline, exposed as `ai_nn_rank_value_v1`
- Scores dynamic legal `(public state, candidate action)` rows with a NumPy MLP
- Keeps the same `choose_intent(...)` and session legality boundary as rule-based AIs

Implemented behavior:

- Constructs a 513-dimensional actor-relative public observation
  - excludes seed and hidden deck identities/order
  - zero-pads to six players and fixed market/plant slots
- Constructs a 42-dimensional candidate-action vector
- Generates deterministic phase-specific candidates
  - all explicit discard and generation plans
  - minimum bid/pass auction decisions
  - all affordable resource quantities
  - single-city/finish build decisions
- Runs a `555 -> 128 -> 64 -> 2` two-head MLP
  - sigmoid win-probability head
  - tanh normalized-rank head
- Uses `0.7 * P(win) + 0.3 * normalized_rank` at inference
- Loads `data/ai_models/ai_nn_rank_value_v1.npz` by default
  - override with `POWERGRID_NN_RANK_VALUE_CHECKPOINT`
  - verifies feature names embedded in the checkpoint
- Is deterministic for a fixed checkpoint and state

Training support:

- behavior-policy self-play produces selected state/action rows with terminal labels
- optional `GameSession.fork()` rollouts label counterfactual candidates
- final labels are `is_winner` and normalized `rank_value`
- generation streams one completed game at a time into Zstandard-compressed Parquet
  shards; parallel workers are bounded, so memory does not scale with dataset size
- each game is one Parquet row group, and shards target 512 MiB by default
- a checksummed manifest records Parquet/feature schemas and per-shard rows/games/bytes
- only the first three complete games are retained as human-readable JSONL examples
- train/validation/test assignment is a deterministic SHA-256 hash of complete game id
- training normalization, optimizer batches, and split evaluation all stream from Parquet
- loss is binary cross entropy plus rank-value mean squared error
- bundled checkpoint is a functional bootstrap artifact, not a strength-qualified replacement for `ai_heuristics`

### `NnRlBasedAiController`

Purpose:

- Offline approximate-policy-iteration baseline exposed as `ai_nn_rl_based_v1`
- Removes a separate V head and computes `V_i(s) = E_{a~policy}[Q_i(s,a)]`
- Uses one listwise Policy head and one six-slot, multi-player vector-Q head

Implemented behavior:

- Reuses the 513-state/42-action public feature and runtime candidate schemas
- Scores every legal runtime candidate in one NumPy batch
- Chooses only by maximum Policy logit online, with candidate order as stable tie-break
- Logs Policy probability, current-actor Q, and Q mapped to every player id
- Supports only Germany with three players in v1
- Loads `data/ai_models/ai_nn_rl_based_v1.npz` by default
  - override with `POWERGRID_NN_RL_BASED_CHECKPOINT`

Training/search support:

- Behavior cloning uses canonical `ai_deterministic` actions
- Chosen actions receive all-player terminal-rank Monte Carlo Q labels
- Deterministically sampled roots receive labels for every candidate from a frozen target Q
- Every edge applies one candidate, then continues to an auction/resource/build/bureaucracy/pending semantic boundary
- Depth 1 is complete; adaptive depth 2 is accepted only when every action fits the node budget
- Player values are remapped by player id whenever the current actor changes
- Sibling forks share one hidden-deck determinization while observations continue to exclude hidden order
- Data uses one decision per Parquet row, one complete game per row group, checksummed shards, game-exclusive splits, and three JSONL examples
- Stage-1 conservative improvement can retain every searched row plus an equal deterministic
  non-search anchor sample with `--training-sampling balanced_search`
- `advantage_gate` keeps the deterministic one-hot target unless a representable searched action
  clears the configured actor-Q margin; accepted targets assign 0.75 to that action and 0.25 to
  the deterministic action
- Germany/3-player generation cycles by seed over all 13 legal contiguous region sets unless an explicit set/list is supplied
- Parallel generation keeps at most `2 * workers` complete games in flight and falls back to a bounded thread pool where process semaphores are unavailable
- Online controller inference never forks or searches
- `powergrid.tools.evaluate_nn_rl_paired_rollouts` audits every Policy deviation on
  held-out deterministic trajectories by pairing the RL/baseline first action from the
  same session state and continuing both branches with `ai_deterministic` to terminal
- `powergrid.tools.evaluate_nn_rl_deterministic_suite` runs the selected checkpoint
  against the canonical, efficiency, expansion, and reserve deterministic controllers
  with balanced seats and all-region cycling; the current bundled v1 checkpoint is the
  `delta=0.10` model selected by paired-rollout plus end-to-end point-score gates

## Important Supporting Patterns

The stronger AI relies heavily on these internal patterns:

- State cloning
  - `_clone_state(...)` uses `GameState.from_dict(state.to_dict())`
  - This keeps search/simulation side effects isolated
- Rule-engine reuse
  - Simulations use existing model functions like:
    - `purchase_resources(...)`
    - `apply_builds(...)`
    - `replace_plant_if_needed(...)`
    - `choose_plants_to_run(...)`
  - This is critical: AI should not re-implement rule legality by hand
- Deterministic tie-breaking
  - Many comparisons include stable tie-breakers like plant price, city id, tuple order
  - This keeps fixed-seed runs reproducible

## Current Defaults and Naming

There are two different “default” concepts in the codebase:

- Generic controller alias
  - `ai` currently resolves to `ai_deterministic`
- Explicit advanced controller
  - `ai_heuristics` resolves to `StrategicAiController`
- Explicit trainable controller
  - `ai_nn_rank_value_v1` resolves to `NnRankValueAiController`
  - `ai_nn_rl_based_v1` resolves to `NnRlBasedAiController`
- Explicit fast data-generation controllers
  - `ai_deterministic_efficiency`
  - `ai_deterministic_expansion`
  - `ai_deterministic_reserve`

Also note:

- Helper-generated AI seats from `make_default_seat_configs(..., ai_players=...)` currently use `ai_deterministic`
- The GUI setup still exposes explicit AI versions and currently defaults its version picker to `ai_heuristics`

## AI Evaluation

The AI package now also contains an offline Elo evaluation subsystem for AI-vs-AI testing.

Current evaluation scope:

- V1 supports one rating bucket only:
  - `map_id="germany"`
  - `player_count=3`
- Default evaluated controllers:
  - `ai_deterministic`
  - `ai_heuristics`
- Ratings are recomputed fresh on each run from a baseline rating

Core entrypoints:

- `AiEvaluationBucketConfig`
- `build_default_evaluation_lineups(...)`
- `derive_final_standings(...)`
- `evaluate_ai_bucket(...)`

Current lineup schedule:

- `("ai_deterministic", "ai_deterministic", "ai_heuristics")`
- `("ai_deterministic", "ai_heuristics", "ai_heuristics")`

Pure mirrors are intentionally excluded from the default schedule because they do not provide cross-AI rating signal.

Current rating algorithm:

- controller-level pairwise Elo derived from final multi-player placements
- final standings use the same winner tuple logic as the rules engine:
  - powered cities
  - money
  - connected cities
  - `state.player_order` as deterministic fallback
- equal `(powered, money, connected)` signatures are treated as pairwise draws
- for each game, only distinct controller pairs are rated
- for each unordered controller pair:
  - compare all seat-vs-seat cross-pairings
  - average the seat-pair outcomes into one controller-pair score
  - compute Elo expectation with rating scale `400`
  - use `pair_k = base_k / (distinct_controller_count - 1)`

Reporting:

- JSON report includes:
  - bucket metadata
  - resolved selected regions
  - algorithm parameters
  - schedule metadata
  - per-controller summaries
  - per-controller-pair diagnostics
  - per-game summaries

CLI workflow:

- Tool: [../tools/evaluate_ai_ratings.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/tools/evaluate_ai_ratings.py)
- Example:

```bash
PYTHONPATH=src python -m powergrid.tools.evaluate_ai_ratings \
  --map germany \
  --players 3 \
  --controllers ai_deterministic ai_heuristics \
  --games-per-lineup 20 \
  --seed-start 1
```

- Default output path:
  - `artifacts/ai_ratings/germany_3p.json`

If you change naming or defaults, update all of:

- [__init__.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/__init__.py)
- [../model.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/model.py)
- [../gui/app.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/app.py)
- tests in [../../../tests/test_ai.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_ai.py), [../../../tests/test_ai_evaluation.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_ai_evaluation.py), [../../../tests/test_session.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_session.py), and [../../../tests/test_gui.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_gui.py)

## Guidance For Further Development

### 1. Keep the public seam unchanged

Prefer to keep:

- `BaseAiController.choose_intent(request, snapshot)`
- registry-based controller construction
- session-owned state mutation

This makes new controllers easy to plug in and keeps GUI/CLI/session logic stable.

### 2. Reuse model legality helpers

When adding new AI logic:

- Use `legal_*` helpers to enumerate options
- Use model transition helpers to simulate outcomes
- Do not hand-build “almost legal” shortcuts unless performance forces it and you also preserve a validated path

Good pattern:

- enumerate legal actions
- clone state
- apply one legal action through rules/model helpers
- score resulting state

### 3. Preserve determinism

Avoid:

- random tie-breaking
- iteration over unordered structures without sorting
- depending on dictionary order where strategic choices matter

Always add stable tie-breakers to `max(...)`, `min(...)`, and search rankings.

### 4. Extend heuristics by cluster, not by scattering constants

In [strategic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/strategic.py), keep changes grouped by concern:

- auction logic
- resource search
- build search
- bureaucracy/generation
- evaluation

If a new scoring dimension becomes important, prefer:

- adding it to `_evaluate_player_strength(...)`
- possibly adjusting `STAGE_WEIGHTS`
- then updating the local search heuristics only if needed

That keeps the AI coherent instead of accumulating one-off conditionals.

### 5. Watch performance

The heuristic AI already does multiple cloned-state simulations.

High-risk areas:

- `_search_resource_purchase(...)`
- `_search_best_build_plan(...)`
- `_enumerate_generation_summaries(...)`
- repeated calls to `_best_generation_summary(...)`

If performance becomes an issue:

- reduce branching before adding deeper search
- cache repeated derived values within a decision call
- avoid recomputing full evaluations when only a small feature changed

But preserve correctness first.

### 6. Add scenario tests for every strategic change

Use focused scenario/state tests rather than relying only on full-game smoke runs.

Current AI tests already cover:

- registry/controller construction
- deterministic opening/build behavior
- heuristic auction preferences
- heuristic pass logic
- smarter discard choices
- useful resource buying
- multi-city build behavior
- final-round bureaucracy behavior
- AI-vs-AI smoke completion
- Elo evaluation helpers, reporting, and CLI output
- custom AI state logging and structured game-log dumping

When changing behavior, extend [../../../tests/test_ai.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_ai.py) first.

### 7. Keep human-facing controller names intentional

If you add a new AI:

1. implement a new `BaseAiController` subclass
2. register it in `AI_CONTROLLER_REGISTRY`
3. decide whether it should be:
   - explicit only, or
   - the generic `ai` alias, or
   - the default helper-generated AI for `ai_players`
4. update GUI option ordering if users should see it
5. add regression tests for registry, session creation, and launcher behavior

## Suggested Next Improvement Areas

Reasonable next steps if we want a stronger AI without changing architecture:

- collect candidate-level counterfactual labels or add a behavior-cloning policy head;
  the 1 GiB behavior-only rank/value run generalized on labels but failed held-out Elo
- extend the current multi-seed, multi-policy Germany data to more player counts/maps
- benchmark new NN checkpoints by held-out-seed Elo before changing any default
- add dataset merge/balancing and calibration diagnostics for the win head
- add memoization inside strategic evaluation for repeated summaries
- improve opponent modeling during auctions and endgame trigger decisions
- improve build planning with better frontier/network centrality features
- add richer hybrid-fuel valuation and future storage planning
- split `strategic.py` into smaller modules once the file becomes hard to maintain

## Validation

Useful commands after AI changes:

```bash
PYTHONPATH=src python -m unittest tests.test_ai tests.test_ai_evaluation tests.test_model tests.test_session tests.test_gui -q
```

Neural rank-value component checks:

```bash
PYTHONPATH=src python -m unittest tests.test_nn_rank_value -v
PYTHONPATH=src python -m powergrid.tools.validate_nn_observation
PYTHONPATH=src python -m powergrid.tools.validate_nn_candidates
PYTHONPATH=src python -m powergrid.tools.validate_nn_dataset
PYTHONPATH=src python -m powergrid.tools.validate_nn_model
PYTHONPATH=src python -m powergrid.tools.validate_nn_training
PYTHONPATH=src python -m powergrid.tools.validate_nn_controller
```

Neural RL Policy/vector-Q checks:

```bash
PYTHONPATH=src python -m unittest tests.test_nn_rl_based -v
PYTHONPATH=src python -m powergrid.tools.validate_nn_rl_based \
  --section all --output artifacts/validation/ai_nn_rl_based_v1.json

# Formal model thresholds, including 50 full-rollout calibration roots:
PYTHONPATH=src python -m powergrid.tools.validate_nn_rl_based \
  --section training --dataset artifacts/datasets/nn_rl_bootstrap \
  --search-dataset artifacts/datasets/nn_rl_search_1 \
  --bootstrap-checkpoint artifacts/models/ai_nn_rl_based_bootstrap.npz \
  --checkpoint src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz \
  --calibration-roots 50 --enforce-acceptance
```

Profile behavior and held-out strength calibration:

```bash
PYTHONPATH=src python -m unittest tests.test_profiled_deterministic_ai -v
PYTHONPATH=src python -m powergrid.tools.validate_profiled_deterministic_ai \
  --games-per-lineup 30 --seed-start 9001
```

If the change is isolated to AI logic, start with:

```bash
PYTHONPATH=src python -m unittest tests.test_ai tests.test_ai_evaluation tests.test_session -q
```
