# AI Agent Guide

This directory contains the rule-based AI controllers for Power Grid.

## Documentation Rule

Treat this file as part of the AI implementation surface.

Whenever an agent changes any of the following, it should also update this document in the same change:

- controller names, aliases, or defaults
- registry wiring
- session/controller integration behavior
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
  - `ai_heuristics` -> `StrategicAiController`
  - `ai_deterministic` -> `DeterministicAiController`
  - `ai` -> `DeterministicAiController` as the generic/default alias
- Session integration in [../session.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/session.py)
  - `GameSession.default_seat_agents()` resolves non-human controllers through `build_ai_controller(...)`
  - `advance_one_ai_action()` and `advance_until_blocked()` call `choose_intent(...)`
  - The AI layer does not mutate session state directly; it only returns intents

This means the core contract is:

1. Session constructs a `TurnRequest` plus full `GameSnapshot`
2. AI chooses one legal `GuiIntent`
3. Session validates and applies that intent through model/rules helpers

## Files

- [base.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/base.py): abstract controller base
- [deterministic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/deterministic.py): simple baseline AI
- [strategic.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/strategic.py): stronger heuristic AI
- [__init__.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/__init__.py): registry and controller construction

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
  - Uses dynamic reserve prices via `_auction_reserve(...)`
  - Incorporates plant quality, portfolio fit, resource pressure, cash floors, and game stage
  - Can pass on overpriced bids
  - In round 1, avoids passing when forced to buy
  - Supports “bait” style starts when opponents value a plant more than the AI does
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

Also note:

- Helper-generated AI seats from `make_default_seat_configs(..., ai_players=...)` currently use `ai_deterministic`
- The GUI setup still exposes explicit AI versions and currently defaults its version picker to `ai_heuristics`

If you change naming or defaults, update all of:

- [__init__.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/ai/__init__.py)
- [../model.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/model.py)
- [../gui/app.py](/Users/mac/Desktop/syt/Projects/PowerGrid/src/powergrid/gui/app.py)
- tests in [../../../tests/test_ai.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_ai.py), [../../../tests/test_session.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_session.py), and [../../../tests/test_gui.py](/Users/mac/Desktop/syt/Projects/PowerGrid/tests/test_gui.py)

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

- add memoization inside strategic evaluation for repeated summaries
- improve opponent modeling during auctions and endgame trigger decisions
- improve build planning with better frontier/network centrality features
- add richer hybrid-fuel valuation and future storage planning
- split `strategic.py` into smaller modules once the file becomes hard to maintain

## Validation

Useful commands after AI changes:

```bash
PYTHONPATH=src python -m unittest tests.test_ai tests.test_model tests.test_session tests.test_gui -q
```

If the change is isolated to AI logic, start with:

```bash
PYTHONPATH=src python -m unittest tests.test_ai -q
```
