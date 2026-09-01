# `ai_nn_rank_value_v1`

## Status and intended use

`ai_nn_rank_value_v1` is the repository's first trainable AI controller. It is a
feature-engineered, two-head neural value baseline that scores the legal candidate
actions at each session decision.

The bundled checkpoint is a **bootstrap pipeline checkpoint**, not a claim that the
NN is stronger than `ai_heuristics`. It exists so a clean checkout can exercise
observation building, candidate scoring, checkpoint loading, logging, and full-game
inference without first training a model. Its validation split is deliberately
reported below; more diverse games and behavior policies are required before using
the model as a strength baseline.

The existing controller meanings are unchanged:

- `ai_deterministic`: simple deterministic rule baseline
- `ai_heuristics`: stronger hand-built evaluation/search controller
- `ai_nn_rank_value_v1`: trainable rank/value baseline
- `ai`: alias of `ai_deterministic`

## Runtime design

For every `TurnRequest`, the controller performs this pipeline:

1. Build an actor-relative observation containing only public game information.
2. Generate legal atomic candidate intents for the current phase.
3. Encode the state once and each candidate action separately.
4. Run all `(state, action)` rows through the MLP in one NumPy batch.
5. Score each candidate as
   `0.7 * P(win) + 0.3 * ((rank_value + 1) / 2)`.
6. Return the highest-scoring intent, using candidate order as a deterministic tie-break.

The controller does not mutate game state. `GameSession` remains responsible for
validating and applying the returned `GuiIntent`.

### Public-information boundary

The observation excludes:

- `GameConfig.seed`
- hidden draw-stack identities and order
- hidden bottom-stack identities and order

It includes the number of hidden plant cards, because that count is public. Visible
markets, player holdings, resources, networks, auction state, resource prices, map
occupancy, and legal build-cost summaries are included.

`validate_nn_observation` verifies that changing the seed and reversing both hidden
stacks leaves the serialized public observation and all encoded features unchanged.

### Candidate actions

Candidate generation is deterministic and emits intents that can be submitted
directly to a session:

- pending plant/resource discards: every explicitly legal discard option
- auction opening: every auctionable plant at its minimum opening bid
- active auction: minimum legal raise when affordable, plus pass
- resources: every affordable amount for every currently legal resource, plus finish
- building: each currently legal single-city build, plus finish
- bureaucracy: every legal plant/resource-mix combination, including skip

Minimum-only auction raises are intentional. A player can continue raising on later
auction turns, while the candidate count remains small and stable. The generator
filters the session's range-shaped bid action when `min_bid > max_bid`; passing is
then the only legal choice.

Resource and build actions are atomic because the same player receives another
request after a purchase/build until it chooses the corresponding finish action.

## Feature design

Feature schemas are named, versioned, and embedded in both dataset metadata and the
checkpoint. Runtime loading rejects a checkpoint whose feature names do not exactly
match the current encoder.

The exhaustive index-by-index data dictionary, including semantic types, source
values, divisors, zero-padding rules, and repeated-slot expansion formulas, is in
[`ai_nn_rank_value_v1_feature_dictionary.md`](ai_nn_rank_value_v1_feature_dictionary.md).

### State features: 513

| Group | Count | Contents |
| --- | ---: | --- |
| Global | 24 | round, step, phase, decision type, map, player/end thresholds, hidden-card count |
| Resource market | 40 | availability, supply, and first eight unit-price slots for four fuels |
| Current market | 60 | six plant slots, ten attributes per plant |
| Future market | 40 | four plant slots, ten attributes per plant |
| Players | 324 | six actor-relative slots, 54 public economy/network/plant/resource attributes each |
| Auction | 8 | active status, plant/bid, actor roles, bidder/pass counts |
| Map/build summary | 17 | occupancy distribution, connection statistics, first eight legal build costs |

Player slot zero is always the acting player. Opponents are ordered by public turn
order and player id; unused slots are zero-padded. This makes one schema work for
three through six players.

### Action features: 42

The action vector contains:

- 11-way action-type one-hot
- bid, plant price, amount, direct cost, and cash after action
- four-way purchased-resource one-hot
- city count, connection/build/total cost, and occupancy
- powered cities and resulting income
- four resource-mix amounts and discarded-unit count
- ten public attributes for the referenced plant

Continuous values use fixed game-scale divisors in the encoder. Training additionally
stores per-input training-set mean and standard deviation; zero-variance columns use
a scale of one.

### Feature types and encoding choice

The distinction below is semantic. One categorical variable becomes several binary
columns after one-hot encoding, so those columns should not be confused with
independent boolean variables.

| Input | Independent bool columns | Enum one-hot columns | Set multi-hot columns | Numeric/ordinal columns | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| State | 120 | 17 | 136 | 240 | 513 |
| Action | 3 | 15 | 4 | 20 | 42 |
| Combined | 123 | 32 | 140 | 260 | 555 |

The groups are:

- independent booleans
  - step-3 pending flags
  - player/plant-slot presence and actor flags
  - plant ecological/hybrid flags
  - auction active/actor-role flags
- multi-valued enums encoded one-hot
  - step: 3 columns
  - phase: 4 columns
  - decision type: 7 columns
  - map: 3 columns
  - action type: 11 columns
  - purchased resource type: 4 columns
- set-valued categories encoded multi-hot
  - four fuel-membership columns for each visible/owned/referenced plant
  - hybrid plants can therefore set both coal and oil instead of forcing one enum value
- numeric or ordinal values encoded as normalized scalars
  - round, player count, thresholds, counts, turn order, cash, resource amounts
  - plant price/cost/output, resource prices/supply
  - bid, purchase/build costs, occupancy, powered cities, income, resource mix

Many values in the last group are integer-valued in the rules engine, but they are
treated as ordered numeric values by the network.

### Why v1 does not use embedding tables

The v1 design deliberately keeps low-cardinality categories as one-hot/multi-hot and
ordered values as scalars:

- the largest ordinary enum has only 11 values, and all ordinary enum columns total 32
- one-hot preserves exact distinctions without adding an arbitrary latent geometry
- scalar prices, costs, counts, and cash preserve order and allow interpolation;
  discretizing them into embedding indexes would lose that useful structure
- plant fuel is set-valued, so multi-hot is a more faithful representation than a
  single categorical embedding
- the current bootstrap already overfits a small dataset; extra embedding parameters
  would increase that risk without solving the data-coverage problem

Embeddings are reasonable v2 experiments for genuinely identity-like fields that are
currently absent:

- a 42-card plant-id embedding, added alongside price/fuel/output features
- a map-scoped city-id embedding for the 42 Germany, 42 USA, and 3 test-map cities,
  added alongside cost/occupancy and preferably graph-derived topology features

Those should be residual identity embeddings, not replacements for structured
features. Map/phase/action/resource embeddings are not recommended unless a controlled
ablation beats the simpler one-hot model on held-out games and Elo.

## Model and objective

The default network is a small MLP implemented in NumPy:

```text
513 state + 42 action = 555 inputs
    -> Linear(128) -> ReLU
    -> Linear(64)  -> ReLU
    -> two output logits
       -> sigmoid: P(win)
       -> tanh: rank_value in [-1, 1]
```

The two equal-weight training losses are:

- binary cross entropy for `is_winner`
- mean squared error for `rank_value`

Adam and a `1e-5` weight-decay term train the parameters. NumPy implements the model
and PyArrow provides compressed, streaming Parquet data I/O; no PyTorch runtime is
needed.

### Terminal labels

Each decision made by player `p` receives the game's final outcome:

```text
is_winner = 1 iff p is in WinnerResult.winner_ids, else 0
rank_value = (N + 1 - 2 * final_place) / (N - 1)
```

For three players, places 1/2/3 map to `+1/0/-1`. `final_place` comes from the real
winner ordering: powered cities, money, then connected cities. Exact ties retain the
same place. The dataset also records final powered cities, money, and connected
cities for diagnostics.

### Behavior and counterfactual samples

Normal data records the action selected by a behavior controller and applies the
same final label to that chosen `(state, action)` row. Optional counterfactual data
forks `GameSession`, submits another candidate, finishes the game with a rollout
controller, and labels that candidate from its own terminal result.

`GameSession.fork()` copies both rules state and session-owned phase cursor/
bureaucracy choices. Fork logs are omitted by default. Counterfactual rollouts are
expensive but reduce the action-selection confounding present in behavior-only data.

Train, validation, and test assignment is a stable SHA-256 hash of complete
`game_id`, never an individual-row split. Every decision and counterfactual from one
game therefore stays in exactly one partition.

## Installation

Install the explicit NumPy and PyArrow dependencies:

```bash
python -m pip install -r requirements-ml.txt
```

All repository commands require `PYTHONPATH=src`.

## Generate training data

Example with behavior data plus sparse counterfactual rollouts:

```bash
PYTHONPATH=src python -m powergrid.tools.generate_nn_rank_value_dataset \
  --output artifacts/nn_rank_value/train \
  --games 100 \
  --seed-start 1 \
  --map germany \
  --players 3 \
  --controllers ai_deterministic \
    ai_deterministic_efficiency \
    ai_deterministic_expansion \
    ai_deterministic_reserve \
  --region-set blue,magenta,black \
  --region-set black,yellow,green \
  --region-set yellow,green,cyan \
  --target-shard-size-mib 512 \
  --train-fraction 0.8 \
  --validation-fraction 0.1 \
  --test-fraction 0.1 \
  --split-seed 17 \
  --workers 4
```

This writes a directory:

```text
train/
  manifest.json
  train/part-00000.parquet
  validation/part-00000.parquet
  test/part-00000.parquet
  examples/game-01-....jsonl
  examples/game-02-....jsonl
  examples/game-03-....jsonl
```

- generation holds only one completed game per worker before writing it
- each game becomes one Parquet row group, so no game crosses a split boundary
- Zstandard-compressed shards close after reaching the configured target size; the
  final shard of each split can be smaller
- the manifest records feature and Parquet schemas plus every shard's row count,
  game count, byte size, and SHA-256 checksum
- exactly the first three complete games are retained as readable JSONL examples
- repeatable `--region-set` values rotate across games, just like behavior controllers
- `--workers` uses a bounded process queue; peak generation memory therefore depends
  on worker count and one-game size, not total dataset size

The four no-search deterministic policies above are the generator defaults. Each new
profile is held-out strength-calibrated against the original baseline while retaining
deterministic generation speed. See
[profiled_deterministic_ai.md](profiled_deterministic_ai.md). Add `ai_heuristics`
explicitly only when its slower search-based teacher behavior is required.

The output directory must not already exist, preventing an interrupted or mismatched
run from silently mixing shards. Generated data under `artifacts/nn_rank_value/` is
ignored by Git.

## Train a checkpoint

```bash
PYTHONPATH=src python -m powergrid.tools.train_nn_rank_value \
  --dataset artifacts/nn_rank_value/train \
  --output artifacts/nn_rank_value/ai_nn_rank_value_v1.npz \
  --epochs 25 \
  --batch-size 2048 \
  --learning-rate 0.001 \
  --hidden-dims 128,64 \
  --scan-batch-size 8192 \
  --seed 17
```

Normalization statistics, optimizer batches, and final split evaluation all stream
from Parquet. The trainer never constructs an all-row Python list or full-dataset
feature array. The compressed `.npz` contains weights, normalization statistics,
full feature-name schemas, architecture metadata, dataset path and manifest SHA-256,
sample counts, label definitions, and final train/validation/test metrics.

## Use the controller

The bundled checkpoint is loaded automatically:

```bash
PYTHONPATH=src python -m powergrid.tools.run_ai_game \
  --controllers ai_nn_rank_value_v1 ai_deterministic ai_deterministic \
  --seed 401
```

To use another checkpoint without changing code:

```bash
POWERGRID_NN_RANK_VALUE_CHECKPOINT=artifacts/nn_rank_value/ai_nn_rank_value_v1.npz \
PYTHONPATH=src python -m powergrid.tools.run_ai_game \
  --controllers ai_nn_rank_value_v1 ai_deterministic ai_deterministic
```

The controller is also available in the GUI AI-version picker. Every decision emits
an `nn_rank_value_decision` structured log containing the checkpoint, candidate
scores, both head predictions, and selected intent.

For a comparative Elo run (the evaluator currently accepts exactly two controller
types in the Germany/three-player bucket):

```bash
PYTHONPATH=src python -m powergrid.tools.evaluate_ai_ratings \
  --controllers ai_nn_rank_value_v1 ai_deterministic \
  --games-per-lineup 20 \
  --seed-start 1
```

Elo 评测默认按 game seed 从全部合法连续区域组合中可复现地随机采样，并保证相同
seed 的换座 lineup 使用同一组合。传 `--regions black,blue,magenta` 可固定区域；
`--region-sampling-seed` 只改变区域采样调度。

## Validation scripts and current results

Run each component independently:

```bash
PYTHONPATH=src python -m powergrid.tools.validate_nn_observation
PYTHONPATH=src python -m powergrid.tools.validate_nn_candidates
PYTHONPATH=src python -m powergrid.tools.validate_nn_dataset
PYTHONPATH=src python -m powergrid.tools.validate_nn_model
PYTHONPATH=src python -m powergrid.tools.validate_nn_training
PYTHONPATH=src python -m powergrid.tools.validate_nn_controller
```

Results recorded for the bundled implementation:

| Component | Result |
| --- | --- |
| Public observation | PASS; 513 features; seed and hidden deck-order invariance |
| Candidate generation | PASS; opening 4, resource 9, build 3, endgame 4 candidates replayed; cash-limited raise filtered |
| Dataset/labels | PASS; 1,507 behavior + 16 counterfactual rows; streaming Parquet schemas/checksums; three whole-game JSONL examples; game-exclusive train/validation/test split |
| MLP/checkpoint | PASS; synthetic loss `1.712759 -> 0.122652`, win accuracy `0.988281`, exact save/load predictions |
| Training pipeline | PASS; 1,415/519/940 train/validation/test rows; final loss 0.587921/0.928044/0.973421; streaming normalization/training/evaluation and manifest hash verified |
| Full controller | PASS; seed 401 completed in 439 actions, 110 NN decisions, zero intent errors |

Automated regression tests:

```bash
PYTHONPATH=src python -m unittest tests.test_nn_rank_value -v
```

Current result: 8 tests passed.

## 1 GiB Parquet generation/training benchmark

The following same-machine point measurements used an Apple M4 MacBook Air, Python
3.12 arm64, NumPy 2.3.5, and PyArrow 21.0.0.

### Dataset generation

The full run rotated all four deterministic behavior policies and all 13 legal
three-region Germany selections. Seeds 30,001 through 71,000 were assigned 80/10/10
by `sha256("1701:" + game_id)`.

| Metric | Result |
| --- | ---: |
| Games | 41,000 |
| Behavior rows | 20,143,881 |
| Counterfactual rows | 0 |
| Wall time | 3,492.18 s (58.20 min) |
| Sustained throughput | 11.74 games/s |
| Parquet size | 1,067,965,552 bytes (1,018.5 MiB) |
| Splits by games | 32,758 / 4,091 / 4,151 |
| Splits by rows | 16,095,160 / 2,010,852 / 2,037,869 |
| Shards | 2 train + 1 validation + 1 test |
| Whole-game JSONL examples | 3 |

The first train shard closed after reaching its target and was 602,025,481 bytes
(574.1 MiB) after its footer; the second train shard was 251,467,099 bytes. The
validation and test tail shards were about 102 MiB each. All four Parquet shards and
all three examples passed manifest byte-size and SHA-256 verification.

The 200-game cold calibration initially reached 23.49 games/s. Sustained full-load
throughput fell to 11.74 games/s as the fanless machine reached thermal equilibrium,
so short benchmark extrapolation materially understated total generation time.

### Streaming training

The `513 + 42 -> 128 -> 64 -> 2` network used batch size 2,048, five epochs, learning
rate `1e-3`, and seed 1,701.

| Stage | Result |
| --- | ---: |
| Streaming normalization | 37.0 s over 16,095,160 rows |
| Five optimizer epochs | 368.1 s; about 218,600 samples/s |
| Final train/validation/test evaluation | 53.2 s |
| Total | 458.3 s (7.64 min) |
| Checkpoint | 298 KiB, SHA-256 `78695a1ea416645ac4d558212e268c48380c5ed22ece823a36746c7f34103379` |

| Split | Loss | Rank MAE | Win accuracy |
| --- | ---: | ---: | ---: |
| Train | 0.672811 | 0.413573 | 0.822253 |
| Validation | 0.676594 | 0.415465 | 0.821174 |
| Test | 0.682884 | 0.415650 | 0.819860 |

The small generalization gap confirms that the larger dataset removes the bootstrap
checkpoint's row-label overfitting. It does **not** establish action-selection
strength, as the held-out games below demonstrate.

### Held-out game strength

Each matchup used seeds outside the generation range and two balanced lineups:
`(NN, NN, opponent)` and `(NN, opponent, opponent)`, 30 games each.

| Opponent | NN game wins | NN per-seat firsts | NN average finish | NN seat-pair score |
| --- | ---: | ---: | ---: | ---: |
| Original deterministic | 4/60 (6.7%) | 4/90 (4.4%) | 2.478 | 0.142 |
| Efficiency | 1/60 (1.7%) | 1/90 (1.1%) | 2.589 | 0.058 |
| Expansion | 1/60 (1.7%) | 1/90 (1.1%) | 2.622 | 0.033 |
| Reserve | 4/60 (6.7%) | 4/90 (4.4%) | 2.500 | 0.125 |

The checkpoint is therefore retained only under ignored `artifacts/` and is **not**
promoted over the bundled bootstrap. A diagnostic seed showed 48 NN `auction_bid`
decisions versus only three `auction_pass` decisions, with mean continued bid 19.5
and maximum 52. In a late auction, increasing the bid from 17 to 44 monotonically
increased the model's predicted advantage over passing. This is action confounding:
behavior-only rows contain the teacher's selected action, so high bids correlate with
valuable plants and successful states, while alternative pass/bid outcomes are absent.

The next strength iteration should collect candidate-level counterfactual labels (or
train a direct behavior-cloning policy head) and gate promotion on held-out Elo, not
terminal-label loss alone.

## Bundled bootstrap checkpoint provenance

The checked-in `src/powergrid/data/ai_models/ai_nn_rank_value_v1.npz` has SHA-256
`8150cef5f53fe1205cd7c3a98528b708b1eb2565a5ca6de539a9d18da19fb072`.
It predates the Parquet migration and was produced by the retired dataset format v1
JSONL pipeline. It remains checked in only as a small controller/runtime smoke-test
artifact; current generation and training commands are documented above.

Dataset: 24 Germany/three-player deterministic games, 11,635 behavior samples and
179 counterfactual rollout samples. The game-level split contained 9,347 training
and 2,467 validation rows.

| Split | Loss | Rank MAE | Win accuracy |
| --- | ---: | ---: | ---: |
| Train | 0.021966 | 0.056471 | 0.996683 |
| Validation | 2.291603 | 0.488867 | 0.699635 |

The large train/validation gap is the reason this artifact is called a bootstrap
checkpoint. The next strength iteration should collect substantially more seeds,
mix behavior policies, cover player counts/maps, increase counterfactual coverage,
and compare by held-out-seed Elo—not tune against this small validation set.
