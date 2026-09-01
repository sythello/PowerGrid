# `ai_nn_rank_value_v1` feature dictionary

## Scope

This document is the exhaustive data dictionary for observation schema v1 and
action-feature schema v1 used by `ai_nn_rank_value_v1`.

> 中文读者可以直接跳转到[中文特征说明](#中文特征说明)。中文部分同样完整覆盖
> 513 个 state 特征和 42 个 action 特征，而不是英文部分的简略摘要。

- State vector: `state_features[0:513]`
- Candidate-action vector: `action_features[0:42]`
- Combined model input: 555 columns
- Authoritative encoder:
  [`observation.py`](../src/powergrid/ai/nn_rank_value/observation.py)

The tables use zero-based vector indexes. A row containing a parameter such as `r`,
`i`, `j`, or `k` describes every physical feature obtained by expanding the stated
range. Base-index tables make every expanded column uniquely identifiable without
duplicating hundreds of structurally identical rows.

## Types and value transformation

Every feature is ultimately serialized to Parquet and passed to the model as a
`float32`. The **semantic type** in this document describes the source value before
conversion:

| Semantic type | Meaning in the feature vector |
| --- | --- |
| Boolean | `false -> 0.0`, `true -> 1.0` |
| Enum one-hot | Exactly one column for the applicable enum normally equals `1.0`; the others equal `0.0` |
| Set multi-hot | Each membership is an independent `0.0/1.0`; a hybrid plant can set both coal and oil |
| Integer count | A non-negative integer divided by the stated fixed divisor |
| Integer ordinal | An ordered integer such as round, bid, price, or turn position, divided by the stated divisor |
| Continuous summary | A derived mean or another real-valued summary divided by the stated divisor |

The fixed divisor does not clip values to `[0, 1]`. For example, cash can exceed 200,
so a cash feature can exceed `1.0`. Training performs a second transformation using
the training split:

```text
encoded_value = source_value / fixed_divisor
model_input = (encoded_value - training_mean) / training_stddev
```

If training standard deviation is below `1e-6`, it is replaced with `1.0`. The mean
and scale arrays are stored in the checkpoint and reused during inference.

The semantic-type distribution is:

| Vector | Independent Boolean | Enum one-hot columns | Set multi-hot columns | Numeric/ordinal | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| State | 120 | 17 | 136 | 240 | 513 |
| Action | 3 | 15 | 4 | 20 | 42 |
| Combined | 123 | 32 | 140 | 260 | 555 |

An enum column is counted separately from an independent Boolean even though both
are physically stored as `0.0/1.0`, because the enum columns jointly represent one
multi-valued source variable.

### Zero-value conventions

- A false Boolean is zero.
- An absent fixed-size player or plant slot is all zero, including its `present`
  column. Consumers must use `present` to distinguish absence from real numeric zero.
- A numeric action feature that is irrelevant to the current action type is zero.
  Consumers must use the action-type one-hot columns to interpret it.
- A padded price or build-cost slot is zero. Zero therefore means “no slot value,” not
  a free resource or free city.
- For enum groups that are not applicable, all columns may be zero. For example,
  `action.resource.*` is all zero for a build action.

## Ordering conventions

These conventions determine the meaning of repeated slots:

- Resources always use `coal`, `oil`, `garbage`, `uranium` order.
- Plant slots are sorted by ascending plant `price`.
- `player_0` is always the player making the current decision.
- Opponents occupy `player_1` onward, sorted by public turn-order position and then
  player ID.
- Unused player slots are zero-padded through `player_5`, allowing one schema for
  three through six players.
- Four owned-plant slots are reserved because a player can temporarily hold an extra
  plant while resolving a mandatory discard, even though the normal ownership limit
  is three in the current rules.

## State vector overview

| Index range | Feature group | Columns |
| --- | --- | ---: |
| `0..23` | Global game/request context | 24 |
| `24..63` | Resource market | 40 |
| `64..123` | Current power-plant market | 60 |
| `124..163` | Future power-plant market | 40 |
| `164..487` | Actor and opponent slots | 324 |
| `488..495` | Auction context | 8 |
| `496..512` | Map and legal-build summary | 17 |
| **Total** |  | **513** |

## Global game and request context: state indexes 0–23

| Index | Feature name | Semantic type | Source and encoding | Actual meaning |
| ---: | --- | --- | --- | --- |
| 0 | `global.round` | Integer ordinal | `round_number / 20` | Current game round, starting at round 1. The divisor is a scale, not a maximum-round rule. |
| 1 | `global.step_1` | Enum one-hot | `step == 1` | The game is in Step 1. |
| 2 | `global.step_2` | Enum one-hot | `step == 2` | The game is in Step 2. |
| 3 | `global.step_3` | Enum one-hot | `step == 3` | The game is in Step 3. |
| 4 | `global.player_count` | Integer count | `number_of_players / 6` | Number of seats participating in the game. |
| 5 | `global.end_game_cities` | Integer count | `end_game_city_threshold / 22` | Connected-city count that triggers the end of the game for this player count. |
| 6 | `global.hidden_plant_count` | Integer count | `(draw_stack + bottom_stack + pending_step_3_card) / 42` | Public count of plant cards not currently visible. Hidden card identities and order are not included. |
| 7 | `global.step_3_pending` | Boolean | Direct Boolean | The Step 3 marker is still pending behind the draw stack and has not yet been surfaced/processed. The marker is represented by this flag rather than a normal hidden plant object. |
| 8 | `global.auction_step_3_pending` | Boolean | Direct Boolean | The Step 3 placeholder has surfaced during the auction flow, but the transition into Step 3 is deferred until the current auction handling finishes. |
| 9 | `global.phase.auction` | Enum one-hot | `phase == "auction"` | The session is in the auction phase. |
| 10 | `global.phase.buy_resources` | Enum one-hot | `phase == "buy_resources"` | The session is in the resource-purchase phase. |
| 11 | `global.phase.build_houses` | Enum one-hot | `phase == "build_houses"` | The session is in the network-building phase. |
| 12 | `global.phase.bureaucracy` | Enum one-hot | `phase == "bureaucracy"` | The session is in the generation/income phase. |
| 13 | `global.decision.auction_start` | Enum one-hot | `decision_type == "auction_start"` | The actor must select a plant and opening bid or otherwise resolve the auction-start request. |
| 14 | `global.decision.auction_bid` | Enum one-hot | `decision_type == "auction_bid"` | The actor must raise or pass in an active auction. |
| 15 | `global.decision.buy_resources` | Enum one-hot | `decision_type == "buy_resources"` | The actor may buy a resource quantity or finish buying. |
| 16 | `global.decision.build_houses` | Enum one-hot | `decision_type == "build_houses"` | The actor may build a legal city or finish building. |
| 17 | `global.decision.bureaucracy` | Enum one-hot | `decision_type == "bureaucracy"` | The actor must select a legal plant-run/resource-mix plan. |
| 18 | `global.decision.discard_power_plant` | Enum one-hot | `decision_type == "discard_power_plant"` | The actor must discard a plant after exceeding the ownership limit. |
| 19 | `global.decision.discard_hybrid_resources` | Enum one-hot | `decision_type == "discard_hybrid_resources"` | The actor must discard coal/oil after a hybrid-storage legality change. |
| 20 | `global.map.germany` | Enum one-hot | `map_id == "germany"` | Germany map is active. |
| 21 | `global.map.usa` | Enum one-hot | `map_id == "usa"` | USA map is active. |
| 22 | `global.map.test` | Enum one-hot | `map_id == "test"` | Development test map is active. |
| 23 | `global.selected_region_count` | Integer count | `len(selected_regions) / 6` | Number of enabled map regions. Region identities are not encoded in schema v1. |

`phase` describes the broad rules phase; `decision_type` describes the exact request.
They are intentionally both present because pending discard decisions can occur while
the broad phase remains unchanged.

## Resource market: state indexes 24–63

Each resource contributes ten consecutive columns. The base index `B(r)` is:

| Resource `r` | `B(r)` | Expanded feature range |
| --- | ---: | --- |
| `coal` | 24 | `market.coal.*` at `24..33` |
| `oil` | 34 | `market.oil.*` at `34..43` |
| `garbage` | 44 | `market.garbage.*` at `44..53` |
| `uranium` | 54 | `market.uranium.*` at `54..63` |

| Relative index | Feature pattern | Semantic type | Source and encoding | Actual meaning |
| ---: | --- | --- | --- | --- |
| `B(r)+0` | `market.{r}.available` | Integer count | `len(available_unit_prices(r)) / 24` | Number of units of resource `r` currently present in purchasable market price bands. |
| `B(r)+1` | `market.{r}.supply` | Integer count | `resource_supply[r] / 24` | Units of `r` in the central refill supply, excluding units in the purchasable market and units held by players. |
| `B(r)+2+k`, `k=0..7` | `market.{r}.unit_price_{k}` | Integer ordinal | `k`-th available unit price `/ 20`; zero when absent | Price of the `(k+1)`-th cheapest currently purchasable unit. Repeated prices represent multiple units in one band. Only the first eight units are encoded. |

Expansion count: `4 resources * (2 summaries + 8 price slots) = 40` features.

## Reusable power-plant feature block

Every visible, owned, or action-referenced plant uses the same ten-column block. If
the block starts at index `P`, the columns are:

| Offset | Feature suffix | Semantic type | Source and encoding | Actual meaning |
| ---: | --- | --- | --- | --- |
| `P+0` | `.present` | Boolean | Plant object exists | Distinguishes a real plant from a padded or inapplicable slot. |
| `P+1` | `.price` | Integer ordinal / identity-like | `plant.price / 50` | Printed plant number, minimum auction price, and unique card identifier in the current deck. |
| `P+2` | `.resource_cost` | Integer count | `resource_cost / 3` | Number of fuel units required to run the plant once; zero for ecological plants. |
| `P+3` | `.output_cities` | Integer count | `output_cities / 10` | Maximum cities powered when the plant is run once. |
| `P+4` | `.resource.coal` | Set multi-hot | `"coal" in resource_types` | Plant can consume coal. |
| `P+5` | `.resource.oil` | Set multi-hot | `"oil" in resource_types` | Plant can consume oil. Hybrid plants can set both coal and oil. |
| `P+6` | `.resource.garbage` | Set multi-hot | `"garbage" in resource_types` | Plant can consume garbage. |
| `P+7` | `.resource.uranium` | Set multi-hot | `"uranium" in resource_types` | Plant can consume uranium. |
| `P+8` | `.ecological` | Boolean | `is_ecological` | Plant requires no purchased fuel. |
| `P+9` | `.hybrid` | Boolean | `is_hybrid` | Plant accepts a coal/oil mixture. |

The prefix before each suffix is supplied by the market, player, or action slot below.

## Current plant market: state indexes 64–123

Six visible current-market plants are sorted by ascending price. Slot `j=0..5` starts
at `P(j) = 64 + 10*j`.

| Slot `j` | Base index | Exact prefix | Index range |
| ---: | ---: | --- | --- |
| 0 | 64 | `current_market_0` | `64..73` |
| 1 | 74 | `current_market_1` | `74..83` |
| 2 | 84 | `current_market_2` | `84..93` |
| 3 | 94 | `current_market_3` | `94..103` |
| 4 | 104 | `current_market_4` | `104..113` |
| 5 | 114 | `current_market_5` | `114..123` |

For every row, append each suffix in the reusable plant block. For example,
`current_market_2.output_cities` is `state_features[87]` because slot 2 starts at 84
and `output_cities` has offset 3.

## Future plant market: state indexes 124–163

Four visible future-market plants are sorted by ascending price. Slot `j=0..3` starts
at `P(j) = 124 + 10*j`.

| Slot `j` | Base index | Exact prefix | Index range |
| ---: | ---: | --- | --- |
| 0 | 124 | `future_market_0` | `124..133` |
| 1 | 134 | `future_market_1` | `134..143` |
| 2 | 144 | `future_market_2` | `144..153` |
| 3 | 154 | `future_market_3` | `154..163` |

Each slot expands using the same ten suffixes. Step-specific market rules may leave
some slots absent; absent slots are all zero.

## Player slots: state indexes 164–487

Each of six player slots contributes 54 columns. Player slot `i=0..5` starts at
`B(i) = 164 + 54*i`:

| Slot `i` | Base index | Exact prefix | Role | Index range |
| ---: | ---: | --- | --- | --- |
| 0 | 164 | `player_0` | Acting player | `164..217` |
| 1 | 218 | `player_1` | First opponent by turn order | `218..271` |
| 2 | 272 | `player_2` | Second opponent by turn order | `272..325` |
| 3 | 326 | `player_3` | Third opponent or padding | `326..379` |
| 4 | 380 | `player_4` | Fourth opponent or padding | `380..433` |
| 5 | 434 | `player_5` | Fifth opponent or padding | `434..487` |

### Player-level fields

| Relative index | Feature pattern | Semantic type | Source and encoding | Actual meaning |
| ---: | --- | --- | --- | --- |
| `B(i)+0` | `player_{i}.present` | Boolean | Player exists in this slot | Distinguishes a real player from a zero-padded slot. |
| `B(i)+1` | `player_{i}.is_actor` | Boolean | `player_id == actor_id` | Whether this slot is the current decision maker. By construction true only for `player_0`. |
| `B(i)+2` | `player_{i}.turn_order` | Integer ordinal | `turn_order_position / 6` | Public one-based position in the current turn order; lower positions act earlier in phases that use forward order. |
| `B(i)+3` | `player_{i}.cash` | Integer ordinal | `elektro / 200` | Elektro currently held by the player. |
| `B(i)+4` | `player_{i}.houses_in_supply` | Integer count | `houses_in_supply / 22` | Unbuilt houses remaining in the player's personal supply. |
| `B(i)+5` | `player_{i}.connected_cities` | Integer count | `len(network_city_ids) / 22` | Number of cities in the player's network. City identities and topology are not encoded. |
| `B(i)+6` | `player_{i}.last_powered` | Integer count | `last_powered_cities[player_id] / 22` | Cities powered by this player in the most recently resolved bureaucracy round; zero before one has resolved. |
| `B(i)+7` | `player_{i}.last_income` | Integer ordinal | `last_income_paid[player_id] / 150` | Income paid to this player in the most recently resolved bureaucracy round; zero before one has resolved. |
| `B(i)+8` | `player_{i}.resource.coal` | Integer count | Stored coal `/ 8` | Total coal currently stored across all compatible plants. |
| `B(i)+9` | `player_{i}.resource.oil` | Integer count | Stored oil `/ 8` | Total oil currently stored across all compatible plants. |
| `B(i)+10` | `player_{i}.resource.garbage` | Integer count | Stored garbage `/ 8` | Total garbage currently stored across all compatible plants. |
| `B(i)+11` | `player_{i}.resource.uranium` | Integer count | Stored uranium `/ 8` | Total uranium currently stored across all compatible plants. |
| `B(i)+12` | `player_{i}.total_output` | Integer count | Sum of owned plant outputs `/ 22` | Nominal portfolio capacity if every plant can run; it does not check current stored fuel or connected-city limit. |
| `B(i)+13` | `player_{i}.largest_plant` | Integer ordinal | Largest owned plant price `/ 50` | Highest printed number among owned plants, used by normal turn-order rules. Zero when no plant is owned. |

### Owned-plant fields

Each player owns four sorted plant slots. Within player slot `i`, owned-plant slot
`j=0..3` begins at:

```text
P(i, j) = B(i) + 14 + 10*j
```

| Owned slot `j` | Relative range | Exact prefix pattern | Meaning |
| ---: | --- | --- | --- |
| 0 | `B(i)+14 .. B(i)+23` | `player_{i}.plant_0` | Lowest-price owned plant |
| 1 | `B(i)+24 .. B(i)+33` | `player_{i}.plant_1` | Second-lowest-price owned plant |
| 2 | `B(i)+34 .. B(i)+43` | `player_{i}.plant_2` | Third-lowest-price owned plant |
| 3 | `B(i)+44 .. B(i)+53` | `player_{i}.plant_3` | Fourth-lowest-price plant, normally present only during pending discard |

Every owned slot expands with the ten reusable plant suffixes. For example,
`player_2.plant_1.resource.oil` has index `272 + 24 + 5 = 301`.

Expansion count: `6 players * (14 player fields + 4 plants * 10 fields) = 324`.

## Auction context: state indexes 488–495

| Index | Feature name | Semantic type | Source and encoding | Actual meaning |
| ---: | --- | --- | --- | --- |
| 488 | `auction.present` | Boolean | `auction_state is not None` | An auction phase state object exists, even if no individual plant auction is active. |
| 489 | `auction.active` | Boolean | Active plant price exists | A specific plant is currently being bid on. |
| 490 | `auction.plant_price` | Integer ordinal | Active plant's printed price `/ 50`; otherwise zero | Identity/base price of the plant in the active auction. |
| 491 | `auction.current_bid` | Integer ordinal | Current highest bid `/ 100`; otherwise zero | Highest accepted bid so far in the active auction. |
| 492 | `auction.actor_is_highest` | Boolean | `highest_bidder_id == actor_id` | Whether the current actor is recorded as highest bidder. It is commonly false when the request is addressed to the next bidder. |
| 493 | `auction.actor_is_chooser` | Boolean | `current_chooser_id == actor_id` | Whether the actor selected, or is selecting, the plant for the current auction turn. |
| 494 | `auction.active_bidder_count` | Integer count | `len(active_bidders) / 6` | Players still eligible in the currently active plant auction. |
| 495 | `auction.passed_player_count` | Integer count | `len(players_passed_phase) / 6` | Players who have passed out of the auction phase for the round, not merely the current plant auction. |

Only counts and actor-relative flags are encoded. The identities and individual
attributes of active bidders, the highest bidder, players who already bought plants,
and phase-passers are not represented directly.

## Map and build summary: state indexes 496–512

All map summaries are restricted to cities in the selected regions.

| Index | Feature name | Semantic type | Source and encoding | Actual meaning |
| ---: | --- | --- | --- | --- |
| 496 | `map.allowed_city_count` | Integer count | Allowed city count `/ 42` | Number of cities enabled by the selected regions. |
| 497 | `map.city_occupancy_0` | Integer count | Count `/ 42` | Enabled cities occupied by exactly zero players. |
| 498 | `map.city_occupancy_1` | Integer count | Count `/ 42` | Enabled cities occupied by exactly one player. |
| 499 | `map.city_occupancy_2` | Integer count | Count `/ 42` | Enabled cities occupied by exactly two players. |
| 500 | `map.city_occupancy_3` | Integer count | Count `/ 42` | Enabled cities occupied by exactly three players. |
| 501 | `map.connection_min` | Integer ordinal | Minimum allowed edge cost `/ 30` | Lowest printed connection cost among all edges whose endpoints are both enabled. This is a global map statistic, not the actor's cheapest build connection. |
| 502 | `map.connection_mean` | Continuous summary | Mean allowed edge cost `/ 30` | Arithmetic mean of all connection costs in the enabled subgraph. |
| 503 | `map.connection_max` | Integer ordinal | Maximum allowed edge cost `/ 30` | Highest printed connection cost in the enabled subgraph. |
| 504 | `map.actor_build_target_count` | Integer count | Number of legal actor build targets `/ 42` | Cities the actor can currently reach, occupy, afford, and supply with a remaining house according to `legal_build_targets`. |
| `505+k`, `k=0..7` | `map.actor_build_cost_{k}` | Integer ordinal | `k`-th smallest legal total build cost `/ 100`; zero when absent | First eight sorted costs returned by the legal-build helper. Each total combines connection and house-placement cost, but city identity is discarded. |

The four occupancy counts describe the board globally. They do not preserve which
player occupies which city, actor/opponent network topology, selected-region identity,
or the city associated with a sorted build cost.

## Candidate-action vector overview

| Index range | Feature group | Columns |
| --- | --- | ---: |
| `0..10` | Action type | 11 |
| `11..15` | Core amount/cost fields | 5 |
| `16..19` | Purchased resource type | 4 |
| `20..24` | Build fields | 5 |
| `25..31` | Generation/resource-mix/discard fields | 7 |
| `32..41` | Referenced plant | 10 |
| **Total** |  | **42** |

## Action type: action indexes 0–10

These columns form one enum one-hot group for a normal candidate. Their meanings are:

| Index | Feature name | Candidate meaning |
| ---: | --- | --- |
| 0 | `action.type.auction_start` | Select a visible plant and start its auction at the candidate opening bid. |
| 1 | `action.type.auction_bid` | Raise the active auction to the candidate bid. |
| 2 | `action.type.auction_pass` | During `auction_bid`, leave the active plant auction; during `auction_start`, pass out of the entire auction phase for this round. |
| 3 | `action.type.buy_resource` | Buy the candidate quantity of one resource. |
| 4 | `action.type.finish_buying` | End the actor's resource-purchase turn without another purchase. |
| 5 | `action.type.commit_build` | Build the listed city or cities. V1-generated policy candidates contain one city. |
| 6 | `action.type.finish_building` | End the actor's build turn without another city. |
| 7 | `action.type.run_plants` | Run the candidate plant/resource-mix plan during bureaucracy. |
| 8 | `action.type.skip_bureaucracy` | Run no plants and power zero cities. |
| 9 | `action.type.discard_power_plant` | Discard the referenced owned plant. |
| 10 | `action.type.discard_hybrid_resources` | Discard the candidate coal/oil quantities to restore valid storage. |

All eleven features have semantic type **enum one-hot** and use direct `0.0/1.0`
encoding.

## Core candidate fields: action indexes 11–19

| Index | Feature name | Semantic type | Source and encoding | Actual meaning |
| ---: | --- | --- | --- | --- |
| 11 | `action.bid` | Integer ordinal | Candidate `bid / 100`; otherwise zero | Full opening/new bid amount, not the increment above `auction.current_bid`. |
| 12 | `action.plant_price` | Integer ordinal | Referenced plant price `/ 50`; otherwise zero | Printed number of the plant selected, bid on, passed on, or discarded. Auction pass still references the active plant. |
| 13 | `action.amount` | Integer count | Resource purchase amount `/ 8`; otherwise zero | Number of units bought by a `buy_resource` candidate. The value may exceed 1.0 when more than eight units are legal. |
| 14 | `action.direct_cost` | Integer ordinal | Resource `cost`, else build `total_cost`, else full `bid`, divided by 200 | Immediate/hypothetical Elektro commitment associated with the candidate. Zero for finish, pass, generation, and discard actions. |
| 15 | `action.cash_after` | Integer ordinal | `max(0, actor_cash - direct_cost) / 200` | Actor cash after reserving/paying the candidate's direct cost. For auction bids this is the cash that would remain if the bid wins; for zero-cost actions it equals current cash. |
| 16 | `action.resource.coal` | Enum one-hot | Purchased resource is coal | `buy_resource` targets coal. |
| 17 | `action.resource.oil` | Enum one-hot | Purchased resource is oil | `buy_resource` targets oil. |
| 18 | `action.resource.garbage` | Enum one-hot | Purchased resource is garbage | `buy_resource` targets garbage. |
| 19 | `action.resource.uranium` | Enum one-hot | Purchased resource is uranium | `buy_resource` targets uranium. |

The resource one-hot group is all zero for non-purchase actions.

## Build candidate fields: action indexes 20–24

| Index | Feature name | Semantic type | Source and encoding | Actual meaning |
| ---: | --- | --- | --- | --- |
| 20 | `action.city_count` | Integer count | `len(city_ids) / 10` | Number of cities committed by the intent. Runtime V1 candidates normally use one; teacher-only intents could contain more. |
| 21 | `action.connection_cost` | Integer ordinal | Candidate connection cost `/ 100` | Network-connection component of a single-city legal build. Zero for non-build actions. |
| 22 | `action.build_cost` | Integer ordinal | Candidate house-placement cost `/ 100` | Occupancy/step-dependent house cost for the candidate city. |
| 23 | `action.total_build_cost` | Integer ordinal | Candidate total build cost `/ 200` | Connection cost plus house-placement cost. |
| 24 | `action.city_occupancy` | Integer count | Current city occupancy `/ 3` | Number of players already occupying the referenced city before this build. The city ID itself is not encoded. |

For a teacher-only multi-city intent, metadata may not contain the same per-city cost
breakdown as a generated single-city candidate; the action type and `city_count` still
identify the intent, while missing numeric metadata remains zero.

## Generation and discard fields: action indexes 25–31

| Index | Feature name | Semantic type | Source and encoding | Actual meaning |
| ---: | --- | --- | --- | --- |
| 25 | `action.powered_cities` | Integer count | Candidate projected powered cities `/ 22` | Cities powered by the complete bureaucracy run plan, capped by the player's connected cities through the rules helper. Zero outside bureaucracy. |
| 26 | `action.income` | Integer ordinal | Income table result `/ 150` | Elektro income resulting from `action.powered_cities`; for a skip candidate this is the rules-table income for powering zero cities. |
| 27 | `action.resource_mix.coal` | Integer count | Candidate coal amount `/ 8` | Coal consumed by all selected run plans; for hybrid-resource discard, coal discarded from the intent payload. |
| 28 | `action.resource_mix.oil` | Integer count | Candidate oil amount `/ 8` | Oil consumed by all selected run plans; for hybrid-resource discard, oil discarded from the intent payload. |
| 29 | `action.resource_mix.garbage` | Integer count | Candidate garbage amount `/ 8` | Garbage consumed by all selected run plans. |
| 30 | `action.resource_mix.uranium` | Integer count | Candidate uranium amount `/ 8` | Uranium consumed by all selected run plans. |
| 31 | `action.discarded_units` | Integer count | `(intent.coal + intent.oil) / 8` | Total units discarded by `discard_hybrid_resources`; zero for other current action types. |

Despite their names, indexes 27–30 are numeric amounts, not the plant fuel-type
multi-hot fields described earlier.

## Referenced plant: action indexes 32–41

The encoder finds a public plant whose printed price equals `action.plant_price`,
searching current market, future market, and player-owned plants. It then emits the
reusable ten-column plant block with prefix `action.plant`:

| Index | Feature name | Reusable block field |
| ---: | --- | --- |
| 32 | `action.plant.present` | `.present` |
| 33 | `action.plant.price` | `.price` |
| 34 | `action.plant.resource_cost` | `.resource_cost` |
| 35 | `action.plant.output_cities` | `.output_cities` |
| 36 | `action.plant.resource.coal` | `.resource.coal` |
| 37 | `action.plant.resource.oil` | `.resource.oil` |
| 38 | `action.plant.resource.garbage` | `.resource.garbage` |
| 39 | `action.plant.resource.uranium` | `.resource.uranium` |
| 40 | `action.plant.ecological` | `.ecological` |
| 41 | `action.plant.hybrid` | `.hybrid` |

The semantic types, divisors, and meanings are exactly those in the reusable plant
block. If no plant is referenced or found, all ten values are zero. The scalar
`action.plant_price` and `action.plant.price` intentionally duplicate the referenced
price when the lookup succeeds.

## Information present in the public payload but absent from schema v1

The following public information is collected before encoding but is not represented
as model features. This distinction is important when interpreting what the model can
learn:

- identities of selected map regions; only their count is encoded
- city IDs and region membership
- actor and opponent network city IDs and network topology
- city identity corresponding to a legal build cost
- identities of active bidders, highest bidder, players who bought plants, and players
  who passed the auction phase; only selected counts/actor flags are encoded
- full resource-market price bands after the first eight purchasable units
- controller/profile identity used to generate a behavior sample
- hidden plant identities and deck order; these are intentionally excluded because
  they are not public information

## 中文特征说明

本节是上述数据字典的完整中文解释。索引仍然从 0 开始；`r`、`i`、`j`、`k`
表示按表中范围展开的物理列，并不是额外的抽象输入。英文特征名与代码、Parquet
manifest 和 checkpoint 中保存的名称完全一致，不做翻译。

### 数据类型、归一化和零值

模型最终接收的所有列都是 `float32`，但它们在业务语义上分为：

| 中文类型 | 编码方式 | 如何理解 |
| --- | --- | --- |
| 独立布尔值 | `false=0.0`，`true=1.0` | 每一列都是可以独立成立的条件，例如“存在”“生态电厂” |
| 枚举 one-hot | 同一组中匹配当前取值的列为 1，其余为 0 | 多列共同表达一个多值变量，例如当前阶段 |
| 集合 multi-hot | 每种成员资格单独取 0 或 1 | 同时允许多个值成立，例如混合电厂同时接受煤和石油 |
| 整数计数 | 原始非负整数除以固定除数 | 玩家数、城市数、资源数、产能等 |
| 整数序数 | 有大小顺序的整数除以固定除数 | 轮次、现金、价格、竞价和顺位等 |
| 连续统计值 | 派生统计量除以固定除数 | 当前只有地图连接费用均值等少数特征 |

固定除数只调整数值尺度，不会截断到 `[0,1]`。训练时还会用训练集均值和标准差
进行第二次标准化；checkpoint 保存相同的均值和标准差供推理使用。

零值必须结合上下文理解：

- 对布尔列，0 表示条件不成立。
- 对空玩家/空电厂 slot，整个 slot 都为 0；必须先查看对应 `.present`。
- 对动作数值列，0 通常表示“不适用于该动作”，必须结合 `action.type.*` 判断。
- 对价格和建造费用 slot，0 表示没有这一项，而不是实际价格为零。
- 对不适用的枚举组可以全部为 0，例如建城动作的 `action.resource.*`。

### State：全局与当前请求，索引 0–23

| 索引 | 特征 | 类型 | 中文解释 |
| ---: | --- | --- | --- |
| 0 | `global.round` | 整数序数，`/20` | 当前游戏轮次。20 只是缩放因子，不代表游戏最多 20 轮。 |
| 1 | `global.step_1` | 枚举 one-hot | 当前处于游戏 Step 1。 |
| 2 | `global.step_2` | 枚举 one-hot | 当前处于游戏 Step 2。 |
| 3 | `global.step_3` | 枚举 one-hot | 当前处于游戏 Step 3。 |
| 4 | `global.player_count` | 整数计数，`/6` | 本局玩家总数。 |
| 5 | `global.end_game_cities` | 整数计数，`/22` | 当前玩家数规则下，触发游戏结束所需的联网城市数。 |
| 6 | `global.hidden_plant_count` | 整数计数，`/42` | 当前不可见的电厂牌数量，加上尚未出现的 Step 3 标记；只包含数量，不包含隐藏牌身份和顺序。 |
| 7 | `global.step_3_pending` | 独立布尔值 | Step 3 标记仍位于抽牌堆之后、尚未出现或处理。 |
| 8 | `global.auction_step_3_pending` | 独立布尔值 | Step 3 占位牌已在拍卖流程中出现，但需要等当前拍卖处理完成后再切换 Step 3。 |
| 9 | `global.phase.auction` | 枚举 one-hot | 当前大阶段是电厂拍卖。 |
| 10 | `global.phase.buy_resources` | 枚举 one-hot | 当前大阶段是购买资源。 |
| 11 | `global.phase.build_houses` | 枚举 one-hot | 当前大阶段是建设城市。 |
| 12 | `global.phase.bureaucracy` | 枚举 one-hot | 当前大阶段是发电、收入和资源补充。 |
| 13 | `global.decision.auction_start` | 枚举 one-hot | 当前请求要求选择电厂并开拍，或在允许时放弃本轮拍卖阶段。 |
| 14 | `global.decision.auction_bid` | 枚举 one-hot | 当前请求要求在正在进行的拍卖中加价或退出。 |
| 15 | `global.decision.buy_resources` | 枚举 one-hot | 当前请求允许购买一种资源或结束购买。 |
| 16 | `global.decision.build_houses` | 枚举 one-hot | 当前请求允许建设一座合法城市或结束建设。 |
| 17 | `global.decision.bureaucracy` | 枚举 one-hot | 当前请求要求选择电厂运行和燃料组合。 |
| 18 | `global.decision.discard_power_plant` | 枚举 one-hot | 因电厂数量超过上限，当前必须弃掉一座电厂。 |
| 19 | `global.decision.discard_hybrid_resources` | 枚举 one-hot | 因混合储存空间不再合法，当前必须弃掉一定煤/石油。 |
| 20 | `global.map.germany` | 枚举 one-hot | 当前使用德国地图。 |
| 21 | `global.map.usa` | 枚举 one-hot | 当前使用美国地图。 |
| 22 | `global.map.test` | 枚举 one-hot | 当前使用开发测试地图。 |
| 23 | `global.selected_region_count` | 整数计数，`/6` | 启用的地图区域数量；schema v1 没有编码具体启用了哪些区域。 |

`phase` 表示规则层的大阶段，`decision_type` 表示当前需要处理的具体请求。两者同时存在，
是因为强制弃电厂/弃资源等 pending decision 可以发生在某个大阶段内部。

### State：资源市场，索引 24–63

四种资源各占连续 10 列：

| 资源 `r` | 基址 `B(r)` | 实际范围 |
| --- | ---: | --- |
| `coal`（煤） | 24 | `24..33` |
| `oil`（石油） | 34 | `34..43` |
| `garbage`（垃圾） | 44 | `44..53` |
| `uranium`（铀） | 54 | `54..63` |

| 相对索引 | 特征模式 | 类型 | 中文解释 |
| ---: | --- | --- | --- |
| `B(r)+0` | `market.{r}.available` | 整数计数，`/24` | 当前价格轨道上可以直接购买的 `r` 单位数。 |
| `B(r)+1` | `market.{r}.supply` | 整数计数，`/24` | 中央补充供应池里的 `r` 单位数；不包含市场轨道上的资源，也不包含玩家已经储存的资源。 |
| `B(r)+2+k`，`k=0..7` | `market.{r}.unit_price_{k}` | 整数序数，`/20` | 当前第 `k+1` 个最便宜单位的价格。同一价格出现多次表示该价位有多个单位；不足八个时后续 slot 补 0。 |

因此这里完整覆盖 `4 * (2 + 8) = 40` 个 state 特征。schema v1 只保存每种资源
最便宜的前八个单位价格。

### State/Action 共用的电厂 10 列模板

任意电厂 block 的起始索引记为 `P`，其十个字段含义如下：

| 偏移 | 后缀 | 类型 | 中文解释 |
| ---: | --- | --- | --- |
| `P+0` | `.present` | 独立布尔值 | 此 slot 是否实际存在一张电厂牌；为 0 时其余字段都是 padding。 |
| `P+1` | `.price` | 整数序数，`/50` | 电厂牌面编号，同时是最低拍卖价，并且在当前牌组中可作为唯一电厂身份。 |
| `P+2` | `.resource_cost` | 整数计数，`/3` | 电厂运行一次需要消耗的燃料单位数；生态电厂为 0。 |
| `P+3` | `.output_cities` | 整数计数，`/10` | 电厂运行一次最多可供电的城市数。 |
| `P+4` | `.resource.coal` | 集合 multi-hot | 该电厂能否使用煤。 |
| `P+5` | `.resource.oil` | 集合 multi-hot | 该电厂能否使用石油；混合电厂会同时设置煤和石油。 |
| `P+6` | `.resource.garbage` | 集合 multi-hot | 该电厂能否使用垃圾。 |
| `P+7` | `.resource.uranium` | 集合 multi-hot | 该电厂能否使用铀。 |
| `P+8` | `.ecological` | 独立布尔值 | 是否为不消耗市场燃料的生态电厂。 |
| `P+9` | `.hybrid` | 独立布尔值 | 是否为可以混用煤和石油的混合电厂。 |

当前市场按价格升序放入六个 slot：

| slot | 基址 | 前缀 | 范围 |
| ---: | ---: | --- | --- |
| 0 | 64 | `current_market_0` | `64..73` |
| 1 | 74 | `current_market_1` | `74..83` |
| 2 | 84 | `current_market_2` | `84..93` |
| 3 | 94 | `current_market_3` | `94..103` |
| 4 | 104 | `current_market_4` | `104..113` |
| 5 | 114 | `current_market_5` | `114..123` |

未来市场按价格升序放入四个 slot：

| slot | 基址 | 前缀 | 范围 |
| ---: | ---: | --- | --- |
| 0 | 124 | `future_market_0` | `124..133` |
| 1 | 134 | `future_market_1` | `134..143` |
| 2 | 144 | `future_market_2` | `144..153` |
| 3 | 154 | `future_market_3` | `154..163` |

每个 slot 都严格按上面的十个后缀展开。Step 1/2 通常只有四张当前市场牌；Step 3
没有未来市场并且当前市场最多六张，因此固定 slot 中可能出现全 0 padding。

### State：玩家，索引 164–487

每个玩家占 54 列，`B(i)=164+54*i`：

| `i` | 基址 | 前缀 | 实际角色 | 范围 |
| ---: | ---: | --- | --- | --- |
| 0 | 164 | `player_0` | 当前做决策的玩家 | `164..217` |
| 1 | 218 | `player_1` | 按公开顺位排序的第一个对手 | `218..271` |
| 2 | 272 | `player_2` | 第二个对手 | `272..325` |
| 3 | 326 | `player_3` | 第三个对手或 padding | `326..379` |
| 4 | 380 | `player_4` | 第四个对手或 padding | `380..433` |
| 5 | 434 | `player_5` | 第五个对手或 padding | `434..487` |

对手先按 `turn_order_position`，再按 `player_id` 排序；因此 `player_1` 并不是某个
固定座位，而是当前状态下顺位最靠前的对手。

| 相对索引 | 特征模式 | 类型 | 中文解释 |
| ---: | --- | --- | --- |
| `B(i)+0` | `player_{i}.present` | 独立布尔值 | 该玩家 slot 是否真实存在。 |
| `B(i)+1` | `player_{i}.is_actor` | 独立布尔值 | 是否为当前决策者；按构造方式只会在 `player_0` 为 1。 |
| `B(i)+2` | `player_{i}.turn_order` | 整数序数，`/6` | 当前公开行动顺位，原值从 1 开始；数值越小，在正向阶段越早行动。 |
| `B(i)+3` | `player_{i}.cash` | 整数序数，`/200` | 当前持有的 Elektro；现金可以超过 200，因此编码后也可以超过 1。 |
| `B(i)+4` | `player_{i}.houses_in_supply` | 整数计数，`/22` | 个人供应中尚未建到地图上的房屋数。 |
| `B(i)+5` | `player_{i}.connected_cities` | 整数计数，`/22` | 玩家网络中的城市数量；不包含城市身份和网络形状。 |
| `B(i)+6` | `player_{i}.last_powered` | 整数计数，`/22` | 上一次已结算 bureaucracy 中实际供电的城市数；首轮结算前为 0。 |
| `B(i)+7` | `player_{i}.last_income` | 整数序数，`/150` | 上一次 bureaucracy 获得的收入；首轮结算前为 0。 |
| `B(i)+8` | `player_{i}.resource.coal` | 整数计数，`/8` | 玩家所有兼容电厂中储存的煤总量。 |
| `B(i)+9` | `player_{i}.resource.oil` | 整数计数，`/8` | 玩家储存的石油总量。 |
| `B(i)+10` | `player_{i}.resource.garbage` | 整数计数，`/8` | 玩家储存的垃圾总量。 |
| `B(i)+11` | `player_{i}.resource.uranium` | 整数计数，`/8` | 玩家储存的铀总量。 |
| `B(i)+12` | `player_{i}.total_output` | 整数计数，`/22` | 所有持有电厂的牌面供电量之和；不检查燃料是否足够，也不受当前联网城市数限制。 |
| `B(i)+13` | `player_{i}.largest_plant` | 整数序数，`/50` | 持有电厂中最大的牌面编号；正常顺位规则会使用这一值，无电厂时为 0。 |

玩家的四个电厂 slot 从 `B(i)+14` 开始，并按电厂价格升序排列：

```text
P(i,j) = B(i) + 14 + 10*j,  j=0..3
```

| `j` | 相对范围 | 特征前缀 | 中文解释 |
| ---: | --- | --- | --- |
| 0 | `B(i)+14 .. B(i)+23` | `player_{i}.plant_0` | 玩家价格最低的电厂。 |
| 1 | `B(i)+24 .. B(i)+33` | `player_{i}.plant_1` | 玩家价格第二低的电厂。 |
| 2 | `B(i)+34 .. B(i)+43` | `player_{i}.plant_2` | 玩家价格第三低的电厂。 |
| 3 | `B(i)+44 .. B(i)+53` | `player_{i}.plant_3` | 第四座电厂，通常只在购买后等待强制弃牌时短暂存在。 |

每个玩家电厂 slot 再按共用的十列电厂模板展开，所以玩家部分完整覆盖
`6 * (14 + 4*10) = 324` 列。

### State：拍卖状态，索引 488–495

| 索引 | 特征 | 类型 | 中文解释 |
| ---: | --- | --- | --- |
| 488 | `auction.present` | 独立布尔值 | 是否存在本轮拍卖阶段状态；即使当前没有具体电厂正在竞拍，也可能为 1。 |
| 489 | `auction.active` | 独立布尔值 | 当前是否有一座具体电厂正在竞拍。 |
| 490 | `auction.plant_price` | 整数序数，`/50` | 正在竞拍电厂的牌面编号/底价；没有 active auction 时为 0。 |
| 491 | `auction.current_bid` | 整数序数，`/100` | 当前已经接受的最高出价。 |
| 492 | `auction.actor_is_highest` | 独立布尔值 | 当前决策者是否是最高出价者；轮到下一名竞价者时通常为 0。 |
| 493 | `auction.actor_is_chooser` | 独立布尔值 | 当前决策者是否是本次选择开拍电厂的 chooser。 |
| 494 | `auction.active_bidder_count` | 整数计数，`/6` | 仍留在当前这座电厂竞拍中的玩家数。 |
| 495 | `auction.passed_player_count` | 整数计数，`/6` | 已经放弃本轮整个拍卖阶段的玩家数，不是只退出当前电厂竞拍的人数。 |

schema v1 没有直接编码 active bidders、最高出价者、已买到电厂者和整阶段 pass 者的
具体身份，只编码数量和少量 actor-relative 标志。

### State：地图与建造摘要，索引 496–512

所有统计只针对当前选中区域内的城市和连接。

| 索引 | 特征 | 类型 | 中文解释 |
| ---: | --- | --- | --- |
| 496 | `map.allowed_city_count` | 整数计数，`/42` | 当前选中区域内启用的城市总数。 |
| 497 | `map.city_occupancy_0` | 整数计数，`/42` | 当前没有任何玩家建房的启用城市数。 |
| 498 | `map.city_occupancy_1` | 整数计数，`/42` | 当前恰有一名玩家建房的启用城市数。 |
| 499 | `map.city_occupancy_2` | 整数计数，`/42` | 当前恰有两名玩家建房的启用城市数。 |
| 500 | `map.city_occupancy_3` | 整数计数，`/42` | 当前恰有三名玩家建房的启用城市数。 |
| 501 | `map.connection_min` | 整数序数，`/30` | 启用子图中所有边的最小印刷连接费；不是 actor 下一次建造的最低连接费。 |
| 502 | `map.connection_mean` | 连续统计值，`/30` | 启用子图全部边连接费的算术平均值。 |
| 503 | `map.connection_max` | 整数序数，`/30` | 启用子图中所有边的最大印刷连接费。 |
| 504 | `map.actor_build_target_count` | 整数计数，`/42` | actor 当前满足可到达、城市未满、有剩余房屋并且付得起费用的合法目标城市数。 |
| `505+k`，`k=0..7` | `map.actor_build_cost_{k}` | 整数序数，`/100` | actor 当前第 `k+1` 个最便宜合法目标的总建造费，包含连接费和房屋费；不足八项时补 0。 |

这里丢弃了每个费用对应的城市身份，也没有保留玩家网络拓扑、城市所属区域以及每座城市
由哪些玩家占据。

### Action：动作类型，索引 0–10

这 11 列共同构成动作类型 one-hot：

| 索引 | 特征 | 中文解释 |
| ---: | --- | --- |
| 0 | `action.type.auction_start` | 选择一座可见电厂，并以候选开价开始拍卖。 |
| 1 | `action.type.auction_bid` | 把当前拍卖价格提高到候选 bid。 |
| 2 | `action.type.auction_pass` | 在 `auction_bid` 请求中退出当前电厂竞拍；在 `auction_start` 请求中放弃本轮整个拍卖阶段。 |
| 3 | `action.type.buy_resource` | 购买候选种类和数量的资源。 |
| 4 | `action.type.finish_buying` | 不再购买资源，结束 actor 本阶段行动。 |
| 5 | `action.type.commit_build` | 建造 intent 中列出的城市；V1 正常候选只包含一个城市。 |
| 6 | `action.type.finish_building` | 不再建城，结束 actor 本阶段行动。 |
| 7 | `action.type.run_plants` | 按候选电厂组合和燃料分配进行发电。 |
| 8 | `action.type.skip_bureaucracy` | 不运行任何电厂，供电城市数为 0。 |
| 9 | `action.type.discard_power_plant` | 弃掉候选引用的一座已拥有电厂。 |
| 10 | `action.type.discard_hybrid_resources` | 弃掉候选数量的煤/石油，使混合储存重新合法。 |

### Action：核心金额、资源类型，索引 11–19

| 索引 | 特征 | 类型 | 中文解释 |
| ---: | --- | --- | --- |
| 11 | `action.bid` | 整数序数，`/100` | 候选的完整开价/新出价，不是相对 `auction.current_bid` 的加价幅度。 |
| 12 | `action.plant_price` | 整数序数，`/50` | 被开拍、竞价、pass 或弃掉的电厂牌面编号；竞价 pass 仍会引用 active plant。 |
| 13 | `action.amount` | 整数计数，`/8` | `buy_resource` 候选购买的单位数；合法购买量超过 8 时编码可大于 1。 |
| 14 | `action.direct_cost` | 整数序数，`/200` | 买资源时为总资源费，建城时为总建造费，拍卖时为完整 bid；finish/pass/发电/弃牌通常为 0。 |
| 15 | `action.cash_after` | 整数序数，`/200` | `max(0, 当前现金-direct_cost)`；拍卖候选表示中标并付款后剩余的预算，零成本动作则等于当前现金。 |
| 16 | `action.resource.coal` | 枚举 one-hot | `buy_resource` 候选购买煤。 |
| 17 | `action.resource.oil` | 枚举 one-hot | `buy_resource` 候选购买石油。 |
| 18 | `action.resource.garbage` | 枚举 one-hot | `buy_resource` 候选购买垃圾。 |
| 19 | `action.resource.uranium` | 枚举 one-hot | `buy_resource` 候选购买铀。 |

非购买动作的 `action.resource.*` 四列全部为 0。

### Action：建造，索引 20–24

| 索引 | 特征 | 类型 | 中文解释 |
| ---: | --- | --- | --- |
| 20 | `action.city_count` | 整数计数，`/10` | intent 一次提交的城市数；V1 运行时候选通常为 1，finish 为 0。 |
| 21 | `action.connection_cost` | 整数序数，`/100` | 单城市候选从 actor 当前网络连接到目标城市的费用。 |
| 22 | `action.build_cost` | 整数序数，`/100` | 由当前 Step 和目标城市已有房屋数决定的房屋放置费用。 |
| 23 | `action.total_build_cost` | 整数序数，`/200` | 连接费与房屋放置费之和。 |
| 24 | `action.city_occupancy` | 整数计数，`/3` | 建造前已经在目标城市中建房的玩家数；目标城市 ID 本身没有编码。 |

如果 behavior teacher 产生候选生成器之外的多城市 intent，`city_count` 仍有值，但
teacher-only metadata 可能没有同等详细的逐城市费用，因此费用列可能为 0。

### Action：发电、燃料组合与弃资源，索引 25–31

| 索引 | 特征 | 类型 | 中文解释 |
| ---: | --- | --- | --- |
| 25 | `action.powered_cities` | 整数计数，`/22` | 完整发电候选根据规则实际可供电的城市数，已经受联网城市数限制。 |
| 26 | `action.income` | 整数序数，`/150` | 对应供电城市数在收入表中的收入；skip 候选仍会得到“供电 0 城市”的规则收入。 |
| 27 | `action.resource_mix.coal` | 整数计数，`/8` | 发电候选消耗的煤总量；弃混合资源动作中表示弃掉的煤。 |
| 28 | `action.resource_mix.oil` | 整数计数，`/8` | 发电候选消耗的石油总量；弃混合资源动作中表示弃掉的石油。 |
| 29 | `action.resource_mix.garbage` | 整数计数，`/8` | 发电候选消耗的垃圾总量。 |
| 30 | `action.resource_mix.uranium` | 整数计数，`/8` | 发电候选消耗的铀总量。 |
| 31 | `action.discarded_units` | 整数计数，`/8` | `discard_hybrid_resources` 候选弃掉的煤和石油单位数之和，其余动作通常为 0。 |

`resource_mix.*` 是资源数量，不是电厂模板里的燃料类型 multi-hot。

### Action：引用电厂，索引 32–41

编码器用 `action.plant_price` 在当前市场、未来市场和所有玩家已拥有电厂中查找公开电厂，
并按共用的十列电厂模板输出：

| 索引 | 特征 | 类型与中文含义 |
| ---: | --- | --- |
| 32 | `action.plant.present` | 独立布尔值；是否成功找到被引用电厂。 |
| 33 | `action.plant.price` | 整数序数，`/50`；被引用电厂的牌面编号。 |
| 34 | `action.plant.resource_cost` | 整数计数，`/3`；运行一次需要的燃料数。 |
| 35 | `action.plant.output_cities` | 整数计数，`/10`；运行一次的供电能力。 |
| 36 | `action.plant.resource.coal` | 集合 multi-hot；是否可用煤。 |
| 37 | `action.plant.resource.oil` | 集合 multi-hot；是否可用石油。 |
| 38 | `action.plant.resource.garbage` | 集合 multi-hot；是否可用垃圾。 |
| 39 | `action.plant.resource.uranium` | 集合 multi-hot；是否可用铀。 |
| 40 | `action.plant.ecological` | 独立布尔值；是否为生态电厂。 |
| 41 | `action.plant.hybrid` | 独立布尔值；是否为煤/石油混合电厂。 |

查找不到或动作不引用电厂时，32–41 全部为 0。查找成功时，`action.plant_price`
和 `action.plant.price` 会重复表达同一个牌面编号，这是 schema v1 的有意冗余。

### 中文说明：明确没有编码的信息

下面的信息虽然部分存在于 public observation payload，但没有进入当前 555 维输入：

- 选中区域的具体身份，只保留区域数量；
- 城市 ID、城市所属区域和地图拓扑；
- actor/对手各自的网络城市 ID 与网络形状；
- 排序后的建造费用分别对应哪个城市；
- active bidders、最高出价者、已经买到电厂者和整阶段 pass 者的具体身份；
- 每种资源第九个及之后可购买单位的逐单位价格；
- 生成 behavior sample 的 controller/profile 身份；
- 隐藏电厂牌身份和牌堆顺序——这部分是为了遵守公开信息边界而有意排除。

## Coverage check

The patterns above cover every physical vector column:

```text
state:
  24 global
  + 4 * 10 resource-market
  + 6 * 10 current-market plants
  + 4 * 10 future-market plants
  + 6 * (14 player fields + 4 * 10 owned-plant fields)
  + 8 auction
  + 17 map/build
  = 513

action:
  11 action type
  + 5 core amount/cost
  + 4 purchased-resource type
  + 5 build
  + 7 generation/discard
  + 10 referenced-plant
  = 42
```

The exact ordered feature-name arrays are also stored in every dataset manifest and
checkpoint. Runtime rejects a checkpoint if those arrays do not match the encoder.
