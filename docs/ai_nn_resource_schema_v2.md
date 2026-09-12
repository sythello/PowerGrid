# NN resource feature schema v2

This document describes the resource-related runtime schema shared by
`ai_nn_rank_value_v1` and `ai_nn_rl_based_v1`. The controller names are unchanged,
but the observation and action feature schema versions are both `2`.

## Compatibility

Schema v2 has 520 state features and 52 action features. Existing schema-v1
datasets and 513/42 checkpoints are intentionally incompatible. The bundled
`ai_nn_rl_based_v1` checkpoint now uses schema v2; the legacy bundled
`ai_nn_rank_value_v1` checkpoint still requires retraining before that controller
can be used with this runtime.

## Resource market: 16 state features

Each resource contributes exactly four features, in the fixed order coal, oil,
garbage, uranium:

| Feature | Meaning | Encoding |
| --- | --- | --- |
| `empty` | No unit can currently be bought from the market | 0/1 |
| `cheapest_price` | Price of the cheapest occupied band; zero when empty | price / 16 |
| `units_at_cheapest` | Units remaining in that cheapest band; zero when empty | amount / band capacity (3; uranium 1) |
| `supply` | Units outside the market in the supply pile | amount / total game supply |

`total_available` is not encoded. Under the market refill and purchase rules, all
bands above the cheapest occupied band are full and all bands below it are empty.
The cheapest band and its remaining quantity therefore reconstruct the entire
in-market quantity. `total_supply` is not an input field; the rule totals (24, or 12
for uranium) are used only as normalization denominators.

The observation builder validates this frontier invariant instead of silently
encoding a malformed market.

## Four atomic resource decisions

For each player, the resource phase visits coal, oil, garbage, and uranium in that
order. A request exposes only the current resource. Its candidates are every legal
quantity from zero through `max_affordable_units`:

- quantity `0` means skip this resource;
- a resource with no positive legal quantity is skipped by the rules without an AI
  decision;
- after uranium, play moves to the next player in reverse turn order;
- after the final player's uranium step, play advances to house building.

The active resource is encoded by four state one-hot features. Search treats the
next resource request as the next atomic decision boundary, so the learned policy
is consulted again rather than using a deterministic continuation for the rest of
the phase.

## Derived resource-planning features

The state adds 13 actor-relative features:

- free coal-only, oil-only, shared hybrid, garbage, and uranium storage;
- maximum additional amount of each of the four resources if bought alone;
- maximum output runnable with fuel currently stored;
- nominal plant output not runnable because of missing fuel;
- maximum connected cities that can currently be powered;
- connected-city power shortfall.

Raw resource inventory remains present in the existing `player_0.resource.*`
features. The derived block makes the storage constraints and generation gap
explicit instead of requiring the MLP to reproduce hybrid-storage allocation and a
combinatorial plant-subset calculation.

Each resource-purchase candidate adds 11 post-action features: the five resulting
free-storage values, runnable output, fuel-output shortfall, change in runnable
output, powerable connected cities, connected-city shortfall, and change in
powerable cities. These values are computed by applying the candidate purchase to
an isolated immutable state.

Coal, oil, and garbage quantities are normalized by 24; uranium by 12. Storage and
aggregate output features use conservative fixed bounds of 24 and 22 respectively.

## Pending hybrid-discard context

The forced `discard_hybrid_resources` decision that can follow a plant discard is
not the normal buying phase. Schema v2 adds 14 state features for the discarded
plant price, the five post-discard storage capacities, four target resource totals,
and four automatic discard amounts. The current four-plant state plus the discarded
plant price uniquely identifies the retained plant set. Each legal coal/oil discard
candidate is simulated and receives the same post-action resource-planning block as
a purchase candidate.
