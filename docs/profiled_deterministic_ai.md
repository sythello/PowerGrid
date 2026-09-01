# Profiled deterministic data-generation AIs

## Purpose

These controllers provide diverse, reproducible behavior policies for rank-value
dataset generation without the forward search cost of `ai_heuristics`:

- `ai_deterministic_efficiency`
- `ai_deterministic_expansion`
- `ai_deterministic_reserve`

They share the session/rules legality boundary and deterministic tie-breaking of
`ai_deterministic`. They do not clone states or search future action sequences.
Bureaucracy still enumerates the small set of legal combinations across at most four
owned plants, matching the original deterministic baseline; this is legal plan
selection, not multi-turn lookahead.

The dataset generator now rotates these three profiles with `ai_deterministic` by
default. `ai_heuristics` remains available when higher-cost teacher data is wanted.

## Strategy profiles

| Parameter | Efficiency | Expansion | Reserve |
| --- | ---: | ---: | ---: |
| Output weight | 6.5 | 8.0 | 4.5 |
| Estimated run-cost penalty | 1.75 | 0.8 | 2.8 |
| Purchase-price penalty | 0.2 | 0.16 | 1.8 |
| Ecological bonus | 7.0 | 3.0 | 10.0 |
| Hybrid bonus | 2.0 | 1.0 | 2.0 |
| Bid markup | 3 | 2 | 2 |
| Target fuel runs | 1 | 1 | 1 |
| Cheap-fuel buffer | 1 | 0 | 0 |
| Build buffer above plant output | 3 | 3 | 2 |
| Normal build cash reserve | 2 | 0 | 10 |
| Build ordering | total cost | connection cost | total cost/cash reserve |
| Resource ordering | cheapest unit | largest deficit | one unit at a time |

### Efficiency

- Scores auction plants by output, expected run cost, purchase cost, and fuel flexibility.
- Adds one stored unit when a required fuel currently costs at most two Elektro.
- Builds toward total plant output plus three cities, choosing the cheapest total cost.

### Expansion

- Places the highest weight on plant output while still penalizing expensive fuel.
- Prioritizes the largest current fuel deficit.
- Builds toward output plus three cities and prioritizes low connection cost.

### Reserve

- Strongly penalizes plant purchase/running cost and favors ecological plants.
- Buys required resources one unit per decision instead of committing to a full basket.
- Maintains ten Elektro during normal building and targets output plus two cities.

All profiles open/bid only through current legal auction actions, buy only current
legal resources, and build only current legal cities.

## Behavior-diversity check

On seed-7 fixed scenarios, the behavior signatures differ:

| Scenario | Baseline | Efficiency | Expansion | Reserve |
| --- | --- | --- | --- | --- |
| Opening auction | plant 6 @ 1 | plant 10 @ 10 | plant 10 @ 10 | plant 6 @ 1 |
| Resource | oil ×3 | oil ×3 | oil ×3 | oil ×1 |
| Build test | Cinder Grove | Cinder Grove | Amber Falls | Cinder Grove |

Although some individual decisions coincide, each profile's multi-scenario signature
differs from the baseline and from both other profiles. Full mixed-profile games also
complete with zero invalid intents.

## Strength calibration

Calibration uses Germany, three players, the two balanced lineups
`(baseline, baseline, profile)` and `(baseline, profile, profile)`, and 30 games per
lineup on held-out seeds 9001–9030.

Acceptance gates:

- profile controller winner share: 35%–65%
- profile pairwise seat-comparison score: 0.35–0.65
- average-finish difference from baseline: at most 0.4
- final sequential Elo difference: at most 110

Final results:

| Profile | Games | Profile wins | Winner share | Pair score | Profile/base average finish | Elo gap |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Efficiency | 60 | 23 | 0.383 | 0.358 | 2.189 / 1.811 | 95.91 |
| Expansion | 60 | 32 | 0.533 | 0.475 | 2.033 / 1.967 | 95.48 |
| Reserve | 60 | 37 | 0.617 | 0.583 | 1.889 / 2.111 | 83.52 |

### Efficiency build-buffer ablation

An isolated experiment compared the current `total plant output + 3` city target
against `total plant output + 0`. Every other Efficiency parameter was identical.
Both variants used the same balanced lineups and 30 seeds per lineup.

| Seed set | `+3` profile wins | `+0` profile wins | `+3` pair score | `+0` pair score | `+3` / `+0` profile average finish |
| --- | ---: | ---: | ---: | ---: | --- |
| 9001–9030 | 23/60 | 16/60 | 0.358 | 0.292 | 2.189 / 2.278 |
| 10001–10030 | 27/60 | 17/60 | 0.425 | 0.342 | 2.100 / 2.211 |
| Combined | 50/120 | 33/120 | 0.392 | 0.317 | 2.144 / 2.244 |

Across 180 balanced seat appearances per variant, seat-level first-place rate fell
from `50/180 = 27.8%` to `33/180 = 18.3%`. The cash saved by strict capacity-bound
building did not compensate for slower network growth and reduced readiness for later
plant capacity. Efficiency therefore retains a build buffer of three.

Reproduce the behavior and strength gates with:

```bash
PYTHONPATH=src python -m powergrid.tools.validate_profiled_deterministic_ai \
  --games-per-lineup 30 --seed-start 9001
```

## `ai_heuristics` speed controls

`ai_heuristics` currently has no public CLI/config speed profile. Its important search
limits are hard-coded in `strategic.py`:

| Area | Current value | Faster experimental value | Expected tradeoff |
| --- | --- | --- | --- |
| Resource recursion | depth 3, or 2 when branching is high | depth 1 | Largest resource-phase speedup; loses multi-purchase planning |
| Build plan depth | normally up to 4, sprint up to 5 | 2 | Loses long build-chain planning |
| Build candidate cities | top 10 | top 5–6 | May miss a strategic but initially expensive frontier |
| Build beam width | 6 | 2–3 | Less plan diversity |
| Auction fallback plants | top 6 | top 2–3 | Weaker pass/later-plant comparison |
| Auction reserve scan | stride 1/4/8 plus refinement | coarser capped samples | Less precise reserve prices |
| Opponent refuel threat | enumerate feasible refuel baskets | cap/disable projection | Faster evaluation but weaker opponent model |

`MAX_LOGGED_CANDIDATE_ACTIONS=24` only limits serialized diagnostics; lowering it does
not materially reduce search work. `MAX_OPPONENT_THREAT_CACHE_ENTRIES` mostly changes
memory/cache retention rather than per-new-state computation.

For bulk data generation, the profiled deterministic mix is preferable to globally
weakening `ai_heuristics`. A future `StrategicAiTuning`/`ai_heuristics_fast` controller
should expose the limits above explicitly and be strength-calibrated separately.

## Data-generation performance

Same-machine point measurements use an Apple M4 MacBook Air, Python 3.9.5 x86-64,
and NumPy 1.23.3.

| Behavior setup | Games | Rows | Wall time | Games/s | Rows/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original deterministic only | 50 | 25,251 | 24.69 s | 2.03 | 1,022.72 |
| Original + three profiled deterministic AIs | 50 | 23,771 | 24.01 s | 2.08 | 990.05 |
| Deterministic + heuristic rotation | 3 | 1,284 | 36.33 s | 0.083 | 35.34 |
| Diverse deterministic + sparse counterfactual | 10 | 4,861 + 78 | 9.65 s | 1.04 | 511.81 total |

The diverse deterministic mix retained all-deterministic game throughput and was
about 25.2 times faster per game than the small heuristic-rotation measurement.
Different policies changed game length, so rows/s is slightly lower even though
games/s is slightly higher.

```bash
/usr/bin/time -p env PYTHONPATH=src \
  python -m powergrid.tools.generate_nn_rank_value_dataset \
  --output artifacts/nn_rank_value/diverse_50g \
  --games 50 --seed-start 2001 \
  --target-shard-size-mib 512 \
  --controllers \
    ai_deterministic \
    ai_deterministic_efficiency \
    ai_deterministic_expansion \
    ai_deterministic_reserve
```
