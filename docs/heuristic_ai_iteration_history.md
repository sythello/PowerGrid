# Heuristic AI Iteration History

This document tracks strategy iterations for `ai_heuristics` / `StrategicAiController`.

## v0.1-observability-baseline

- Date: 2026-07-18
- Status: implemented logging only; no scoring or decision formula changes
- Code focus:
  - Added detailed `heuristic_decision` strategy traces.
  - Added current relative-state evaluation details with raw metrics, weights, and weighted terms.
  - Added ranked candidate action scores and selected-action projected evaluation details.
  - Added optional `run_ai_game --strategy-output` extraction for AI-only strategy logs.
- Evaluation setup:
  - Map: Germany
  - Players: 3
  - Lineup: `p1=ai_heuristics`, `p2=ai_deterministic`, `p3=ai_deterministic`
  - Seeds: 1-10
- Saved logs:
  - Full game logs: `artifacts/game_logs/heuristic_v001/germany_3p_seed*.json`
  - Strategy logs: `artifacts/strategy_logs/heuristic_v001/germany_3p_seed*.json`
  - Discrepancy report: `artifacts/strategy_logs/heuristic_v001/discrepancy_report.json`
- Performance:
  - Wins: 3 / 10
  - Win rate: 30%
  - Average finish: 2.0
  - Winning seeds: 4, 8, 9

### Largest Discrepancy Found

Definition used:

```text
abs(next heuristic current_evaluation.relative_score - selected_action.projected_relative_score)
```

Largest case:

- Seed: 1
- Log entry: 827
- Round / step / phase: round 9, step 3, `buy_resources`
- Action: `finish_buying`
- Predicted projected relative score: `227.856`
- Next heuristic current relative score: `-197.4792`
- Absolute delta: `425.3352`
- Next phase: `build_houses`

Main component movement:

- Own strength delta: `-16.4565`
- Opponent adjustment delta: `-408.8788`

The projected state treated the strongest opponent as `p2` with:

- powered: 3
- income: 44
- cash: 2
- resources: coal 3, garbage 2, oil 0, uranium 0

At the next heuristic decision, the strongest opponent became `p3` with:

- powered: 13
- income: 124
- cash: 70
- resources: coal 0, garbage 0, oil 6, uranium 0

### Diagnosis

The largest errors are mostly not caused by the selected resource action itself. They come from a phase-horizon mismatch in `_evaluate_relative_state(...)`.

During `buy_resources`, opponents who have not bought fuel yet can look artificially weak because `_best_generation_summary(...)` only counts currently stored resources. If an opponent has strong plants, enough cash, and a later resource-buying turn, the current evaluation may score them as nearly unable to power cities. By the next heuristic decision, that opponent has bought fuel and their powered cities / income terms jump sharply.

The same issue appears in the opposite direction after `bureaucracy`: opponents consume fuel, so auction-phase evaluation can temporarily treat them as weak even though they can likely refuel in the upcoming resource phase.

Top-30 discrepancy aggregation supports this:

- 19 / 30 largest discrepancies are `buy_resources -> finish_buying -> build_houses`.
- Average own-strength movement in the top 30 is only about `-1.82`.
- Average opponent-adjustment movement in the top 30 is about `-111.8`.

## v0.2-opponent-refuel-threat

Status: implemented after approval on 2026-07-18.

Implemented change:

- Added an opponent-only refuel threat projection to relative evaluation.
- For each opponent, estimate one-run generation after conservatively buying the current full resource deficit:
  - uses existing plant requirements through `_resource_need_by_type(...)`,
  - uses current market prices and resource availability,
  - constrains by opponent cash,
  - requires legal storage through `can_store_resources(...)`,
  - applies the purchase with `purchase_resources(...)` on a cloned state.
- In `_evaluate_relative_state(...)`, compare against:

```text
opponent_threat_strength = max(current_opponent_strength, affordable_refuel_projected_strength)
```

- Kept the current own-player strength calculation unchanged, so the AI does not become complacent about its own fuel needs.
- Kept `STAGE_WEIGHTS` unchanged to isolate the fix to opponent threat modeling.
- Extended strategy trace schema to `schema_version=2`.
- Added `opponent_threats` to detailed evaluations with:
  - current strength,
  - refuel-projected strength,
  - applied threat strength,
  - refuel basket and cost,
  - projected generation.

Evaluation setup:

- Map: Germany
- Players: 3
- Lineup: `p1=ai_heuristics`, `p2=ai_deterministic`, `p3=ai_deterministic`
- Seeds: 1-10
- Saved logs:
  - Full game logs: `artifacts/game_logs/heuristic_v002/germany_3p_seed*.json`
  - Strategy logs: `artifacts/strategy_logs/heuristic_v002/germany_3p_seed*.json`
  - Discrepancy report: `artifacts/strategy_logs/heuristic_v002/discrepancy_report.json`

Performance:

- Wins: 1 / 10
- Win rate: 10%
- Winning seed: 9

Observed effect:

- The original largest-error cluster improved materially:
  - v0.1 top-30 discrepancies had 19 `buy_resources -> finish_buying -> build_houses` cases.
  - v0.2 top-30 discrepancies had 2 such cases.
  - Average signed delta for `buy_resources / finish_buying` improved from about `-81.19` to about `-9.33`.
- Overall playing strength regressed:
  - v0.1 won 3 / 10.
  - v0.2 won 1 / 10.
- New top-30 discrepancies are dominated by auction bidding chains:
  - 26 / 30 largest discrepancies are `auction -> auction_bid -> auction`.

Diagnosis:

- Hard `max(current_strength, refuel_projected_strength)` reduced false optimism, but appears too punitive in phases where opponent refueling is not the immediate next state.
- Because the relative score subtracts both max-opponent and average-opponent strength, threat projection can double-count the same near-future risk.
- This can make the heuristic too pessimistic while evaluating auctions and midgame expansion, reducing willingness to invest enough to win tempo races.

Suggested v0.3 direction:

- Keep threat logging.
- Replace the hard max with a phase-aware blended threat:

```text
opponent_strength_for_relative =
  current_strength + threat_weight_by_phase * max(0, refuel_projected_strength - current_strength)
```

- Candidate starting weights:
  - `buy_resources`: `0.75`
  - `build_houses`: `0.50`
  - `bureaucracy`: `0.35`
  - `auction`: `0.20`
- Keep final-round winner tuple logic unchanged.
- Re-run the same 10-seed bucket before accepting.

## v0.3-scoreboard-own-first-analysis

Status: implemented logging and analysis changes only on 2026-07-18; no strategy formula changes beyond v0.2.

Implemented instrumentation change:

- Extended strategy trace schema to `schema_version=3`.
- Added `scoreboard` details to each detailed current/projected situation score:
  - own score,
  - each opponent's current score,
  - each opponent's refuel-projected score,
  - each opponent score actually used in the relative formula,
  - opponent adjustment and final relative score.
- Updated `analyze_strategy_logs` to classify prediction gaps in this order:
  - `own_score_miss` first, when own-score movement is material and explains enough of the relative movement,
  - otherwise `opponent_score_or_relative_miss`,
  - otherwise `mixed_or_small_relative_miss`,
  - `no_material_miss` for zero-delta pairs.

Evaluation setup:

- Map: Germany
- Players: 3
- Lineup: `p1=ai_heuristics`, `p2=ai_deterministic`, `p3=ai_deterministic`
- Seeds: 1-10
- Saved logs:
  - Full game logs: `artifacts/game_logs/heuristic_v003/germany_3p_seed*.json`
  - Strategy logs: `artifacts/strategy_logs/heuristic_v003/germany_3p_seed*.json`
  - Discrepancy report: `artifacts/strategy_logs/heuristic_v003/discrepancy_report.json`

Performance:

- Wins: 1 / 10
- Win rate: 10%
- Winning seed: 9
- Decision pairs analyzed: 1363
- Strategy trace schema counts: `schema_version=3`: 1373 entries

Primary-driver counts:

- `opponent_score_or_relative_miss`: 409
- `own_score_miss`: 335
- `mixed_or_small_relative_miss`: 242
- `no_material_miss`: 377

Largest discrepancy:

- Seed: 4
- Log entry: 1031
- Round / step / phase: round 11, step 3, `buy_resources`
- Action: `finish_buying`
- Next phase: `build_houses`
- Relative signed delta: `-374.6326`
- Own-score delta: `+22.0`
- Opponent-adjustment delta: `-396.6326`
- Primary driver: `opponent_score_or_relative_miss`

Scoreboard movement:

- Projected p1 own score: `571.8176`
- Actual next p1 own score: `593.8176`
- Projected p3 score-for-relative: `626.4825`
- Actual next p3 score-for-relative: `1202.7593`
- p3 threat flag: `false` in both projected and actual scoreboards

Event sequence between the two p1 decisions:

- p1 finished buying resources.
- p3 bought 3 oil for 7 Elektro.
- p3 bought 1 uranium for 16 Elektro.
- p3 finished buying resources.
- p2 finished building.
- p1 next evaluated the board.

Own-score-first diagnosis:

- The largest own-score misses are mostly `auction_bid` chains, especially seed 10 round 11.
- Example: entry 937 projected p1 own score `809.7685`, but next p1 auction decision saw actual own score `633.5143`.
- The reason is a logging/analysis horizon mismatch: the selected `auction_bid` projected evaluation assumes a hypothetical immediate plant purchase if p1 wins the plant at the current minimum bid, while the next actual p1 decision can still be inside the same auction before p1 owns that plant.
- This is not yet strong evidence that the own scoring formula is wrong; the analyzer should separate terminal auction projections from same-auction continuation states before using auction own-score misses to tune scoring.

Opponent/relative diagnosis:

- After excluding the auction horizon artifact, the largest actionable discrepancy remains the seed 4 resource-phase case above.
- v0.2's opponent refuel threat is too all-or-nothing:
  - it computes the full resource deficit for all opponent plants,
  - rejects the threat if the whole basket cannot be stored, bought, or afforded,
  - and therefore misses partial purchases that still unlock a large powered-cities / income jump.
- In seed 4, p3 could not or did not buy a full refuel basket, but buying 3 oil + 1 uranium was enough to move p3's score-for-relative by `+576.2768`.

Proposed v0.4 direction, implemented below:

- Keep the simple relative-score formula shape.
- Replace the all-or-nothing opponent refuel projection with a best affordable partial-refuel projection:
  - enumerate or greedily search legal purchase baskets up to each opponent's deficits,
  - constrain by current market availability, storage, and cash,
  - evaluate projected generation after each feasible basket,
  - use the best projected opponent strength instead of returning no threat when full refuel is impossible.
- Add a small logging-only field such as `projection_horizon` / `projected_kind` on selected actions, then let the analyzer partition auction terminal projections separately from immediate next-state comparisons.
- Re-run the same 10-seed bucket and compare against v0.2/v0.3 before considering additional weight changes.

## v0.4-partial-refuel-threat

Status: implemented on 2026-07-18.

Implemented change:

- Replaced the all-or-nothing opponent refuel projection with a best affordable partial-refuel projection.
- Candidate refuel baskets are generated from feasible plant-run resource requirements, rather than arbitrary stockpiling:
  - includes partial subsets of an opponent's plants,
  - includes hybrid coal/oil mixes,
  - constrains by market availability, storage legality, and cash,
  - ignores baskets that do not improve immediate powered cities or income.
- Kept the relative-score formula shape:

```text
relative_score = own_score - 0.65 * max_opponent_score - 0.2 * average_opponent_score
```

- Updated detailed logs to `schema_version=4`.
- Added selected-action `projected_kind` / `projection_horizon`, especially to separate auction `terminal_if_won` projections from `immediate_state` projections.
- Added a bounded cache for opponent threat projections so resource-search evaluation remains practical.

Evaluation setup:

- Map: Germany
- Players: 3
- Lineup: `p1=ai_heuristics`, `p2=ai_deterministic`, `p3=ai_deterministic`
- Seeds: 1-10
- Saved logs:
  - Full game logs: `artifacts/game_logs/heuristic_v004/germany_3p_seed*.json`
  - Strategy logs: `artifacts/strategy_logs/heuristic_v004/germany_3p_seed*.json`
  - Discrepancy report: `artifacts/strategy_logs/heuristic_v004/discrepancy_report.json`

Performance:

- Wins: 2 / 10
- Win rate: 20%
- Winning seeds: 4, 8
- Decision pairs analyzed: 1339
- Strategy trace schema counts: `schema_version=4`: 1349 entries

Primary-driver counts:

- `opponent_score_or_relative_miss`: 422
- `own_score_miss`: 269
- `mixed_or_small_relative_miss`: 269
- `no_material_miss`: 379

Projection-horizon split:

- `immediate_state`
  - count: 810
  - max relative absolute delta: `118.7062`
  - own-score misses: 8
  - opponent/relative misses: 391
- `terminal_if_won`
  - count: 529
  - max relative absolute delta: `151.3846`
  - own-score misses: 261
  - this remains mostly the known auction projection horizon artifact.

Observed effect:

- v003 largest relative discrepancy was `374.6326`; v004 largest is `151.3846`.
- v003 largest actionable resource-phase discrepancy was seed 4 `finish_buying`, where p3's partial resource buy was not projected; v004 seed 4 is now won by the heuristic AI.
- `buy_resources / finish_buying` max relative absolute delta dropped from `374.6326` to `98.6232`.
- The largest remaining discrepancies are mostly auction `terminal_if_won` comparisons, which the new horizon field now makes explicit.

Diagnosis after v0.4:

- The partial-refuel change fixed the specific all-or-nothing blind spot and improved the 10-seed bucket from v002/v003's 1 / 10 to 2 / 10.
- It still does not recover v001's 3 / 10 baseline, so opponent threat modeling is better calibrated than v002 but still likely too pessimistic in some phases.
- The next useful analysis should focus on immediate-state opponent/relative misses, especially post-build and bureaucracy transitions, while treating auction `terminal_if_won` own-score gaps as a separate horizon category rather than a scoring error.

## v0.5-auction-economy-reserve

Status: implemented on 2026-07-19.

Implemented change:

- Replaced the heuristic AI's own auction reserve calculation with a fallback-aware post-auction economy projection.
- The reserve model now scans candidate purchase prices and projects:
  - plant purchase at that price,
  - feasible refuel baskets,
  - greedy city builds affordable after refueling,
  - generation and income after the build,
  - final relative score compared with pass/later-buy fallback projections.
- Removed the old auction-specific cash-slope/floor logic from the AI's own reserve cap; own reserve scanning is capped by available cash, while the projected resource/build/generation chain determines whether spending that cash is good.
- Kept `_auction_reserve_level0(...)` as a cheap opponent interest estimate only for contest pressure.
- Added a viability adjustment inside auction projections:
  - opening-round no-city/no-generation projections are heavily penalized,
  - newly purchased plants that cannot run in the projected chain are penalized.
- Disabled bait starts in round 1 when the AI has no plant yet.
- Updated detailed logs to `schema_version=5`.
- Auction logs now include:
  - fallback score and target/fallback surplus,
  - target/fallback projections,
  - reserve price samples,
  - raw score and viability adjustment,
  - cash after purchase/resources/build/income,
  - chosen resource plan, build plan, and generation summary.
- Updated the analyzer to keep `post_auction_economy`, `auction_fallback`, and `terminal_if_won` horizons in all/horizon summaries, but exclude them from the default actionable top-discrepancy list because they intentionally look beyond the next heuristic decision.
- Optimized reserve price scanning with coarse scan plus boundary refinement to keep auction decisions under the 2-second target.

Evaluation setup:

- Map: Germany
- Players: 3
- Lineup: `p1=ai_heuristics`, `p2=ai_deterministic`, `p3=ai_deterministic`
- Seeds: 1-10
- Saved logs:
  - Full game logs: `artifacts/game_logs/heuristic_v005/germany_3p_seed*.json`
  - Strategy logs: `artifacts/strategy_logs/heuristic_v005/germany_3p_seed*.json`
  - Discrepancy report: `artifacts/strategy_logs/heuristic_v005/discrepancy_report.json`

Performance:

- Wins: 6 / 10
- Win rate: 60%
- Winning seeds: 2, 5, 6, 7, 9, 10
- Decision pairs analyzed: 1048
- Actionable decision pairs analyzed: 750
- Strategy trace schema counts: `schema_version=5`: 1058 entries

Auction timing check:

- Measured across the same 10 full games.
- Auction decisions timed: 426
- Max auction decision time: `1.0703s`
- Mean auction decision time: `0.1743s`
- P95 auction decision time: `0.5976s`

Primary-driver counts:

- All decision pairs:
  - `opponent_score_or_relative_miss`: 305
  - `own_score_miss`: 206
  - `mixed_or_small_relative_miss`: 83
  - `no_material_miss`: 454
- Actionable immediate-horizon pairs:
  - `opponent_score_or_relative_miss`: 278
  - `own_score_miss`: 7
  - `mixed_or_small_relative_miss`: 11
  - `no_material_miss`: 454

Projection-horizon split:

- `immediate_state`
  - count: 750
  - max relative absolute delta: `129.9172`
  - own-score misses: 7
  - opponent/relative misses: 278
- `post_auction_economy`
  - count: 199
  - max relative absolute delta: `234.8723`
  - these are intentionally cross-stage projections and are not used as immediate actionable misses.
- `auction_fallback`
  - count: 99
  - max relative absolute delta: `246.9367`
  - these are also intentionally cross-stage projections.

Observed effect:

- The seed 5 opening plant-5 reserve case now projects buying resources, building one city, and generating income before deciding the reserve.
- In that seed 5 opening, plant 5 reserve dropped from the old `35` to `30`; the reserve scan fails at `31` because the projected score falls below the fallback purchase score.
- The previous overbid regression case where p1 would follow a 40 bid on plant 10 now passes; the fallback projection buys plant 11 at minimum price, buys uranium, builds one city, and generates.
- The 10-seed bucket improved from v004's 2 / 10 to 6 / 10.

Diagnosis after v0.5:

- The auction reserve issue identified by the user is substantially improved: bids are now tied to whether the remaining cash can complete resource/build/generation plans.
- The largest actionable miss is no longer an auction reserve own-score miss. It is seed 2, entry 937, `bureaucracy/run_plants`, where opponent strength increases before p1's next auction decision:
  - relative delta: `-129.9172`
  - own delta: `+6.7319`
  - opponent adjustment delta: `-136.6492`
  - primary driver: `opponent_score_or_relative_miss`
- The next likely improvement area is opponent-turn projection between p1 decisions, especially after p1 finishes resources or bureaucracy and opponents still have resource/build/bureaucracy actions before p1's next decision.
