# Manual Tests

This directory is for interactive, human-driven checks that sit outside the automated
`unittest` suite.

These scripts are useful when we want to:
- drive the engine from the terminal by hand
- verify that legal actions are accepted
- verify that illegal actions are rejected with a clear explanation
- inspect intermediate game state after each human-entered action

## Auction Phase Manual Test

Run:

```bash
PYTHONPATH=src python tests/manual_test/run_auction_phase.py
```

You can also pre-fill the setup values:

```bash
PYTHONPATH=src python tests/manual_test/run_auction_phase.py --players 4 --step 2 --first-round no --seed 11
```

The script will:
- build a valid seeded auction state
- let you choose first round vs later round
- let you choose Step 1, 2, or 3
- show the current market, future market, and draw-stack preview
- show whether the `$1` token is currently on a specific plant or off the market
- prompt the active player for each auction decision
- accept only valid auction commands according to the current state
- print a rejection message for invalid commands without changing the state

Supported commands depend on the current state:
- chooser with no active auction: `start <plant_price> <bid>` or `pass`
- bidder in an active auction: `bid <amount>` or `pass`
- discard decision: `discard <plant_price>`
- utility commands at any time: `status`, `help`, `quit`

## Resource Buying Manual Test

Run:

```bash
PYTHONPATH=src python tests/manual_test/run_resource_buying.py
```

You can also pre-fill the setup values:

```bash
PYTHONPATH=src python tests/manual_test/run_resource_buying.py --players 4 --step 2 --seed 11
```

The script will:
- build a valid seeded state in the `buy_resources` phase
- seed each player with real power plants from the game deck
- show reverse turn order for resource buying
- show each player's current Elektro and player-level resource storage
- let you inspect legal purchases for the active buyer
- let you buy resources into the shared player storage and inspect the updated state
- let you trigger a market refill manually to inspect the result

Supported commands:
- `options`
- `buy <resource> <amount>`
- `done`
- `refill`
- `status`
- `help`
- `quit`

## Build Phase Manual Test

Run:

```bash
PYTHONPATH=src python tests/manual_test/run_build_phase.py
```

You can also choose the map and step:

```bash
PYTHONPATH=src python tests/manual_test/run_build_phase.py --map germany --seed 7 --step 2
```

The script will:
- build a valid seeded state in the `build_houses` phase
- seed each player with an existing city network
- show reverse turn order for building
- let you inspect legal single-city build targets
- let you quote multi-city builds before committing
- let you execute single-city or multi-city builds and inspect the updated state

Supported commands:
- `options`
- `quote <city_id> [city_id ...]`
- `build <city_id> [city_id ...]`
- `done`
- `status`
- `help`
- `quit`

## Bureaucracy Phase Manual Test

Run:

```bash
PYTHONPATH=src python tests/manual_test/run_bureaucracy_phase.py
```

You can also choose a focused scenario:

```bash
PYTHONPATH=src python tests/manual_test/run_bureaucracy_phase.py --scenario step3 --seed 7
```

Available scenarios:
- `normal`
- `step2`
- `step3`
- `endgame`

The script will:
- build a valid seeded state in the `bureaucracy` phase
- seed players with city networks, power plants, and shared resource storage
- let each player choose which power plants to run
- validate hybrid-fuel mixes when you provide them
- resolve the whole Bureaucracy phase and print step changes, incomes, market updates, and winner information

Supported commands:
- `options`
- `run <plant_price>[:resource=amount,...] ...`
- `skip`
- `done`
- `status`
- `resolve`
- `help`
- `quit`
