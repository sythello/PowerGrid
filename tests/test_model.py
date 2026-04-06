from __future__ import annotations

from dataclasses import replace
import unittest

from powergrid.model import (
    Action,
    apply_builds,
    AuctionState,
    BureaucracySummary,
    build_city,
    can_store_resources,
    check_step_2_trigger,
    check_step_3_trigger,
    choose_plants_to_run,
    compute_all_targets_connection_cost,
    compute_connection_cost,
    compute_powered_cities,
    consume_resources,
    DecisionRequest,
    GameConfig,
    GameState,
    legal_build_targets,
    ModelValidationError,
    legal_resource_purchases,
    plant_storage_capacity,
    PlantRunPlan,
    PlayerState,
    PowerPlantCard,
    pay_income,
    purchase_resources,
    refill_resource_market,
    ResourceStorage,
    ResourceMarket,
    advance_phase,
    advance_round,
    create_initial_state,
    initialize_game,
    list_auctionable_plants,
    make_default_seat_configs,
    pass_auction,
    prepare_plant_deck,
    raise_bid,
    replace_plant_if_needed,
    resolve_bureaucracy,
    resolve_auction_round,
    resolve_winner,
    select_play_areas,
    start_auction,
    update_plant_market_after_bureaucracy,
)
from powergrid.rules_data import MapDefinition, RegionDefinition, load_power_plants


def _player(state: GameState, player_id: str) -> PlayerState:
    return next(player for player in state.players if player.player_id == player_id)


def _player_prices(state: GameState, player_id: str) -> tuple[int, ...]:
    return tuple(plant.price for plant in _player(state, player_id).power_plants)


def _player_elektro(state: GameState, player_id: str) -> int:
    return _player(state, player_id).elektro


def _resource_test_state(seed: int = 7) -> GameState:
    base_state = create_initial_state(
        GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=seed)
    )
    definitions = {definition.price: definition for definition in load_power_plants()}
    player_plants = {
        "p1": (5, 10),
        "p2": (7, 11),
        "p3": (6, 13),
    }
    updated_players = []
    for player in base_state.players:
        plants = tuple(
            PowerPlantCard.from_definition(definitions[price])
            for price in player_plants[player.player_id]
        )
        updated_players.append(replace(player, power_plants=plants))
    return replace(base_state, players=tuple(updated_players), phase="buy_resources", auction_state=None)


def _build_test_state(step: int = 1) -> GameState:
    base_state = create_initial_state(
        GameConfig(
            map_id="test",
            players=make_default_seat_configs(3),
            seed=11,
            selected_regions=("alpha", "beta", "gamma"),
        )
    )
    updated_players = []
    for player in base_state.players:
        if player.player_id == "p1":
            updated_players.append(
                replace(
                    player,
                    elektro=60,
                    houses_in_supply=21,
                    network_city_ids=("amber_falls",),
                )
            )
        elif player.player_id == "p2":
            updated_players.append(
                replace(
                    player,
                    elektro=60,
                    houses_in_supply=21,
                    network_city_ids=("brass_harbor",),
                )
            )
        else:
            updated_players.append(replace(player, elektro=60))
    return replace(base_state, players=tuple(updated_players), phase="build_houses", step=step, auction_state=None)


def _bureaucracy_test_state(
    *,
    step: int = 1,
    round_number: int = 2,
    player_specs: dict[str, dict[str, object]] | None = None,
    draw_stack_prices: tuple[int, ...] | None = None,
    bottom_stack_prices: tuple[int, ...] = (),
) -> GameState:
    base_state = create_initial_state(
        GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
    )
    definitions = {definition.price: definition for definition in load_power_plants()}
    available_city_ids = [
        city.id for city in base_state.game_map.cities if city.region in base_state.selected_regions
    ]
    if not available_city_ids:
        raise AssertionError("expected at least one city inside the selected play area")

    default_specs: dict[str, dict[str, object]] = {
        "p1": {"cities": 3, "plants": (10, 13), "storage": {"coal": 2}, "elektro": 40},
        "p2": {"cities": 4, "plants": (11, 18), "storage": {"uranium": 1}, "elektro": 35},
        "p3": {"cities": 2, "plants": (6, 15), "storage": {"garbage": 1, "coal": 2}, "elektro": 30},
    }
    specs = player_specs or default_specs
    updated_players = []
    for player in base_state.players:
        spec = specs[player.player_id]
        city_count = int(spec["cities"])
        owned_cities = tuple(available_city_ids[:city_count])
        plants = tuple(
            PowerPlantCard.from_definition(definitions[price])
            for price in spec["plants"]  # type: ignore[index]
        )
        updated_players.append(
            replace(
                player,
                elektro=int(spec.get("elektro", 50)),
                houses_in_supply=22 - city_count,
                network_city_ids=owned_cities,
                power_plants=plants,
                resource_storage=ResourceStorage.from_dict(spec.get("storage", {})),
            )
        )

    draw_stack = (
        tuple(PowerPlantCard.from_definition(definitions[price]) for price in draw_stack_prices)
        if draw_stack_prices is not None
        else base_state.power_plant_draw_stack
    )
    bottom_stack = tuple(
        PowerPlantCard.from_definition(definitions[price]) for price in bottom_stack_prices
    )

    return replace(
        base_state,
        players=tuple(updated_players),
        phase="bureaucracy",
        step=step,
        round_number=round_number,
        auction_state=None,
        pending_decision=None,
        power_plant_draw_stack=draw_stack,
        power_plant_bottom_stack=bottom_stack,
        last_powered_cities={},
        last_income_paid={},
    )


class ModelTests(unittest.TestCase):
    def test_create_initial_state_three_players(self) -> None:
        config = GameConfig(
            map_id="germany",
            players=make_default_seat_configs(3, ai_players=1),
            seed=7,
        )

        state = create_initial_state(config)

        self.assertEqual(state.phase, "setup")
        self.assertEqual(state.step, 1)
        self.assertEqual(len(state.players), 3)
        self.assertEqual(len(state.current_market), 4)
        self.assertEqual(len(state.future_market), 4)
        self.assertEqual(state.resource_market.total_in_market("coal"), 24)
        self.assertEqual(state.resource_market.total_in_market("oil"), 18)
        self.assertEqual(state.resource_market.total_in_market("garbage"), 9)
        self.assertEqual(state.resource_market.total_in_market("uranium"), 2)
        self.assertTrue(all(player.elektro == 50 for player in state.players))
        self.assertTrue(all(player.houses_in_supply == 22 for player in state.players))
        self.assertTrue(all(player.connected_city_count == 0 for player in state.players))
        self.assertEqual(state.selected_regions, ("black", "blue", "magenta"))
        self.assertEqual(state.player_order, ("p3", "p1", "p2"))
        self.assertEqual(tuple(plant.price for plant in state.current_market), (6, 7, 10, 11))
        self.assertEqual(tuple(plant.price for plant in state.future_market), (12, 13, 14, 15))
        self.assertEqual(len(state.power_plant_draw_stack), 26)
        self.assertEqual(tuple(plant.price for plant in state.power_plant_draw_stack[:4]), (4, 44, 29, 33))
        self.assertTrue(state.step_3_card_pending)
        self.assertEqual(
            {player.player_id: player.turn_order_position for player in state.players},
            {"p1": 2, "p2": 3, "p3": 1},
        )

    def test_prepare_plant_deck_three_players(self) -> None:
        deck = prepare_plant_deck(3, seed=7)
        self.assertEqual(tuple(plant.price for plant in deck.current_market), (6, 7, 10, 11))
        self.assertEqual(tuple(plant.price for plant in deck.future_market), (12, 13, 14, 15))
        self.assertEqual(len(deck.draw_stack), 26)
        self.assertEqual(tuple(plant.price for plant in deck.draw_stack[:5]), (4, 44, 29, 33, 28))
        self.assertEqual(deck.removed_plant_prices, (3, 9, 21, 25, 30, 32, 36, 38))
        self.assertTrue(deck.step_3_card_pending)

    def test_prepare_plant_deck_six_players(self) -> None:
        deck = prepare_plant_deck(6, seed=19)
        self.assertEqual(tuple(plant.price for plant in deck.current_market), (5, 6, 7, 8))
        self.assertEqual(tuple(plant.price for plant in deck.future_market), (9, 10, 12, 14))
        self.assertEqual(len(deck.draw_stack), 34)
        self.assertEqual(deck.removed_plant_prices, ())

    def test_same_seed_produces_same_initial_state(self) -> None:
        config = GameConfig(
            map_id="usa",
            players=make_default_seat_configs(6, ai_players=2),
            seed=19,
        )

        first = create_initial_state(config)
        second = create_initial_state(config)

        self.assertEqual(first, second)

    def test_different_seeds_change_initial_randomness(self) -> None:
        base_config = GameConfig(
            map_id="germany",
            players=make_default_seat_configs(4, ai_players=1),
            seed=3,
        )
        other_config = GameConfig(
            map_id="germany",
            players=make_default_seat_configs(4, ai_players=1),
            seed=4,
        )

        first = create_initial_state(base_config)
        second = create_initial_state(other_config)

        self.assertNotEqual(
            (first.player_order, tuple(plant.price for plant in first.current_market)),
            (second.player_order, tuple(plant.price for plant in second.current_market)),
        )

    def test_state_round_trip_serialization(self) -> None:
        config = GameConfig(
            map_id="test",
            players=make_default_seat_configs(3),
            seed=11,
            selected_regions=("alpha", "beta", "gamma"),
        )
        state = create_initial_state(config)

        restored = GameState.from_dict(state.to_dict())

        self.assertEqual(restored, state)

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            GameConfig(map_id="germany", players=make_default_seat_configs(2))

    def test_invalid_selected_region_is_rejected(self) -> None:
        config = GameConfig(
            map_id="test",
            players=make_default_seat_configs(3),
            seed=5,
            selected_regions=("missing",),
        )
        with self.assertRaises(ModelValidationError):
            create_initial_state(config)

    def test_select_play_areas_uses_contiguous_regions(self) -> None:
        state = create_initial_state(
            GameConfig(map_id="test", players=make_default_seat_configs(3), seed=5)
        )
        self.assertEqual(state.selected_regions, ("alpha", "beta", "gamma"))

    def test_select_play_areas_rejects_non_contiguous_choice(self) -> None:
        custom_map = MapDefinition(
            id="custom",
            name="Custom",
            regions=tuple(
                RegionDefinition(id=region_id, label=region_id.upper(), color=f"#{index}{index}{index}")
                for index, region_id in enumerate(("a", "b", "c", "d"), start=1)
            ),
            cities=(),
            connections=(),
            region_adjacency={
                "a": ("b",),
                "b": ("a", "c"),
                "c": ("b",),
                "d": (),
            },
            special_rules=(),
        )
        with self.assertRaises(ModelValidationError):
            select_play_areas(custom_map, 4, chosen_region_ids=("a", "b", "c", "d"))

    def test_initialize_game_validates_controller_registry(self) -> None:
        config = GameConfig(map_id="test", players=make_default_seat_configs(3), seed=2)
        controllers = {seat.player_id: object() for seat in config.players}
        state = initialize_game(config, controllers)
        self.assertEqual(state.selected_regions, ("alpha", "beta", "gamma"))
        with self.assertRaises(ModelValidationError):
            initialize_game(config, {"p1": object()})

    def test_advance_phase_moves_through_round_skeleton(self) -> None:
        state = create_initial_state(
            GameConfig(map_id="test", players=make_default_seat_configs(3), seed=1)
        )
        self.assertEqual((state.round_number, state.phase), (0, "setup"))
        state = advance_phase(state)
        self.assertEqual((state.round_number, state.phase), (1, "auction"))
        self.assertIsNotNone(state.auction_state)
        self.assertEqual(state.auction_state.current_chooser_id, state.player_order[0])
        with self.assertRaises(ModelValidationError):
            advance_phase(state)
        state = replace(
            state,
            round_number=2,
            auction_state=AuctionState(players_passed_phase=state.player_order),
        )
        state = resolve_auction_round(state)
        self.assertEqual(state.phase, "buy_resources")
        state = advance_phase(state)
        self.assertEqual(state.phase, "build_houses")
        state = advance_phase(state)
        self.assertEqual(state.phase, "bureaucracy")
        state = advance_phase(state)
        self.assertEqual((state.round_number, state.phase), (3, "determine_order"))

    def test_advance_round_requires_round_boundary(self) -> None:
        state = create_initial_state(
            GameConfig(map_id="test", players=make_default_seat_configs(3), seed=1)
        )
        self.assertEqual(advance_round(state).phase, "auction")
        mid_round = advance_phase(state)
        with self.assertRaises(ModelValidationError):
            advance_round(mid_round)

    def test_action_and_decision_request_are_typed_containers(self) -> None:
        action = Action(action_type="choose_power_plant", player_id="p1", payload={"price": 13})
        request = DecisionRequest(
            player_id="p1",
            decision_type="auction_choice",
            prompt="Choose a power plant to start the auction.",
            legal_actions=(action,),
            metadata={"phase": "auction"},
        )

        self.assertEqual(request.legal_actions[0].payload["price"], 13)
        self.assertEqual(request.metadata["phase"], "auction")

    def test_resource_market_round_trip(self) -> None:
        state = create_initial_state(
            GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=1)
        )
        restored = ResourceMarket.from_dict(state.resource_market.to_dict())
        self.assertEqual(restored, state.resource_market)

    def test_plant_storage_capacity_and_can_store_resources(self) -> None:
        state = _resource_test_state()
        hybrid = next(plant for plant in _player(state, "p1").power_plants if plant.price == 5)
        uranium = next(plant for plant in _player(state, "p2").power_plants if plant.price == 11)
        player_one = _player(state, "p1")
        player_two = _player(state, "p2")
        player_three = _player(state, "p3")

        self.assertEqual(plant_storage_capacity(hybrid), 4)
        self.assertEqual(plant_storage_capacity(uranium), 2)
        self.assertTrue(can_store_resources(player_one, {"coal": 4, "oil": 4}))
        self.assertFalse(can_store_resources(player_one, {"coal": 5, "oil": 4}))
        self.assertTrue(can_store_resources(player_two, {"uranium": 2}))
        self.assertFalse(can_store_resources(player_two, {"coal": 1}))
        self.assertFalse(can_store_resources(player_three, {"coal": 1}))

    def test_legal_resource_purchases_describes_capacity_and_prices(self) -> None:
        state = _resource_test_state()
        actions = legal_resource_purchases(state, "p1")
        action_map = {
            action.payload["resource"]: action.payload
            for action in actions
        }

        self.assertEqual(action_map["coal"]["max_units"], 8)
        self.assertEqual(action_map["coal"]["unit_prices"], [1, 1, 1, 2, 2, 2, 3, 3])
        self.assertEqual(action_map["oil"]["max_units"], 4)
        self.assertEqual(action_map["oil"]["unit_prices"], [3, 3, 3, 4])
        self.assertNotIn("garbage", action_map)

    def test_purchase_resources_updates_storage_market_and_money(self) -> None:
        state = _resource_test_state()

        state = purchase_resources(
            state,
            "p1",
            {"coal": 6, "oil": 2},
        )

        self.assertEqual(_player_elektro(state, "p1"), 35)
        player = _player(state, "p1")
        self.assertEqual(
            player.resource_storage,
            ResourceStorage(coal=4, oil=0, hybrid_coal=2, hybrid_oil=2),
        )
        self.assertEqual(state.resource_market.total_in_market("coal"), 18)
        self.assertEqual(state.resource_market.total_in_market("oil"), 16)
        self.assertEqual(state.resource_market.market["coal"][1], 0)
        self.assertEqual(state.resource_market.market["coal"][2], 0)
        self.assertEqual(state.resource_market.market["oil"][3], 1)

    def test_purchase_resources_rejects_capacity_and_supply_errors(self) -> None:
        state = _resource_test_state()

        with self.assertRaises(ModelValidationError):
            purchase_resources(state, "p1", {"coal": 5, "oil": 4})
        with self.assertRaises(ModelValidationError):
            purchase_resources(state, "p2", {"uranium": 3})
        with self.assertRaises(ModelValidationError):
            purchase_resources(state, "p3", {"coal": 1})

    def test_purchase_resources_rejects_unaffordable_basket(self) -> None:
        state = _resource_test_state()
        poor_player = replace(_player(state, "p2"), elektro=10)
        state = replace(
            state,
            players=tuple(
                poor_player if player.player_id == "p2" else player for player in state.players
            ),
        )

        with self.assertRaises(ModelValidationError):
            purchase_resources(state, "p2", {"uranium": 1})

    def test_refill_resource_market_uses_highest_empty_slots(self) -> None:
        state = _resource_test_state()
        market = state.resource_market.remove_from_market("oil", 4).remove_from_market("coal", 3)

        refilled = refill_resource_market(market, state.rules, step=1, player_count=3)

        self.assertEqual(refilled.market["oil"][4], 3)
        self.assertEqual(refilled.market["oil"][3], 1)
        self.assertEqual(refilled.market["coal"][1], 0)
        self.assertEqual(refilled.market["coal"][2], 3)
        self.assertEqual(refilled.supply["coal"], 0)
        self.assertEqual(refilled.supply["oil"], state.resource_market.supply["oil"] - 2)

    def test_resource_market_available_prices_and_quote(self) -> None:
        state = _resource_test_state()
        self.assertEqual(state.resource_market.available_unit_prices("uranium"), (14, 16))
        self.assertEqual(state.resource_market.quote_purchase_cost("oil", 4), 13)

    def test_choose_plants_to_run_defaults_hybrid_mix_and_consumes_shared_storage(self) -> None:
        state = _resource_test_state()
        player = replace(
            _player(state, "p1"),
            houses_in_supply=21,
            network_city_ids=(state.game_map.cities[0].id,),
            resource_storage=ResourceStorage(coal=1, hybrid_oil=1),
        )
        state = replace(
            state,
            players=tuple(player if existing.player_id == "p1" else existing for existing in state.players),
        )

        plans = choose_plants_to_run(state, "p1", (5,))
        self.assertEqual(plans, (PlantRunPlan(plant_price=5, resource_mix={"coal": 1, "oil": 1}),))
        self.assertEqual(compute_powered_cities(state, "p1", plans), 1)

        updated_state = consume_resources(state, "p1", plans)
        self.assertEqual(_player(updated_state, "p1").resource_storage, ResourceStorage())

    def test_pay_income_uses_payment_schedule(self) -> None:
        rules = create_initial_state(
            GameConfig(map_id="test", players=make_default_seat_configs(3), seed=1)
        ).rules
        self.assertEqual(pay_income(rules, 0), 10)
        self.assertEqual(pay_income(rules, 4), 54)
        self.assertEqual(pay_income(rules, 25), 150)

    def test_compute_connection_cost_uses_shortest_path_through_occupied_city(self) -> None:
        state = _build_test_state(step=1)
        self.assertEqual(compute_connection_cost(state, "p1", "cinder_grove"), 11)
        self.assertEqual(compute_connection_cost(state, "p3", "cinder_grove"), 0)

    def test_compute_all_targets_connection_cost_returns_shared_distances(self) -> None:
        state = _build_test_state(step=1)
        self.assertEqual(
            compute_all_targets_connection_cost(state, "p1"),
            {
                "amber_falls": 0,
                "brass_harbor": 4,
                "cinder_grove": 11,
            },
        )
        self.assertEqual(
            compute_all_targets_connection_cost(state, "p3"),
            {
                "amber_falls": 0,
                "brass_harbor": 0,
                "cinder_grove": 0,
            },
        )

    def test_legal_build_targets_respect_step_occupancy(self) -> None:
        step_one = _build_test_state(step=1)
        step_two = _build_test_state(step=2)

        self.assertEqual(
            {action.payload["city_id"] for action in legal_build_targets(step_one, "p3")},
            {"cinder_grove"},
        )
        brass_target = {
            action.payload["city_id"]: action.payload for action in legal_build_targets(step_two, "p3")
        }["brass_harbor"]
        self.assertEqual(brass_target["build_cost"], 15)
        self.assertEqual(brass_target["total_cost"], 15)

    def test_build_city_updates_network_houses_and_money(self) -> None:
        state = _build_test_state(step=1)
        state = build_city(state, "p1", "cinder_grove")

        player = _player(state, "p1")
        self.assertEqual(player.network_city_ids, ("amber_falls", "cinder_grove"))
        self.assertEqual(player.houses_in_supply, 20)
        self.assertEqual(player.elektro, 39)

    def test_apply_builds_uses_cheapest_multi_city_sequence(self) -> None:
        state = _build_test_state(step=2)
        rich_player = replace(_player(state, "p1"), elektro=80)
        state = replace(
            state,
            players=tuple(
                rich_player if existing.player_id == "p1" else existing for existing in state.players
            ),
        )

        state = apply_builds(state, "p1", ("cinder_grove", "brass_harbor"))

        updated_player = _player(state, "p1")
        self.assertEqual(updated_player.network_city_ids, ("amber_falls", "brass_harbor", "cinder_grove"))
        self.assertEqual(updated_player.elektro, 44)
        self.assertEqual(updated_player.houses_in_supply, 19)

    def test_apply_builds_rejects_full_city_and_out_of_area_city(self) -> None:
        state = _build_test_state(step=1)
        with self.assertRaises(ModelValidationError):
            build_city(state, "p3", "brass_harbor")

        germany_state = create_initial_state(
            GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
        )
        germany_state = replace(germany_state, phase="build_houses", step=1)
        with self.assertRaises(ModelValidationError):
            build_city(germany_state, germany_state.player_order[0], "aachen")

    def test_resolve_bureaucracy_updates_income_storage_market_and_order(self) -> None:
        state = _bureaucracy_test_state()

        updated_state, summary = resolve_bureaucracy(
            state,
            generation_choices={
                "p1": (10, 13),
                "p2": (11, 18),
                "p3": (6, 15),
            },
        )

        self.assertEqual(summary.powered_cities, {"p3": 2, "p1": 3, "p2": 4})
        self.assertEqual(summary.income_paid, {"p3": 33, "p1": 44, "p2": 54})
        self.assertFalse(summary.triggered_step_2)
        self.assertFalse(summary.triggered_step_3)
        self.assertFalse(summary.game_end_triggered)
        self.assertEqual(updated_state.phase, "determine_order")
        self.assertEqual(updated_state.round_number, 3)
        self.assertEqual(updated_state.player_order, ("p2", "p1", "p3"))
        self.assertEqual(_player(updated_state, "p1").resource_storage, ResourceStorage())
        self.assertEqual(_player(updated_state, "p2").resource_storage, ResourceStorage())
        self.assertEqual(_player(updated_state, "p3").resource_storage, ResourceStorage())
        self.assertEqual(_player(updated_state, "p1").elektro, 84)
        self.assertEqual(_player(updated_state, "p2").elektro, 89)
        self.assertEqual(_player(updated_state, "p3").elektro, 63)
        self.assertEqual(updated_state.last_powered_cities, summary.powered_cities)
        self.assertEqual(updated_state.last_income_paid, summary.income_paid)
        self.assertEqual(tuple(plant.price for plant in updated_state.future_market), (12, 13, 14, 44))
        self.assertEqual(tuple(plant.price for plant in updated_state.power_plant_bottom_stack), (15,))

    def test_step_2_trigger_starts_before_income_and_updates_market(self) -> None:
        state = _bureaucracy_test_state(
            player_specs={
                "p1": {"cities": 7, "plants": (13,), "storage": {}, "elektro": 40},
                "p2": {"cities": 4, "plants": (18,), "storage": {}, "elektro": 35},
                "p3": {"cities": 2, "plants": (22,), "storage": {}, "elektro": 30},
            }
        )

        self.assertTrue(check_step_2_trigger(state))
        updated_state, summary = resolve_bureaucracy(
            state,
            generation_choices={"p1": (13,), "p2": (18,), "p3": (22,)},
        )

        self.assertTrue(summary.triggered_step_2)
        self.assertEqual(summary.refill_step_used, 2)
        self.assertEqual(updated_state.step, 2)
        self.assertNotIn(6, tuple(plant.price for plant in (*updated_state.current_market, *updated_state.future_market)))
        self.assertEqual(tuple(plant.price for plant in updated_state.future_market), (13, 14, 15, 29))

    def test_step_3_trigger_reduces_market_and_uses_previous_refill_step(self) -> None:
        state = _bureaucracy_test_state(
            step=2,
            player_specs={
                "p1": {"cities": 3, "plants": (13,), "storage": {}, "elektro": 40},
                "p2": {"cities": 4, "plants": (18,), "storage": {}, "elektro": 35},
                "p3": {"cities": 2, "plants": (22,), "storage": {}, "elektro": 30},
            },
            draw_stack_prices=(),
            bottom_stack_prices=(25, 31, 33),
        )
        depleted_market = state.resource_market.remove_from_market("oil", 4)
        state = replace(state, resource_market=depleted_market)

        self.assertTrue(check_step_3_trigger(state))
        updated_state, summary = resolve_bureaucracy(
            state,
            generation_choices={"p1": (13,), "p2": (18,), "p3": (22,)},
        )

        self.assertTrue(summary.triggered_step_3)
        self.assertEqual(summary.refill_step_used, 2)
        self.assertEqual(updated_state.step, 3)
        self.assertFalse(updated_state.step_3_card_pending)
        self.assertEqual(len(updated_state.current_market), 6)
        self.assertEqual(updated_state.future_market, ())
        self.assertEqual(updated_state.resource_market.total_in_market("oil"), 17)
        self.assertEqual(updated_state.power_plant_bottom_stack, ())

    def test_resolve_winner_uses_powered_cities_then_money_then_connected_cities(self) -> None:
        state = _bureaucracy_test_state(
            player_specs={
                "p1": {"cities": 17, "plants": (25, 13), "storage": {"coal": 2}, "elektro": 40},
                "p2": {"cities": 15, "plants": (20, 23), "storage": {"coal": 3, "uranium": 1}, "elektro": 20},
                "p3": {"cities": 10, "plants": (18, 22), "storage": {}, "elektro": 50},
            },
        )
        updated_state, summary = resolve_bureaucracy(
            state,
            generation_choices={"p1": (25, 13), "p2": (20, 23), "p3": (18, 22)},
        )

        self.assertTrue(summary.game_end_triggered)
        self.assertIsNotNone(summary.winner_result)
        assert summary.winner_result is not None
        self.assertEqual(summary.winner_result.winner_ids, ("p2",))
        self.assertEqual(summary.powered_cities, {"p3": 4, "p1": 6, "p2": 8})
        self.assertEqual(updated_state.phase, "bureaucracy")

        tiebreak_state = replace(
            state,
            players=tuple(
                replace(player, elektro=70 if player.player_id == "p2" else 60 if player.player_id == "p1" else 10)
                for player in state.players
            ),
            last_powered_cities={"p1": 6, "p2": 6, "p3": 1},
        )
        self.assertEqual(resolve_winner(tiebreak_state).winner_ids, ("p2",))
        second_tiebreak_state = replace(
            tiebreak_state,
            players=tuple(replace(player, elektro=60 if player.player_id in {"p1", "p2"} else player.elektro) for player in tiebreak_state.players),
        )
        self.assertEqual(resolve_winner(second_tiebreak_state).winner_ids, ("p1",))

    def test_replace_plant_if_needed_discards_excess_resources_when_capacity_shrinks(self) -> None:
        base_state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
            )
        )
        player = _player(base_state, "p3")
        definitions = {definition.price: definition for definition in load_power_plants()}
        seeded_player = PlayerState(
            player_id=player.player_id,
            name=player.name,
            controller=player.controller,
            color=player.color,
            elektro=60,
            houses_in_supply=player.houses_in_supply,
            network_city_ids=player.network_city_ids,
            power_plants=tuple(
                PowerPlantCard.from_definition(definitions[price]) for price in (5, 10, 11, 13)
            ),
            resource_storage=ResourceStorage(coal=4, hybrid_coal=4),
            turn_order_position=player.turn_order_position,
        )
        state = replace(
            base_state,
            players=tuple(
                seeded_player if existing.player_id == "p3" else existing for existing in base_state.players
            ),
            pending_decision=DecisionRequest(
                player_id="p3",
                decision_type="discard_power_plant",
                prompt="Choose a power plant to discard.",
                legal_actions=tuple(
                    Action(
                        action_type="discard_power_plant",
                        player_id="p3",
                        payload={"price": price},
                    )
                    for price in (5, 10, 11, 13)
                ),
            ),
        )

        state = replace_plant_if_needed(state, "p3", 10)
        self.assertEqual(_player(state, "p3").resource_storage, ResourceStorage(hybrid_coal=4))

    def test_list_auctionable_plants_returns_current_market(self) -> None:
        state = advance_phase(
            create_initial_state(
                GameConfig(map_id="test", players=make_default_seat_configs(3), seed=1)
            )
        )
        self.assertEqual(
            tuple(plant.price for plant in list_auctionable_plants(state)),
            tuple(plant.price for plant in state.current_market),
        )

    def test_start_auction_uses_discount_token_on_lowest_plant(self) -> None:
        state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
            )
        )
        assert state.auction_state is not None
        self.assertEqual(state.auction_state.discount_token_plant_price, 6)
        state = start_auction(state, "p3", 6, 1)
        assert state.auction_state is not None
        self.assertEqual(state.auction_state.active_plant_price, 6)
        self.assertEqual(state.auction_state.current_bid, 1)
        self.assertEqual(state.auction_state.highest_bidder_id, "p3")
        self.assertEqual(state.auction_state.next_bidder_id, "p1")

    def test_discount_token_does_not_move_after_discounted_plant_is_bought(self) -> None:
        state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
            )
        )
        state = start_auction(state, "p3", 6, 1)
        state = pass_auction(state, "p1")
        state = pass_auction(state, "p2")

        assert state.auction_state is not None
        self.assertIsNone(state.auction_state.discount_token_plant_price)
        self.assertEqual(tuple(plant.price for plant in state.current_market), (4, 7, 10, 11))
        with self.assertRaises(ModelValidationError):
            start_auction(state, "p1", 4, 1)

    def test_cheaper_revealed_plant_is_discarded_and_clears_discount_token(self) -> None:
        state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
            )
        )
        state = replace(state, round_number=2)
        state = start_auction(state, "p3", 7, 7)
        state = pass_auction(state, "p1")
        state = pass_auction(state, "p2")

        assert state.auction_state is not None
        self.assertIsNone(state.auction_state.discount_token_plant_price)
        self.assertNotIn(4, tuple(plant.price for plant in (*state.current_market, *state.future_market)))
        self.assertEqual(tuple(plant.price for plant in state.current_market), (6, 10, 11, 12))
        self.assertEqual(tuple(plant.price for plant in state.future_market), (13, 14, 15, 44))
        with self.assertRaises(ModelValidationError):
            start_auction(state, "p1", 6, 1)

    def test_first_round_pass_is_rejected(self) -> None:
        state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
            )
        )
        with self.assertRaises(ModelValidationError):
            pass_auction(state, "p3")

    def test_non_opening_winner_keeps_opener_as_next_chooser(self) -> None:
        state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
            )
        )
        state = start_auction(state, "p3", 6, 1)
        state = raise_bid(state, "p1", 7)
        state = pass_auction(state, "p2")
        state = pass_auction(state, "p3")

        assert state.auction_state is not None
        self.assertEqual(_player_prices(state, "p1"), (6,))
        self.assertEqual(_player_elektro(state, "p1"), 43)
        self.assertEqual(state.auction_state.players_with_plants, ("p1",))
        self.assertEqual(state.auction_state.current_chooser_id, "p3")

    def test_first_round_auction_phase_auto_advances_after_all_players_buy(self) -> None:
        state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
            )
        )
        state = start_auction(state, "p3", 6, 1)
        state = pass_auction(state, "p1")
        state = pass_auction(state, "p2")
        self.assertEqual(_player_prices(state, "p3"), (6,))
        assert state.auction_state is not None
        self.assertEqual(state.auction_state.current_chooser_id, "p1")

        state = start_auction(state, "p1", 7, 7)
        state = pass_auction(state, "p2")
        self.assertEqual(_player_prices(state, "p1"), (7,))
        assert state.auction_state is not None
        self.assertEqual(state.auction_state.current_chooser_id, "p2")

        state = start_auction(state, "p2", 10, 10)
        self.assertEqual(state.phase, "buy_resources")
        self.assertIsNone(state.auction_state)
        self.assertEqual(_player_prices(state, "p2"), (10,))
        self.assertEqual(state.player_order, ("p2", "p1", "p3"))

    def test_passing_entire_phase_without_sales_removes_lowest_market_plant(self) -> None:
        state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=0)
            )
        )
        state = replace(state, round_number=2)
        original_discounted = min(plant.price for plant in state.current_market)
        chooser, second, third = state.player_order

        state = pass_auction(state, chooser)
        assert state.auction_state is not None
        self.assertEqual(state.auction_state.current_chooser_id, second)
        state = pass_auction(state, second)
        state = pass_auction(state, third)

        self.assertEqual(state.phase, "buy_resources")
        self.assertNotIn(
            original_discounted,
            tuple(plant.price for plant in (*state.current_market, *state.future_market)),
        )
        self.assertIn(11, tuple(plant.price for plant in (*state.current_market, *state.future_market)))
        self.assertEqual(len(state.current_market), 4)
        self.assertEqual(len(state.future_market), 4)

    def test_unsold_discounted_plant_is_removed_at_end_of_phase_after_other_sales(self) -> None:
        state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=0)
            )
        )
        state = replace(state, round_number=2)
        chooser, second, third = state.player_order
        state = start_auction(state, chooser, 5, 5)
        state = pass_auction(state, second)
        state = pass_auction(state, third)

        assert state.auction_state is not None
        self.assertEqual(state.auction_state.discount_token_plant_price, 4)
        self.assertEqual(state.auction_state.current_chooser_id, second)
        state = pass_auction(state, second)
        state = pass_auction(state, third)

        self.assertEqual(state.phase, "buy_resources")
        self.assertNotIn(4, tuple(plant.price for plant in (*state.current_market, *state.future_market)))
        self.assertIn(11, tuple(plant.price for plant in (*state.current_market, *state.future_market)))

    def test_replace_plant_if_needed_discards_after_fourth_purchase(self) -> None:
        base_state = advance_phase(
            create_initial_state(
                GameConfig(map_id="germany", players=make_default_seat_configs(3), seed=7)
            )
        )
        player = _player(base_state, "p3")
        seeded_player = PlayerState(
            player_id=player.player_id,
            name=player.name,
            controller=player.controller,
            color=player.color,
            elektro=60,
            houses_in_supply=player.houses_in_supply,
            network_city_ids=player.network_city_ids,
            power_plants=tuple(
                PowerPlantCard.from_dict(power_plant.to_dict())
                for power_plant in base_state.current_market[1:4]
            ),
            turn_order_position=player.turn_order_position,
        )
        other_players = tuple(
            seeded_player if existing.player_id == "p3" else existing for existing in base_state.players
        )
        state = replace(
            base_state,
            players=other_players,
            round_number=2,
            auction_state=AuctionState(
                current_chooser_id="p3",
                discount_token_plant_price=6,
                players_passed_phase=("p1", "p2"),
            ),
        )

        state = start_auction(state, "p3", 6, 1)
        self.assertIsNotNone(state.pending_decision)
        self.assertEqual(state.pending_decision.player_id, "p3")
        self.assertEqual(_player_prices(state, "p3"), (6, 7, 10, 11))

        state = replace_plant_if_needed(state, "p3", 6)
        self.assertIsNone(state.pending_decision)
        self.assertEqual(_player_prices(state, "p3"), (7, 10, 11))
        self.assertEqual(state.phase, "buy_resources")


if __name__ == "__main__":
    unittest.main()
