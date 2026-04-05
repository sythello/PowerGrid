from __future__ import annotations

from dataclasses import replace
import unittest

from powergrid.model import (
    Action,
    AuctionState,
    can_store_resources,
    DecisionRequest,
    GameConfig,
    GameState,
    ModelValidationError,
    legal_resource_purchases,
    plant_storage_capacity,
    PlayerState,
    PowerPlantCard,
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
    resolve_auction_round,
    select_play_areas,
    start_auction,
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
