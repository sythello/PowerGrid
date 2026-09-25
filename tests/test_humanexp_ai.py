from __future__ import annotations

from dataclasses import replace
import json
import random
import unittest
from unittest.mock import patch

from powergrid.ai import HumanExpHeuristicsAiController, build_ai_controller
from powergrid.ai.humanexp import (
    CONTROLLER, BuildProjection, Competition, _build_projection, _can_overbuild, _can_stockpile, _catalog,
    _competition, _contest_adjustments, _endgame_plant, _estimated_market,
    _fuel_options, _opening_priority, _opening_range, _opponent_endgame_threats,
    _portfolio_after_purchase, _prioritized_inventory, _replacement_plant,
    _stockpile_demand_shares, _with_player, _worst,
)
from powergrid.model import (
    AuctionState, GameConfig, ResourceStorage, SeatConfig,
    add_power_plant_to_player, compute_powered_cities, create_initial_state,
    purchase_resources, start_auction,
)
from powergrid.session import GameSession, default_seat_agents
from powergrid.web.server import PowerGridWebController


PLANTS = {p.price: p for p in _catalog()}


def state_for(*, count=3, map_id="germany", seed=7):
    state = create_initial_state(GameConfig(
        map_id=map_id, seed=seed,
        players=tuple(SeatConfig(f"p{i}", f"Player {i}", CONTROLLER) for i in range(1, count + 1)),
    ))
    players = tuple(replace(p, turn_order_position=i + 1) for i, p in enumerate(state.players))
    return replace(state, players=players, player_order=tuple(p.player_id for p in players),
                   phase="auction", round_number=1, auction_state=AuctionState(current_chooser_id="p1"))


def market(state, current=(3, 4, 5, 6), future=(7, 8, 9, 10), *, discount=None):
    visible = {*current, *future}
    return replace(state, current_market=tuple(PLANTS[p] for p in current),
                   future_market=tuple(PLANTS[p] for p in future),
                   power_plant_draw_stack=tuple(p for p in _catalog() if p.price not in visible),
                   power_plant_bottom_stack=(),
                   auction_state=replace(state.auction_state, discount_token_plant_price=discount))


def player(state, pid, *, plants=(), money=50, cities=(), storage=None):
    original = next(p for p in state.players if p.player_id == pid)
    return _with_player(state, replace(original, power_plants=tuple(PLANTS[p] for p in plants),
                                      elektro=money, network_city_ids=tuple(cities),
                                      houses_in_supply=22-len(cities),
                                      resource_storage=ResourceStorage.from_dict(storage or {})))


def choose(controller, state):
    session = GameSession(state, default_seat_agents(state.config))
    snapshot = session.snapshot()
    return controller.choose_intent(snapshot.active_request, snapshot)


class OpeningTests(unittest.TestCase):
    def test_competition_matches_example_and_excludes_bought_plant(self):
        competition = _competition(market(state_for()), "p1", 4)
        self.assertTrue(competition.has(5, 8))
        self.assertFalse(competition.has(9))
        self.assertFalse(competition.has(4))
        self.assertFalse(competition.has(3, 5, 6))

    def test_competition_ignores_finished_players_but_not_single_auction_passes(self):
        state = market(state_for())
        state = replace(state, auction_state=replace(state.auction_state, players_with_plants=("p3",)))
        competition = _competition(state, "p1", 4)
        self.assertTrue(competition.has(7))
        self.assertFalse(competition.has(8))
        self.assertFalse(competition.has(3, 5))

    def test_every_opening_price_branch(self):
        cases = [
            (3, (), (3,5)), (4, (), (4,8)), (5, (4,), (5,6)), (5, (), (6,9)),
            (6, (), (1,2)), (7, (3,), (0,0)), (7, (), (5,8)),
            (8, (4,5), (0,0)), (8, (4,), (8,10)), (8, (), (10,12)),
            (9, (), (9,11)), (10, (4,8), (0,0)), (10, (5,8), (0,0)),
            (10, (8,), (9,10)), (10, (4,5), (9,10)), (10, (5,), (10,12)),
            (10, (), (13,16)), (11, (), (0,0)), (12, (), (12,14)), (13, (), (15,20)),
        ]
        for price, rivals, expected in cases:
            with self.subTest(price=price, rivals=rivals):
                self.assertEqual(_opening_range(price, Competition(rivals, (), 2)), expected)

    def test_preference_branches_have_specified_order(self):
        cases = [(13, ()), (10, ()), (8, ()), (9, ()), (4, ()), (5, ()), (12, ()),
                 (10, (4,)), (8, (4,)), (3, ()), (7, ()), (10, (4,5)), (6, ())]
        self.assertEqual([_opening_priority(p, Competition(c, (), 2)) for p,c in cases], list(range(13)))
        self.assertEqual(_opening_priority(10, Competition((8,), (), 2)), 11)

    def test_nomination_minimum_and_last_buyer_respects_cap(self):
        controller = HumanExpHeuristicsAiController()
        state = market(state_for())
        intent = choose(controller, state)
        self.assertEqual(intent.payload, {"plant_price": 4, "bid": 4})
        state = market(state, (6, 7, 11, 12), (14,15,16,17), discount=6)
        state = replace(state, auction_state=replace(state.auction_state, players_with_plants=("p2","p3")))
        with patch.object(controller, "_sample", return_value=10):
            self.assertEqual(choose(controller, state).payload, {"plant_price": 7, "bid": 7})

    def test_last_buyer_chooses_cheapest_when_no_plant_fits_cap(self):
        state = market(state_for())
        state = replace(state, auction_state=replace(state.auction_state, players_with_plants=("p2","p3")))
        controller = HumanExpHeuristicsAiController()
        with patch.object(controller, "_sample", return_value=0):
            intent = choose(controller, state)
        self.assertEqual(intent.payload, {"plant_price": 3, "bid": 3})

    def test_mandatory_opening_chooses_cheapest_when_all_caps_too_low(self):
        controller = HumanExpHeuristicsAiController()
        with patch.object(controller, "_sample", return_value=0):
            intent = choose(controller, market(state_for()))
        self.assertEqual(intent.intent_type, "auction_start")
        self.assertEqual(intent.payload, {"plant_price": 3, "bid": 3})

    def test_mandatory_opening_fallback_uses_discounted_price(self):
        controller = HumanExpHeuristicsAiController()
        with patch.object(controller, "_sample", return_value=0):
            intent = choose(controller, market(state_for(), discount=3))
        self.assertEqual(intent.payload, {"plant_price": 3, "bid": 1})

    def test_bid_cap_and_near_cap_wait_boundaries(self):
        controller = HumanExpHeuristicsAiController()
        for standing, probability, expected in ((5,1,"auction_bid"), (6,0.5,"auction_pass"),
                                               (7,0.49,"auction_bid"), (8,0,"auction_pass")):
            state = start_auction(market(state_for()), "p1", 4, standing)
            with patch.object(controller, "_sample", return_value=8), patch.object(controller, "_better_next_probability", return_value=probability):
                self.assertEqual(choose(controller, state).intent_type, expected)

    def test_unseen_plugs_are_uniform_and_hidden_card_identity_is_ignored(self):
        state = market(state_for(), (3,4,5,6), (8,9,10,11))
        controller = HumanExpHeuristicsAiController()
        controller._seen.update((3,4,5,6,8,9,10,11))
        with patch("powergrid.ai.humanexp._catalog", return_value=(PLANTS[12], PLANTS[13])):
            # #8 appears for either draw and outranks #6.
            self.assertEqual(controller._better_next_probability(state, "p1", 6), 1.0)
        state = market(state, (3,4,6,9), (12,14,15,16))
        controller._seen = {3,4,6,9,12,14,15,16}
        with patch("powergrid.ai.humanexp._catalog", return_value=(PLANTS[8],PLANTS[13])):
            # Unseen #8 beats #4; otherwise #12 enters and ranks lower.
            self.assertEqual(controller._better_next_probability(state, "p1", 4), 0.5)
            swapped = replace(state, power_plant_draw_stack=tuple(reversed(state.power_plant_draw_stack)))
            swapped = replace(swapped, power_plant_draw_stack=(PLANTS[13],))
            self.assertEqual(controller._better_next_probability(swapped, "p1", 4), 0.5)

    def test_seeded_prices_stay_fixed_within_game(self):
        controller = HumanExpHeuristicsAiController()
        state = state_for()
        samples = [controller._sample(state, "p1", "opening:13", 15,20) for _ in range(20)]
        self.assertEqual(len(set(samples)), 1)
        self.assertTrue(15 <= samples[0] <= 20)
        self.assertGreater(len({controller._sample(replace(state, config=replace(state.config, seed=i)), "p1", "opening:13",15,20) for i in range(20)}),1)

    def test_socket_back_always_promotes_lowest_future_plant(self):
        state = market(state_for(), (3,4,5,6), (8,9,10,11))
        state = replace(state, power_plant_draw_stack=(PLANTS[20],PLANTS[25]))
        controller = HumanExpHeuristicsAiController()
        self.assertEqual(controller._better_next_probability(state,"p1",6),1.0)
        swapped = replace(state,power_plant_draw_stack=(PLANTS[25],PLANTS[20]))
        self.assertEqual(controller._better_next_probability(swapped,"p1",6),1.0)
        self.assertEqual(controller._better_next_probability(state,"p1",4),0.0)


class EconomyTests(unittest.TestCase):
    def test_endgame_threat_reserves_fuel_and_uses_map_build_cost(self):
        state = player(state_for(map_id="test"), "p2", plants=(4,), money=15,
                       cities=("amber_falls",))
        rules = replace(state.rules, player_count_rules={
            **state.rules.player_count_rules,
            3: {**state.rules.player_count_rules[3], "end_game_cities": 2},
        })
        state = replace(state, rules=rules, phase="buy_resources", auction_state=None)
        state = player(state, "p3", money=0)
        # The next city costs 14; two coal cost 2, so 15 is insufficient.
        self.assertEqual(_opponent_endgame_threats(state, "p1"), [])
        state = _with_player(state, replace(state.players[1], elektro=16))
        self.assertEqual(_opponent_endgame_threats(state, "p1"), [
            {"player_id": "p2", "fuel_cost": 2, "projected_cities": 2, "build_cost": 14},
        ])
        state = _with_player(state, replace(state.players[1], elektro=14,
                                           resource_storage=ResourceStorage(coal=2)))
        self.assertEqual(_opponent_endgame_threats(state, "p1")[0]["fuel_cost"], 0)

    def test_endgame_threat_excludes_self_and_requires_reachable_cities(self):
        state = player(state_for(map_id="test"), "p1", plants=(50,), money=1000)
        state = player(state, "p2", money=0)
        state = player(state, "p3", money=0)
        self.assertEqual(_opponent_endgame_threats(state, "p1"), [])
        # The test map cannot reach the normal 17-city threshold, even with cash.
        state = player(state, "p2", plants=(50,), money=1000)
        self.assertEqual(_opponent_endgame_threats(state, "p1"), [])

    def test_stockpile_capacity_excludes_unproductive_plants(self):
        state = player(state_for(), "p1", plants=(4,8,9), storage={"coal":4, "oil":2})
        holder = state.players[0]
        # #8's six extra coal spaces must not count; unrelated retained oil
        # must not prevent a valid coal purchase when coal space is available.
        self.assertFalse(_can_stockpile(holder, (PLANTS[4],), "coal"))
        self.assertTrue(_can_stockpile(holder, (PLANTS[8],), "coal"))
        self.assertFalse(_can_stockpile(holder, (PLANTS[4],), "oil"))
        self.assertEqual(holder.resource_storage.resource_totals()["oil"], 2)

    def test_stockpile_capacity_shares_coal_oil_space_in_eligible_hybrid_plants(self):
        state = player(state_for(), "p1", plants=(5,7), storage={"hybrid_coal":2,"oil":2})
        self.assertFalse(_can_stockpile(state.players[0], (PLANTS[5],), "coal"))
        self.assertFalse(_can_stockpile(state.players[0], (PLANTS[5],), "oil"))
        holder = replace(state.players[0], resource_storage=ResourceStorage(hybrid_coal=1, oil=2))
        self.assertTrue(_can_stockpile(holder, (PLANTS[5],), "coal"))
        self.assertTrue(_can_stockpile(holder, (PLANTS[5],), "oil"))

    def test_fuel_allocation_does_not_double_use_coal_between_hybrid_and_coal_plant(self):
        state = player(state_for(), "p1", plants=(4,5), storage={"coal":2})
        owner = state.players[0]
        options = list(_fuel_options(owner, owner.power_plants, state.resource_market))
        self.assertTrue(options)
        self.assertTrue(all(sum(basket.values()) == 2 for basket,_ in options))
        self.assertEqual(min(cost for _,cost in options), 2)

    def test_resource_estimate_respects_order_storage_and_round_up(self):
        state = player(state_for(), "p3", plants=(4,), storage={"coal":1})
        state = replace(state, phase="buy_resources", auction_state=None)
        estimated = _estimated_market(state, "p2")
        self.assertEqual(state.resource_market.total_in_market("coal") - estimated.total_in_market("coal"), 2)
        self.assertEqual(estimated.quote_purchase_cost("coal",2),3)
        self.assertEqual(_estimated_market(state,"p3"),state.resource_market)

    def test_auction_estimate_includes_competing_plants(self):
        state = market(state_for())
        estimated = _estimated_market(state,"p1",4)
        # Competitor #8 takes ceil(3/2)=2 coal; #5 may add hybrid coal too.
        self.assertGreaterEqual(state.resource_market.total_in_market("coal") - estimated.total_in_market("coal"),2)
        self.assertLess(estimated.total_in_market("garbage"), state.resource_market.total_in_market("garbage"))

    def test_build_forecast_counts_total_cities_and_per_city_surcharge(self):
        state = player(state_for(map_id="test"), "p1", cities=("amber_falls",))
        curve = _build_projection(state,"p1",surcharge=2)
        self.assertEqual(curve.costs,(16,19))
        self.assertEqual(curve.affordable(15),(1,0))
        self.assertEqual(curve.affordable(35),(3,35))

    def test_endgame_thresholds(self):
        state = state_for()
        self.assertTrue(_endgame_plant(state,(),PLANTS[20]))
        self.assertFalse(_endgame_plant(state,(),PLANTS[19]))
        self.assertTrue(_endgame_plant(state,(PLANTS[30],),PLANTS[25]))
        self.assertTrue(_endgame_plant(state,(PLANTS[20],PLANTS[25]),PLANTS[30]))
        self.assertFalse(_endgame_plant(state,(PLANTS[20],PLANTS[25]),PLANTS[29]))

    def test_replacement_preserves_resources_that_fit_new_plant(self):
        state = player(state_for(),"p1",plants=(4,13,18),storage={"coal":4})
        # #4 and #13 both output 1: lower-number #4 is replaced.
        self.assertEqual(_worst(state.players[0].power_plants).price,4)
        updated = _portfolio_after_purchase(state,"p1",PLANTS[25])
        self.assertEqual([p.price for p in updated.power_plants],[13,18,25])
        self.assertEqual(updated.resource_storage.total("coal"),4)
        self.assertEqual(state.players[0].resource_storage.total("coal"),4)

    def test_worst_plant_uses_number_even_when_higher_number_has_less_output(self):
        self.assertGreater(PLANTS[10].output_cities, PLANTS[13].output_cities)
        self.assertEqual(_worst((PLANTS[13], PLANTS[10], PLANTS[18])).price, 10)
        state = player(state_for(), "p1", plants=(10,13,18))
        updated = _portfolio_after_purchase(state, "p1", PLANTS[25])
        self.assertEqual([plant.price for plant in updated.power_plants], [13,18,25])

    def test_terminal_history_preserves_acquisition_order_and_discarded_values(self):
        controller = HumanExpHeuristicsAiController()
        state = player(state_for(), "p1", plants=(20,))
        controller._observe_terminal_history(state, "p1")
        state = player(state, "p1", plants=(20,36))
        controller._observe_terminal_history(state, "p1")
        self.assertEqual([p.output_cities for p in controller._terminal_history["p1"]], [5,7])
        state = player(state, "p1", plants=(29,36))
        controller._observe_terminal_history(state, "p1")
        self.assertEqual([p.price for p in controller._terminal_history["p1"]], [20,36,29])
        controller._observe_terminal_history(state, "p1")
        self.assertEqual(len(controller._terminal_history["p1"]), 3)

    def test_old_plant_becomes_terminal_when_new_purchase_lowers_threshold(self):
        controller = HumanExpHeuristicsAiController()
        state = player(state_for(), "p1", plants=(29,))
        controller._observe_terminal_history(state, "p1")
        self.assertEqual(controller._terminal_history["p1"], [])
        state = player(state, "p1", plants=(29,36))
        controller._observe_terminal_history(state, "p1")
        self.assertEqual([p.price for p in controller._terminal_history["p1"]], [36,29])

    def test_old_plant_is_recognized_before_mandatory_discard(self):
        controller = HumanExpHeuristicsAiController()
        state = player(state_for(), "p1", plants=(21,23,27))
        controller._observe_terminal_history(state, "p1")
        self.assertEqual(controller._terminal_history["p1"], [])
        awarded = add_power_plant_to_player(state, "p1", 36)
        self.assertIsNotNone(awarded.pending_decision)
        controller._observe_terminal_history(awarded, "p1")
        self.assertEqual([p.price for p in controller._terminal_history["p1"]], [36,21])
        from powergrid.model import replace_plant_if_needed
        discarded = replace_plant_if_needed(awarded, "p1", 21)
        controller._observe_terminal_history(discarded, "p1")
        self.assertEqual([p.price for p in controller._terminal_history["p1"]], [36,21])

    def test_after_three_terminals_candidate_must_exceed_lowest_current_output(self):
        state = state_for()
        history = (PLANTS[20], PLANTS[25], PLANTS[30])
        owned = (PLANTS[13], PLANTS[18], PLANTS[22])
        # The former third-plant threshold is 6; after three terminals it no
        # longer applies. Output 3 beats the current minimum of 1.
        self.assertTrue(_endgame_plant(state, history, PLANTS[27], owned=owned))
        self.assertFalse(_endgame_plant(state, history, PLANTS[13], owned=owned))

    def test_after_three_terminals_requires_one_more_city_even_for_discount(self):
        for discount in (None,27):
            for cities, expected in ((5,False),(6,True)):
                with self.subTest(discount=discount,cities=cities):
                    state = market(state_for(), (27,29,31,32), (33,34,35,36), discount=discount)
                    base = 1 if discount else 27
                    state = player(state,"p1",plants=(13,18,22),money=base + cities)
                    state = replace(state,round_number=6)
                    controller = HumanExpHeuristicsAiController()
                    controller._terminal_history["p1"] = [PLANTS[20],PLANTS[25],PLANTS[30]]
                    curve = BuildProjection(tuple(str(i) for i in range(20)), (1,)*20, 0)
                    projection = controller._auction_projection(state,"p1",PLANTS[27],curve)
                    self.assertEqual(projection.cities,cities)
                    self.assertTrue(projection.endgame)
                    self.assertEqual(projection.eligible,expected)
                    if discount:
                        self.assertEqual(projection.discount_eligible,expected)
                    self.assertEqual([p.price for p in controller._terminal_history["p1"]], [20,25,30])

    def test_after_three_terminals_rejects_nonupgrade_discount(self):
        state = market(state_for(),(13,27,29,31),(32,33,34,35),discount=13)
        state = player(state,"p1",plants=(3,18,22),money=100)
        controller = HumanExpHeuristicsAiController()
        controller._terminal_history["p1"] = [PLANTS[20],PLANTS[25],PLANTS[30]]
        curve = BuildProjection(tuple(str(i) for i in range(20)), (1,)*20, 0)
        projection = controller._auction_projection(state,"p1",PLANTS[13],curve)
        self.assertFalse(projection.endgame)
        self.assertFalse(projection.eligible)
        self.assertFalse(projection.discount_eligible)

    def test_after_three_terminals_drops_old_output_and_number_upgrade_gates(self):
        state = market(state_for(),(27,29,31,32),(34,35,36,37))
        state = player(state,"p1",plants=(18,22,33),money=36)
        state = replace(state,round_number=6)
        controller = HumanExpHeuristicsAiController()
        controller._terminal_history["p1"] = [PLANTS[20],PLANTS[25],PLANTS[30]]
        curve = BuildProjection(tuple(str(i) for i in range(9)), (1,)*9, 0)
        projection = controller._auction_projection(state,"p1",PLANTS[27],curve)
        self.assertTrue(projection.eligible)  # +1 output; exactly 1.5x number.
        self.assertEqual(projection.cities,9)
        # C=P+1 would block a nomination under the old pre-purchase capacity
        # gate, but after three terminals only the new two conditions apply.
        request = GameSession(state,default_seat_agents(state.config)).current_request()
        with patch("powergrid.ai.humanexp._build_projection",return_value=curve):
            intent,_ = controller._later_auction(state,request)
        self.assertEqual(intent.payload,{"plant_price":27,"bid":27})

    def test_after_three_terminals_replaces_lowest_output_in_forecast_and_real_purchase(self):
        controller = HumanExpHeuristicsAiController()
        state = replace(state_for(), round_number=6)
        # Observe real holdings in the acquisition order from the confirmed
        # example: #36=7, then #35=5, then #37=4.
        for plants in ((36,), (35,36), (35,36,37)):
            state = player(state,"p1",plants=plants,money=1000)
            choose(controller,state)
        self.assertEqual([p.price for p in controller._terminal_history["p1"]], [36,35,37])
        portfolio = _portfolio_after_purchase(state,"p1",PLANTS[44],terminal_complete=True)
        self.assertEqual([p.price for p in portfolio.power_plants], [35,36,44])
        self.assertEqual(sum(p.output_cities for p in portfolio.power_plants),17)
        state = replace(state, step=3, current_market=(PLANTS[44],), future_market=(),
                        power_plant_draw_stack=(), power_plant_bottom_stack=(),
                        auction_state=AuctionState(current_chooser_id="p1",players_with_plants=("p2","p3")))
        intent = choose(controller,state)
        self.assertEqual(intent.payload,{"plant_price":44,"bid":44})
        awarded = start_auction(state,"p1",44,44)
        self.assertIsNotNone(awarded.pending_decision)
        discarded = choose(controller,awarded)
        self.assertEqual(discarded.payload,{"plant_price":37})
        self.assertEqual([p.price for p in controller._terminal_history["p1"]], [36,35,37])

    def test_after_three_terminals_breaks_equal_output_discard_ties_by_number(self):
        choice = _replacement_plant((PLANTS[25],PLANTS[20],PLANTS[30]),terminal_complete=True)
        self.assertEqual(choice.price,20)

    def test_later_auction_capacity_gate_and_discount_exception(self):
        state = market(state_for(map_id="test"),(3,13,18,22),(24,25,26,27),discount=3)
        state = player(state,"p1",plants=(50,),money=50)
        state = replace(state,round_number=2)
        controller=HumanExpHeuristicsAiController()
        self.assertEqual(choose(controller,state).payload,{"plant_price":3,"bid":1})
        state=replace(state,auction_state=replace(state.auction_state,discount_token_plant_price=None))
        self.assertEqual(choose(controller,state).intent_type,"auction_pass")

    def test_full_portfolio_rejects_small_discount_upgrade(self):
        state = market(state_for(),(3,13,18,22),(24,25,26,27),discount=3)
        state = player(state,"p1",plants=(20,25,30),money=50)
        state = replace(state,round_number=2)
        self.assertEqual(choose(HumanExpHeuristicsAiController(),state).intent_type,"auction_pass")

    def test_nonterminal_auction_margin_reserves_fuel_and_required_cities(self):
        state = market(state_for(map_id="test"),(3,4,18,22),(24,25,26,27))
        state = player(state,"p1",plants=(13,),money=50)
        state = replace(state,round_number=2)
        curve = _build_projection(state,"p1",surcharge=2)
        candidate = HumanExpHeuristicsAiController()._auction_projection(state,"p1",PLANTS[18],curve)
        self.assertTrue(candidate.eligible)
        self.assertEqual(candidate.cities,2)
        self.assertEqual(candidate.margin,4)  # 50 - 18 - (12 + 16)

    def test_resource_plan_maximizes_income_less_fuel_and_preserves_build_money(self):
        state = player(state_for(map_id="test"),"p2",plants=(4,),money=35)
        state = replace(state,phase="buy_resources",round_number=2,auction_state=None)
        controller=HumanExpHeuristicsAiController()
        basket,details=controller._plan_resources(state,"p2",("coal","oil","garbage","uranium"))
        self.assertEqual(basket["coal"],2)
        self.assertEqual(details["net_income"],20)
        self.assertEqual(details["build_reserve"],11)
        self.assertEqual(details["projected_cities"],1)

    def test_stockpiling_only_positive_gaps_and_below_reserved_budget(self):
        state = player(state_for(map_id="test"),"p2",plants=(4,),money=15)
        state = player(state,"p1",plants=(8,20),money=50)
        state = replace(state,phase="buy_resources",round_number=2,auction_state=None)
        basket,details=HumanExpHeuristicsAiController()._plan_resources(state,"p2",("coal","oil","garbage","uranium"))
        self.assertGreater(details["resource_gaps"]["coal"],0)
        self.assertEqual(basket["coal"],3)
        result=purchase_resources(state,"p2",basket)
        self.assertGreaterEqual(result.players[1].elektro,details["build_reserve"])
        self.assertEqual((basket["oil"],basket["garbage"],basket["uranium"]),(0,0,0))

    def test_resource_plan_does_not_revisit_already_passed_resource(self):
        state=player(state_for(),"p2",plants=(4,5),money=30)
        state=replace(state,phase="buy_resources",auction_state=None)
        basket,details=HumanExpHeuristicsAiController()._plan_resources(state,"p2",("oil","garbage","uranium"))
        self.assertEqual(basket["coal"],0)
        self.assertNotIn(4,details["plants"])

    def test_forecast_skips_unaffordable_contested_city(self):
        from powergrid.rules_data import ConnectionDefinition
        state = player(state_for(map_id="test"),"p1",cities=("amber_falls",))
        state = replace(state,game_map=replace(state.game_map,connections=(
            ConnectionDefinition("amber_falls","brass_harbor",2),
            ConnectionDefinition("amber_falls","cinder_grove",1),
        )))
        # Brass would win the adjusted ranking (12-2), but only Cinder fits 11.
        curve = _build_projection(state,"p1",adjustments={"brass_harbor":2},budget=11)
        self.assertEqual(curve.cities,("cinder_grove",))
        self.assertEqual(curve.affordable(11),(2,11))

    def test_real_replacement_discards_old_plant_even_if_new_discount_has_less_output(self):
        state = market(state_for(),(13,18,22,24),(26,28,29,30),discount=13)
        state = player(state,"p1",plants=(7,15,25),money=50)
        state = replace(state,round_number=2,
                        auction_state=replace(state.auction_state,players_with_plants=("p2","p3")))
        controller = HumanExpHeuristicsAiController()
        intent = choose(controller,state)
        self.assertEqual(intent.payload,{"plant_price":13,"bid":1})
        awarded = start_auction(state,"p1",13,1)
        self.assertIsNotNone(awarded.pending_decision)
        first = choose(controller,awarded)
        second = choose(controller,awarded)
        self.assertEqual(first,second)
        self.assertEqual(first.payload,{"plant_price":7})


class ResourceMarginalTests(unittest.TestCase):
    def plan(self, state, *, cities, last_round=False):
        curve = BuildProjection(tuple(str(i) for i in range(cities)), (1,) * cities, 0)
        threats = [{"player_id": "p1", "projected_cities": 17}] if last_round else []
        with patch("powergrid.ai.humanexp._build_projection", return_value=curve), \
             patch("powergrid.ai.humanexp._opponent_endgame_threats", return_value=threats):
            return HumanExpHeuristicsAiController()._plan_resources(
                state, "p2", ("coal", "oil", "garbage", "uranium"),
            )

    def coal_state(self, stock=0):
        state = player(state_for(), "p2", plants=(4,25), money=100, storage={"coal":stock})
        state = player(state, "p1", plants=(8,20), money=0)
        return replace(state, phase="buy_resources", round_number=6, auction_state=None)

    def test_demand_share_blocks_only_extra_stock_above_threshold(self):
        state = player(state_for(), "p2", plants=(8,25), money=100)
        state = player(state, "p1", plants=(4,), money=0)
        for threshold, blocked in ((0.5, True), (5/7, False), (0.75, False)):
            with self.subTest(threshold=threshold), patch.object(
                HumanExpHeuristicsAiController, "_stockpile_share_threshold", return_value=threshold,
            ):
                basket, details = self.plan(state, cities=8)
                self.assertEqual(details["fuel_basket"]["coal"], 5)
                self.assertEqual(basket["coal"], 5 if blocked else 10)
                self.assertEqual(details["stockpile_demand_shares"]["coal"],
                                 {"own": 5, "total": 7, "share": 5/7, "blocked": blocked})
                self.assertEqual(details["stockpile_share_threshold"], threshold)
                json.dumps(details, allow_nan=False)

    def test_high_coal_share_does_not_block_oil_stockpiling(self):
        state = player(state_for(), "p2", plants=(20,16), money=100)
        state = player(state, "p1", plants=(4,7), money=0)
        with patch.object(HumanExpHeuristicsAiController, "_stockpile_share_threshold", return_value=0.5):
            basket, details = self.plan(state, cities=8)
        self.assertEqual(basket["coal"], details["fuel_basket"]["coal"])
        self.assertGreater(basket["oil"], details["fuel_basket"]["oil"])
        self.assertTrue(details["stockpile_demand_shares"]["coal"]["blocked"])
        self.assertFalse(details["stockpile_demand_shares"]["oil"]["blocked"])

    def test_demand_share_counts_all_plants_without_subtracting_inventory(self):
        state = self.coal_state(stock=6)
        _, details = self.plan(state, cities=5)
        self.assertEqual(details["stockpile_plants"], [25])
        self.assertEqual(details["stockpile_demand_shares"]["coal"]["own"], 4)
        self.assertEqual(details["stockpile_demand_shares"]["coal"]["total"], 10)
        self.assertEqual(details["stockpile_demand_shares"]["uranium"]["share"], 0)

    def test_hybrid_demand_goes_wholly_to_current_cheaper_track(self):
        state = player(state_for(), "p2", plants=(5,))
        state = player(state, "p1", plants=(12,))
        for removed, expected in ((0, "coal"), (6, "coal"), (9, "oil"), (24, "oil")):
            with self.subTest(removed=removed):
                current = replace(state, resource_market=state.resource_market.remove_from_market("coal", removed))
                shares, resource = _stockpile_demand_shares(current, "p2")
                self.assertEqual(resource, expected)
                self.assertEqual(shares[expected], {"own": 2, "total": 4, "share": 0.5})
                self.assertEqual(shares["oil" if expected == "coal" else "coal"]["total"], 0)
        empty = state.resource_market.remove_from_market("coal",24).remove_from_market("oil",18)
        shares, resource = _stockpile_demand_shares(replace(state, resource_market=empty), "p2")
        self.assertEqual(resource, "coal")
        self.assertEqual(shares["coal"]["total"], 4)
        # Even if a mixed purchase would be cheaper, demand classification uses
        # the next unit's price and never splits a hybrid across both tracks.
        uneven = replace(state.resource_market, market={
            **state.resource_market.market, "coal": {1: 1, 8: 3},
        })
        shares, resource = _stockpile_demand_shares(replace(state, resource_market=uneven), "p2")
        self.assertEqual(resource, "coal")
        self.assertEqual(shares["coal"]["total"], 4)
        self.assertEqual(shares["oil"]["total"], 0)

    def test_stockpile_threshold_is_reproducible_varies_by_round_and_preserves_global_rng(self):
        controller = HumanExpHeuristicsAiController()
        state = state_for()
        before = random.getstate()
        draws = [controller._stockpile_share_threshold(replace(state, round_number=i), "p2")
                 for i in range(1, 21)]
        self.assertTrue(all(0.5 <= value <= 0.75 for value in draws))
        self.assertEqual(len(set(draws)), 20)
        self.assertEqual(draws[0], controller._stockpile_share_threshold(state, "p2"))
        self.assertNotEqual(draws[0], controller._stockpile_share_threshold(state, "p1"))
        self.assertEqual(random.getstate(), before)

    def test_zero_marginal_plant_is_not_refueled_or_counted_for_stockpiling(self):
        basket, details = self.plan(self.coal_state(), cities=5)
        margins = {row["plant"]: row for row in details["plant_margins"]}
        self.assertEqual(margins[4]["marginal_income"], 0)
        self.assertFalse(margins[4]["refuel_eligible"])
        self.assertFalse(margins[4]["stockpile_eligible"])
        self.assertEqual(details["plants"], [25])
        self.assertEqual(details["fuel_basket"]["coal"], 2)
        self.assertEqual(basket["coal"], 4)  # Only #25's storage, not #4's extra four.
        self.assertGreater(details["resource_gaps"]["coal"], 0)

    def test_shared_inventory_fills_productive_storage_before_the_other_plant(self):
        state = self.coal_state(stock=4)
        state = replace(state, resource_market=state.resource_market.remove_from_market("coal",20))
        basket, details = self.plan(state, cities=6)
        margins = {row["plant"]: row for row in details["plant_margins"]}
        self.assertEqual(details["inventory_allocation"][0]["stored"]["coal"], 4)
        self.assertEqual(details["inventory_allocation"][1]["stored"]["coal"], 0)
        self.assertEqual(margins[4]["marginal_income"], 9)
        self.assertEqual(margins[4]["refuel_cost"], 15)
        self.assertFalse(margins[4]["refuel_eligible"])
        self.assertEqual(details["plants"], [25])
        self.assertEqual(basket["coal"], 0)

    def test_surplus_stock_can_make_refueling_profitable_without_allowing_stockpiling(self):
        state = self.coal_state(stock=5)
        state = replace(state, resource_market=state.resource_market.remove_from_market("coal",20))
        basket, details = self.plan(state, cities=6)
        margins = {row["plant"]: row for row in details["plant_margins"]}
        self.assertEqual(details["inventory_allocation"][0]["stored"]["coal"], 4)
        self.assertEqual(details["inventory_allocation"][1]["stored"]["coal"], 1)
        self.assertEqual(margins[4]["full_run_cost"], 15)
        self.assertEqual(margins[4]["refuel_cost"], 7)
        self.assertTrue(margins[4]["refuel_eligible"])
        self.assertFalse(margins[4]["stockpile_eligible"])
        self.assertEqual(details["plants"], [4,25])
        self.assertEqual(basket["coal"], 1)

    def test_existing_stock_does_not_give_a_zero_marginal_plant_stockpile_capacity(self):
        basket, details = self.plan(self.coal_state(stock=6), cities=5)
        margins = {row["plant"]: row for row in details["plant_margins"]}
        self.assertEqual(margins[4]["refuel_cost"], 0)
        self.assertTrue(margins[4]["refuel_eligible"])
        self.assertFalse(margins[4]["stockpile_eligible"])
        self.assertEqual(details["stockpile_plants"], [25])
        self.assertEqual(basket["coal"], 0)

    def test_prioritized_hybrid_allocation_keeps_all_existing_resources(self):
        state = player(state_for(), "p2", plants=(4,5), storage={"coal":4, "hybrid_oil":4})
        groups = _prioritized_inventory(state.players[1], (PLANTS[5],))
        # Putting coal in the productive hybrid would strand four oil in #4.
        self.assertEqual(groups[0][1]["oil"], 4)
        self.assertEqual(groups[0][1]["coal"], 0)
        self.assertEqual(groups[1][1]["coal"], 4)

    def test_final_round_buys_expensive_generation_even_if_net_income_is_lower(self):
        state = player(state_for(), "p2", plants=(7,13), money=100)
        state = replace(state, phase="buy_resources", auction_state=None,
                        resource_market=state.resource_market.remove_from_market("oil",15))
        regular, normal_details = self.plan(state, cities=3)
        final, final_details = self.plan(state, cities=3, last_round=True)
        self.assertEqual(regular["oil"], 0)
        self.assertEqual(normal_details["powered_cities"], 1)
        self.assertEqual(final["oil"], 3)
        self.assertEqual(final_details["powered_cities"], 3)
        self.assertEqual(final_details["resource_mode"], "final_round")
        self.assertLess(final_details["net_income"], normal_details["net_income"])
        self.assertEqual(final_details["stockpile_plants"], [])

    def test_final_round_only_refuels_once_despite_scarcity_and_spare_money(self):
        state = self.coal_state()
        regular, _ = self.plan(state, cities=5)
        final, details = self.plan(state, cities=5, last_round=True)
        self.assertEqual(regular["coal"], 4)
        self.assertEqual(final["coal"], 2)
        self.assertEqual(final, details["fuel_basket"])
        self.assertEqual(details["plants"], [25])
        self.assertGreater(details["cash_after_income"], 0)

    def test_real_endgame_projection_switches_resource_plan_to_final_round(self):
        state = player(state_for(map_id="test"), "p2", plants=(4,), money=15)
        state = player(state, "p1", plants=(4,), money=16, cities=("amber_falls",))
        state = player(state, "p3", plants=(8,20), money=0)
        rules = replace(state.rules, player_count_rules={
            **state.rules.player_count_rules,
            3: {**state.rules.player_count_rules[3], "end_game_cities": 2},
        })
        state = replace(state, rules=rules, phase="buy_resources", auction_state=None)
        basket, details = HumanExpHeuristicsAiController()._plan_resources(
            state, "p2", ("coal", "oil", "garbage", "uranium"),
        )
        self.assertEqual(details["resource_mode"], "final_round")
        self.assertEqual(details["endgame_threats"], [
            {"player_id": "p1", "fuel_cost": 2, "projected_cities": 2, "build_cost": 14},
        ])
        self.assertEqual(basket["coal"], 2)

    def test_full_cost_equal_to_marginal_income_is_not_excluded_from_stockpiling(self):
        state = self.coal_state()
        state = replace(state, resource_market=state.resource_market.remove_from_market("coal",11))
        _, details = self.plan(state, cities=6)
        margin = next(row for row in details["plant_margins"] if row["plant"] == 4)
        self.assertEqual(margin["full_run_cost"], 9)
        self.assertEqual(margin["marginal_income"], 9)
        self.assertTrue(margin["stockpile_eligible"])

    def test_unavailable_market_cost_is_logged_as_null_even_with_enough_stored_fuel(self):
        state = self.coal_state(stock=6)
        state = replace(state, resource_market=state.resource_market.remove_from_market("coal",24))
        basket, details = self.plan(state, cities=6)
        self.assertTrue(all(row["full_run_cost"] is None for row in details["plant_margins"]))
        self.assertEqual(details["stockpile_plants"], [])
        self.assertEqual(basket["coal"], 0)
        json.dumps(details, allow_nan=False)


class BuildAndIntegrationTests(unittest.TestCase):
    def test_contested_cities_receive_ranked_discounts(self):
        state=player(state_for(map_id="test"),"p2",cities=("amber_falls",))
        self.assertEqual(_contest_adjustments(state,"p1"),{"brass_harbor":2,"cinder_grove":1})

    def test_build_limit_and_first_order_exception(self):
        state=player(state_for(map_id="test"),"p2",plants=(13,),cities=("amber_falls",),money=100)
        state=replace(state,phase="build_houses",auction_state=None)
        controller=HumanExpHeuristicsAiController()
        self.assertEqual(controller._build(state,"p2")[0].intent_type,"finish_building")
        state=player(state,"p1",plants=(13,),cities=("brass_harbor",),money=100)
        self.assertTrue(_can_overbuild(state,"p1",2))
        self.assertEqual(controller._build(state,"p1")[0].intent_type,"commit_build")
        state=player(state,"p2",plants=(50,),cities=("amber_falls","cinder_grove"))
        self.assertFalse(_can_overbuild(state,"p1",2))

    def test_generation_maximizes_powered_cities_without_double_spending(self):
        state=player(state_for(map_id="test"),"p1",plants=(4,5,13),
                     cities=("amber_falls","brass_harbor","cinder_grove"),storage={"coal":2})
        state=replace(state,phase="bureaucracy",auction_state=None)
        intent=choose(HumanExpHeuristicsAiController(),state)
        self.assertEqual(intent.intent_type,"run_plants")
        from powergrid.model import PlantRunPlan
        plans=tuple(PlantRunPlan.from_dict(p) for p in intent.payload["plans"])
        self.assertEqual(compute_powered_cities(state,"p1",plans),2)
        self.assertLessEqual(sum(p.resource_mix.get("coal",0) for p in plans),2)

    def test_registry_and_web_launch_with_chinese_label(self):
        self.assertIsInstance(build_ai_controller(CONTROLLER),HumanExpHeuristicsAiController)
        web=PowerGridWebController()
        self.assertIn({"id":CONTROLLER,"name":"经验启发式v1"},web.metadata()["controllers"])
        payload=web.new_game({"map_id":"usa","players":[{"controller":CONTROLLER} for _ in range(4)],"seed":3})
        self.assertTrue(all(p["controller"]==CONTROLLER for p in payload["state"]["players"]))

    def test_tk_label_round_trip(self):
        from powergrid.gui.app import _controller_from_label, _available_ai_controller_names
        self.assertEqual(_controller_from_label("经验启发式v1"),CONTROLLER)
        self.assertIn(CONTROLLER,_available_ai_controller_names())

    def test_decisions_leave_snapshot_unchanged(self):
        for phase in ("auction","buy_resources","build_houses","bureaucracy"):
            with self.subTest(phase=phase):
                state = player(market(state_for()),"p1",plants=(14,15,18),money=50)
                state = replace(state,phase=phase,round_number=2,
                                auction_state=state.auction_state if phase=="auction" else None)
                before = state.to_dict()
                choose(HumanExpHeuristicsAiController(),state)
                self.assertEqual(state.to_dict(),before)

    def test_full_games_complete_without_illegal_actions_and_log_decisions(self):
        for map_id in ("germany","usa"):
            for count in (3,4,5,6):
                with self.subTest(map_id=map_id,count=count):
                    session=GameSession.new_game(state_for(map_id=map_id,count=count,seed=count).config)
                    for _ in range(2000):
                        snapshot,applied=session.advance_one_ai_action()
                        if not applied:
                            break
                    self.assertIsNotNone(snapshot.winner_result)
                    self.assertFalse(any(e.level=="error" for e in snapshot.event_log))
                    entries=[e for e in session.game_log_entries() if e.source=="ai"]
                    self.assertTrue(entries)
                    self.assertTrue(all(e.payload["label"]=="humanexp_heuristics_decision" for e in entries))
                    json.dumps([e.to_dict() for e in entries],allow_nan=False)


if __name__ == "__main__":
    unittest.main()
