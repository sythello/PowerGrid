from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from powergrid.ai import NnRlBasedAiController, build_ai_controller
from powergrid.ai.nn_rank_value.candidates import generate_candidate_actions
from powergrid.ai.nn_rank_value.observation import (
    build_public_observation,
    encode_action_features,
    encode_state_features,
    player_slot_ids,
)
from powergrid.ai.nn_rl_based.dataset import (
    generate_rl_dataset,
    legal_region_sets,
    load_rl_dataset_metadata,
    load_rl_dataset_records,
    verify_rl_dataset_manifest,
)
from powergrid.ai.nn_rl_based.model import (
    NumpyRlPolicyQNetwork,
    PolicyQPredictions,
    build_policy_targets,
)
from powergrid.ai.nn_rl_based.search import (
    FullActionSemanticSearcher,
    SearchConfig,
    advance_to_semantic_boundary,
)
from powergrid.ai.nn_rl_based.training import _iter_array_batches, train_rl_model
from powergrid.model import ModelValidationError, add_power_plant_to_player
from powergrid.session import GameSession, default_seat_agents
from powergrid.tools.evaluate_nn_rl_paired_rollouts import (
    PairedRolloutRecord,
    _rollout_terminal_values,
    _summarize_checkpoint,
)
from powergrid.tools.evaluate_nn_rl_deterministic_suite import _summarize_opponent


class RlPolicyQModelTests(unittest.TestCase):
    def test_advantage_gate_targets_and_feature_collision_fallback(self) -> None:
        offsets = np.asarray([0, 3, 6, 9, 12], dtype=np.int32)
        teacher = np.asarray([0, 0, 0, 0], dtype=np.int32)
        searched = np.asarray([False, True, True, True], dtype=bool)
        actions = np.asarray(
            [
                [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
                [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
                [0.0, 3.0], [0.0, 3.0], [2.0, 3.0],
            ],
            dtype=np.float32,
        )
        search_q = np.zeros((12, 6), dtype=np.float32)
        search_q[3:6, 0] = [0.4, 0.45, 0.2]
        search_q[6:9, 0] = [0.1, 0.4, 0.2]
        search_q[9:12, 0] = [0.1, 0.8, 0.25]
        targets, accepted, improved = build_policy_targets(
            offsets,
            teacher,
            searched,
            search_q,
            actions,
            policy_target_mode="advantage_gate",
            improved_action_weight=0.75,
            min_search_advantage=0.1,
        )
        np.testing.assert_array_equal(accepted, [False, False, True, True])
        np.testing.assert_array_equal(improved, [0, 0, 1, 2])
        np.testing.assert_allclose(targets[0:3], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(targets[3:6], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(targets[6:9], [0.25, 0.75, 0.0])
        np.testing.assert_allclose(targets[9:12], [0.25, 0.0, 0.75])
        for start, end in zip(offsets[:-1], offsets[1:]):
            self.assertAlmostEqual(float(targets[start:end].sum()), 1.0)

    def test_legacy_policy_target_is_unchanged(self) -> None:
        offsets = np.asarray([0, 3], dtype=np.int32)
        teacher = np.asarray([1], dtype=np.int32)
        searched = np.asarray([True], dtype=bool)
        search_q = np.zeros((3, 6), dtype=np.float32)
        search_q[:, 0] = [0.1, 0.2, 0.6]
        actions = np.arange(6, dtype=np.float32).reshape(3, 2)
        targets, accepted, improved = build_policy_targets(
            offsets,
            teacher,
            searched,
            search_q,
            actions,
            policy_target_mode="legacy_soft_mix",
            search_policy_mix=0.25,
            search_temperature=0.5,
        )
        logits = search_q[:, 0] / 0.5
        soft = np.exp(logits - np.max(logits))
        expected = 0.25 * soft / soft.sum()
        expected[1] += 0.75
        np.testing.assert_allclose(targets, expected, atol=1e-7)
        np.testing.assert_array_equal(accepted, [False])
        np.testing.assert_array_equal(improved, teacher)

    def test_advantage_gate_policy_overfits_accepted_and_fallback_targets(self) -> None:
        decisions = 32
        offsets = np.arange(0, (decisions + 1) * 2, 2, dtype=np.int32)
        states = np.zeros((decisions, 2), dtype=np.float32)
        states[:, 0] = np.arange(decisions) % 2
        actions = np.tile(
            np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            (decisions, 1),
        )
        teacher = np.zeros(decisions, dtype=np.int32)
        terminal = np.zeros((decisions, 6), dtype=np.float32)
        masks = np.zeros((decisions, 6), dtype=bool)
        masks[:, :2] = True
        searched = np.ones(decisions, dtype=bool)
        search_q = np.zeros((decisions * 2, 6), dtype=np.float32)
        for index in range(decisions):
            if index % 2 == 0:
                search_q[offsets[index] : offsets[index + 1], 0] = [0.0, 0.4]
            else:
                search_q[offsets[index] : offsets[index + 1], 0] = [0.4, 0.0]
        model = NumpyRlPolicyQNetwork(2, 2, hidden_dims=(16, 8, 8), seed=43)
        for _ in range(160):
            model.train_batch(
                states,
                actions,
                offsets,
                teacher,
                terminal,
                masks,
                searched,
                search_q,
                learning_rate=3e-3,
                policy_weight=1.0,
                q_mc_weight=0.0,
                q_search_weight=0.0,
                policy_target_mode="advantage_gate",
                improved_action_weight=0.75,
                min_search_advantage=0.1,
            )
        metrics = model.evaluate_batch(
            states,
            actions,
            offsets,
            teacher,
            terminal,
            masks,
            searched,
            search_q,
            policy_target_mode="advantage_gate",
            improved_action_weight=0.75,
            min_search_advantage=0.1,
        )
        self.assertGreater(metrics["accepted_policy_top1_accuracy"], 0.95)
        self.assertGreater(metrics["searched_fallback_teacher_accuracy"], 0.95)

    def test_grouped_policy_q_learns_and_round_trips(self) -> None:
        rng = np.random.default_rng(31)
        decisions = 24
        candidates_per_decision = 3
        offsets = np.arange(0, (decisions + 1) * candidates_per_decision, candidates_per_decision)
        states = rng.normal(size=(decisions, 5)).astype(np.float32)
        actions = rng.normal(size=(offsets[-1], 2)).astype(np.float32)
        teacher = np.argmax(actions[:, 0].reshape(decisions, candidates_per_decision), axis=1)
        terminal = np.zeros((decisions, 6), dtype=np.float32)
        terminal[:, 0] = np.tanh(states[:, 0])
        terminal[:, 1] = -terminal[:, 0]
        masks = np.zeros((decisions, 6), dtype=bool)
        masks[:, :2] = True
        searched = np.zeros(decisions, dtype=bool)
        search_q = np.zeros((len(actions), 6), dtype=np.float32)
        model = NumpyRlPolicyQNetwork(5, 2, hidden_dims=(24, 12, 12), seed=31)
        before = model.evaluate_batch(
            states, actions, offsets, teacher, terminal, masks, searched, search_q
        )
        for _ in range(100):
            model.train_batch(
                states,
                actions,
                offsets,
                teacher,
                terminal,
                masks,
                searched,
                search_q,
                learning_rate=3e-3,
                q_search_weight=0.0,
            )
        after = model.evaluate_batch(
            states, actions, offsets, teacher, terminal, masks, searched, search_q
        )
        self.assertGreater(after["policy_accuracy"], before["policy_accuracy"])
        self.assertLess(after["q_mc_mae"], before["q_mc_mae"])
        predictions = model.predict(states, actions, offsets)
        for start, end in zip(offsets[:-1], offsets[1:]):
            self.assertAlmostEqual(
                float(predictions.policy_probabilities[start:end].sum()), 1.0, places=6
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = NumpyRlPolicyQNetwork.load(path)
            actual = loaded.predict(states, actions, offsets)
        np.testing.assert_array_equal(predictions.policy_logits, actual.policy_logits)
        np.testing.assert_array_equal(predictions.q_values, actual.q_values)

    def test_search_q_head_overfits_synthetic_full_action_labels(self) -> None:
        rng = np.random.default_rng(37)
        decisions = 16
        offsets = np.arange(0, (decisions + 1) * 2, 2, dtype=np.int32)
        states = rng.normal(size=(decisions, 4)).astype(np.float32)
        actions = rng.normal(size=(offsets[-1], 2)).astype(np.float32)
        teacher = np.zeros(decisions, dtype=np.int32)
        terminal = np.zeros((decisions, 6), dtype=np.float32)
        masks = np.zeros((decisions, 6), dtype=bool)
        masks[:, :2] = True
        searched = np.ones(decisions, dtype=bool)
        search_q = np.zeros((len(actions), 6), dtype=np.float32)
        search_q[:, 0] = np.tanh(actions[:, 0])
        search_q[:, 1] = -search_q[:, 0]
        model = NumpyRlPolicyQNetwork(4, 2, hidden_dims=(24, 12, 12), seed=37)
        before = model.evaluate_batch(
            states, actions, offsets, teacher, terminal, masks, searched, search_q
        )
        for _ in range(120):
            model.train_batch(
                states,
                actions,
                offsets,
                teacher,
                terminal,
                masks,
                searched,
                search_q,
                learning_rate=3e-3,
                policy_weight=0.0,
                q_mc_weight=0.0,
            )
        after = model.evaluate_batch(
            states, actions, offsets, teacher, terminal, masks, searched, search_q
        )
        self.assertLess(after["q_search_mae"], before["q_search_mae"] * 0.35)
        self.assertEqual(after["q_mc_elements"], decisions * 2)
        self.assertEqual(after["q_search_elements"], len(actions) * 2)

    def test_player_slots_match_actor_relative_feature_order(self) -> None:
        snapshot = GameSession.from_scenario("opening", seed=7).snapshot()
        assert snapshot.active_request is not None
        observation = build_public_observation(snapshot.state, snapshot.active_request)
        slots = player_slot_ids(observation)
        self.assertEqual(slots[0], snapshot.active_request.player_id)
        self.assertEqual(set(slots), {player.player_id for player in snapshot.state.players})


class RlPairedTerminalRolloutTests(unittest.TestCase):
    def test_paired_terminal_rollout_is_repeatable_and_does_not_mutate_root(self) -> None:
        session = GameSession.from_scenario("opening", seed=59)
        before = session.snapshot()
        assert before.active_request is not None
        agents = {
            player.player_id: build_ai_controller("ai_deterministic")
            for player in before.state.players
        }
        intent = agents[before.active_request.player_id].choose_intent(
            before.active_request, before
        )
        first = _rollout_terminal_values(session, intent, agents, max_actions=5000)
        second = _rollout_terminal_values(session, intent, agents, max_actions=5000)
        self.assertEqual(first, second)
        self.assertEqual(session.snapshot().state, before.state)
        self.assertIsNone(session.snapshot().winner_result)

    def test_paired_summary_uses_deviations_and_baseline_decision_coverage(self) -> None:
        def record(
            game_index: int, advantage: float, decision_type: str
        ) -> PairedRolloutRecord:
            return PairedRolloutRecord(
                checkpoint="candidate",
                game_index=game_index,
                seed=70000 + game_index,
                selected_regions=("black", "blue", "magenta"),
                decision_index=game_index,
                round_number=1,
                phase="auction",
                decision_type=decision_type,
                actor_id="p1",
                baseline_intent={"intent_type": "auction_pass"},
                rl_intent={"intent_type": "auction_start"},
                baseline_rank_value=0.0,
                rl_rank_value=advantage,
                advantage=advantage,
            )

        result = _summarize_checkpoint(
            [
                record(1, 1.0, "auction_start"),
                record(1, 0.0, "auction_start"),
                record(2, -1.0, "buy_resources"),
                record(2, 1.0, "buy_resources"),
            ],
            games=2,
            total_decisions=30,
            game_decisions={1: 10, 2: 20},
            bootstrap_samples=100,
            bootstrap_seed=11,
        )
        self.assertEqual(result["deviations"], 4)
        self.assertAlmostEqual(result["deviation_rate"], 4 / 30)
        self.assertEqual(
            (result["improved"], result["tied"], result["harmed"]),
            (2, 1, 1),
        )
        self.assertAlmostEqual(result["mean_advantage_on_deviations"], 0.25)
        self.assertAlmostEqual(
            result["mean_advantage_per_baseline_decision"], 1 / 30
        )
        self.assertAlmostEqual(result["paired_score"], 0.625)
        self.assertEqual(
            result["by_decision_type"]["auction_start"]["deviations"], 2
        )

    def test_deterministic_suite_summary_uses_cross_controller_pairs(self) -> None:
        games = [
            {
                "selected_regions": ["black", "blue", "magenta"],
                "score": 2.0,
                "comparisons": 2,
                "wins": 2,
                "draws": 0,
                "losses": 0,
                "standings": [
                    {"controller_name": "ai_nn_rl_based_v1", "place": 1},
                    {"controller_name": "ai_nn_rl_based_v1", "place": 2},
                    {"controller_name": "ai_deterministic", "place": 3},
                ],
            },
            {
                "selected_regions": ["black", "blue", "yellow"],
                "score": 1.0,
                "comparisons": 2,
                "wins": 1,
                "draws": 0,
                "losses": 1,
                "standings": [
                    {"controller_name": "ai_deterministic", "place": 1},
                    {"controller_name": "ai_nn_rl_based_v1", "place": 2},
                    {"controller_name": "ai_deterministic", "place": 3},
                ],
            },
        ]
        result = _summarize_opponent(
            games, bootstrap_samples=100, bootstrap_seed=17
        )
        self.assertEqual(result["games_completed"], 2)
        self.assertEqual(result["seat_pair_comparisons"], 4)
        self.assertEqual((result["wins"], result["draws"], result["losses"]), (3, 0, 1))
        self.assertAlmostEqual(result["pairwise_score"], 0.75)
        self.assertTrue(result["score_above_0_50"])
        self.assertAlmostEqual(result["rl_average_finish"], 5 / 3)
        self.assertAlmostEqual(result["opponent_average_finish"], 7 / 3)


class RlSemanticSearchTests(unittest.TestCase):
    def _agents(self, session: GameSession) -> dict[str, object]:
        return {
            player.player_id: build_ai_controller("ai_deterministic")
            for player in session.snapshot().state.players
        }

    def test_phase_boundaries_complete_without_mutating_root(self) -> None:
        for scenario in ("opening", "resource", "build_test", "endgame"):
            with self.subTest(scenario=scenario):
                session = GameSession.from_scenario(scenario, seed=7)
                before = session.snapshot()
                assert before.active_request is not None
                candidates = generate_candidate_actions(before.active_request, before)
                child = advance_to_semantic_boundary(
                    session,
                    candidates[0],
                    continuation_agents=self._agents(session),  # type: ignore[arg-type]
                )
                self.assertEqual(session.snapshot().state, before.state)
                after = child.snapshot()
                if scenario == "endgame":
                    self.assertIsNotNone(after.winner_result)
                self.assertTrue(
                    after.winner_result is not None
                    or after.state.phase != before.state.phase
                    or after.active_request is not None
                    and after.active_request.player_id != before.active_request.player_id
                    or before.state.phase == "auction"
                    and after.state.auction_state is not None
                    and after.state.auction_state.active_plant_price is None
                )

    def test_full_width_depth_two_and_complete_fallback(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)
        snapshot = session.snapshot()
        assert snapshot.active_request is not None
        observation = build_public_observation(snapshot.state, snapshot.active_request)
        states, state_names = encode_state_features(observation)
        candidates = generate_candidate_actions(snapshot.active_request, snapshot)
        actions, action_names = encode_action_features(observation, candidates[0])
        model = NumpyRlPolicyQNetwork(
            len(states),
            len(actions),
            hidden_dims=(16, 8, 8),
            state_feature_names=state_names,
            action_feature_names=action_names,
        )
        complete = FullActionSemanticSearcher(
            model, SearchConfig(max_search_nodes=512)
        ).search(session)
        fallback = FullActionSemanticSearcher(
            model, SearchConfig(max_search_nodes=len(candidates))
        ).search(session)
        self.assertEqual(len(complete.q_values), len(candidates))
        self.assertEqual(complete.depth_used, 2)
        self.assertTrue(complete.depth_2_completed)
        self.assertEqual(len(fallback.q_values), len(candidates))
        self.assertEqual(fallback.depth_used, 1)
        self.assertFalse(fallback.depth_2_completed)

    def test_sibling_forks_share_hidden_deck_and_are_isolated(self) -> None:
        session = GameSession.from_scenario("opening", seed=19)
        first = session.fork().snapshot().state
        second = session.fork().snapshot().state
        self.assertEqual(first.power_plant_draw_stack, second.power_plant_draw_stack)
        self.assertEqual(session.snapshot().state, first)

        snapshot = session.snapshot()
        assert snapshot.active_request is not None
        public_before = build_public_observation(snapshot.state, snapshot.active_request)
        reversed_state = replace(
            snapshot.state,
            power_plant_draw_stack=tuple(reversed(snapshot.state.power_plant_draw_stack)),
        )
        public_after = build_public_observation(reversed_state, snapshot.active_request)
        self.assertEqual(public_before.payload, public_after.payload)

    def test_boundary_guard_auction_pass_and_pending_discard(self) -> None:
        opening = GameSession.from_scenario("opening", seed=7)
        snapshot = opening.snapshot()
        assert snapshot.active_request is not None
        start = generate_candidate_actions(snapshot.active_request, snapshot)[0]
        with self.assertRaisesRegex(ModelValidationError, "exceeded 1"):
            advance_to_semantic_boundary(
                opening,
                start,
                continuation_agents=self._agents(opening),  # type: ignore[arg-type]
                max_actions=1,
            )

        round_two_state = replace(snapshot.state, round_number=2)
        round_two = GameSession(
            round_two_state, default_seat_agents(round_two_state.config)
        )
        round_two_snapshot = round_two.snapshot()
        assert round_two_snapshot.active_request is not None
        pass_candidate = next(
            candidate
            for candidate in generate_candidate_actions(
                round_two_snapshot.active_request, round_two_snapshot
            )
            if candidate.intent.intent_type == "auction_pass"
        )
        passed = advance_to_semantic_boundary(
            round_two,
            pass_candidate,
            continuation_agents=self._agents(round_two),  # type: ignore[arg-type]
        ).snapshot()
        assert passed.active_request is not None
        self.assertNotEqual(
            passed.active_request.player_id, round_two_snapshot.active_request.player_id
        )

        resource = GameSession.from_scenario("resource", seed=7).snapshot().state
        pending_state = add_power_plant_to_player(resource, "p1", 7)
        pending_state = add_power_plant_to_player(pending_state, "p1", 11)
        self.assertIsNotNone(pending_state.pending_decision)
        pending = GameSession(pending_state, default_seat_agents(pending_state.config))
        pending_snapshot = pending.snapshot()
        assert pending_snapshot.active_request is not None
        discard = generate_candidate_actions(
            pending_snapshot.active_request, pending_snapshot
        )[0]
        resolved = advance_to_semantic_boundary(
            pending,
            discard,
            continuation_agents=self._agents(pending),  # type: ignore[arg-type]
        ).snapshot()
        self.assertIsNone(resolved.state.pending_decision)

    def test_child_actor_q_slots_are_remapped_by_player_id(self) -> None:
        class SlotValueModel:
            state_dim: int = 0
            action_dim: int = 0
            state_feature_names: tuple[str, ...] = ()
            action_feature_names: tuple[str, ...] = ()

            def predict_one(
                self, state_features: np.ndarray, action_features: np.ndarray
            ) -> PolicyQPredictions:
                count = len(action_features)
                q_row = np.asarray([0.9, 0.2, -0.7, 0.0, 0.0, 0.0], dtype=np.float32)
                return PolicyQPredictions(
                    policy_logits=np.zeros(count, dtype=np.float32),
                    policy_probabilities=np.full(count, 1.0 / count, dtype=np.float32),
                    q_values=np.tile(q_row, (count, 1)),
                )

        session = GameSession.from_scenario("opening", seed=7)
        snapshot = session.snapshot()
        assert snapshot.active_request is not None
        root_observation = build_public_observation(snapshot.state, snapshot.active_request)
        root_state_features, root_state_names = encode_state_features(root_observation)
        root_slots = player_slot_ids(root_observation)
        candidates = generate_candidate_actions(snapshot.active_request, snapshot)
        root_action_features, root_action_names = encode_action_features(
            root_observation, candidates[0]
        )
        agents = self._agents(session)
        slot_model = SlotValueModel()
        slot_model.state_dim = len(root_state_features)
        slot_model.action_dim = len(root_action_features)
        slot_model.state_feature_names = root_state_names
        slot_model.action_feature_names = root_action_names
        result = FullActionSemanticSearcher(
            slot_model,  # type: ignore[arg-type]
            SearchConfig(adaptive_depth_2=False),
        ).search(session)
        slot_values = (0.9, 0.2, -0.7)
        for candidate_index, candidate in enumerate(candidates):
            child = advance_to_semantic_boundary(
                session,
                candidate,
                continuation_agents=agents,  # type: ignore[arg-type]
            ).snapshot()
            assert child.active_request is not None
            child_slots = player_slot_ids(
                build_public_observation(child.state, child.active_request)
            )
            expected = [slot_values[child_slots.index(player_id)] for player_id in root_slots]
            np.testing.assert_allclose(
                result.q_values[candidate_index][:3], expected, atol=1e-6
            )


class RlDatasetTrainingControllerTests(unittest.TestCase):
    def test_balanced_search_sampling_is_exact_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap_dataset = root / "bootstrap"
            generate_rl_dataset(
                bootstrap_dataset,
                games=2,
                seed_start=8051,
                search_fraction=0.0,
                split_fractions=(1.0, 0.0, 0.0),
                target_shard_size_bytes=64 * 1024,
            )
            bootstrap_checkpoint = root / "bootstrap.npz"
            train_rl_model(
                bootstrap_dataset,
                bootstrap_checkpoint,
                epochs=1,
                batch_decisions=64,
                hidden_dims=(16, 8, 8),
                q_search_weight=0.0,
            )
            search_dataset = root / "search"
            generate_rl_dataset(
                search_dataset,
                games=2,
                seed_start=8061,
                target_checkpoint=bootstrap_checkpoint,
                search_fraction=0.2,
                split_fractions=(1.0, 0.0, 0.0),
                target_shard_size_bytes=64 * 1024,
            )

            def sampled(seed: int) -> tuple[int, int, list[tuple[float, ...]]]:
                searched_count = non_search_count = 0
                anchors: list[tuple[float, ...]] = []
                for arrays in _iter_array_batches(
                    search_dataset,
                    "train",
                    batch_decisions=64,
                    shuffle_seed=seed,
                    training_sampling="balanced_search",
                ):
                    batch_searched = int(np.sum(arrays["searched"]))
                    self.assertEqual(batch_searched * 2, len(arrays["states"]))
                    searched_count += batch_searched
                    non_search_count += len(arrays["states"]) - batch_searched
                    anchors.extend(
                        tuple(float(value) for value in row)
                        for row, is_searched in zip(
                            arrays["states"], arrays["searched"]
                        )
                        if not is_searched
                    )
                return searched_count, non_search_count, anchors

            first = sampled(17)
            duplicate = sampled(17)
            next_epoch = sampled(18)
            records = load_rl_dataset_records(search_dataset, split="train")
            expected_searched = sum(
                bool(row["has_search_targets"]) for row in records
            )
            self.assertEqual(first[0], expected_searched)
            self.assertEqual(first[0], first[1])
            self.assertEqual(first, duplicate)
            self.assertNotEqual(first[2], next_epoch[2])

            gated_checkpoint = root / "gated.npz"
            train_rl_model(
                search_dataset,
                gated_checkpoint,
                init_checkpoint=bootstrap_checkpoint,
                epochs=1,
                batch_decisions=64,
                policy_target_mode="advantage_gate",
                improved_action_weight=0.75,
                min_search_advantage=0.1,
                training_sampling="balanced_search",
            )
            metadata = NumpyRlPolicyQNetwork.load(gated_checkpoint).metadata
            epoch_counts = metadata["training_sampling_epoch_counts"][0]
            self.assertEqual(epoch_counts["searched"], expected_searched)
            self.assertEqual(epoch_counts["non_search"], expected_searched)
            self.assertEqual(metadata["policy_target_mode"], "advantage_gate")

    def test_streamed_dataset_training_and_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset"
            summary = generate_rl_dataset(
                dataset_path,
                games=3,
                seed_start=8101,
                search_fraction=0.0,
                split_fractions=(1.0, 0.0, 0.0),
                target_shard_size_bytes=64 * 1024,
            )
            duplicate_path = Path(directory) / "dataset_duplicate"
            generate_rl_dataset(
                duplicate_path,
                games=3,
                seed_start=8101,
                search_fraction=0.0,
                split_fractions=(1.0, 0.0, 0.0),
                target_shard_size_bytes=64 * 1024,
            )
            records = load_rl_dataset_records(dataset_path, split="train")
            manifest = load_rl_dataset_metadata(dataset_path)
            duplicate_records = load_rl_dataset_records(
                duplicate_path, split="train"
            )
            verification = verify_rl_dataset_manifest(dataset_path)
            checkpoint = Path(directory) / "model.npz"
            training = train_rl_model(
                dataset_path,
                checkpoint,
                epochs=1,
                batch_decisions=128,
                hidden_dims=(32, 16, 16),
                q_search_weight=0.0,
            )
            controller = NnRlBasedAiController(checkpoint)
            session = GameSession.from_scenario("opening", seed=7)
            snapshot = session.snapshot()
            assert snapshot.active_request is not None
            intent = controller.choose_intent(snapshot.active_request, snapshot)
            result = session.submit_intent(intent, auto_advance=False)

        self.assertEqual(len(records), summary.decisions)
        self.assertEqual(records, duplicate_records)
        self.assertEqual(len(summary.example_jsonl_paths), 3)
        self.assertEqual(verification["examples"], 3)
        self.assertEqual(verification["games"], 3)
        self.assertEqual(verification["rows"], summary.decisions)
        self.assertGreater(len(legal_region_sets("germany", 3)), 3)
        self.assertGreaterEqual(
            len(manifest["generation"]["resolved_region_sets"]), 2
        )
        self.assertEqual(
            manifest["generation"]["region_selection"],
            "seed_cycle_all_legal_sets",
        )
        self.assertTrue(all(len(row["terminal_rank_values"]) == 6 for row in records))
        self.assertTrue(
            all(
                row["player_ids_in_slot_order"][0] == row["player_id"]
                for row in records
            )
        )
        self.assertTrue(all(len(row["candidate_action_features"]) > 0 for row in records))
        self.assertEqual(training.train_decisions, summary.decisions)
        self.assertNotEqual(result.event_log[-1].level, "error")
        self.assertIsInstance(
            build_ai_controller("ai_nn_rl_based_v1"), NnRlBasedAiController
        )

    def test_controller_stable_tie_break_and_validation_errors(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)
        snapshot = session.snapshot()
        assert snapshot.active_request is not None
        observation = build_public_observation(snapshot.state, snapshot.active_request)
        state_features, state_names = encode_state_features(observation)
        candidates = generate_candidate_actions(snapshot.active_request, snapshot)
        action_features, action_names = encode_action_features(observation, candidates[0])
        model = NumpyRlPolicyQNetwork(
            len(state_features),
            len(action_features),
            hidden_dims=(8, 8, 8),
            state_feature_names=state_names,
            action_feature_names=action_names,
        )
        for parameter in model.parameters.values():
            parameter.fill(0.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tie.npz"
            model.save(
                checkpoint,
                metadata={"supported_map": "germany", "supported_player_count": 3},
            )
            chosen = NnRlBasedAiController(checkpoint).choose_intent(
                snapshot.active_request, snapshot
            )
            self.assertEqual(chosen.to_dict(), candidates[0].intent.to_dict())
            with self.assertRaisesRegex(ModelValidationError, "cannot load"):
                NnRlBasedAiController(Path(directory) / "missing.npz").choose_intent(
                    snapshot.active_request, snapshot
                )
            test_map = GameSession.from_scenario("build_test", seed=7).snapshot()
            assert test_map.active_request is not None
            with self.assertRaisesRegex(ModelValidationError, "supports only"):
                NnRlBasedAiController(checkpoint).choose_intent(
                    test_map.active_request, test_map
                )


if __name__ == "__main__":
    unittest.main()
