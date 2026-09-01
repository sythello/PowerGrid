from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from powergrid.ai import NnRankValueAiController, build_ai_controller
from powergrid.ai.nn_rank_value.candidates import generate_candidate_actions
from powergrid.ai.nn_rank_value.dataset import (
    DATASET_SPLITS,
    assign_game_split,
    generate_rank_value_dataset,
    iter_parquet_batches,
    load_dataset_metadata,
    load_dataset_records,
    verify_dataset_manifest,
)
from powergrid.ai.nn_rank_value.model import NumpyRankValueNetwork
from powergrid.ai.nn_rank_value.observation import (
    build_public_observation,
    encode_action_features,
    encode_state_features,
)
from powergrid.session import GameSession, GuiIntent


class NnObservationAndCandidateTests(unittest.TestCase):
    def test_public_features_exclude_seed_and_hidden_deck_order(self) -> None:
        snapshot = GameSession.from_scenario("opening", seed=17).snapshot()
        assert snapshot.active_request is not None
        state = snapshot.state
        hidden_changed = replace(
            state,
            config=replace(state.config, seed=123456),
            power_plant_draw_stack=tuple(reversed(state.power_plant_draw_stack)),
            power_plant_bottom_stack=tuple(reversed(state.power_plant_bottom_stack)),
        )

        first = build_public_observation(state, snapshot.active_request)
        second = build_public_observation(hidden_changed, snapshot.active_request)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(encode_state_features(first), encode_state_features(second))

    def test_every_generated_scenario_candidate_is_accepted_by_a_fork(self) -> None:
        for scenario in ("opening", "resource", "build_test", "endgame"):
            with self.subTest(scenario=scenario):
                session = GameSession.from_scenario(scenario, seed=7)
                snapshot = session.snapshot()
                assert snapshot.active_request is not None
                candidates = generate_candidate_actions(snapshot.active_request, snapshot)
                self.assertTrue(candidates)
                for candidate in candidates:
                    result = session.fork().submit_intent(
                        candidate.intent,
                        auto_advance=False,
                    )
                    self.assertNotEqual(result.event_log[-1].level, "error")

    def test_unaffordable_minimum_raise_is_not_a_candidate(self) -> None:
        session = GameSession.from_scenario("opening", seed=7)
        request = session.current_request()
        assert request is not None
        plant_price = int(
            next(
                action.payload["plant_price"]
                for action in request.legal_actions
                if action.action_type == "auction_start"
            )
        )
        session.submit_intent(
            GuiIntent.auction_start(request.player_id, plant_price, 49),
            auto_advance=False,
        )
        request = session.current_request()
        assert request is not None
        session.submit_intent(
            GuiIntent.auction_bid(request.player_id, 50),
            auto_advance=False,
        )
        snapshot = session.snapshot()
        assert snapshot.active_request is not None

        candidates = generate_candidate_actions(snapshot.active_request, snapshot)

        self.assertEqual(
            [candidate.intent.intent_type for candidate in candidates],
            ["auction_pass"],
        )

    def test_session_fork_is_isolated_and_preserves_turn_cursor(self) -> None:
        session = GameSession.from_scenario("resource", seed=7)
        before = session.snapshot()
        assert before.active_request is not None
        candidate = generate_candidate_actions(before.active_request, before)[0]
        fork = session.fork()

        fork.submit_intent(candidate.intent, auto_advance=False)

        self.assertEqual(session.snapshot().state, before.state)
        self.assertEqual(session.current_request(), before.active_request)
        self.assertNotEqual(fork.snapshot().state, before.state)


class NnModelDatasetAndControllerTests(unittest.TestCase):
    def test_numpy_mlp_learns_and_checkpoint_round_trips(self) -> None:
        rng = np.random.default_rng(19)
        states = rng.normal(size=(256, 5)).astype(np.float32)
        actions = rng.normal(size=(256, 2)).astype(np.float32)
        signal = states[:, 0] - actions[:, 0]
        wins = (signal > 0).astype(np.float32)
        ranks = np.tanh(signal).astype(np.float32)
        model = NumpyRankValueNetwork(5, 2, hidden_dims=(16, 8), seed=19)
        model.set_normalization(np.concatenate([states, actions], axis=1))
        before = model.evaluate(states, actions, wins, ranks)["loss"]
        for _ in range(80):
            model.train_batch(states, actions, wins, ranks, learning_rate=3e-3)
        after = model.evaluate(states, actions, wins, ranks)["loss"]
        self.assertLess(after, before * 0.4)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            expected = model.predict(states[:8], actions[:8])
            model.save(path)
            actual = NumpyRankValueNetwork.load(path).predict(states[:8], actions[:8])
        np.testing.assert_array_equal(expected.win_probability, actual.win_probability)
        np.testing.assert_array_equal(expected.rank_value, actual.rank_value)

    def test_dataset_generation_attaches_terminal_rank_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset"
            summary = generate_rank_value_dataset(
                path,
                games=3,
                seed_start=313,
                behavior_controllers=("ai_deterministic",),
                target_shard_size_bytes=64 * 1024,
                split_fractions=(0.5, 0.25, 0.25),
                split_seed=9,
            )
            records = load_dataset_records(path)
            manifest = load_dataset_metadata(path)
            verification = verify_dataset_manifest(path)

        self.assertEqual(len(records), summary.behavior_samples)
        self.assertTrue(records)
        self.assertEqual({row["rank_value"] for row in records} - {-1.0, 0.0, 1.0}, set())
        self.assertEqual({row["is_winner"] for row in records} - {0, 1}, set())
        self.assertEqual(len(summary.example_jsonl_paths), 3)
        self.assertEqual(verification, {"shards": summary.shards, "examples": 3})
        self.assertEqual(manifest["storage"]["row_group_unit"], "complete_game")

    def test_dataset_split_is_stable_and_game_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset"
            generate_rank_value_dataset(
                path,
                games=4,
                seed_start=421,
                behavior_controllers=("ai_deterministic",),
                split_fractions=(0.5, 0.25, 0.25),
                split_seed=13,
            )
            split_game_ids = {
                split: {
                    str(game_id)
                    for batch in iter_parquet_batches(
                        path,
                        split,
                        batch_size=1024,
                        columns=("game_id",),
                    )
                    for game_id in batch.column(0).to_pylist()
                }
                for split in DATASET_SPLITS
            }

        self.assertFalse(split_game_ids["train"] & split_game_ids["validation"])
        self.assertFalse(split_game_ids["train"] & split_game_ids["test"])
        self.assertFalse(split_game_ids["validation"] & split_game_ids["test"])
        for split, game_ids in split_game_ids.items():
            for game_id in game_ids:
                self.assertEqual(
                    assign_game_split(
                        game_id,
                        split_fractions=(0.5, 0.25, 0.25),
                        split_seed=13,
                    ),
                    split,
                )

    def test_registered_controller_scores_and_returns_a_legal_candidate(self) -> None:
        snapshot = GameSession.from_scenario("opening", seed=7).snapshot()
        assert snapshot.active_request is not None
        observation = build_public_observation(snapshot.state, snapshot.active_request)
        state_features, state_names = encode_state_features(observation)
        candidates = generate_candidate_actions(snapshot.active_request, snapshot)
        action_features, action_names = encode_action_features(observation, candidates[0])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.npz"
            NumpyRankValueNetwork(
                len(state_features),
                len(action_features),
                hidden_dims=(8, 4),
                seed=3,
                state_feature_names=state_names,
                action_feature_names=action_names,
            ).save(checkpoint)
            controller = NnRankValueAiController(checkpoint)
            intent = controller.choose_intent(snapshot.active_request, snapshot)

        self.assertIsInstance(build_ai_controller("ai_nn_rank_value_v1"), NnRankValueAiController)
        result = GameSession.from_scenario("opening", seed=7).submit_intent(
            intent,
            auto_advance=False,
        )
        self.assertNotEqual(result.event_log[-1].level, "error")


if __name__ == "__main__":
    unittest.main()
