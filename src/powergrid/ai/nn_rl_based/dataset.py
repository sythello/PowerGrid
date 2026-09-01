from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator

from ...model import GameConfig, ModelValidationError, SeatConfig, legal_region_sets
from ...session import GameSession
from ...session_types import GameSnapshot
from .. import build_ai_controller
from ..nn_rank_value.candidates import find_candidate_for_intent, generate_candidate_actions
from ..nn_rank_value.dataset import (
    DATASET_SPLITS,
    DEFAULT_SPLIT_FRACTIONS,
    DEFAULT_TARGET_SHARD_SIZE_BYTES,
    StreamingParquetDatasetWriter,
    assign_game_split,
    import_pyarrow,
    sha256_file,
)
from ..nn_rank_value.observation import (
    ACTION_FEATURE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    build_public_observation,
    encode_action_features,
    encode_state_features,
    player_slot_ids,
)
from .model import MAX_PLAYERS, NumpyRlPolicyQNetwork
from .search import (
    FullActionSemanticSearcher,
    SearchConfig,
    pad_player_values,
    terminal_rank_values,
)


DATASET_FORMAT_NAME = "powergrid.nn_rl_based.parquet"
DATASET_FORMAT_VERSION = 1
EXAMPLE_GAME_COUNT = 3


@dataclass(frozen=True)
class RlDatasetProgress:
    games_completed: int
    games_total: int
    decisions: int
    searched_decisions: int
    search_nodes: int
    parquet_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True)
class RlDatasetSummary:
    output_path: Path
    manifest_path: Path
    games: int
    decisions: int
    searched_decisions: int
    search_nodes: int
    depth_2_completed: int
    shards: int
    parquet_bytes: int
    state_dim: int
    action_dim: int
    split_games: dict[str, int]
    split_rows: dict[str, int]
    example_jsonl_paths: tuple[Path, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class _GameTask:
    game_offset: int
    seed: int
    map_id: str
    player_count: int
    selected_regions: tuple[str, ...]
    behavior_controller: str
    target_checkpoint: str
    search_fraction: float
    split_seed: int
    search_config: SearchConfig
    max_actions: int


@dataclass(frozen=True)
class _CompletedGame:
    game_offset: int
    game_id: str
    records: list[dict[str, Any]]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    selected_regions: tuple[str, ...]
    searched_decisions: int
    search_nodes: int
    depth_2_completed: int


_TARGET_MODEL_CACHE: dict[str, NumpyRlPolicyQNetwork] = {}


def generate_rl_dataset(
    output_path: str | Path,
    *,
    games: int,
    seed_start: int = 1,
    map_id: str = "germany",
    player_count: int = 3,
    behavior_controller: str = "ai_deterministic",
    continuation_controller: str = "ai_deterministic",
    selected_regions: tuple[str, ...] = (),
    region_sets: tuple[tuple[str, ...], ...] = (),
    target_checkpoint: str | Path | None = None,
    search_fraction: float = 0.0,
    search_depth: int = 1,
    adaptive_depth_2: bool = True,
    max_search_nodes: int = 512,
    max_boundary_actions: int = 128,
    leaf_policy: str = "deterministic",
    search_policy_mix: float = 0.5,
    search_temperature: float = 0.25,
    max_actions_per_game: int = 5000,
    target_shard_size_bytes: int = DEFAULT_TARGET_SHARD_SIZE_BYTES,
    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
    split_seed: int = 0,
    workers: int = 1,
    progress_callback: Callable[[RlDatasetProgress], None] | None = None,
) -> RlDatasetSummary:
    if games <= 0:
        raise ValueError("games must be positive")
    if map_id != "germany" or player_count != 3:
        raise ValueError("ai_nn_rl_based_v1 datasets support Germany with 3 players")
    if behavior_controller != "ai_deterministic":
        raise ValueError("ai_nn_rl_based_v1 requires canonical ai_deterministic behavior")
    if continuation_controller != "ai_deterministic":
        raise ValueError(
            "ai_nn_rl_based_v1 requires canonical ai_deterministic continuation"
        )
    if not 0.0 <= search_fraction <= 1.0:
        raise ValueError("search_fraction must be between 0 and 1")
    if search_fraction > 0.0 and target_checkpoint is None:
        raise ValueError("searched datasets require --target-checkpoint")
    if workers <= 0 or target_shard_size_bytes <= 0 or max_actions_per_game <= 0:
        raise ValueError("workers, shard size, and max actions must be positive")
    if selected_regions and region_sets:
        raise ValueError("selected_regions and region_sets may not both be supplied")
    all_legal_region_sets = legal_region_sets(map_id, player_count)
    requested_region_sets = tuple(
        tuple(sorted(values)) for values in region_sets
    )
    if selected_regions:
        resolved_task_region_sets = (tuple(sorted(selected_regions)),)
        region_selection = "fixed_explicit"
    elif requested_region_sets:
        resolved_task_region_sets = requested_region_sets
        region_selection = "seed_cycle_explicit_sets"
    else:
        resolved_task_region_sets = all_legal_region_sets
        region_selection = "seed_cycle_all_legal_sets"
    invalid_region_sets = tuple(
        values for values in resolved_task_region_sets if values not in all_legal_region_sets
    )
    if invalid_region_sets:
        raise ValueError(f"invalid Germany/3-player region set(s): {invalid_region_sets!r}")
    fractions = _validate_split_fractions(split_fractions)
    checkpoint = str(Path(target_checkpoint).resolve()) if target_checkpoint else ""
    if checkpoint and not Path(checkpoint).exists():
        raise FileNotFoundError(f"target checkpoint does not exist: {checkpoint}")
    search_config = SearchConfig(
        depth=search_depth,
        adaptive_depth_2=adaptive_depth_2,
        max_search_nodes=max_search_nodes,
        max_boundary_actions=max_boundary_actions,
        continuation_controller=continuation_controller,
        leaf_policy=leaf_policy,
        search_policy_mix=search_policy_mix,
        search_temperature=search_temperature,
    )

    pa, parquet = import_pyarrow()
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    examples_dir = output / "examples"
    examples_dir.mkdir()
    started = time.perf_counter()
    tasks = tuple(
        _GameTask(
            game_offset=offset,
            seed=seed_start + offset,
            map_id=map_id,
            player_count=player_count,
            selected_regions=resolved_task_region_sets[
                (seed_start + offset) % len(resolved_task_region_sets)
            ],
            behavior_controller=behavior_controller,
            target_checkpoint=checkpoint,
            search_fraction=search_fraction,
            split_seed=split_seed,
            search_config=search_config,
            max_actions=max_actions_per_game,
        )
        for offset in range(games)
    )
    writer: Any | None = None
    schema: Any | None = None
    state_names: tuple[str, ...] | None = None
    action_names: tuple[str, ...] | None = None
    split_games = {split: 0 for split in DATASET_SPLITS}
    split_rows = {split: 0 for split in DATASET_SPLITS}
    total_decisions = 0
    total_searched = 0
    total_nodes = 0
    total_depth_2 = 0
    resolved_region_sets: set[tuple[str, ...]] = set()
    example_paths: list[Path] = []
    example_descriptors: list[dict[str, Any]] = []
    try:
        for completed_index, completed in enumerate(
            _iter_completed_games(tasks, workers=workers), start=1
        ):
            if state_names is None:
                state_names = completed.state_names
                action_names = completed.action_names
                schema = _build_schema(pa, len(state_names), len(action_names))
                writer = StreamingParquetDatasetWriter(
                    output,
                    schema,
                    target_shard_size_bytes=target_shard_size_bytes,
                    parquet_module=parquet,
                )
            elif (state_names, action_names) != (
                completed.state_names,
                completed.action_names,
            ):
                raise ModelValidationError("RL feature schema changed during generation")
            assert writer is not None and schema is not None
            split = assign_game_split(
                completed.game_id,
                split_fractions=fractions,
                split_seed=split_seed,
            )
            table = _records_to_table(pa, completed.records, schema)
            writer.write_game(split, completed.records, table)
            split_games[split] += 1
            split_rows[split] += len(completed.records)
            total_decisions += len(completed.records)
            total_searched += completed.searched_decisions
            total_nodes += completed.search_nodes
            total_depth_2 += completed.depth_2_completed
            resolved_region_sets.add(completed.selected_regions)
            if completed.game_offset < EXAMPLE_GAME_COUNT:
                path = examples_dir / f"game-{completed.game_offset + 1:02d}-{completed.game_id}.jsonl"
                _write_jsonl(path, completed.records)
                example_paths.append(path)
                example_descriptors.append(
                    {
                        "path": path.relative_to(output).as_posix(),
                        "game_id": completed.game_id,
                        "rows": len(completed.records),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    RlDatasetProgress(
                        games_completed=completed_index,
                        games_total=games,
                        decisions=total_decisions,
                        searched_decisions=total_searched,
                        search_nodes=total_nodes,
                        parquet_bytes=writer.current_bytes(),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
    finally:
        if writer is not None:
            writer.close()

    assert writer is not None and schema is not None
    assert state_names is not None and action_names is not None
    split_payload: dict[str, Any] = {}
    all_descriptors = []
    for split in DATASET_SPLITS:
        descriptors = writer.writers[split].descriptors
        all_descriptors.extend(descriptors)
        split_payload[split] = {
            "games": split_games[split],
            "rows": split_rows[split],
            "bytes": sum(item.bytes for item in descriptors),
            "shards": [item.to_dict() for item in descriptors],
        }
    parquet_bytes = sum(item.bytes for item in all_descriptors)
    elapsed = time.perf_counter() - started
    manifest = {
        "format_name": DATASET_FORMAT_NAME,
        "format_version": DATASET_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        "state_dim": len(state_names),
        "action_dim": len(action_names),
        "max_players": MAX_PLAYERS,
        "state_feature_names": list(state_names),
        "action_feature_names": list(action_names),
        "parquet_schema": [_field_payload(field) for field in schema],
        "storage": {
            "compression": "zstd",
            "compression_level": 3,
            "target_shard_size_bytes": target_shard_size_bytes,
            "row_group_unit": "complete_game",
            "parquet_bytes": parquet_bytes,
            "shards": len(all_descriptors),
        },
        "generation": {
            "games": games,
            "decisions": total_decisions,
            "searched_decisions": total_searched,
            "search_nodes": total_nodes,
            "depth_2_completed": total_depth_2,
            "map_id": map_id,
            "player_count": player_count,
            "seed_start": seed_start,
            "behavior_controller": behavior_controller,
            "continuation_controller": continuation_controller,
            "requested_selected_regions": list(selected_regions),
            "requested_region_sets": [list(values) for values in requested_region_sets],
            "region_selection": region_selection,
            "available_region_sets": [
                list(values) for values in resolved_task_region_sets
            ],
            "resolved_region_sets": [list(values) for values in sorted(resolved_region_sets)],
            "search_fraction": search_fraction,
            "search_depth": search_depth,
            "adaptive_depth_2": adaptive_depth_2,
            "max_search_nodes": max_search_nodes,
            "max_boundary_actions": max_boundary_actions,
            "leaf_policy": leaf_policy,
            "search_policy_mix": search_policy_mix,
            "search_temperature": search_temperature,
            "hidden_state_sampling": "single_common_determinization",
            "target_checkpoint": checkpoint,
            "target_checkpoint_sha256": sha256_file(checkpoint) if checkpoint else "",
            "max_actions_per_game": max_actions_per_game,
            "workers": workers,
            "parallel_execution": "bounded_process_pool_with_thread_fallback",
            "max_games_in_flight": max(1, workers * 2),
            "elapsed_seconds": elapsed,
        },
        "split": {
            "algorithm": "sha256_uniform_v1",
            "key": "game_id",
            "seed": split_seed,
            "fractions": dict(zip(DATASET_SPLITS, fractions)),
        },
        "splits": split_payload,
        "example_jsonl": example_descriptors,
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return RlDatasetSummary(
        output_path=output,
        manifest_path=manifest_path,
        games=games,
        decisions=total_decisions,
        searched_decisions=total_searched,
        search_nodes=total_nodes,
        depth_2_completed=total_depth_2,
        shards=len(all_descriptors),
        parquet_bytes=parquet_bytes,
        state_dim=len(state_names),
        action_dim=len(action_names),
        split_games=split_games,
        split_rows=split_rows,
        example_jsonl_paths=tuple(sorted(example_paths)),
        elapsed_seconds=elapsed,
    )


def load_rl_dataset_metadata(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_name") != DATASET_FORMAT_NAME:
        raise ValueError("unsupported RL dataset format")
    if int(manifest.get("format_version", -1)) != DATASET_FORMAT_VERSION:
        raise ValueError("unsupported RL dataset version")
    return manifest


def iter_rl_parquet_batches(
    path: str | Path,
    split: str,
    *,
    batch_size: int,
    shuffle_seed: int | None = None,
    columns: tuple[str, ...] | None = None,
) -> Iterator[Any]:
    if split not in DATASET_SPLITS or batch_size <= 0:
        raise ValueError("invalid split or batch size")
    _, parquet = import_pyarrow()
    root = Path(path)
    manifest = load_rl_dataset_metadata(root)
    paths = [root / item["path"] for item in manifest["splits"][split]["shards"]]
    if shuffle_seed is not None:
        import random

        random.Random(shuffle_seed).shuffle(paths)
    for shard_index, shard_path in enumerate(paths):
        parquet_file = parquet.ParquetFile(shard_path)
        groups = list(range(parquet_file.num_row_groups))
        if shuffle_seed is not None:
            import random

            random.Random((shuffle_seed + 1) * 1_000_003 + shard_index).shuffle(groups)
        for group in groups:
            yield from parquet_file.iter_batches(
                batch_size=batch_size,
                row_groups=[group],
                columns=list(columns) if columns else None,
                use_threads=True,
            )


def load_rl_dataset_records(
    path: str | Path, *, split: str | None = None
) -> list[dict[str, Any]]:
    splits = DATASET_SPLITS if split is None else (split,)
    records: list[dict[str, Any]] = []
    for split_name in splits:
        for batch in iter_rl_parquet_batches(path, split_name, batch_size=4096):
            records.extend(batch.to_pylist())
    if not records:
        raise ValueError("RL dataset contains no records")
    return records


def verify_rl_dataset_manifest(path: str | Path) -> dict[str, int]:
    root = Path(path)
    manifest = load_rl_dataset_metadata(root)
    _, parquet = import_pyarrow()
    checked_shards = 0
    checked_examples = 0
    checked_rows = 0
    checked_games: set[str] = set()
    split_fractions = tuple(
        float(manifest["split"]["fractions"][split]) for split in DATASET_SPLITS
    )
    split_seed = int(manifest["split"]["seed"])
    for split in DATASET_SPLITS:
        rows = 0
        games = 0
        for descriptor in manifest["splits"][split]["shards"]:
            shard_path = root / descriptor["path"]
            if shard_path.stat().st_size != int(descriptor["bytes"]):
                raise ValueError(f"RL shard size mismatch: {shard_path}")
            if sha256_file(shard_path) != descriptor["sha256"]:
                raise ValueError(f"RL shard checksum mismatch: {shard_path}")
            parquet_file = parquet.ParquetFile(shard_path)
            if [_field_payload(field) for field in parquet_file.schema_arrow] != manifest["parquet_schema"]:
                raise ValueError(f"RL shard schema mismatch: {shard_path}")
            if int(descriptor["rows"]) != int(parquet_file.metadata.num_rows):
                raise ValueError(f"RL shard descriptor row mismatch: {shard_path}")
            if int(descriptor["games"]) != int(parquet_file.num_row_groups):
                raise ValueError(f"RL shard descriptor game mismatch: {shard_path}")
            for row_group in range(parquet_file.num_row_groups):
                game_ids = set(
                    parquet_file.read_row_group(
                        row_group, columns=["game_id"]
                    ).column("game_id").to_pylist()
                )
                if len(game_ids) != 1:
                    raise ValueError(
                        f"RL row group must contain one complete game: {shard_path}"
                    )
                game_id = next(iter(game_ids))
                if game_id in checked_games:
                    raise ValueError(f"RL game occurs in multiple row groups: {game_id}")
                expected_split = assign_game_split(
                    game_id,
                    split_fractions=split_fractions,  # type: ignore[arg-type]
                    split_seed=split_seed,
                )
                if expected_split != split:
                    raise ValueError(
                        f"RL game {game_id} is in {split}, expected {expected_split}"
                    )
                checked_games.add(game_id)
            rows += int(parquet_file.metadata.num_rows)
            games += int(parquet_file.num_row_groups)
            checked_shards += 1
        if rows != int(manifest["splits"][split]["rows"]):
            raise ValueError(f"RL split row mismatch: {split}")
        if games != int(manifest["splits"][split]["games"]):
            raise ValueError(f"RL split game mismatch: {split}")
        checked_rows += rows
    for descriptor in manifest["example_jsonl"]:
        example_path = root / descriptor["path"]
        if example_path.stat().st_size != int(descriptor["bytes"]):
            raise ValueError(f"RL example size mismatch: {example_path}")
        if sha256_file(example_path) != descriptor["sha256"]:
            raise ValueError(f"RL example checksum mismatch: {example_path}")
        checked_examples += 1
    return {
        "shards": checked_shards,
        "examples": checked_examples,
        "games": len(checked_games),
        "rows": checked_rows,
    }


def _iter_completed_games(
    tasks: tuple[_GameTask, ...], *, workers: int
) -> Iterator[_CompletedGame]:
    if workers == 1:
        for task in tasks:
            yield _generate_one_game(task)
        return
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
    except (OSError, PermissionError):
        # Some sandboxed runtimes deny the semaphore sysconf probe performed by
        # ProcessPoolExecutor. Preserve the requested bounded parallelism there
        # instead of making the documented --workers command unusable.
        with ThreadPoolExecutor(max_workers=workers) as thread_executor:
            yield from _bounded_executor_results(
                thread_executor, tasks, workers=workers
            )
        return
    with executor:
        yield from _bounded_executor_results(executor, tasks, workers=workers)


def _bounded_executor_results(
    executor: Any, tasks: tuple[_GameTask, ...], *, workers: int
) -> Iterator[_CompletedGame]:
    max_in_flight = max(workers, workers * 2)
    task_iterator = iter(tasks)
    pending: dict[Any, int] = {}
    buffered: dict[int, _CompletedGame] = {}
    next_offset = 0
    exhausted = False
    while pending or buffered or not exhausted:
        while not exhausted and len(pending) + len(buffered) < max_in_flight:
            try:
                task = next(task_iterator)
            except StopIteration:
                exhausted = True
                break
            pending[executor.submit(_generate_one_game, task)] = task.game_offset
        if pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                game_offset = pending.pop(future)
                buffered[game_offset] = future.result()
        while next_offset in buffered:
            yield buffered.pop(next_offset)
            next_offset += 1


def _generate_one_game(task: _GameTask) -> _CompletedGame:
    config = GameConfig(
        map_id=task.map_id,
        players=tuple(
            SeatConfig(f"p{index + 1}", f"Player {index + 1}", controller="human")
            for index in range(task.player_count)
        ),
        seed=task.seed,
        selected_regions=task.selected_regions,
    )
    session = GameSession.new_game(config)
    behavior_agents = {
        player.player_id: build_ai_controller(task.behavior_controller)
        for player in session.snapshot().state.players
    }
    target_model = _load_target_model(task.target_checkpoint) if task.target_checkpoint else None
    searcher = (
        FullActionSemanticSearcher(target_model, task.search_config)
        if target_model is not None
        else None
    )
    game_id = f"rl-{task.map_id}-{task.player_count}p-{task.seed:010d}"
    records: list[dict[str, Any]] = []
    state_names: tuple[str, ...] | None = None
    action_names: tuple[str, ...] | None = None
    searched_count = 0
    search_nodes = 0
    depth_2_count = 0
    for decision_index in range(task.max_actions):
        snapshot = session.advance_until_blocked()
        if snapshot.winner_result is not None:
            terminal = terminal_rank_values(snapshot)
            for record in records:
                values, _ = pad_player_values(
                    tuple(record["player_ids_in_slot_order"]), terminal
                )
                record["terminal_rank_values"] = values
            assert state_names is not None and action_names is not None
            return _CompletedGame(
                game_offset=task.game_offset,
                game_id=game_id,
                records=records,
                state_names=state_names,
                action_names=action_names,
                selected_regions=tuple(snapshot.state.selected_regions),
                searched_decisions=searched_count,
                search_nodes=search_nodes,
                depth_2_completed=depth_2_count,
            )
        request = snapshot.active_request
        if request is None:
            raise ModelValidationError(f"RL game {game_id} stopped without request")
        observation = build_public_observation(snapshot.state, request)
        state_features, current_state_names = encode_state_features(observation)
        slot_ids = player_slot_ids(observation)
        candidates = generate_candidate_actions(request, snapshot)
        action_rows: list[list[float]] = []
        current_action_names: tuple[str, ...] | None = None
        for candidate in candidates:
            features, names = encode_action_features(observation, candidate)
            action_rows.append(features)
            current_action_names = current_action_names or names
            if names != current_action_names:
                raise ModelValidationError("RL action feature schema changed within decision")
        state_names = state_names or current_state_names
        action_names = action_names or tuple(current_action_names or ())
        if current_state_names != state_names or tuple(current_action_names or ()) != action_names:
            raise ModelValidationError("RL feature schema changed within game")
        chosen_intent = behavior_agents[request.player_id].choose_intent(request, snapshot)
        chosen = find_candidate_for_intent(candidates, chosen_intent)
        if chosen is None:
            raise ModelValidationError(
                f"teacher action outside candidate set in {game_id} decision {decision_index}"
            )
        teacher_index = next(index for index, item in enumerate(candidates) if item.key == chosen.key)
        do_search = bool(
            searcher is not None
            and _should_search(game_id, decision_index, task.split_seed, task.search_fraction)
        )
        search_result = searcher.search(session) if do_search and searcher is not None else None
        if search_result is not None:
            if search_result.player_ids != slot_ids:
                raise ModelValidationError("search root player slots do not match dataset observation")
            if len(search_result.q_values) != len(candidates):
                raise ModelValidationError("search did not label every root candidate")
            searched_count += 1
            search_nodes += search_result.nodes_evaluated
            depth_2_count += int(search_result.depth_2_completed)
        player_mask = [True] * len(slot_ids) + [False] * (MAX_PLAYERS - len(slot_ids))
        records.append(
            {
                "format_version": DATASET_FORMAT_VERSION,
                "game_id": game_id,
                "seed": task.seed,
                "decision_index": decision_index,
                "player_id": request.player_id,
                "phase": request.phase,
                "decision_type": request.decision_type,
                "behavior_controller": task.behavior_controller,
                "continuation_controller": task.search_config.continuation_controller,
                "selected_regions": list(snapshot.state.selected_regions),
                "player_ids_in_slot_order": list(slot_ids),
                "player_mask": player_mask,
                "state_features": state_features,
                "candidate_jsons": [
                    json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":"))
                    for item in candidates
                ],
                "candidate_action_features": action_rows,
                "teacher_action_index": teacher_index,
                "terminal_rank_values": [0.0] * MAX_PLAYERS,
                "has_search_targets": search_result is not None,
                "search_q_values": (
                    [list(row) for row in search_result.q_values]
                    if search_result is not None
                    else []
                ),
                "search_depth_used": search_result.depth_used if search_result else 0,
                "search_nodes_evaluated": search_result.nodes_evaluated if search_result else 0,
                "depth_2_completed": search_result.depth_2_completed if search_result else False,
            }
        )
        result = session.submit_intent(chosen_intent, auto_advance=False)
        if result.event_log and result.event_log[-1].level == "error":
            raise ModelValidationError(
                f"teacher produced invalid action in {game_id}: {result.event_log[-1].message}"
            )
    raise ModelValidationError(f"RL game {game_id} exceeded {task.max_actions} actions")


def _load_target_model(path: str) -> NumpyRlPolicyQNetwork:
    model = _TARGET_MODEL_CACHE.get(path)
    if model is None:
        model = NumpyRlPolicyQNetwork.load(path)
        if (
            model.max_players != MAX_PLAYERS
            or model.metadata.get("supported_map") != "germany"
            or int(model.metadata.get("supported_player_count", -1)) != 3
        ):
            raise ModelValidationError("target RL checkpoint support metadata is incompatible")
        _TARGET_MODEL_CACHE[path] = model
    return model


def _should_search(game_id: str, decision_index: int, seed: int, fraction: float) -> bool:
    if fraction <= 0.0:
        return False
    digest = hashlib.sha256(f"{seed}:{game_id}:{decision_index}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < fraction


def _build_schema(pa: Any, state_dim: int, action_dim: int) -> Any:
    player_vector = pa.list_(pa.float32(), MAX_PLAYERS)
    action_vector = pa.list_(pa.float32(), action_dim)
    return pa.schema(
        [
            pa.field("format_version", pa.int16(), nullable=False),
            pa.field("game_id", pa.string(), nullable=False),
            pa.field("seed", pa.int64(), nullable=False),
            pa.field("decision_index", pa.int32(), nullable=False),
            pa.field("player_id", pa.string(), nullable=False),
            pa.field("phase", pa.string(), nullable=False),
            pa.field("decision_type", pa.string(), nullable=False),
            pa.field("behavior_controller", pa.string(), nullable=False),
            pa.field("continuation_controller", pa.string(), nullable=False),
            pa.field("selected_regions", pa.list_(pa.string()), nullable=False),
            pa.field(
                "player_ids_in_slot_order", pa.list_(pa.string()), nullable=False
            ),
            pa.field("player_mask", pa.list_(pa.bool_(), MAX_PLAYERS), nullable=False),
            pa.field("state_features", pa.list_(pa.float32(), state_dim), nullable=False),
            pa.field("candidate_jsons", pa.list_(pa.string()), nullable=False),
            pa.field("candidate_action_features", pa.list_(action_vector), nullable=False),
            pa.field("teacher_action_index", pa.int16(), nullable=False),
            pa.field("terminal_rank_values", player_vector, nullable=False),
            pa.field("has_search_targets", pa.bool_(), nullable=False),
            pa.field("search_q_values", pa.list_(player_vector), nullable=False),
            pa.field("search_depth_used", pa.int8(), nullable=False),
            pa.field("search_nodes_evaluated", pa.int32(), nullable=False),
            pa.field("depth_2_completed", pa.bool_(), nullable=False),
        ]
    )


def _records_to_table(pa: Any, records: list[dict[str, Any]], schema: Any) -> Any:
    state_dim = int(schema.field("state_features").type.list_size)
    action_dim = int(
        schema.field("candidate_action_features").type.value_type.list_size
    )
    for record in records:
        _validate_record(record, state_dim=state_dim, action_dim=action_dim)
    return pa.Table.from_pylist(records, schema=schema)


def _validate_record(
    record: dict[str, Any], *, state_dim: int, action_dim: int
) -> None:
    candidates = record["candidate_action_features"]
    candidate_count = len(candidates)
    teacher_index = int(record["teacher_action_index"])
    if len(record["state_features"]) != state_dim:
        raise ModelValidationError("RL state vector has the wrong length")
    if candidate_count <= 0 or any(len(row) != action_dim for row in candidates):
        raise ModelValidationError("RL candidate action vectors have the wrong shape")
    if len(record["candidate_jsons"]) != candidate_count:
        raise ModelValidationError("RL candidate JSON/features length mismatch")
    if not 0 <= teacher_index < candidate_count:
        raise ModelValidationError("RL teacher action index is outside its candidate group")
    slot_ids = record["player_ids_in_slot_order"]
    mask = record["player_mask"]
    terminal = record["terminal_rank_values"]
    if len(slot_ids) != len(set(slot_ids)) or not 1 <= len(slot_ids) <= MAX_PLAYERS:
        raise ModelValidationError("RL player slot ids are invalid")
    if len(mask) != MAX_PLAYERS or mask != [True] * len(slot_ids) + [False] * (
        MAX_PLAYERS - len(slot_ids)
    ):
        raise ModelValidationError("RL player mask does not match the slot order")
    if len(terminal) != MAX_PLAYERS or any(abs(float(value)) > 1.0 for value in terminal):
        raise ModelValidationError("RL terminal rank vector is invalid")
    searched = bool(record["has_search_targets"])
    search_q = record["search_q_values"]
    depth = int(record["search_depth_used"])
    nodes = int(record["search_nodes_evaluated"])
    depth_2_completed = bool(record["depth_2_completed"])
    if searched:
        if len(search_q) != candidate_count or any(
            len(row) != MAX_PLAYERS for row in search_q
        ):
            raise ModelValidationError("RL searched decision does not label every candidate")
        if depth not in (1, 2) or nodes <= 0 or depth_2_completed != (depth == 2):
            raise ModelValidationError("RL search metadata is inconsistent")
    elif search_q or depth != 0 or nodes != 0 or depth_2_completed:
        raise ModelValidationError("RL non-search decision contains search targets")


def _field_payload(field: Any) -> dict[str, Any]:
    return {
        "name": field.name,
        "type": str(field.type).replace("<element:", "<item:"),
        "nullable": field.nullable,
    }


def _validate_split_fractions(
    fractions: tuple[float, float, float],
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in fractions)
    if len(values) != 3 or any(value < 0 for value in values):
        raise ValueError("split fractions must contain three non-negative values")
    if abs(sum(values) - 1.0) > 1e-9 or values[0] <= 0:
        raise ValueError("split fractions must sum to one with positive train fraction")
    return values  # type: ignore[return-value]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


__all__ = [
    "DATASET_FORMAT_NAME",
    "DATASET_FORMAT_VERSION",
    "RlDatasetProgress",
    "RlDatasetSummary",
    "generate_rl_dataset",
    "iter_rl_parquet_batches",
    "legal_region_sets",
    "load_rl_dataset_metadata",
    "load_rl_dataset_records",
    "verify_rl_dataset_manifest",
]
