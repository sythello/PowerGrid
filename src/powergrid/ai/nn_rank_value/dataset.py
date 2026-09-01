from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator

from ...model import GameConfig, ModelValidationError, SeatConfig
from ...session import GameSession
from ...session_types import GameSnapshot
from .. import build_ai_controller, derive_final_standings
from .candidates import (
    CandidateAction,
    candidate_from_intent,
    find_candidate_for_intent,
    generate_candidate_actions,
)
from .observation import (
    ACTION_FEATURE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    build_public_observation,
    encode_action_features,
    encode_state_features,
)


DATASET_FORMAT_VERSION = 2
DATASET_FORMAT_NAME = "powergrid.nn_rank_value.parquet"
DEFAULT_TARGET_SHARD_SIZE_BYTES = 512 * 1024 * 1024
DEFAULT_SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
DATASET_SPLITS = ("train", "validation", "test")
EXAMPLE_GAME_COUNT = 3
DEFAULT_BEHAVIOR_CONTROLLERS = (
    "ai_deterministic",
    "ai_deterministic_efficiency",
    "ai_deterministic_expansion",
    "ai_deterministic_reserve",
)


@dataclass(frozen=True)
class DatasetGenerationProgress:
    games_completed: int
    games_total: int
    rows: int
    parquet_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True)
class DatasetGenerationSummary:
    output_path: Path
    metadata_path: Path
    games: int
    behavior_samples: int
    counterfactual_samples: int
    state_dim: int
    action_dim: int
    shards: int
    parquet_bytes: int
    split_games: dict[str, int]
    split_rows: dict[str, int]
    example_jsonl_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _GameTask:
    game_offset: int
    seed: int
    map_id: str
    player_count: int
    selected_regions: tuple[str, ...]
    assigned_controllers: tuple[str, ...]
    counterfactual_every: int
    counterfactual_max_candidates: int
    rollout_controller: str
    max_actions: int


@dataclass(frozen=True)
class _CompletedGame:
    game_offset: int
    game_id: str
    records: list[dict[str, Any]]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    behavior_samples: int
    counterfactual_samples: int
    selected_regions: tuple[str, ...]


@dataclass(frozen=True)
class _ShardDescriptor:
    relative_path: str
    rows: int
    games: int
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "rows": self.rows,
            "games": self.games,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


class _SplitShardWriter:
    def __init__(
        self,
        root: Path,
        split: str,
        schema: Any,
        *,
        target_shard_size_bytes: int,
        parquet_module: Any,
    ) -> None:
        self.root = root
        self.split = split
        self.schema = schema
        self.target_shard_size_bytes = target_shard_size_bytes
        self.parquet_module = parquet_module
        self.split_dir = root / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Any | None = None
        self._path: Path | None = None
        self._shard_index = 0
        self._rows = 0
        self._games = 0
        self.descriptors: list[_ShardDescriptor] = []

    def write_game(self, records: list[dict[str, Any]], table: Any) -> None:
        if not records:
            return
        if self._writer is None:
            self._open_shard()
        assert self._writer is not None
        self._writer.write_table(table, row_group_size=len(records))
        self._rows += len(records)
        self._games += 1
        assert self._path is not None
        if self._path.stat().st_size >= self.target_shard_size_bytes:
            self._close_shard()

    def current_bytes(self) -> int:
        open_bytes = self._path.stat().st_size if self._path is not None else 0
        return sum(item.bytes for item in self.descriptors) + open_bytes

    def close(self) -> None:
        self._close_shard()

    def _open_shard(self) -> None:
        self._path = self.split_dir / f"part-{self._shard_index:05d}.parquet"
        self._writer = self.parquet_module.ParquetWriter(
            self._path,
            self.schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
        self._rows = 0
        self._games = 0

    def _close_shard(self) -> None:
        if self._writer is None:
            return
        assert self._path is not None
        self._writer.close()
        size = self._path.stat().st_size
        self.descriptors.append(
            _ShardDescriptor(
                relative_path=self._path.relative_to(self.root).as_posix(),
                rows=self._rows,
                games=self._games,
                bytes=size,
                sha256=_sha256_file(self._path),
            )
        )
        self._writer = None
        self._path = None
        self._shard_index += 1
        self._rows = 0
        self._games = 0


class _ParquetDatasetWriter:
    def __init__(
        self,
        root: Path,
        schema: Any,
        *,
        target_shard_size_bytes: int,
        parquet_module: Any,
    ) -> None:
        self.writers = {
            split: _SplitShardWriter(
                root,
                split,
                schema,
                target_shard_size_bytes=target_shard_size_bytes,
                parquet_module=parquet_module,
            )
            for split in DATASET_SPLITS
        }

    def write_game(self, split: str, records: list[dict[str, Any]], table: Any) -> None:
        self.writers[split].write_game(records, table)

    def current_bytes(self) -> int:
        return sum(writer.current_bytes() for writer in self.writers.values())

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def generate_rank_value_dataset(
    output_path: str | Path,
    *,
    games: int,
    seed_start: int = 1,
    map_id: str = "germany",
    player_count: int = 3,
    behavior_controllers: tuple[str, ...] = DEFAULT_BEHAVIOR_CONTROLLERS,
    selected_regions: tuple[str, ...] = (),
    region_sets: tuple[tuple[str, ...], ...] = (),
    counterfactual_every: int = 0,
    counterfactual_max_candidates: int = 8,
    rollout_controller: str = "ai_deterministic",
    max_actions_per_game: int = 5000,
    target_shard_size_bytes: int = DEFAULT_TARGET_SHARD_SIZE_BYTES,
    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
    split_seed: int = 0,
    workers: int = 1,
    progress_callback: Callable[[DatasetGenerationProgress], None] | None = None,
) -> DatasetGenerationSummary:
    """Generate a streaming, game-grouped Parquet dataset directory."""
    if games <= 0:
        raise ValueError("games must be positive")
    if not 3 <= player_count <= 6:
        raise ValueError("player_count must be between 3 and 6")
    if not behavior_controllers:
        raise ValueError("at least one behavior controller is required")
    if counterfactual_every < 0:
        raise ValueError("counterfactual_every may not be negative")
    if target_shard_size_bytes <= 0:
        raise ValueError("target_shard_size_bytes must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if selected_regions and region_sets:
        raise ValueError("selected_regions and region_sets may not both be supplied")
    normalized_fractions = _validate_split_fractions(split_fractions)
    normalized_region_sets = tuple(tuple(values) for values in region_sets)
    if not normalized_region_sets:
        normalized_region_sets = (tuple(selected_regions),)

    pa, parquet = _import_pyarrow()
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    examples_dir = output / "examples"
    examples_dir.mkdir()
    started = time.perf_counter()
    writer: _ParquetDatasetWriter | None = None
    parquet_schema: Any | None = None
    state_names: tuple[str, ...] | None = None
    action_names: tuple[str, ...] | None = None
    behavior_count = 0
    counterfactual_count = 0
    total_rows = 0
    split_games = {split: 0 for split in DATASET_SPLITS}
    split_rows = {split: 0 for split in DATASET_SPLITS}
    resolved_region_sets: set[tuple[str, ...]] = set()
    example_descriptors: list[dict[str, Any]] = []
    example_paths: list[Path] = []

    try:
        tasks = _build_game_tasks(
            games=games,
            seed_start=seed_start,
            map_id=map_id,
            player_count=player_count,
            behavior_controllers=behavior_controllers,
            region_sets=normalized_region_sets,
            counterfactual_every=counterfactual_every,
            counterfactual_max_candidates=counterfactual_max_candidates,
            rollout_controller=rollout_controller,
            max_actions=max_actions_per_game,
        )
        for completed_index, completed in enumerate(
            _iter_completed_games(tasks, workers=workers), start=1
        ):
            if state_names is None:
                state_names = completed.state_names
                action_names = completed.action_names
                parquet_schema = _build_parquet_schema(
                    pa,
                    state_dim=len(state_names),
                    action_dim=len(action_names),
                )
                writer = _ParquetDatasetWriter(
                    output,
                    parquet_schema,
                    target_shard_size_bytes=target_shard_size_bytes,
                    parquet_module=parquet,
                )
            elif (completed.state_names, completed.action_names) != (
                state_names,
                action_names,
            ):
                raise ModelValidationError("feature schema changed during dataset generation")

            assert writer is not None and parquet_schema is not None
            split = assign_game_split(
                completed.game_id,
                split_fractions=normalized_fractions,
                split_seed=split_seed,
            )
            table = _records_to_parquet_table(pa, completed.records, parquet_schema)
            writer.write_game(split, completed.records, table)
            split_games[split] += 1
            split_rows[split] += len(completed.records)
            behavior_count += completed.behavior_samples
            counterfactual_count += completed.counterfactual_samples
            total_rows += len(completed.records)
            resolved_region_sets.add(completed.selected_regions)

            if completed.game_offset < EXAMPLE_GAME_COUNT:
                example_path = examples_dir / (
                    f"game-{completed.game_offset + 1:02d}-{completed.game_id}.jsonl"
                )
                _write_jsonl(example_path, completed.records)
                example_paths.append(example_path)
                example_descriptors.append(
                    {
                        "path": example_path.relative_to(output).as_posix(),
                        "game_id": completed.game_id,
                        "rows": len(completed.records),
                        "bytes": example_path.stat().st_size,
                        "sha256": _sha256_file(example_path),
                    }
                )

            if progress_callback is not None:
                progress_callback(
                    DatasetGenerationProgress(
                        games_completed=completed_index,
                        games_total=games,
                        rows=total_rows,
                        parquet_bytes=writer.current_bytes(),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
    finally:
        if writer is not None:
            writer.close()

    assert state_names is not None and action_names is not None
    assert parquet_schema is not None and writer is not None
    split_payload: dict[str, Any] = {}
    all_descriptors: list[_ShardDescriptor] = []
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
    manifest_path = output / "manifest.json"
    manifest = {
        "format_name": DATASET_FORMAT_NAME,
        "format_version": DATASET_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        "state_dim": len(state_names),
        "action_dim": len(action_names),
        "state_feature_names": list(state_names),
        "action_feature_names": list(action_names),
        "parquet_schema": [
            {
                "name": field.name,
                "type": _canonical_arrow_type(field.type),
                "nullable": field.nullable,
            }
            for field in parquet_schema
        ],
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
            "behavior_samples": behavior_count,
            "counterfactual_samples": counterfactual_count,
            "rows": total_rows,
            "map_id": map_id,
            "player_count": player_count,
            "seed_start": seed_start,
            "behavior_controllers": list(behavior_controllers),
            "requested_region_sets": [list(values) for values in normalized_region_sets],
            "resolved_region_sets": [list(values) for values in sorted(resolved_region_sets)],
            "rollout_controller": rollout_controller,
            "counterfactual_every": counterfactual_every,
            "counterfactual_max_candidates": counterfactual_max_candidates,
            "max_actions_per_game": max_actions_per_game,
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "split": {
            "algorithm": "sha256_uniform_v1",
            "key": "game_id",
            "seed": split_seed,
            "fractions": dict(zip(DATASET_SPLITS, normalized_fractions)),
        },
        "splits": split_payload,
        "example_jsonl": example_descriptors,
    }
    _write_json(manifest_path, manifest)
    return DatasetGenerationSummary(
        output_path=output,
        metadata_path=manifest_path,
        games=games,
        behavior_samples=behavior_count,
        counterfactual_samples=counterfactual_count,
        state_dim=len(state_names),
        action_dim=len(action_names),
        shards=len(all_descriptors),
        parquet_bytes=parquet_bytes,
        split_games=split_games,
        split_rows=split_rows,
        example_jsonl_paths=tuple(sorted(example_paths)),
    )


def assign_game_split(
    game_id: str,
    *,
    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
    split_seed: int = 0,
) -> str:
    fractions = _validate_split_fractions(split_fractions)
    digest = hashlib.sha256(f"{split_seed}:{game_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < fractions[0]:
        return "train"
    if value < fractions[0] + fractions[1]:
        return "validation"
    return "test"


def load_dataset_records(
    path: str | Path,
    *,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Load records for inspection/tests; large-dataset training uses streaming batches."""
    _, parquet = _import_pyarrow()
    root = Path(path)
    manifest = load_dataset_metadata(root)
    requested_splits = DATASET_SPLITS if split is None else (split,)
    records: list[dict[str, Any]] = []
    for split_name in requested_splits:
        if split_name not in DATASET_SPLITS:
            raise ValueError(f"unsupported dataset split {split_name!r}")
        for descriptor in manifest["splits"][split_name]["shards"]:
            table = parquet.read_table(root / descriptor["path"])
            for row in table.to_pylist():
                row["candidate"] = json.loads(row.pop("candidate_json"))
                records.append(row)
    if not records:
        raise ValueError("dataset contains no samples for the requested split(s)")
    return records


def load_dataset_metadata(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    manifest_path = dataset_path / "manifest.json" if dataset_path.is_dir() else dataset_path
    with manifest_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("format_name") != DATASET_FORMAT_NAME:
        raise ValueError("unsupported dataset format")
    if int(metadata.get("format_version", -1)) != DATASET_FORMAT_VERSION:
        raise ValueError("unsupported dataset metadata version")
    return metadata


def iter_parquet_batches(
    path: str | Path,
    split: str,
    *,
    batch_size: int,
    shuffle_seed: int | None = None,
    columns: tuple[str, ...] | None = None,
) -> Iterator[Any]:
    """Yield bounded PyArrow record batches, optionally shuffling shards/row groups."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if split not in DATASET_SPLITS:
        raise ValueError(f"unsupported dataset split {split!r}")
    _, parquet = _import_pyarrow()
    root = Path(path)
    manifest = load_dataset_metadata(root)
    shard_paths = [root / item["path"] for item in manifest["splits"][split]["shards"]]
    if shuffle_seed is not None:
        import random

        random.Random(shuffle_seed).shuffle(shard_paths)
    for shard_index, shard_path in enumerate(shard_paths):
        parquet_file = parquet.ParquetFile(shard_path)
        row_groups = list(range(parquet_file.num_row_groups))
        if shuffle_seed is not None:
            import random

            random.Random((shuffle_seed + 1) * 1_000_003 + shard_index).shuffle(row_groups)
        for row_group in row_groups:
            yield from parquet_file.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group],
                columns=list(columns) if columns is not None else None,
                use_threads=True,
            )


def verify_dataset_manifest(path: str | Path) -> dict[str, int]:
    root = Path(path)
    manifest = load_dataset_metadata(root)
    _, parquet = _import_pyarrow()
    expected_schema = manifest["parquet_schema"]
    checked_shards = 0
    checked_examples = 0
    for split in DATASET_SPLITS:
        split_rows = 0
        split_games = 0
        for descriptor in manifest["splits"][split]["shards"]:
            shard_path = root / descriptor["path"]
            if shard_path.stat().st_size != int(descriptor["bytes"]):
                raise ValueError(f"dataset shard size mismatch: {shard_path}")
            if _sha256_file(shard_path) != descriptor["sha256"]:
                raise ValueError(f"dataset shard checksum mismatch: {shard_path}")
            parquet_file = parquet.ParquetFile(shard_path)
            actual_rows = int(parquet_file.metadata.num_rows)
            actual_games = int(parquet_file.num_row_groups)
            if actual_rows != int(descriptor["rows"]):
                raise ValueError(f"dataset shard row-count mismatch: {shard_path}")
            if actual_games != int(descriptor["games"]):
                raise ValueError(f"dataset shard game-count mismatch: {shard_path}")
            actual_schema = [
                {
                    "name": field.name,
                    "type": _canonical_arrow_type(field.type),
                    "nullable": field.nullable,
                }
                for field in parquet_file.schema_arrow
            ]
            if actual_schema != expected_schema:
                raise ValueError(f"dataset shard schema mismatch: {shard_path}")
            split_rows += actual_rows
            split_games += actual_games
            checked_shards += 1
        if split_rows != int(manifest["splits"][split]["rows"]):
            raise ValueError(f"dataset split row-count mismatch: {split}")
        if split_games != int(manifest["splits"][split]["games"]):
            raise ValueError(f"dataset split game-count mismatch: {split}")
    for descriptor in manifest["example_jsonl"]:
        example_path = root / descriptor["path"]
        if example_path.stat().st_size != int(descriptor["bytes"]):
            raise ValueError(f"example JSONL size mismatch: {example_path}")
        if _sha256_file(example_path) != descriptor["sha256"]:
            raise ValueError(f"example JSONL checksum mismatch: {example_path}")
        checked_examples += 1
    return {"shards": checked_shards, "examples": checked_examples}


def _build_game_tasks(
    *,
    games: int,
    seed_start: int,
    map_id: str,
    player_count: int,
    behavior_controllers: tuple[str, ...],
    region_sets: tuple[tuple[str, ...], ...],
    counterfactual_every: int,
    counterfactual_max_candidates: int,
    rollout_controller: str,
    max_actions: int,
) -> Iterator[_GameTask]:
    for game_offset in range(games):
        yield _GameTask(
            game_offset=game_offset,
            seed=seed_start + game_offset,
            map_id=map_id,
            player_count=player_count,
            selected_regions=region_sets[game_offset % len(region_sets)],
            assigned_controllers=tuple(
                behavior_controllers[(game_offset + seat_index) % len(behavior_controllers)]
                for seat_index in range(player_count)
            ),
            counterfactual_every=counterfactual_every,
            counterfactual_max_candidates=counterfactual_max_candidates,
            rollout_controller=rollout_controller,
            max_actions=max_actions,
        )


def _iter_completed_games(tasks: Iterator[_GameTask], *, workers: int) -> Iterator[_CompletedGame]:
    if workers == 1:
        for task in tasks:
            yield _generate_completed_game(task)
        return

    max_in_flight = max(workers, workers * 2)
    task_iterator = iter(tasks)
    pending: dict[Any, int] = {}
    buffered: dict[int, _CompletedGame] = {}
    next_offset = 0
    exhausted = False
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while pending or buffered or not exhausted:
            while not exhausted and len(pending) + len(buffered) < max_in_flight:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(_generate_completed_game, task)
                pending[future] = task.game_offset
            if pending:
                completed_futures, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    game_offset = pending.pop(future)
                    buffered[game_offset] = future.result()
            while next_offset in buffered:
                yield buffered.pop(next_offset)
                next_offset += 1


def _generate_completed_game(task: _GameTask) -> _CompletedGame:
    region_token = "-".join(sorted(task.selected_regions)) or "default"
    game_id = f"{task.map_id}-{task.player_count}p-regions-{region_token}-seed-{task.seed}"
    records, final_snapshot, names, counterfactual_count = _generate_one_game(
        game_id=game_id,
        seed=task.seed,
        map_id=task.map_id,
        player_count=task.player_count,
        selected_regions=task.selected_regions,
        assigned_controllers=task.assigned_controllers,
        counterfactual_every=task.counterfactual_every,
        counterfactual_max_candidates=task.counterfactual_max_candidates,
        rollout_controller=task.rollout_controller,
        max_actions=task.max_actions,
    )
    if final_snapshot.winner_result is None:
        raise ModelValidationError(f"dataset game {game_id} ended without a winner")
    completed = _attach_terminal_labels(records, final_snapshot)
    return _CompletedGame(
        game_offset=task.game_offset,
        game_id=game_id,
        records=completed,
        state_names=names[0],
        action_names=names[1],
        behavior_samples=sum(row["sample_source"] == "behavior" for row in completed),
        counterfactual_samples=counterfactual_count,
        selected_regions=tuple(final_snapshot.state.selected_regions),
    )


def _generate_one_game(
    *,
    game_id: str,
    seed: int,
    map_id: str,
    player_count: int,
    selected_regions: tuple[str, ...],
    assigned_controllers: tuple[str, ...],
    counterfactual_every: int,
    counterfactual_max_candidates: int,
    rollout_controller: str,
    max_actions: int,
) -> tuple[list[dict[str, Any]], GameSnapshot, tuple[tuple[str, ...], tuple[str, ...]], int]:
    config = GameConfig(
        map_id=map_id,
        players=tuple(
            SeatConfig(f"p{index + 1}", f"Player {index + 1}", controller="human")
            for index in range(player_count)
        ),
        seed=seed,
        selected_regions=selected_regions,
    )
    session = GameSession.new_game(config)
    resolved_regions = tuple(session.snapshot().state.selected_regions)
    behavior_agents = {
        f"p{index + 1}": build_ai_controller(controller_name)
        for index, controller_name in enumerate(assigned_controllers)
    }
    records: list[dict[str, Any]] = []
    state_names: tuple[str, ...] | None = None
    action_names: tuple[str, ...] | None = None
    counterfactual_count = 0

    for decision_index in range(max_actions):
        snapshot = session.advance_until_blocked()
        if snapshot.winner_result is not None:
            assert state_names is not None and action_names is not None
            return records, snapshot, (state_names, action_names), counterfactual_count
        request = snapshot.active_request
        if request is None:
            raise ModelValidationError(f"dataset game {game_id} stopped without an active request")
        observation = build_public_observation(snapshot.state, request)
        state_features, current_state_names = encode_state_features(observation)
        candidates = generate_candidate_actions(request, snapshot)
        behavior_agent = behavior_agents[request.player_id]
        chosen_intent = behavior_agent.choose_intent(request, snapshot)
        chosen_candidate = find_candidate_for_intent(candidates, chosen_intent)
        if chosen_candidate is None:
            chosen_candidate = candidate_from_intent(chosen_intent)
        action_features, current_action_names = encode_action_features(observation, chosen_candidate)
        state_names = state_names or current_state_names
        action_names = action_names or current_action_names
        if current_state_names != state_names or current_action_names != action_names:
            raise ModelValidationError("feature schema changed within a game")
        records.append(
            _base_record(
                game_id=game_id,
                seed=seed,
                decision_index=decision_index,
                request=request,
                assigned_controllers=assigned_controllers,
                selected_regions=resolved_regions,
                sample_source="behavior",
                candidate=chosen_candidate,
                state_features=state_features,
                action_features=action_features,
            )
        )

        if counterfactual_every and decision_index % counterfactual_every == 0:
            selected_candidates = _select_counterfactual_candidates(
                candidates, chosen_candidate, counterfactual_max_candidates
            )
            for candidate_index, candidate in enumerate(selected_candidates):
                rollout = session.fork()
                rollout.submit_intent(candidate.intent, auto_advance=False)
                rollout_snapshot = _finish_external_rollout(
                    rollout, controller_name=rollout_controller, max_actions=max_actions
                )
                if rollout_snapshot.winner_result is None:
                    continue
                cf_action_features, cf_action_names = encode_action_features(observation, candidate)
                if cf_action_names != action_names:
                    raise ModelValidationError("counterfactual action feature schema changed")
                record = _base_record(
                    game_id=game_id,
                    seed=seed,
                    decision_index=decision_index,
                    request=request,
                    assigned_controllers=assigned_controllers,
                    selected_regions=resolved_regions,
                    sample_source="counterfactual_rollout",
                    candidate=candidate,
                    state_features=state_features,
                    action_features=cf_action_features,
                )
                record["counterfactual_candidate_index"] = candidate_index
                record.update(_terminal_label_for_player(rollout_snapshot, request.player_id))
                records.append(record)
                counterfactual_count += 1

        before_request = session.current_request()
        session.submit_intent(chosen_intent, auto_advance=False)
        after_request = session.current_request()
        if before_request == after_request and session.snapshot().winner_result is None:
            last_event = session.snapshot().event_log[-1] if session.snapshot().event_log else None
            if last_event is not None and last_event.level == "error":
                raise ModelValidationError(
                    f"teacher produced invalid intent in {game_id}: {last_event.message}"
                )
    raise ModelValidationError(f"dataset game {game_id} exceeded {max_actions} actions")


def _finish_external_rollout(
    session: GameSession, *, controller_name: str, max_actions: int
) -> GameSnapshot:
    agents = {
        player.player_id: build_ai_controller(controller_name)
        for player in session.snapshot().state.players
    }
    for _ in range(max_actions):
        snapshot = session.advance_until_blocked()
        if snapshot.winner_result is not None:
            return snapshot
        request = snapshot.active_request
        if request is None:
            return snapshot
        intent = agents[request.player_id].choose_intent(request, snapshot)
        session.submit_intent(intent, auto_advance=False)
    return session.snapshot()


def _select_counterfactual_candidates(
    candidates: tuple[CandidateAction, ...], chosen: CandidateAction, maximum: int
) -> tuple[CandidateAction, ...]:
    if maximum <= 0 or len(candidates) <= maximum:
        return candidates
    selected = list(candidates[:maximum])
    if all(candidate.key != chosen.key for candidate in selected):
        selected[-1] = chosen
    return tuple(selected)


def _attach_terminal_labels(
    records: list[dict[str, Any]], snapshot: GameSnapshot
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for record in records:
        if "rank_value" in record:
            completed.append(record)
            continue
        enriched = dict(record)
        enriched.update(_terminal_label_for_player(snapshot, str(record["player_id"])))
        completed.append(enriched)
    return completed


def _terminal_label_for_player(snapshot: GameSnapshot, player_id: str) -> dict[str, int | float]:
    if snapshot.winner_result is None:
        raise ModelValidationError("terminal labels require a completed game")
    standings = derive_final_standings(snapshot.state, snapshot.winner_result)
    standing = next(item for item in standings if item.player_id == player_id)
    player_count = len(standings)
    rank_value = (player_count + 1 - (2 * standing.place)) / (player_count - 1)
    return {
        "final_place": standing.place,
        "rank_value": float(rank_value),
        "is_winner": int(player_id in snapshot.winner_result.winner_ids),
        "final_powered_cities": standing.powered_cities,
        "final_money": standing.money,
        "final_connected_cities": standing.connected_cities,
    }


def _base_record(
    *,
    game_id: str,
    seed: int,
    decision_index: int,
    request: Any,
    assigned_controllers: tuple[str, ...],
    selected_regions: tuple[str, ...],
    sample_source: str,
    candidate: CandidateAction,
    state_features: list[float],
    action_features: list[float],
) -> dict[str, Any]:
    player_index = int(request.player_id[1:]) - 1
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "game_id": game_id,
        "seed": seed,
        "decision_index": decision_index,
        "player_id": request.player_id,
        "phase": request.phase,
        "decision_type": request.decision_type,
        "behavior_controller": assigned_controllers[player_index],
        "selected_regions": list(selected_regions),
        "sample_source": sample_source,
        "candidate": candidate.to_dict(),
        "state_features": state_features,
        "action_features": action_features,
    }


def _build_parquet_schema(pa: Any, *, state_dim: int, action_dim: int) -> Any:
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
            pa.field("selected_regions", pa.list_(pa.string()), nullable=False),
            pa.field("sample_source", pa.string(), nullable=False),
            pa.field("candidate_json", pa.string(), nullable=False),
            pa.field("state_features", pa.list_(pa.float32(), state_dim), nullable=False),
            pa.field("action_features", pa.list_(pa.float32(), action_dim), nullable=False),
            pa.field("counterfactual_candidate_index", pa.int16(), nullable=True),
            pa.field("final_place", pa.int16(), nullable=False),
            pa.field("rank_value", pa.float32(), nullable=False),
            pa.field("is_winner", pa.int8(), nullable=False),
            pa.field("final_powered_cities", pa.int16(), nullable=False),
            pa.field("final_money", pa.int32(), nullable=False),
            pa.field("final_connected_cities", pa.int16(), nullable=False),
        ]
    )


def _canonical_arrow_type(arrow_type: Any) -> str:
    # Parquet round-trips may rename nested list children from "item" to
    # "element" without changing the logical type.
    return str(arrow_type).replace("<element:", "<item:")


def _records_to_parquet_table(pa: Any, records: list[dict[str, Any]], schema: Any) -> Any:
    rows = []
    for record in records:
        row = dict(record)
        row["candidate_json"] = json.dumps(
            row.pop("candidate"), sort_keys=True, separators=(",", ":")
        )
        row.setdefault("counterfactual_candidate_index", None)
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema)


def _validate_split_fractions(
    fractions: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(fractions) != 3:
        raise ValueError("split_fractions must contain train, validation, and test values")
    normalized = tuple(float(value) for value in fractions)
    if any(value < 0.0 for value in normalized):
        raise ValueError("split fractions may not be negative")
    if abs(sum(normalized) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    if normalized[0] <= 0.0:
        raise ValueError("training split fraction must be positive")
    return normalized  # type: ignore[return-value]


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "Parquet dataset support requires PyArrow; install requirements-ml.txt"
        ) from exc
    return pa, parquet


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Narrow schema-agnostic compatibility surface for later NN dataset formats.
# The v1 writer behavior and manifest format remain unchanged.
StreamingParquetDatasetWriter = _ParquetDatasetWriter


def import_pyarrow() -> tuple[Any, Any]:
    return _import_pyarrow()


def sha256_file(path: str | Path) -> str:
    return _sha256_file(Path(path))


__all__ = [
    "DATASET_FORMAT_NAME",
    "DATASET_FORMAT_VERSION",
    "DATASET_SPLITS",
    "DEFAULT_BEHAVIOR_CONTROLLERS",
    "DEFAULT_SPLIT_FRACTIONS",
    "DEFAULT_TARGET_SHARD_SIZE_BYTES",
    "DatasetGenerationProgress",
    "DatasetGenerationSummary",
    "assign_game_split",
    "generate_rank_value_dataset",
    "iter_parquet_batches",
    "load_dataset_metadata",
    "load_dataset_records",
    "import_pyarrow",
    "sha256_file",
    "StreamingParquetDatasetWriter",
    "verify_dataset_manifest",
]
