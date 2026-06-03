from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from ..model import GameConfig, GameState, ModelValidationError, SeatConfig, WinnerResult


DEFAULT_EVALUATION_CONTROLLERS = ("ai_deterministic", "ai_heuristics")
DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_K_FACTOR = 24.0
RATING_SCALE = 400.0


@dataclass(frozen=True)
class AiEvaluationBucketConfig:
    map_id: str = "germany"
    player_count: int = 3
    controller_names: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EVALUATION_CONTROLLERS)
    selected_regions: tuple[str, ...] = field(default_factory=tuple)
    games_per_lineup: int = 20
    seed_start: int = 1
    initial_rating: float = DEFAULT_INITIAL_RATING
    k_factor: float = DEFAULT_K_FACTOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "controller_names", tuple(self.controller_names))
        object.__setattr__(self, "selected_regions", tuple(self.selected_regions))
        if self.map_id != "germany" or self.player_count != 3:
            raise ModelValidationError("AI evaluation v1 only supports map_id='germany' with player_count=3")
        if len(self.controller_names) != len(set(self.controller_names)):
            raise ModelValidationError("controller_names must be unique")
        if len(self.controller_names) != 2:
            raise ModelValidationError("AI evaluation v1 requires exactly two controller names")
        if any(not controller_name for controller_name in self.controller_names):
            raise ModelValidationError("controller_names must be non-empty")
        if self.games_per_lineup <= 0:
            raise ModelValidationError("games_per_lineup must be positive")
        if self.seed_start < 0:
            raise ModelValidationError("seed_start may not be negative")
        if self.initial_rating <= 0:
            raise ModelValidationError("initial_rating must be positive")
        if self.k_factor <= 0:
            raise ModelValidationError("k_factor must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "map_id": self.map_id,
            "player_count": self.player_count,
            "controller_names": list(self.controller_names),
            "selected_regions": list(self.selected_regions),
            "games_per_lineup": self.games_per_lineup,
            "seed_start": self.seed_start,
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
        }


@dataclass(frozen=True)
class AiEvaluationStanding:
    player_id: str
    controller_name: str
    place: int
    powered_cities: int
    money: int
    connected_cities: int

    def to_dict(self) -> dict[str, object]:
        return {
            "player_id": self.player_id,
            "controller_name": self.controller_name,
            "place": self.place,
            "powered_cities": self.powered_cities,
            "money": self.money,
            "connected_cities": self.connected_cities,
        }


@dataclass(frozen=True)
class AiControllerRatingSummary:
    controller_name: str
    rating: float
    games: int
    seat_appearances: int
    controller_wins: int
    first_place_seats: int
    average_finish: float

    def to_dict(self) -> dict[str, object]:
        return {
            "controller_name": self.controller_name,
            "rating": round(self.rating, 6),
            "games": self.games,
            "seat_appearances": self.seat_appearances,
            "controller_wins": self.controller_wins,
            "first_place_seats": self.first_place_seats,
            "average_finish": round(self.average_finish, 6),
        }


@dataclass(frozen=True)
class AiControllerPairSummary:
    controller_a: str
    controller_b: str
    games: int
    seat_pair_comparisons: int
    seat_pair_wins_for_a: int
    seat_pair_draws: int
    seat_pair_losses_for_a: int
    actual_score_total_for_a: float
    expected_score_total_for_a: float
    rating_delta_total_for_a: float

    def to_dict(self) -> dict[str, object]:
        games = max(1, self.games)
        return {
            "controller_a": self.controller_a,
            "controller_b": self.controller_b,
            "games": self.games,
            "seat_pair_comparisons": self.seat_pair_comparisons,
            "seat_pair_wins_for_a": self.seat_pair_wins_for_a,
            "seat_pair_draws": self.seat_pair_draws,
            "seat_pair_losses_for_a": self.seat_pair_losses_for_a,
            "actual_score_total_for_a": round(self.actual_score_total_for_a, 6),
            "expected_score_total_for_a": round(self.expected_score_total_for_a, 6),
            "rating_delta_total_for_a": round(self.rating_delta_total_for_a, 6),
            "actual_score_average_for_a": round(self.actual_score_total_for_a / games, 6),
            "expected_score_average_for_a": round(self.expected_score_total_for_a / games, 6),
        }


@dataclass(frozen=True)
class AiEvaluationGameSummary:
    game_index: int
    seed: int
    lineup: tuple[str, ...]
    winner_ids: tuple[str, ...]
    standings: tuple[AiEvaluationStanding, ...]
    powered_cities: dict[str, int]
    money: dict[str, int]
    connected_cities: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "game_index": self.game_index,
            "seed": self.seed,
            "lineup": list(self.lineup),
            "winner_ids": list(self.winner_ids),
            "standings": [standing.to_dict() for standing in self.standings],
            "powered_cities": dict(self.powered_cities),
            "money": dict(self.money),
            "connected_cities": dict(self.connected_cities),
        }


@dataclass(frozen=True)
class AiEvaluationReport:
    config: AiEvaluationBucketConfig
    resolved_selected_regions: tuple[str, ...]
    scheduled_lineups: tuple[tuple[str, ...], ...]
    controller_summaries: tuple[AiControllerRatingSummary, ...]
    controller_pair_summaries: tuple[AiControllerPairSummary, ...]
    game_summaries: tuple[AiEvaluationGameSummary, ...]
    rating_scale: float = RATING_SCALE

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket": {
                **self.config.to_dict(),
                "resolved_selected_regions": list(self.resolved_selected_regions),
            },
            "algorithm": {
                "name": "pairwise_elo",
                "rating_scale": self.rating_scale,
                "draw_rule": "equal powered_cities/money/connected_cities is treated as a draw",
                "pair_k_formula": "base_k / (distinct_controller_count - 1)",
            },
            "schedule": {
                "lineups": [list(lineup) for lineup in self.scheduled_lineups],
                "games_per_lineup": self.config.games_per_lineup,
                "seed_start": self.config.seed_start,
                "total_games": len(self.game_summaries),
            },
            "controllers": [summary.to_dict() for summary in self.controller_summaries],
            "controller_pairs": [summary.to_dict() for summary in self.controller_pair_summaries],
            "games": [summary.to_dict() for summary in self.game_summaries],
        }


@dataclass(frozen=True)
class _ControllerPairGameUpdate:
    controller_a: str
    controller_b: str
    seat_pair_comparisons: int
    seat_pair_wins_for_a: int
    seat_pair_draws: int
    seat_pair_losses_for_a: int
    actual_score_for_a: float
    expected_score_for_a: float
    rating_delta_for_a: float


def build_default_evaluation_lineups(
    controller_names: tuple[str, ...] | list[str],
    player_count: int,
    *,
    map_id: str = "germany",
) -> tuple[tuple[str, ...], ...]:
    ordered = tuple(controller_names)
    if map_id != "germany" or player_count != 3:
        raise ModelValidationError("AI evaluation v1 only supports map_id='germany' with player_count=3")
    if len(ordered) != len(set(ordered)):
        raise ModelValidationError("controller_names must be unique")
    if len(ordered) != 2:
        raise ModelValidationError("AI evaluation v1 requires exactly two controller names")
    first, second = ordered
    return (
        (first, first, second),
        (first, second, second),
    )


def derive_final_standings(
    state: GameState,
    winner_result: WinnerResult,
) -> tuple[AiEvaluationStanding, ...]:
    powered = winner_result.powered_cities
    money = winner_result.money
    connected = winner_result.connected_cities
    prior_index = {player_id: index for index, player_id in enumerate(state.player_order)}
    ordered_player_ids = sorted(
        powered,
        key=lambda player_id: (
            -powered[player_id],
            -money[player_id],
            -connected[player_id],
            prior_index.get(player_id, 0),
        ),
    )

    players_by_id = {player.player_id: player for player in state.players}
    standings: list[AiEvaluationStanding] = []
    previous_signature: tuple[int, int, int] | None = None
    current_place = 0
    for index, player_id in enumerate(ordered_player_ids, start=1):
        signature = (
            powered[player_id],
            money[player_id],
            connected[player_id],
        )
        if signature != previous_signature:
            current_place = index
            previous_signature = signature
        player = players_by_id[player_id]
        standings.append(
            AiEvaluationStanding(
                player_id=player_id,
                controller_name=player.controller,
                place=current_place,
                powered_cities=signature[0],
                money=signature[1],
                connected_cities=signature[2],
            )
        )
    return tuple(standings)


def evaluate_ai_bucket(config: AiEvaluationBucketConfig) -> AiEvaluationReport:
    _validate_controller_names(config.controller_names)
    scheduled_lineups = build_default_evaluation_lineups(
        config.controller_names,
        config.player_count,
        map_id=config.map_id,
    )
    ratings = {controller_name: float(config.initial_rating) for controller_name in config.controller_names}
    controller_stats = {
        controller_name: {
            "games": 0,
            "seat_appearances": 0,
            "controller_wins": 0,
            "first_place_seats": 0,
            "finish_total": 0.0,
        }
        for controller_name in config.controller_names
    }
    controller_pair_stats: dict[tuple[str, str], dict[str, float | int]] = {}
    game_summaries: list[AiEvaluationGameSummary] = []
    resolved_selected_regions: tuple[str, ...] = ()
    game_index = 0

    for lineup in scheduled_lineups:
        for offset in range(config.games_per_lineup):
            seed = config.seed_start + offset
            game_index += 1
            game_summary, selected_regions = _run_evaluation_game(
                config,
                lineup,
                seed=seed,
                game_index=game_index,
            )
            if not resolved_selected_regions:
                resolved_selected_regions = selected_regions
            game_summaries.append(game_summary)
            _accumulate_controller_stats(controller_stats, game_summary)

            pair_updates = _compute_pairwise_game_updates(
                game_summary.standings,
                ratings,
                base_k=config.k_factor,
            )
            pending_rating_deltas = {controller_name: 0.0 for controller_name in ratings}
            for update in pair_updates:
                pending_rating_deltas[update.controller_a] += update.rating_delta_for_a
                pending_rating_deltas[update.controller_b] -= update.rating_delta_for_a
                key = (update.controller_a, update.controller_b)
                pair_stats = controller_pair_stats.setdefault(
                    key,
                    {
                        "games": 0,
                        "seat_pair_comparisons": 0,
                        "seat_pair_wins_for_a": 0,
                        "seat_pair_draws": 0,
                        "seat_pair_losses_for_a": 0,
                        "actual_score_total_for_a": 0.0,
                        "expected_score_total_for_a": 0.0,
                        "rating_delta_total_for_a": 0.0,
                    },
                )
                pair_stats["games"] += 1
                pair_stats["seat_pair_comparisons"] += update.seat_pair_comparisons
                pair_stats["seat_pair_wins_for_a"] += update.seat_pair_wins_for_a
                pair_stats["seat_pair_draws"] += update.seat_pair_draws
                pair_stats["seat_pair_losses_for_a"] += update.seat_pair_losses_for_a
                pair_stats["actual_score_total_for_a"] += update.actual_score_for_a
                pair_stats["expected_score_total_for_a"] += update.expected_score_for_a
                pair_stats["rating_delta_total_for_a"] += update.rating_delta_for_a
            for controller_name, delta in pending_rating_deltas.items():
                ratings[controller_name] += delta

    controller_summaries = tuple(
        sorted(
            (
                AiControllerRatingSummary(
                    controller_name=controller_name,
                    rating=ratings[controller_name],
                    games=int(stats["games"]),
                    seat_appearances=int(stats["seat_appearances"]),
                    controller_wins=int(stats["controller_wins"]),
                    first_place_seats=int(stats["first_place_seats"]),
                    average_finish=(
                        float(stats["finish_total"]) / int(stats["seat_appearances"])
                        if int(stats["seat_appearances"]) > 0
                        else 0.0
                    ),
                )
                for controller_name, stats in controller_stats.items()
            ),
            key=lambda summary: (-summary.rating, summary.controller_name),
        )
    )
    controller_pair_summaries = tuple(
        AiControllerPairSummary(
            controller_a=controller_a,
            controller_b=controller_b,
            games=int(stats["games"]),
            seat_pair_comparisons=int(stats["seat_pair_comparisons"]),
            seat_pair_wins_for_a=int(stats["seat_pair_wins_for_a"]),
            seat_pair_draws=int(stats["seat_pair_draws"]),
            seat_pair_losses_for_a=int(stats["seat_pair_losses_for_a"]),
            actual_score_total_for_a=float(stats["actual_score_total_for_a"]),
            expected_score_total_for_a=float(stats["expected_score_total_for_a"]),
            rating_delta_total_for_a=float(stats["rating_delta_total_for_a"]),
        )
        for (controller_a, controller_b), stats in sorted(controller_pair_stats.items())
    )
    return AiEvaluationReport(
        config=config,
        resolved_selected_regions=resolved_selected_regions,
        scheduled_lineups=scheduled_lineups,
        controller_summaries=controller_summaries,
        controller_pair_summaries=controller_pair_summaries,
        game_summaries=tuple(game_summaries),
    )


def _validate_controller_names(controller_names: tuple[str, ...]) -> None:
    from . import AI_CONTROLLER_REGISTRY

    unsupported = [controller_name for controller_name in controller_names if controller_name not in AI_CONTROLLER_REGISTRY]
    if unsupported:
        raise ModelValidationError(
            "unsupported AI controller(s) for evaluation: " + ", ".join(sorted(unsupported))
        )


def _run_evaluation_game(
    config: AiEvaluationBucketConfig,
    lineup: tuple[str, ...],
    *,
    seed: int,
    game_index: int,
) -> tuple[AiEvaluationGameSummary, tuple[str, ...]]:
    from ..session import GameSession

    seat_configs = tuple(
        SeatConfig(
            player_id=f"p{index + 1}",
            name=f"Player {index + 1}",
            controller=controller_name,
        )
        for index, controller_name in enumerate(lineup)
    )
    game_config = GameConfig(
        map_id=config.map_id,
        players=seat_configs,
        seed=seed,
        selected_regions=config.selected_regions,
    )
    session = GameSession.new_game(game_config)
    snapshot = session.advance_until_blocked()
    if snapshot.winner_result is None:
        message = snapshot.event_log[-1].message if snapshot.event_log else "game ended without a winner"
        raise ModelValidationError(
            f"AI evaluation game failed for lineup={lineup!r} seed={seed}: {message}"
        )
    standings = derive_final_standings(snapshot.state, snapshot.winner_result)
    return (
        AiEvaluationGameSummary(
            game_index=game_index,
            seed=seed,
            lineup=tuple(lineup),
            winner_ids=snapshot.winner_result.winner_ids,
            standings=standings,
            powered_cities=dict(snapshot.winner_result.powered_cities),
            money=dict(snapshot.winner_result.money),
            connected_cities=dict(snapshot.winner_result.connected_cities),
        ),
        tuple(snapshot.state.selected_regions),
    )


def _accumulate_controller_stats(
    controller_stats: dict[str, dict[str, float | int]],
    game_summary: AiEvaluationGameSummary,
) -> None:
    winners = {
        standing.controller_name
        for standing in game_summary.standings
        if standing.player_id in game_summary.winner_ids
    }
    controllers_in_game = {standing.controller_name for standing in game_summary.standings}
    for controller_name in controllers_in_game:
        controller_stats[controller_name]["games"] += 1
    for standing in game_summary.standings:
        stats = controller_stats[standing.controller_name]
        stats["seat_appearances"] += 1
        stats["finish_total"] += standing.place
        if standing.place == 1:
            stats["first_place_seats"] += 1
    for controller_name in winners:
        controller_stats[controller_name]["controller_wins"] += 1


def _compute_pairwise_game_updates(
    standings: tuple[AiEvaluationStanding, ...],
    ratings: dict[str, float],
    *,
    base_k: float,
) -> tuple[_ControllerPairGameUpdate, ...]:
    distinct_controllers = sorted({standing.controller_name for standing in standings})
    if len(distinct_controllers) < 2:
        return ()
    pair_k = _pair_k_factor(base_k, len(distinct_controllers))
    updates: list[_ControllerPairGameUpdate] = []
    for controller_a, controller_b in combinations(distinct_controllers, 2):
        actual_score, wins, draws, losses = _controller_pair_actual_score(
            standings,
            controller_a,
            controller_b,
        )
        expected_score = _expected_score(ratings[controller_a], ratings[controller_b])
        delta = pair_k * (actual_score - expected_score)
        updates.append(
            _ControllerPairGameUpdate(
                controller_a=controller_a,
                controller_b=controller_b,
                seat_pair_comparisons=wins + draws + losses,
                seat_pair_wins_for_a=wins,
                seat_pair_draws=draws,
                seat_pair_losses_for_a=losses,
                actual_score_for_a=actual_score,
                expected_score_for_a=expected_score,
                rating_delta_for_a=delta,
            )
        )
    return tuple(updates)


def _controller_pair_actual_score(
    standings: tuple[AiEvaluationStanding, ...],
    controller_a: str,
    controller_b: str,
) -> tuple[float, int, int, int]:
    seats_a = [standing for standing in standings if standing.controller_name == controller_a]
    seats_b = [standing for standing in standings if standing.controller_name == controller_b]
    if not seats_a or not seats_b:
        raise ModelValidationError(
            f"cannot compare controller pair {controller_a!r}/{controller_b!r} without both controllers present"
        )
    wins = 0
    draws = 0
    losses = 0
    for standing_a in seats_a:
        for standing_b in seats_b:
            if _standing_signature(standing_a) == _standing_signature(standing_b):
                draws += 1
            elif standing_a.place < standing_b.place:
                wins += 1
            else:
                losses += 1
    total = wins + draws + losses
    if total <= 0:
        raise ModelValidationError("controller pair comparison must contain at least one seat pairing")
    actual_score = (wins + (0.5 * draws)) / total
    return actual_score, wins, draws, losses


def _standing_signature(standing: AiEvaluationStanding) -> tuple[int, int, int]:
    return (
        standing.powered_cities,
        standing.money,
        standing.connected_cities,
    )


def _pair_k_factor(base_k: float, distinct_controller_count: int) -> float:
    if distinct_controller_count <= 1:
        raise ModelValidationError("distinct_controller_count must be at least 2")
    return base_k / (distinct_controller_count - 1)


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / RATING_SCALE)))


__all__ = [
    "AiControllerPairSummary",
    "AiControllerRatingSummary",
    "AiEvaluationBucketConfig",
    "AiEvaluationGameSummary",
    "AiEvaluationReport",
    "AiEvaluationStanding",
    "DEFAULT_EVALUATION_CONTROLLERS",
    "DEFAULT_INITIAL_RATING",
    "DEFAULT_K_FACTOR",
    "RATING_SCALE",
    "build_default_evaluation_lineups",
    "derive_final_standings",
    "evaluate_ai_bucket",
]
