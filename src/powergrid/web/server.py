from __future__ import annotations

import json
import math
import mimetypes
import threading
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..ai import AI_CONTROLLER_REGISTRY
from ..board_layout import load_board_layout
from ..model import GameConfig, GameState, ModelValidationError, SeatConfig, apply_builds
from ..session import GameSession
from ..session_types import GuiIntent


STATIC_ROOT = Path(__file__).resolve().parent / "static"
WEB_LAYOUTS_PATH = STATIC_ROOT / "data" / "web_layouts.json"
MAX_REQUEST_BYTES = 1_000_000
MAP_ASSETS = {
    "germany": "/assets/maps/germany.jpg",
    "usa": "/assets/maps/usa.jpg",
}
MAP_SIZES = {
    "germany": {"width": 1424, "height": 2000},
    "usa": {"width": 2000, "height": 1180},
    "test": {"width": 1200, "height": 780},
}
WEB_CONTROLLER_ALIASES = {
    "ai_nn_rl_v2": "ai_nn_rl_based_v1",
}


def _load_web_layouts(path: Path = WEB_LAYOUTS_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class PowerGridWebController:
    """Owns the single local game session exposed to the browser."""

    def __init__(self, *, web_layouts_path: Path = WEB_LAYOUTS_PATH) -> None:
        self._lock = threading.RLock()
        self._session: GameSession | None = None
        self._web_layouts_path = web_layouts_path
        self._web_layouts = _load_web_layouts(self._web_layouts_path)

    def metadata(self) -> dict[str, Any]:
        return {
            "maps": [
                {"id": "germany", "name": "德国"},
                {"id": "usa", "name": "美国"},
                {"id": "test", "name": "测试地图"},
            ],
            "controllers": [
                {"id": "human", "name": "本地玩家"},
                {
                    "id": "ai_nn_rl_v2",
                    "name": "NN RL v2",
                    "supported_maps": ["germany"],
                    "supported_player_counts": [3],
                },
                {"id": "ai_deterministic", "name": "确定性 AI"},
            ],
            "defaults": {
                "map_id": "germany",
                "player_count": 3,
                "seed": 7,
                "ai_controller": "ai_nn_rl_v2",
                "controllers": ["human", "ai_nn_rl_v2", "ai_nn_rl_v2"],
            },
        }

    def new_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            map_id = str(payload.get("map_id", "germany"))
            if map_id not in {"germany", "usa", "test"}:
                raise ModelValidationError(f"unsupported map {map_id!r}")
            raw_seats = payload.get("players")
            if not isinstance(raw_seats, list) or not 3 <= len(raw_seats) <= 6:
                raise ModelValidationError("players must contain between 3 and 6 seats")
            requested_controllers = [str(raw.get("controller") or "human") for raw in raw_seats]
            canonical_controllers = [
                WEB_CONTROLLER_ALIASES.get(controller, controller)
                for controller in requested_controllers
            ]
            if "ai_nn_rl_based_v1" in canonical_controllers and (
                map_id != "germany" or len(raw_seats) != 3
            ):
                raise ModelValidationError("NN RL v2 supports only 3-player Germany games")
            seats = tuple(
                SeatConfig(
                    player_id=f"p{index + 1}",
                    name=str(raw.get("name") or f"玩家 {index + 1}")[:32],
                    controller=canonical_controllers[index],
                )
                for index, raw in enumerate(raw_seats)
            )
            known_controllers = {"human", *AI_CONTROLLER_REGISTRY}
            unknown = [seat.controller for seat in seats if seat.controller not in known_controllers]
            if unknown:
                raise ModelValidationError(f"unknown controller {unknown[0]!r}")
            config = GameConfig(
                map_id=map_id,
                players=seats,
                seed=int(payload.get("seed", 7)),
            )
            self._session = GameSession.new_game(config)
            return self.snapshot_payload()

    def snapshot_payload(self) -> dict[str, Any]:
        with self._lock:
            if self._session is None:
                return {"has_game": False, "meta": self.metadata()}
            snapshot = self._session.snapshot()
            state = snapshot.state.to_dict()
            draw_stack = state.pop("power_plant_draw_stack", [])
            state["deck_count"] = len(draw_stack)
            if draw_stack:
                state["deck_top_back"] = draw_stack[0]["deck_back"]
            elif state["step_3_card_pending"]:
                # "step3" is the engine's face-up placeholder type. The physical
                # Step 3 card is face-down here and shows the standard socket back.
                state["deck_top_back"] = "socket"
            else:
                state["deck_top_back"] = None
            state["bottom_deck_count"] = len(state.pop("power_plant_bottom_stack", []))
            rules = state.pop("rules")
            player_count = len(state["players"])
            player_rules = rules["player_count_rules"][str(player_count)]
            global_parameters = {
                "current_step": state["step"],
                "player_count": player_count,
                "step_2_cities": player_rules["step_2_cities"],
                "end_game_cities": player_rules["end_game_cities"],
                "payment_schedule": rules["payment_schedule"],
                "resource_refill": player_rules["resource_refill"],
            }
            request = snapshot.active_request.to_dict() if snapshot.active_request is not None else None
            active_player = None
            if request is not None:
                active_player = next(
                    (player for player in state["players"] if player["player_id"] == request["player_id"]),
                    None,
                )
            needs_ai_advance = (
                snapshot.winner_result is None
                and (request is None or (active_player is not None and active_player["controller"] != "human"))
            )
            return {
                "has_game": True,
                "state": state,
                "request": request,
                "events": [event.to_dict() for event in snapshot.event_log[-60:]],
                "last_round_summary": _to_payload(snapshot.last_round_summary),
                "winner": _to_payload(snapshot.winner_result),
                "layout": self._layout_payload(state["game_map"]["id"]),
                "global_parameters": global_parameters,
                "needs_ai_advance": needs_ai_advance,
            }

    def submit_intent(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = self._require_session()
            intent_type = str(payload.get("intent_type") or "")
            player_id = str(payload.get("player_id") or "")
            intent_payload = payload.get("payload") or {}
            if not isinstance(intent_payload, dict):
                raise ModelValidationError("intent payload must be an object")
            before_count = len(session.snapshot().event_log)
            session.submit_intent(
                GuiIntent(intent_type=intent_type, player_id=player_id, payload=intent_payload),
                auto_advance=False,
            )
            result = self.snapshot_payload()
            events = session.snapshot().event_log
            if len(events) > before_count and events[-1].level == "error":
                result["error"] = events[-1].message
            return result

    def advance_one(self) -> dict[str, Any]:
        with self._lock:
            session = self._require_session()
            _snapshot, action_applied = session.advance_one_ai_action()
            result = self.snapshot_payload()
            result["action_applied"] = action_applied
            return result

    def quote_build(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = self._require_session()
            snapshot = session.snapshot()
            player_id = str(payload.get("player_id") or "")
            city_ids = [str(city_id) for city_id in payload.get("city_ids", [])]
            if not city_ids:
                return {"valid": False, "cost": 0, "message": "请选择至少一座城市。"}
            cloned = GameState.from_dict(snapshot.state.to_dict())
            before = next(player for player in cloned.players if player.player_id == player_id)
            try:
                quoted = apply_builds(cloned, player_id, city_ids)
            except (ModelValidationError, ValueError) as exc:
                return {"valid": False, "cost": 0, "message": str(exc)}
            after = next(player for player in quoted.players if player.player_id == player_id)
            names = {
                city.id: city.name
                for city in snapshot.state.game_map.cities
            }
            return {
                "valid": True,
                "cost": before.elektro - after.elektro,
                "cities": [names.get(city_id, city_id) for city_id in city_ids],
                "message": " · ".join(names.get(city_id, city_id) for city_id in city_ids),
            }

    def save_city_layout(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = self._require_session()
            if payload.get("confirmed") is not True:
                raise ModelValidationError("layout save requires explicit confirmation")
            snapshot = session.snapshot()
            map_id = str(payload.get("map_id") or "")
            if map_id != snapshot.state.game_map.id:
                raise ModelValidationError("layout map does not match the active game")
            raw_cities = payload.get("cities")
            if not isinstance(raw_cities, dict):
                raise ModelValidationError("cities must be an object")
            expected_ids = [city.id for city in snapshot.state.game_map.cities]
            expected = set(expected_ids)
            supplied = set(raw_cities)
            if supplied != expected:
                missing = sorted(expected - supplied)
                unexpected = sorted(supplied - expected)
                details = []
                if missing:
                    details.append(f"missing: {', '.join(missing)}")
                if unexpected:
                    details.append(f"unexpected: {', '.join(unexpected)}")
                raise ModelValidationError("city coordinate set mismatch (" + "; ".join(details) + ")")

            normalized: dict[str, dict[str, float]] = {}
            for city_id in expected_ids:
                coordinate = raw_cities[city_id]
                if not isinstance(coordinate, dict):
                    raise ModelValidationError(f"invalid coordinate for {city_id}")
                values: dict[str, float] = {}
                for axis in ("x", "y"):
                    raw_value = coordinate.get(axis)
                    if isinstance(raw_value, bool):
                        raise ModelValidationError(f"invalid {axis} coordinate for {city_id}")
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError) as exc:
                        raise ModelValidationError(f"invalid {axis} coordinate for {city_id}") from exc
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ModelValidationError(f"{city_id}.{axis} must be between 0 and 1")
                    values[axis] = round(value, 6)
                normalized[city_id] = values

            layouts = _load_web_layouts(self._web_layouts_path)
            map_layout = layouts.setdefault(map_id, {})
            existing_cities = map_layout.get("cities", {})
            map_layout["cities"] = {
                city_id: {**existing_cities.get(city_id, {}), **normalized[city_id]}
                for city_id in expected_ids
            }
            self._write_web_layouts(layouts)
            self._web_layouts = layouts
            return {
                "saved": True,
                "map_id": map_id,
                "city_count": len(normalized),
                "config_file": self._web_layouts_path.name,
                "layout": self._layout_payload(map_id),
            }

    def _write_web_layouts(self, layouts: dict[str, Any]) -> None:
        temporary = self._web_layouts_path.with_name(f".{self._web_layouts_path.name}.tmp")
        body = json.dumps(layouts, ensure_ascii=False, indent=2) + "\n"
        try:
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(self._web_layouts_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _require_session(self) -> GameSession:
        if self._session is None:
            raise ModelValidationError("start a game before sending actions")
        return self._session

    def _layout_payload(self, map_id: str) -> dict[str, Any]:
        layout = dict(self._web_layouts.get(map_id, {}))
        if map_id in {"germany", "test"}:
            shared = load_board_layout(map_id)
            shared_cities: dict[str, dict[str, Any]] = {}
            for city_id, city_payload in shared.get("cities", {}).items():
                city_layout = dict(city_payload.get("anchor", {}))
                house_slots = [
                    {"x": slot.get("x"), "y": slot.get("y")}
                    for slot in city_payload.get("house_slots", [])
                    if slot.get("x") is not None and slot.get("y") is not None
                ]
                if house_slots:
                    city_layout["house_slots"] = house_slots
                shared_cities[city_id] = city_layout
            overrides = layout.get("cities", {})
            layout["cities"] = {
                city_id: {**city_layout, **overrides.get(city_id, {})}
                for city_id, city_layout in shared_cities.items()
            }
        layout["size"] = MAP_SIZES.get(map_id, {"width": 1600, "height": 1000})
        layout["asset"] = MAP_ASSETS.get(map_id)
        return layout


def _to_payload(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return value


class PowerGridRequestHandler(BaseHTTPRequestHandler):
    controller: PowerGridWebController
    static_root = STATIC_ROOT

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/meta":
            self._write_json(self.controller.metadata())
            return
        if route == "/api/state":
            self._write_json(self.controller.snapshot_payload())
            return
        if route == "/":
            self._write_file(self.static_root / "index.html")
            return
        self._write_static(route)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
            if route == "/api/game":
                result = self.controller.new_game(payload)
            elif route == "/api/intent":
                result = self.controller.submit_intent(payload)
            elif route == "/api/advance":
                result = self.controller.advance_one()
            elif route == "/api/quote-build":
                result = self.controller.quote_build(payload)
            elif route == "/api/layout/cities":
                result = self.controller.save_city_layout(payload)
            else:
                self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            status = HTTPStatus.UNPROCESSABLE_ENTITY if result.get("error") else HTTPStatus.OK
            self._write_json(result, status)
        except (ModelValidationError, ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        length = int(raw_length)
        if length > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_static(self, route: str) -> None:
        relative = unquote(route).lstrip("/")
        candidate = (self.static_root / relative).resolve()
        try:
            candidate.relative_to(self.static_root.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND.value)
            return
        self._write_file(candidate)

    def _write_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND.value)
            return
        body = path.read_bytes()
        content_type, _encoding = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if path.suffix in {".png", ".jpg"} else "no-cache")
        self.end_headers()
        self.wfile.write(body)


def make_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    controller: PowerGridWebController | None = None,
) -> ThreadingHTTPServer:
    active_controller = controller or PowerGridWebController()

    class BoundHandler(PowerGridRequestHandler):
        pass

    BoundHandler.controller = active_controller
    return ThreadingHTTPServer((host, port), BoundHandler)
