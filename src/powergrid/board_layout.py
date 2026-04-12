from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .rules_data import DATA_ROOT


GUI_LAYOUTS_ROOT = DATA_ROOT / "gui_layouts"
DEFAULT_BOARD_LAYOUT_PATH = GUI_LAYOUTS_ROOT / "board_layout_placeholders.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_board_layouts(layout_path: str | Path | None = None) -> dict[str, Any]:
    path = _resolve_layout_path(layout_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_board_layout(map_id: str, layout_path: str | Path | None = None) -> dict[str, Any]:
    payload = load_board_layouts(layout_path)
    try:
        return payload["maps"][map_id]
    except KeyError as exc:
        known_maps = ", ".join(sorted(payload.get("maps", {}).keys())) or "(none)"
        raise ValueError(f"unknown board layout map_id={map_id!r}; known maps: {known_maps}") from exc


def resolve_board_art_path(
    board_art_path: str | None,
    layout_path: str | Path | None = None,
) -> Path | None:
    if not board_art_path:
        return None
    raw_path = Path(board_art_path)
    if raw_path.is_absolute():
        return raw_path

    resolved_layout_path = _resolve_layout_path(layout_path)
    candidates = (
        PROJECT_ROOT / raw_path,
        resolved_layout_path.parent / raw_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_layout_path(layout_path: str | Path | None) -> Path:
    if layout_path is None:
        return DEFAULT_BOARD_LAYOUT_PATH
    path = Path(layout_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()
