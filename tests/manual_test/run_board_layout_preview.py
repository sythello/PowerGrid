from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk

from powergrid.board_layout import (
    DEFAULT_BOARD_LAYOUT_PATH,
    load_board_layout,
    load_board_layouts,
    resolve_board_art_path,
)


CITY_ANCHOR_COLOR = "#facc15"
CITY_SLOT_COLOR = "#38bdf8"
RESOURCE_COLORS = {
    "coal": "#111827",
    "oil": "#7c3aed",
    "garbage": "#a16207",
    "uranium": "#22c55e",
}
FALLBACK_WIDTH = 1600
FALLBACK_HEIGHT = 1000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual overlay preview for board-layout coordinates on a board image."
    )
    parser.add_argument("--map", dest="map_id", default="germany", choices=("germany", "usa"))
    parser.add_argument("--layout", default=str(DEFAULT_BOARD_LAYOUT_PATH))
    parser.add_argument(
        "--image",
        help="Optional image override. PNG is recommended because Tkinter PhotoImage support is limited.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide city and resource labels to reduce clutter.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Render once and exit without entering the interactive main loop.",
    )
    args = parser.parse_args()

    layouts = load_board_layouts(args.layout)
    map_layout = load_board_layout(args.map_id, args.layout)
    board_art_path = resolve_board_art_path(
        args.image or map_layout.get("board_art", {}).get("image_path"),
        args.layout,
    )

    root = tk.Tk()
    root.title(f"Board Layout Preview - {args.map_id}")

    summary_var = tk.StringVar()
    summary = tk.Label(root, textvariable=summary_var, anchor="w", justify="left")
    summary.pack(fill="x", padx=8, pady=(8, 0))

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=8, pady=8)
    canvas = tk.Canvas(frame, background="#0f172a", highlightthickness=0)
    y_scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    x_scroll = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_rowconfigure(0, weight=1)
    canvas.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")

    image, image_status, width, height = _load_canvas_image(canvas, board_art_path)
    if image is not None:
        canvas.image = image
        canvas.create_image(0, 0, anchor="nw", image=image)
    else:
        _draw_fallback_background(canvas, width, height)

    city_count = _draw_city_overlays(canvas, map_layout, width, height, not args.hide_labels)
    resource_count = _draw_resource_market_overlays(
        canvas,
        map_layout,
        layouts.get("resource_market_template"),
        width,
        height,
        not args.hide_labels,
    )
    canvas.configure(scrollregion=(0, 0, width, height))

    summary_var.set(
        "\n".join(
            [
                f"Map: {args.map_id}",
                f"Layout: {Path(args.layout).resolve()}",
                f"Image: {board_art_path if board_art_path is not None else '(none configured)'}",
                image_status,
                f"Rendered overlays: city anchors/slots={city_count}, resource slots={resource_count}",
            ]
        )
    )

    root.update_idletasks()
    root.update()
    if args.smoke_test:
        root.destroy()
        return
    root.mainloop()


def _load_canvas_image(
    canvas: tk.Canvas,
    board_art_path: Path | None,
) -> tuple[tk.PhotoImage | None, str, int, int]:
    if board_art_path is None:
        return None, "No board image configured. Showing fallback canvas.", FALLBACK_WIDTH, FALLBACK_HEIGHT
    if not board_art_path.exists():
        return (
            None,
            f"Board image not found at {board_art_path}. Showing fallback canvas.",
            FALLBACK_WIDTH,
            FALLBACK_HEIGHT,
        )
    try:
        image = tk.PhotoImage(file=str(board_art_path))
    except tk.TclError as exc:
        return (
            None,
            f"Could not load image {board_art_path} ({exc}). Use PNG for the preview tool.",
            FALLBACK_WIDTH,
            FALLBACK_HEIGHT,
        )
    return image, f"Loaded image size: {image.width()}x{image.height()}", image.width(), image.height()


def _draw_fallback_background(canvas: tk.Canvas, width: int, height: int) -> None:
    canvas.create_rectangle(0, 0, width, height, fill="#e2e8f0", outline="")
    step = 100
    for x in range(0, width + 1, step):
        canvas.create_line(x, 0, x, height, fill="#cbd5e1")
    for y in range(0, height + 1, step):
        canvas.create_line(0, y, width, y, fill="#cbd5e1")
    canvas.create_text(
        width / 2,
        40,
        text="Board image missing or unsupported - showing fallback calibration grid",
        fill="#0f172a",
        font=("Helvetica", 16, "bold"),
    )


def _draw_city_overlays(
    canvas: tk.Canvas,
    map_layout: dict,
    width: int,
    height: int,
    show_labels: bool,
) -> int:
    count = 0
    for city_id, city_payload in sorted(map_layout.get("cities", {}).items()):
        anchor = city_payload.get("anchor", {})
        anchor_x = anchor.get("x")
        anchor_y = anchor.get("y")
        if _has_point(anchor_x, anchor_y):
            px, py = _normalize_point(anchor_x, anchor_y, width, height)
            radius = max(6, int(width * map_layout.get("city_defaults", {}).get("hit_radius", 0.018)))
            canvas.create_oval(
                px - radius,
                py - radius,
                px + radius,
                py + radius,
                outline=CITY_ANCHOR_COLOR,
                width=2,
            )
            canvas.create_line(px - radius, py, px + radius, py, fill=CITY_ANCHOR_COLOR)
            canvas.create_line(px, py - radius, px, py + radius, fill=CITY_ANCHOR_COLOR)
            if show_labels:
                label_offset = city_payload.get(
                    "label_offset",
                    map_layout.get("city_defaults", {}).get("label_offset", {"x": 0.0, "y": -0.03}),
                )
                label_x = px + int(width * float(label_offset.get("x", 0.0)))
                label_y = py + int(height * float(label_offset.get("y", -0.03)))
                canvas.create_text(
                    label_x,
                    label_y,
                    text=city_id,
                    fill="#f8fafc",
                    font=("Helvetica", 10, "bold"),
                )
            count += 1

        for slot in city_payload.get("house_slots", ()):
            slot_x = slot.get("x")
            slot_y = slot.get("y")
            if not _has_point(slot_x, slot_y):
                continue
            px, py = _normalize_point(slot_x, slot_y, width, height)
            canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill=CITY_SLOT_COLOR, outline="")
            count += 1
    return count


def _draw_resource_market_overlays(
    canvas: tk.Canvas,
    map_layout: dict,
    template_payload: dict | None,
    width: int,
    height: int,
    show_labels: bool,
) -> int:
    resource_market = map_layout.get("resource_market") or template_payload or {}
    count = 0
    for resource, prices in resource_market.items():
        if resource == "notes":
            continue
        color = RESOURCE_COLORS.get(resource, "#ef4444")
        for price, slots in prices.items():
            first_label_drawn = False
            for slot in slots:
                slot_x = slot.get("x")
                slot_y = slot.get("y")
                if not _has_point(slot_x, slot_y):
                    continue
                px, py = _normalize_point(slot_x, slot_y, width, height)
                canvas.create_rectangle(px - 5, py - 5, px + 5, py + 5, fill=color, outline="#f8fafc")
                if show_labels and not first_label_drawn:
                    canvas.create_text(
                        px + 20,
                        py - 10,
                        text=f"{resource}:{price}",
                        fill="#f8fafc",
                        font=("Helvetica", 9),
                        anchor="w",
                    )
                    first_label_drawn = True
                count += 1
    return count


def _normalize_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return int(float(x) * width), int(float(y) * height)


def _has_point(x: object, y: object) -> bool:
    return x is not None and y is not None


if __name__ == "__main__":
    main()
