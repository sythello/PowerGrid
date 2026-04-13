from __future__ import annotations

import argparse

from powergrid.gui import create_root, PowerGridApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tkinter PowerGrid GUI.")
    parser.add_argument("--scenario")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--board-render-mode", choices=("drawn", "asset"), default="drawn")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Render the GUI once and exit without entering the interactive main loop.",
    )
    args = parser.parse_args()

    root = create_root()
    app = PowerGridApp(root, board_render_mode=args.board_render_mode)
    app.pack(fill="both", expand=True)
    if args.scenario:
        app.load_scenario(args.scenario, seed=args.seed)
    else:
        app.show_launcher()
    root.update_idletasks()
    root.update()
    if args.smoke_test:
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
