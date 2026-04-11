from __future__ import annotations

import unittest

from powergrid.gui import create_root
from powergrid.gui.app import PowerGridApp
from powergrid.model import GameConfig, SeatConfig


class PowerGridGuiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = create_root()
        self.root.withdraw()

    def tearDown(self) -> None:
        self.root.destroy()

    def test_app_creation_smoke(self) -> None:
        app = PowerGridApp(self.root)
        app.pack(fill="both", expand=True)

        self.root.update_idletasks()
        self.root.update()

        self.assertTrue(app.launcher.winfo_exists())
        self.assertTrue(app.shell.winfo_exists())

    def test_launcher_can_start_new_game(self) -> None:
        app = PowerGridApp(self.root)
        app.pack(fill="both", expand=True)
        config = GameConfig(
            map_id="germany",
            players=(
                SeatConfig("p1", "Player 1", controller="human"),
                SeatConfig("p2", "Player 2", controller="human"),
                SeatConfig("p3", "Player 3", controller="human"),
            ),
            seed=7,
        )

        app.start_new_game(config)
        self.root.update_idletasks()
        self.root.update()

        self.assertIsNotNone(app.session)
        self.assertEqual(app.session.snapshot().state.phase, "auction")
        self.assertTrue(app.shell.grid_info())

    def test_launcher_can_load_scenario(self) -> None:
        app = PowerGridApp(self.root)
        app.pack(fill="both", expand=True)

        app.load_scenario("opening", seed=7)
        self.root.update_idletasks()
        self.root.update()

        self.assertIsNotNone(app.session)
        self.assertEqual(app.session.snapshot().state.phase, "auction")
        self.assertTrue(app.shell.grid_info())

    def test_shell_renders_multiple_phase_snapshots(self) -> None:
        app = PowerGridApp(self.root)
        app.pack(fill="both", expand=True)

        for scenario_name in ("opening", "resource", "build_test", "step2", "endgame"):
            app.load_scenario(scenario_name, seed=7)
            snapshot = app.session.snapshot()
            app.shell.render(snapshot)
            self.root.update_idletasks()
            self.root.update()

        self.assertTrue(app.shell.header.winfo_exists())


if __name__ == "__main__":
    unittest.main()
