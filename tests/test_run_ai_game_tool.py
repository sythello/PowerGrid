from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from tempfile import TemporaryDirectory
import unittest

from powergrid.tools.run_ai_game import main as run_ai_game_main


class RunAiGameToolTests(unittest.TestCase):
    def test_cli_tool_runs_game_and_writes_log(self) -> None:
        with TemporaryDirectory() as tempdir:
            output_path = f"{tempdir}/game_log.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = run_ai_game_main(
                    [
                        "--seed",
                        "7",
                        "--controllers",
                        "ai_deterministic",
                        "ai_heuristics",
                        "ai_deterministic",
                        "--output",
                        output_path,
                    ]
                )

            self.assertEqual(result, 0)
            with open(output_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["config"]["map_id"], "germany")
            self.assertEqual(len(payload["config"]["players"]), 3)
            self.assertEqual(payload["game_log"][0]["event_type"], "session_start")
            self.assertTrue(payload["game_log"])
            self.assertIn("AI Game Completed", stdout.getvalue())
            self.assertIn("Winner:", stdout.getvalue())
            self.assertIn("Wrote game log to", stdout.getvalue())

    def test_single_controller_argument_repeats_for_all_seats(self) -> None:
        with TemporaryDirectory() as tempdir:
            output_path = f"{tempdir}/game_log.json"
            result = run_ai_game_main(
                [
                    "--players",
                    "3",
                    "--controllers",
                    "ai_deterministic",
                    "--output",
                    output_path,
                ]
            )

            self.assertEqual(result, 0)
            with open(output_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(
                [player["controller"] for player in payload["config"]["players"]],
                ["ai_deterministic", "ai_deterministic", "ai_deterministic"],
            )


if __name__ == "__main__":
    unittest.main()
