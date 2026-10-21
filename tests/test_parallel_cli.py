"""Opt-in MPI subprocess checks for successful and failing command-line runs.

Run with RUN_MPI_TESTS=1. MPIEXEC may name a different matching MPI launcher;
the default is the project's native Windows Intel MPI installation. Workers use
the interpreter running these tests, so invoke them with the project .venv.
"""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
PROCESS_COUNTS = (1, 2, 4)
MPI_TIMEOUT_SECONDS = 45


@unittest.skipUnless(
    os.environ.get("RUN_MPI_TESTS") == "1",
    "Set RUN_MPI_TESTS=1 to run MPI command-line integration checks.",
)
class ParallelCommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        launcher = os.environ.get(
            "MPIEXEC", str(REPOSITORY / ".venv" / "Library" / "bin" / "mpiexec.exe")
        )
        cls.launcher = shutil.which(launcher)
        if cls.launcher is None:
            raise RuntimeError(
                f"MPI launcher not found: {launcher}. Set MPIEXEC to its executable path."
            )

    def run_mpi(self, process_count, *arguments):
        command = [
            self.launcher,
            "-n",
            str(process_count),
            sys.executable,
            str(REPOSITORY / "dijkstra_parallel.py"),
            *map(str, arguments),
        ]
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name != "nt",
        )
        try:
            stdout, stderr = process.communicate(timeout=MPI_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # Stop only this test's launcher and its workers, including workers
            # blocked inside an MPI collective after an incorrect error path.
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    shell=False,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stdout, stderr = process.communicate(timeout=10)
            self.fail(
                f"MPI command timed out after {MPI_TIMEOUT_SECONDS}s: {command!r}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def assert_success(self, result, expected_distances):
        detail = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        self.assertEqual(result.returncode, 0, detail)
        self.assertEqual(result.stdout.count("Shortest distances from source 0:"), 1, detail)
        self.assertEqual(result.stdout.count("Verification: PASS"), 1, detail)
        self.assertNotIn("Traceback", result.stdout + result.stderr, detail)
        actual_lines = [
            line for line in result.stdout.splitlines() if line.startswith("0 -> ")
        ]
        expected_lines = []
        for vertex, distance in enumerate(expected_distances):
            value = "inf (unreachable)" if np.isinf(distance) else f"{distance:.17g}"
            expected_lines.append(f"0 -> {vertex}: {value}")
        self.assertEqual(actual_lines, expected_lines, detail)

    def assert_clean_error(self, result):
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 2, output)
        self.assertEqual(output.lower().count("error:"), 1, output)
        self.assertNotIn("Traceback", output, output)
        self.assertNotIn("Shortest distances from source 0:", output, output)
        self.assertNotIn("Verification: PASS", output, output)

    def test_load_example_and_verify_every_distance(self):
        # Five vertices create uneven partitions with both two and four ranks.
        for count in PROCESS_COUNTS:
            with self.subTest(processes=count):
                result = self.run_mpi(
                    count,
                    "--graph",
                    REPOSITORY / "examples" / "zero_weight_disconnected.txt",
                    "--verify",
                )
                self.assert_success(result, [0, 0, 2, 3, np.inf])

    def test_generate_directed_graph_with_more_ranks_than_vertices(self):
        for count in PROCESS_COUNTS:
            with self.subTest(processes=count):
                result = self.run_mpi(
                    count,
                    "--vertices", "3",
                    "--density", "1",
                    "--max-weight", "0",
                    "--seed", "42",
                    "--directed",
                    "--verify",
                )
                self.assert_success(result, [0, 0, 0])

    def test_invalid_graph_files_exit_without_waiting_workers(self):
        with tempfile.TemporaryDirectory(prefix="dijkstra_mpi_cli_") as directory:
            folder = Path(directory)
            invalid_graphs = {
                "missing": folder / "does_not_exist.txt",
                "negative_unreachable_edge": folder / "negative.txt",
                "nonsquare": folder / "nonsquare.txt",
                "nan": folder / "nan.txt",
            }
            invalid_graphs["negative_unreachable_edge"].write_text(
                "0 inf inf\ninf 0 -1\ninf 2 0\n", encoding="utf-8"
            )
            invalid_graphs["nonsquare"].write_text("0 1 2\n1 0 3\n", encoding="utf-8")
            invalid_graphs["nan"].write_text("0 nan\ninf 0\n", encoding="utf-8")
            for name, graph_path in invalid_graphs.items():
                for count in PROCESS_COUNTS:
                    with self.subTest(graph=name, processes=count):
                        self.assert_clean_error(
                            self.run_mpi(count, "--graph", graph_path, "--verify")
                        )

    def test_invalid_generation_and_argparse_exit_cleanly(self):
        invalid_options = {
            "zero_vertices": ("--vertices", "0"),
            "unknown_argument": ("--not-an-option",),
        }
        for name, arguments in invalid_options.items():
            for count in PROCESS_COUNTS:
                with self.subTest(options=name, processes=count):
                    self.assert_clean_error(self.run_mpi(count, *arguments))

    def test_help_exits_cleanly_on_all_ranks(self):
        for count in PROCESS_COUNTS:
            with self.subTest(processes=count):
                result = self.run_mpi(count, "--help")
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, output)
                self.assertEqual(output.count("usage:"), 1, output)
                self.assertIn("--verify", output, output)
                self.assertNotIn("Error:", output, output)
                self.assertNotIn("Traceback", output, output)

    def test_unrepresentable_distance_exits_cleanly(self):
        with tempfile.TemporaryDirectory(prefix="dijkstra_mpi_overflow_") as directory:
            graph_path = Path(directory) / "overflow.npy"
            np.save(
                graph_path,
                np.array(
                    [
                        [0, 1e308, np.inf],
                        [np.inf, 0, 1e308],
                        [np.inf, np.inf, 0],
                    ],
                    dtype=np.float64,
                ),
                allow_pickle=False,
            )
            for count in PROCESS_COUNTS:
                with self.subTest(processes=count):
                    self.assert_clean_error(
                        self.run_mpi(count, "--graph", graph_path, "--verify")
                    )


if __name__ == "__main__":
    unittest.main()
