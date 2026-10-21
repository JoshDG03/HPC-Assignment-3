"""Independent summary statistics and rejection of untrustworthy measurements."""

import csv
import math
from pathlib import Path
import tempfile
import unittest

from benchmark_plots import REQUIRED_COLUMNS, load_measurements, summarize, write_summary


def measured_rows(graph_id="small", vertices=10, graph_hash="a" * 64):
    rows = []
    for mode, processes, times in (
        ("sequential", 1, [4, 5, 6, 7, 8]),
        ("parallel", 1, [2, 3, 4, 5, 6]),
        ("parallel", 2, [1, 2, 3, 4, 5]),
    ):
        for repetition, elapsed in enumerate(times, start=1):
            rows.append(dict(
                phase="main", mode=mode, processes=processes, graph_id=graph_id,
                graph_sha256=graph_hash, vertices=vertices, density=0.1,
                max_weight=1000, seed=42, directed=False, sample_kind="measured",
                repetition=repetition, elapsed_seconds=elapsed, verified=True,
            ))
    return rows


class BenchmarkSummaryTests(unittest.TestCase):
    def test_known_statistics_speedups_and_warmup_pilot_exclusion(self):
        rows = measured_rows()
        rows.append(dict(rows[0], sample_kind="warmup", elapsed_seconds=999, verified=False))
        rows.append(dict(rows[0], phase="pilot", elapsed_seconds=888, verified=False))
        summary = summarize(rows)
        self.assertEqual(len(summary), 3)
        by_mode = {(row["mode"], row["processes"]): row for row in summary}
        sequential = by_mode["sequential", 1]
        self.assertEqual(sequential["repetitions"], 5)
        self.assertEqual(sequential["mean_seconds"], 6)
        self.assertEqual(sequential["speedup"], 1)
        self.assertAlmostEqual(sequential["std_seconds"], math.sqrt(2.5))
        self.assertEqual((sequential["min_seconds"], sequential["max_seconds"]), (4, 8))
        self.assertEqual(by_mode["parallel", 1]["speedup"], 1.5)
        self.assertEqual(by_mode["parallel", 2]["mean_seconds"], 3)
        self.assertEqual(by_mode["parallel", 2]["speedup"], 2)
        self.assertEqual(by_mode["parallel", 2]["sequential_mean_seconds"], 6)

    def test_slowdown_retained_and_baseline_matched_per_graph(self):
        first = measured_rows()
        second = measured_rows("large", 20, "b" * 64)
        for row in second:
            row["elapsed_seconds"] *= 0.5 if row["mode"] == "sequential" else 2
        summary = summarize(first + second)
        result = next(row for row in summary if row["graph_id"] == "large"
                      and row["mode"] == "parallel" and row["processes"] == 2)
        self.assertEqual(result["sequential_mean_seconds"], 3)
        self.assertEqual(result["mean_seconds"], 6)
        self.assertEqual(result["speedup"], 0.5)

    def test_rejects_hash_or_parameter_changes_for_one_graph_id(self):
        for changes in ({"graph_sha256": "b" * 64}, {"seed": 43}, {"vertices": 11},
                        {"directed": True}, {"density": 0.5}):
            with self.subTest(changes=changes):
                rows = measured_rows()
                rows[-1].update(changes)
                with self.assertRaisesRegex(ValueError, "inconsistent hashes or parameters"):
                    summarize(rows)

    def test_rejects_duplicate_repetition(self):
        rows = measured_rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate measured repetition"):
            summarize(rows)

    def test_rejects_unverified_or_invalid_elapsed_times(self):
        for value in (0, -0.1, math.nan, math.inf, "not a time"):
            with self.subTest(elapsed=value):
                rows = measured_rows()
                rows[0]["elapsed_seconds"] = value
                with self.assertRaises(ValueError):
                    summarize(rows)
        rows = measured_rows()
        rows[0]["verified"] = False
        with self.assertRaisesRegex(ValueError, "must be verified"):
            summarize(rows)

    def test_rejects_insufficient_samples_and_missing_baseline(self):
        with self.assertRaisesRegex(ValueError, "At least five"):
            summarize(measured_rows()[1:])
        rows = [row for row in measured_rows() if row["mode"] == "parallel"]
        with self.assertRaisesRegex(ValueError, "missing its sequential baseline"):
            summarize(rows)

    def test_requires_one_process_mpi_and_consistent_complete_process_counts(self):
        rows = [row for row in measured_rows()
                if not (row["mode"] == "parallel" and row["processes"] == 1)]
        with self.assertRaisesRegex(ValueError, "one-process MPI"):
            summarize(rows)
        rows = measured_rows()
        rows += [row for row in measured_rows("large", 20, "b" * 64)
                 if not (row["mode"] == "parallel" and row["processes"] == 2)]
        with self.assertRaisesRegex(ValueError, "same complete MPI process counts"):
            summarize(rows)

    def test_rejects_distinct_graph_ids_for_same_configuration(self):
        rows = measured_rows() + measured_rows("other", 10, "b" * 64)
        with self.assertRaisesRegex(ValueError, "multiple IDs"):
            summarize(rows)

    def test_rejects_bad_inputs(self):
        for changes in (
            {"mode": "mystery"}, {"processes": 0}, {"processes": 2},
            {"vertices": 0}, {"vertices": 1.5}, {"graph_id": ""},
            {"graph_sha256": "not a hash"}, {"density": math.nan},
            {"density": 1.01}, {"max_weight": -1}, {"seed": True},
            {"directed": "maybe"}, {"repetition": -1}, {"verified": "maybe"},
        ):
            with self.subTest(changes=changes):
                rows = measured_rows()
                rows[0].update(changes)
                with self.assertRaises(ValueError):
                    summarize(rows)

    def test_rejects_no_measured_main_samples(self):
        with self.assertRaisesRegex(ValueError, "No measured main-phase"):
            summarize([])

    def test_csv_roundtrip_including_string_boolean_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path = Path(temporary) / "raw.csv"
            with raw_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerows(measured_rows())
            summary = summarize(load_measurements(raw_path))
            self.assertEqual(summary[0]["mean_seconds"], 6)
            summary_path = Path(temporary) / "nested" / "summary.csv"
            write_summary(summary, summary_path)
            with summary_path.open(encoding="utf-8", newline="") as source:
                restored = list(csv.DictReader(source))
            self.assertEqual(len(restored), 3)
            self.assertEqual(float(restored[-1]["speedup"]), 2)
            self.assertEqual(restored[0]["graph_sha256"], "a" * 64)

    def test_csv_missing_header_fields_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path = Path(temporary) / "bad.csv"
            raw_path.write_text("phase,mode\nmain,parallel\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_measurements(raw_path)


if __name__ == "__main__":
    unittest.main()
