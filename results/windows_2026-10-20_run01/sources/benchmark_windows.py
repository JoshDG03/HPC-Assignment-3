"""Prepare a fresh Windows pilot, then run a separate reproducible benchmark."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil

from dijkstra_sequential import dijkstra
from graph_utils import generate_graph, save_graph


ROOT = Path(__file__).resolve().parent
CONTROLLED_ENV = {
    "I_MPI_PIN": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
RELEVANT_ENV = [
    *CONTROLLED_ENV, "I_MPI_PIN_DOMAIN", "I_MPI_PIN_ORDER", "I_MPI_WAIT_MODE",
    "I_MPI_ASYNC_PROGRESS", "I_MPI_FABRICS", "I_MPI_DEBUG", "OMP_PROC_BIND",
    "OMP_PLACES", "KMP_AFFINITY", "MSMPI_BIN",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(array):
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def powershell_command(argv):
    return "& " + " ".join("'" + str(item).replace("'", "''") + "'" for item in argv)


def git_output(*arguments):
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def source_hashes():
    paths = sorted([*ROOT.glob("*.py"), *(ROOT / "tests").glob("*.py")])
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths}


def check_sources(metadata):
    for name, expected in metadata["source_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError(f"Source changed after the pilot started: {name}")


def windows_hardware():
    query = """
@{
 processors = @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed)
 computer = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory, NumberOfLogicalProcessors
 operating_system = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, FreePhysicalMemory, TotalVisibleMemorySize
 power_plan = (powercfg /getactivescheme | Out-String).Trim()
} | ConvertTo-Json -Depth 6
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", query],
        capture_output=True, text=True, check=True,
    )
    if result.stderr.strip():
        raise RuntimeError(f"Windows hardware query failed: {result.stderr}")
    hardware = json.loads(result.stdout)
    battery = psutil.sensors_battery()
    hardware.update({
        "psutil_physical_cores": psutil.cpu_count(logical=False),
        "psutil_logical_processors": psutil.cpu_count(),
        "available_ram_bytes": psutil.virtual_memory().available,
        "controller_affinity_logical_cpu_ids": psutil.Process().cpu_affinity(),
        "battery": None if battery is None else battery._asdict(),
    })
    return hardware


def run_command(argv, output, metadata, log_name, environment, timeout=900):
    """Run one child at a time; startup and log I/O are never benchmark timings."""
    record = {
        "argv": list(map(str, argv)), "powershell": powershell_command(argv),
        "environment_overrides": CONTROLLED_ENV, "started_utc": utc_now(),
        "log": f"logs/{log_name}.txt",
    }
    print(record["powershell"], flush=True)
    process = subprocess.Popen(
        list(map(str, argv)), cwd=ROOT, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    started = time.perf_counter()
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, timeout=15, check=False,
        )
        stdout, _ = process.communicate(timeout=15)
        record["timed_out"] = True
    record.update({
        "completed_utc": utc_now(), "exit_code": process.returncode,
        "launcher_wall_seconds_not_a_benchmark": time.perf_counter() - started,
    })
    (output / "logs" / f"{log_name}.txt").write_text(stdout, encoding="utf-8")
    metadata["commands"].append(record)
    write_json(output / "metadata.json", metadata)
    print(stdout.rstrip(), flush=True)
    if process.returncode != 0 or record.get("timed_out"):
        raise RuntimeError(f"Child command failed; inspect {record['log']}")
    return stdout


def merge_csv(paths, target):
    columns = None
    with target.open("x", encoding="utf-8", newline="") as stream:
        writer = None
        for path in paths:
            with path.open(encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if columns is None:
                    columns = reader.fieldnames
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                if columns != reader.fieldnames:
                    raise ValueError(f"CSV schema mismatch: {path}")
                writer.writerows(reader)


def worker_runs(output, manifest, phase, counts, repetitions, metadata, environment):
    folder = output / phase if phase == "pilot" else output
    parts = folder / "raw_parts"
    parts.mkdir(parents=True)
    paths = []
    for mode, processes in [("sequential", 1), *[("parallel", count) for count in counts]]:
        label = f"{mode}_p{processes}"
        csv_path = parts / f"{label}.csv"
        command = [
            sys.executable, str(ROOT / "benchmark_worker.py"),
            "--manifest", str(manifest), "--output", str(csv_path),
            "--mode", mode, "--phase", phase, "--repetitions", str(repetitions),
            "--warmups", "1", "--run-metadata", str(folder / f"{label}.json"),
        ]
        if mode == "parallel":
            command = [metadata["mpi_launcher"], "-n", str(processes), *command]
        run_command(command, output, metadata, f"{phase}_{label}", environment)
        paths.append(csv_path)
    merge_csv(paths, folder / "raw.csv")


def pilot(args):
    output = args.output.resolve()
    if len(set(args.sizes)) < 4 or any(size < 1 for size in args.sizes):
        raise ValueError("The pilot needs at least four distinct positive sizes.")
    if args.processes != sorted(set(args.processes)) or not {1, 2, 4, 8} <= set(args.processes):
        raise ValueError("Process counts must be sorted/unique and include 1, 2, 4, 8.")
    if max(args.processes) > len(psutil.Process().cpu_affinity()):
        raise ValueError("Requested MPI count exceeds this process's available logical CPUs.")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("graphs", "references", "logs", "pilot", "sources"):
        (output / folder).mkdir()
    environment = {**os.environ, **CONTROLLED_ENV}
    metadata = {
        "schema_version": 1, "status": "preparing_pilot", "created_utc": utc_now(),
        "client_timezone": "America/La_Paz",
        "platform": platform.platform(), "hardware": windows_hardware(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_at_start": git_output("status", "--short"),
        "source_sha256": source_hashes(),
        "packages": dict(sorted((item.metadata["Name"], item.version) for item in distributions())),
        "python_executable": sys.executable, "python_version": sys.version,
        "mpi_launcher": str(args.mpiexec.resolve()),
        "mpi_launcher_sha256": file_sha256(args.mpiexec.resolve()),
        "requested_process_counts": args.processes,
        "controlled_environment": CONTROLLED_ENV,
        "inherited_relevant_environment": {key: os.environ.get(key) for key in RELEVANT_ENV},
        "effective_relevant_environment": {key: environment.get(key) for key in RELEVANT_ENV},
        "affinity_policy": "Intel pinning disabled; Windows schedules workers within recorded inherited CPU affinity; library thread pools limited to one.",
        "timing_contract": {
            "clock": "MPI.Wtime", "parallel_statistic": "maximum locally elapsed public-call time, reduced after stop",
            "includes": ["validation", "packing", "Scatter", "distributed computation and reductions", "gather", "result reconstruction"],
            "excludes": ["MPI startup", "input generation/loading", "reference creation/verification", "pre-call Barrier", "elapsed-time reduction", "printing", "disk writes"],
            "sequential_baseline": "plain single-process Python child, complete public dijkstra()",
            "warmups_per_configuration": 1, "warmups_in_statistics": False,
        },
        "commands": [{"argv": [sys.executable, *sys.argv], "powershell": powershell_command([sys.executable, *sys.argv]), "role": "pilot_controller"}],
        "preflight": {},
    }
    for name in metadata["source_sha256"]:
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(output / "metadata.json", metadata)
    # Proof at every included process count, before any pilot measurements.
    for processes in args.processes:
        stdout = run_command(
            [metadata["mpi_launcher"], "-n", str(processes), sys.executable, "-m", "tests.mpi_dijkstra_cases"],
            output, metadata, f"correctness_p{processes}", environment,
        )
        result = json.loads(stdout.strip().splitlines()[-1])
        if result.get("status") != "PASS":
            raise RuntimeError("MPI correctness preflight did not pass.")
        metadata["preflight"][str(processes)] = result
    graphs = []
    for density in args.densities:
        for vertices in args.sizes:
            graph_id = f"n{vertices}_d{str(density).replace('.', 'p')}_seed{args.seed}"
            graph = generate_graph(vertices, density=density, max_weight=args.max_weight, seed=args.seed)
            graph_path = output / "graphs" / f"{graph_id}.npy"
            reference_path = output / "references" / f"{graph_id}.npy"
            save_graph(graph_path, graph)
            reference = dijkstra(graph)  # Untimed independent reference preparation.
            np.save(reference_path, reference, allow_pickle=False)
            graphs.append({
                "graph_id": graph_id, "vertices": vertices, "density": density,
                "max_weight": args.max_weight, "seed": args.seed, "directed": False,
                "path": f"graphs/{graph_id}.npy", "reference_path": f"references/{graph_id}.npy",
                "graph_sha256": canonical_sha256(graph), "file_sha256": file_sha256(graph_path),
                "reference_sha256": canonical_sha256(reference),
                "reference_file_sha256": file_sha256(reference_path),
                "matrix_bytes": graph.nbytes,
                "realized_density": float(np.count_nonzero(np.isfinite(graph) & ~np.eye(vertices, dtype=bool)) / (vertices * (vertices - 1))) if vertices > 1 else 0,
            })
            print(f"Prepared {graph_id}, {graph.nbytes / 2**20:.2f} MiB, SHA256={graphs[-1]['graph_sha256']}", flush=True)
            del graph, reference
    manifest = {"schema_version": 1, "graphs": graphs, "hash_format": "SHA256 of C-order little-endian float64 values"}
    write_json(output / "manifest.json", manifest)
    metadata["status"] = "running_pilot"
    metadata["pilot"] = {"sizes": args.sizes, "densities": args.densities, "processes": sorted(set([1, 4, max(args.processes)])), "measured_repetitions": 1}
    worker_runs(output, output / "manifest.json", "pilot", metadata["pilot"]["processes"], 1, metadata, environment)
    with (output / "pilot" / "raw.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    decision_data = {
        "maximum_public_call_seconds": max(float(row["elapsed_seconds"]) for row in rows),
        "maximum_root_rss_bytes": max(int(row["root_rss_bytes"]) for row in rows),
        "maximum_sum_rank_rss_bytes": max(int(row["total_rank_rss_bytes"]) for row in rows),
        "maximum_sum_rank_lifetime_peak_bytes_not_simultaneous": max(int(row["total_rank_peak_rss_bytes"]) for row in rows),
        "available_ram_bytes_after_pilot": psutil.virtual_memory().available,
        "all_samples_verified": all(row["verified"] == "True" for row in rows),
    }
    write_json(output / "pilot" / "resource_check.json", decision_data)
    check_sources(metadata)
    metadata.update({"status": "pilot_complete", "pilot_completed_utc": utc_now(), "pilot_resource_check": decision_data})
    write_json(output / "metadata.json", metadata)
    print("Pilot complete: " + json.dumps(decision_data), flush=True)


def run(args):
    output = args.output.resolve()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "pilot_complete":
        raise ValueError("Run needs a completed pilot and fresh main output paths.")
    if args.repetitions < 5:
        raise ValueError("At least five measured repetitions are required.")
    check_sources(metadata)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    sizes = sorted(set(args.sizes or metadata["pilot"]["sizes"]))
    if len(sizes) < 4:
        raise ValueError("Final matrix needs at least four sizes.")
    selected = [graph for graph in manifest["graphs"] if graph["vertices"] in sizes]
    if {graph["vertices"] for graph in selected} != set(sizes):
        raise ValueError("Final sizes must have been prepared for the pilot.")
    # Paths remain relative to this manifest's parent, as in the pilot.
    write_json(output / "main_manifest.json", {**manifest, "graphs": selected})
    metadata.update({
        "status": "running_main", "main_started_utc": utc_now(),
        "final_matrix": {"sizes": sizes, "densities": sorted({graph["density"] for graph in selected}), "processes": metadata["requested_process_counts"], "measured_repetitions": args.repetitions, "warmups": 1, "selection_reason": args.decision},
    })
    metadata["commands"].append({"argv": [sys.executable, *sys.argv], "powershell": powershell_command([sys.executable, *sys.argv]), "role": "main_controller"})
    write_json(output / "metadata.json", metadata)
    environment = {**os.environ, **CONTROLLED_ENV}
    metadata["main_effective_relevant_environment"] = {
        key: environment.get(key) for key in RELEVANT_ENV
    }
    metadata["main_hardware_snapshot"] = windows_hardware()
    worker_runs(output, output / "main_manifest.json", "main", metadata["requested_process_counts"], args.repetitions, metadata, environment)
    run_command(
        [sys.executable, str(ROOT / "benchmark_plots.py"), "--raw", str(output / "raw.csv"),
         "--summary", str(output / "summary.csv"), "--plots-dir", str(output / "plots"), "--findings", str(output / "findings.md")],
        output, metadata, "analysis", environment,
    )
    check_sources(metadata)
    metadata.update({"status": "complete", "completed_utc": utc_now(), "source_sha256_after": source_hashes(), "all_original_source_hashes_unchanged": True, "available_ram_bytes_after_main": psutil.virtual_memory().available})
    metadata["artifact_sha256"] = {
        str(path.relative_to(output)).replace("\\", "/"): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    write_json(output / "metadata.json", metadata)
    print(f"Benchmark complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("pilot", help="Create a fresh result directory and run the pilot.")
    first.add_argument("--output", type=Path, required=True)
    first.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000, 2000])
    first.add_argument("--densities", type=float, nargs="+", default=[0.1, 0.5])
    first.add_argument("--seed", type=int, default=42)
    first.add_argument("--max-weight", type=int, default=1000)
    first.add_argument("--processes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    first.add_argument("--mpiexec", type=Path, default=ROOT / ".venv" / "Library" / "bin" / "mpiexec.exe")
    second = commands.add_parser("run", help="Finalize the matrix from pilot evidence, then measure.")
    second.add_argument("--output", type=Path, required=True)
    second.add_argument("--sizes", type=int, nargs="+")
    second.add_argument("--repetitions", type=int, default=5)
    second.add_argument("--decision", required=True, help="Reason for the finalized sizes/process counts.")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This controller captures native Windows hardware and process metadata.")
    (pilot if args.command == "pilot" else run)(args)


if __name__ == "__main__":
    main()
