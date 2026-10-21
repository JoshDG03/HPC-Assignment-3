"""Time complete public Dijkstra calls on a previously saved graph manifest.

Run sequential mode with plain Python and parallel mode through mpiexec. Graph
loading, hash checks, reference comparisons, timing reductions, and disk writes
are outside the measured intervals. Warm-ups are retained in the CSV with an
explicit sample_kind and must be excluded when calculating measured summaries.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from mpi4py import MPI
import psutil

from graph_utils import load_graph


RAW_COLUMNS = [
    "phase", "mode", "processes", "graph_id", "graph_sha256", "vertices",
    "density", "max_weight", "seed", "directed", "sample_kind", "repetition",
    "elapsed_seconds", "rank_elapsed_seconds", "verified", "root_rss_bytes",
    "root_peak_rss_bytes", "total_rank_rss_bytes", "total_rank_peak_rss_bytes",
]


def canonical_sha256(array):
    """Hash C-order little-endian float64 values, independent of .npy headers."""
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _memory_snapshot():
    memory = psutil.Process().memory_info()
    return {
        "rss_bytes": memory.rss,
        # Windows reports the process-lifetime peak working set. It is not a
        # per-solve allocation measurement and includes interpreter/MPI setup.
        "peak_rss_bytes": getattr(memory, "peak_wset", None),
    }


def _rank_metadata(comm):
    process = psutil.Process()
    try:
        affinity = process.cpu_affinity()
    except (AttributeError, psutil.Error):
        affinity = None
    prefixes = (
        "I_MPI_", "MPI_", "PMI_", "PMIX_", "OMP_", "MKL_", "OPENBLAS_",
        "KMP_", "MSMPI_", "HYDRA_", "FI_",
    )
    return {
        "rank": comm.Get_rank(),
        "pid": process.pid,
        "processor_name": MPI.Get_processor_name(),
        "affinity_logical_cpu_ids": affinity,
        "environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith(prefixes)
        },
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_architecture_bits": 8 * np.dtype(np.intp).itemsize,
        "numpy_version": np.__version__,
        "mpi4py_version": __import__("mpi4py").__version__,
        "psutil_version": psutil.__version__,
        "mpi_vendor": MPI.get_vendor(),
        "mpi_library_version": MPI.Get_library_version().strip(),
        "mpi_wtick_seconds": MPI.Wtick(),
        "memory": _memory_snapshot(),
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("sequential", "parallel"), required=True)
    parser.add_argument("--phase", choices=("pilot", "main"), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--run-metadata", type=Path)
    return parser


def _read_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    graphs = manifest.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("Manifest must contain a nonempty graphs list.")
    required = {
        "graph_id", "vertices", "density", "max_weight", "seed", "directed",
        "path", "reference_path", "graph_sha256", "file_sha256", "reference_sha256",
    }
    seen = set()
    for graph in graphs:
        if not isinstance(graph, dict) or not required <= graph.keys():
            raise ValueError(f"Each graph needs all fields: {sorted(required)}")
        if graph["graph_id"] in seen:
            raise ValueError(f"Duplicate graph ID: {graph['graph_id']}")
        seen.add(graph["graph_id"])
        for key in ("path", "reference_path"):
            item_path = Path(graph[key])
            graph[key] = str(
                (path.parent / item_path).resolve()
                if not item_path.is_absolute() else item_path.resolve()
            )
    return manifest


def _load_inputs(spec):
    graph_path = Path(spec["path"])
    actual_file_hash = file_sha256(graph_path)
    if actual_file_hash != spec["file_sha256"]:
        raise ValueError(f"Graph file hash mismatch for {spec['graph_id']}.")
    graph = load_graph(graph_path)
    actual_graph_hash = canonical_sha256(graph)
    if actual_graph_hash != spec["graph_sha256"]:
        raise ValueError(f"Graph value hash mismatch for {spec['graph_id']}.")
    if graph.shape != (spec["vertices"], spec["vertices"]):
        raise ValueError(f"Vertex count mismatch for {spec['graph_id']}.")
    if not spec["directed"] and not np.array_equal(graph, graph.T):
        raise ValueError(f"Undirected graph is not symmetric: {spec['graph_id']}.")
    finite = graph[np.isfinite(graph)]
    if finite.max() > spec["max_weight"]:
        raise ValueError(f"Graph exceeds maximum weight: {spec['graph_id']}.")
    reference = np.load(spec["reference_path"], allow_pickle=False)
    if reference.dtype.kind not in "iuf" or reference.shape != (spec["vertices"],):
        raise ValueError(f"Invalid reference shape or dtype for {spec['graph_id']}.")
    if np.isnan(reference).any() or (reference < 0).any() or reference[0] != 0:
        raise ValueError(f"Invalid reference distances for {spec['graph_id']}.")
    actual_reference_hash = canonical_sha256(reference)
    if actual_reference_hash != spec["reference_sha256"]:
        raise ValueError(f"Reference hash mismatch for {spec['graph_id']}.")
    return graph, reference, {
        "graph_id": spec["graph_id"],
        "file_sha256": actual_file_hash,
        "graph_sha256": actual_graph_hash,
        "reference_sha256": actual_reference_hash,
        "verified_before_timing": True,
    }


def _write_metadata(path, metadata):
    # The caller exclusively creates this path before starting any solves.
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sum_peaks(snapshots):
    values = [snapshot["peak_rss_bytes"] for snapshot in snapshots]
    return sum(values) if all(value is not None for value in values) else None


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    args = None
    manifest = None
    csv_stream = None
    setup_error = None
    exit_code = 0

    # Root handles expected setup errors before peers enter any solver calls.
    if rank == 0:
        try:
            args = _parser().parse_args()
            if args.repetitions < 1 or (args.phase == "main" and args.repetitions < 5):
                raise ValueError("Use at least one pilot repetition and five main repetitions.")
            if args.warmups < 1:
                raise ValueError("At least one discarded warm-up is required.")
            launcher_markers = (
                "PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "MPI_LOCALRANKID",
            )
            launcher_size = max(
                int(os.environ.get(key, "1"))
                for key in ("PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE")
            )
            if args.mode == "sequential" and (
                size != 1 or launcher_size > 1
                or any(key in os.environ for key in launcher_markers)
            ):
                raise ValueError("Sequential mode must use plain Python, outside mpiexec.")
            args.manifest = args.manifest.resolve()
            args.output = args.output.resolve()
            if args.run_metadata is None:
                args.run_metadata = args.output.with_suffix(".metadata.json")
            args.run_metadata = args.run_metadata.resolve()
            if args.output == args.run_metadata:
                raise ValueError("CSV and metadata paths must differ.")
            if args.output.exists() or args.run_metadata.exists():
                raise ValueError("Output CSV and metadata must both be new paths.")
            manifest = _read_manifest(args.manifest)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.run_metadata.parent.mkdir(parents=True, exist_ok=True)
            csv_stream = args.output.open("x", encoding="utf-8", newline="")
            with args.run_metadata.open("x", encoding="utf-8") as stream:
                stream.write("{}\n")
        except SystemExit as exc:
            exit_code = exc.code
        except (OSError, TypeError, ValueError) as exc:
            setup_error = str(exc)
            exit_code = 2

    args, manifest, exit_code, setup_error = comm.bcast(
        (args, manifest, exit_code, setup_error), root=0
    )
    if exit_code or args is None or manifest is None:
        if rank == 0:
            if setup_error:
                print(f"Benchmark setup failed: {setup_error}", file=sys.stderr, flush=True)
            if csv_stream is not None:
                csv_stream.close()
        return exit_code

    if args.mode == "parallel":
        from dijkstra_parallel import dijkstra
    else:
        from dijkstra_sequential import dijkstra

    ranks = comm.gather(_rank_metadata(comm), root=0)
    metadata = None
    writer = None
    if rank == 0:
        metadata = {
            "schema_version": 1,
            "started_utc": _utc_now(),
            "completed_utc": None,
            "status": "running",
            "command_argv": [sys.executable, *sys.argv],
            "working_directory": str(Path.cwd()),
            "manifest_path": str(args.manifest),
            "manifest_file_sha256": file_sha256(args.manifest),
            "raw_csv_path": str(args.output),
            "mode": args.mode,
            "phase": args.phase,
            "processes": size,
            "warmups_per_graph": args.warmups,
            "measured_repetitions_per_graph": args.repetitions,
            "timing_method": (
                "MPI.Wtime immediately around the complete public dijkstra() call; "
                "parallel Barrier before each call; maximum local elapsed time "
                "reduced after stopping the timers."
            ),
            "excluded_from_timing": [
                "MPI startup", "imports", "graph generation/loading", "hash checks",
                "reference verification", "pre-call Barrier", "elapsed-time reduction",
                "memory sampling", "printing", "CSV/metadata writes",
            ],
            "memory_method": (
                "RSS sampled outside timing after each solve; peak RSS is Windows "
                "process-lifetime peak working set, including interpreter/MPI setup. "
                "Summed per-rank peaks are not a simultaneous aggregate peak."
            ),
            "initial_ranks": ranks,
            "verified_graphs": [],
            "samples": [],
        }
        writer = csv.DictWriter(csv_stream, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        csv_stream.flush()
        _write_metadata(args.run_metadata, metadata)

    for spec in manifest["graphs"]:
        graph = reference = None
        load_error = None
        if rank == 0:
            try:
                graph, reference, hash_record = _load_inputs(spec)
                metadata["verified_graphs"].append(hash_record)
            except (OSError, TypeError, ValueError) as exc:
                load_error = str(exc)
        load_error = comm.bcast(load_error, root=0)
        if load_error is not None:
            if rank == 0:
                metadata["status"] = "input_error"
                metadata["error"] = load_error
                metadata["completed_utc"] = _utc_now()
                _write_metadata(args.run_metadata, metadata)
                csv_stream.close()
                print(f"Benchmark input failed: {load_error}", file=sys.stderr, flush=True)
            return 2

        measured_times = []
        root_memory = None
        for sample_index in range(args.warmups + args.repetitions):
            warmup = sample_index < args.warmups
            sample_kind = "warmup" if warmup else "measured"
            repetition = sample_index + 1 if warmup else sample_index - args.warmups + 1

            if args.mode == "parallel":
                comm.Barrier()
                start = MPI.Wtime()
                result = dijkstra(graph, comm)
                local_elapsed = MPI.Wtime() - start
            else:
                start = MPI.Wtime()
                result = dijkstra(graph)
                local_elapsed = MPI.Wtime() - start

            # All communication about timings is outside the interval above.
            elapsed = comm.reduce(local_elapsed, op=MPI.MAX, root=0)
            rank_elapsed = comm.gather(local_elapsed, root=0)
            verification_error = None
            try:
                if rank == 0:
                    np.testing.assert_array_equal(result, reference)
                    if canonical_sha256(graph) != spec["graph_sha256"]:
                        raise AssertionError("Solver modified the input matrix.")
                elif result is not None:
                    raise AssertionError("A non-root rank returned distances.")
            except (AssertionError, TypeError, ValueError) as exc:
                verification_error = f"Rank {rank}: {exc}"
            errors = comm.allgather(verification_error)
            verified = not any(error is not None for error in errors)
            del result
            memories = comm.gather(_memory_snapshot(), root=0)

            if rank == 0:
                root_memory = memories[0]
                row = {
                    key: spec[key] for key in (
                        "graph_id", "graph_sha256", "vertices", "density",
                        "max_weight", "seed", "directed",
                    )
                }
                row.update({
                    "phase": args.phase,
                    "mode": args.mode,
                    "processes": size,
                    "sample_kind": sample_kind,
                    "repetition": repetition,
                    "elapsed_seconds": elapsed,
                    "rank_elapsed_seconds": json.dumps(rank_elapsed),
                    "verified": verified,
                    "root_rss_bytes": root_memory["rss_bytes"],
                    "root_peak_rss_bytes": root_memory["peak_rss_bytes"],
                    "total_rank_rss_bytes": sum(item["rss_bytes"] for item in memories),
                    "total_rank_peak_rss_bytes": _sum_peaks(memories),
                })
                writer.writerow(row)
                csv_stream.flush()
                metadata["samples"].append({
                    **row,
                    "rank_elapsed_seconds": rank_elapsed,
                    "rank_memory": memories,
                    "verification_errors": errors,
                })
                _write_metadata(args.run_metadata, metadata)
                if not warmup:
                    measured_times.append(elapsed)
            if not verified:
                raise RuntimeError(f"Distance verification failed: {errors}")

        if rank == 0:
            print(
                f"{args.phase} {args.mode} p={size} {spec['graph_id']}: "
                f"{args.repetitions} verified repetitions, "
                f"{min(measured_times):.6f}..{max(measured_times):.6f} s; "
                f"root RSS={root_memory['rss_bytes'] / 2**20:.1f} MiB",
                flush=True,
            )
        # Do not retain a previous root graph while loading the next graph.
        del graph, reference

    final_ranks = comm.gather(_rank_metadata(comm), root=0)
    if rank == 0:
        metadata["final_ranks"] = final_ranks
        metadata["completed_utc"] = _utc_now()
        metadata["status"] = "complete"
        _write_metadata(args.run_metadata, metadata)
        csv_stream.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"Rank {MPI.COMM_WORLD.Get_rank()}: benchmark failure: {exc}",
            file=sys.stderr,
            flush=True,
        )
        # An unexpected rank-local error must not strand a peer in a collective.
        MPI.COMM_WORLD.Abort(1)
