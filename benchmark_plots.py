"""Validate raw benchmark measurements, summarize them, and plot actual results.

Only verified, measured samples from the main phase enter the summaries. Pilot
and warm-up samples remain in the raw CSV for auditing but never affect plots.
Matplotlib is imported only when figures are requested.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import statistics
from typing import Iterable, Mapping


REQUIRED_COLUMNS = (
    "phase", "mode", "processes", "graph_id", "graph_sha256", "vertices",
    "density", "max_weight", "seed", "directed", "sample_kind", "repetition",
    "elapsed_seconds", "verified",
)
GRAPH_COLUMNS = (
    "graph_id", "graph_sha256", "vertices", "density", "max_weight", "seed",
    "directed",
)
SUMMARY_COLUMNS = GRAPH_COLUMNS + (
    "mode", "processes", "repetitions", "mean_seconds", "std_seconds",
    "min_seconds", "max_seconds", "sequential_mean_seconds", "speedup",
)


def _boolean(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("true", "false"):
        return normalized == "true"
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _integer(value: object, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(text)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


def _measurement(row: Mapping[str, object]) -> dict[str, object]:
    missing = set(REQUIRED_COLUMNS) - row.keys()
    if missing:
        raise ValueError(f"Missing measurement fields: {', '.join(sorted(missing))}")
    result = {name: row[name] for name in REQUIRED_COLUMNS}
    result["mode"] = str(row["mode"]).strip().lower()
    if result["mode"] not in ("sequential", "parallel"):
        raise ValueError(f"Unknown benchmark mode: {row['mode']!r}")
    result["processes"] = _integer(row["processes"], "processes", 1)
    if result["mode"] == "sequential" and result["processes"] != 1:
        raise ValueError("Sequential measurements must use exactly one process")
    result["graph_id"] = str(row["graph_id"]).strip()
    if not result["graph_id"]:
        raise ValueError("graph_id must not be empty")
    result["graph_sha256"] = str(row["graph_sha256"]).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", result["graph_sha256"]):
        raise ValueError("graph_sha256 must contain 64 hexadecimal characters")
    for name, minimum in (("vertices", 1), ("max_weight", 0), ("seed", None),
                          ("repetition", 0)):
        result[name] = _integer(row[name], name, minimum)
    try:
        result["density"] = float(row["density"])
        result["elapsed_seconds"] = float(row["elapsed_seconds"])
    except (TypeError, ValueError) as exc:
        raise ValueError("density and elapsed_seconds must be numeric") from exc
    if not math.isfinite(result["density"]) or not 0 <= result["density"] <= 1:
        raise ValueError("density must be finite and between 0 and 1")
    if (not math.isfinite(result["elapsed_seconds"])
            or result["elapsed_seconds"] <= 0):
        raise ValueError("elapsed_seconds must be finite and positive")
    result["directed"] = _boolean(row["directed"], "directed")
    result["verified"] = _boolean(row["verified"], "verified")
    if not result["verified"]:
        raise ValueError("Every measured main-phase sample must be verified")
    return result


def load_measurements(path: str | Path) -> list[dict[str, object]]:
    """Read a raw CSV without discarding its pilot or warm-up records."""
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Raw CSV is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def summarize(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return means, sample SDs, and matching-graph speedups after validation.

    Each graph must have a separate sequential baseline and the same complete
    MPI process-count set, including one process. Every group needs five or
    more unique measured repetitions. A graph ID cannot change hash or inputs.
    """
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    graphs: dict[str, tuple[object, ...]] = {}
    configurations: dict[tuple[object, ...], str] = {}
    repetitions: set[tuple[str, int, str, int]] = set()
    for raw in rows:
        if (str(raw.get("phase", "")).strip().lower() != "main"
                or str(raw.get("sample_kind", "")).strip().lower() != "measured"):
            continue
        row = _measurement(raw)
        graph_id = row["graph_id"]
        identity = tuple(row[name] for name in GRAPH_COLUMNS[1:])
        if graph_id in graphs and graphs[graph_id] != identity:
            raise ValueError(f"Graph {graph_id!r} has inconsistent hashes or parameters")
        graphs[graph_id] = identity
        configuration = tuple(row[name] for name in GRAPH_COLUMNS[2:])
        if configuration in configurations and configurations[configuration] != graph_id:
            raise ValueError("A graph configuration has multiple IDs; graphs cannot be mixed")
        configurations[configuration] = graph_id
        key = (row["mode"], row["processes"], graph_id)
        repetition_key = (*key, row["repetition"])
        if repetition_key in repetitions:
            raise ValueError(f"Duplicate measured repetition: {repetition_key}")
        repetitions.add(repetition_key)
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError("No measured main-phase samples were found")

    expected_counts: set[int] | None = None
    means = {}
    for key, samples in groups.items():
        if len(samples) < 5:
            raise ValueError(f"At least five measured repetitions are required for {key}")
        means[key] = statistics.fmean(row["elapsed_seconds"] for row in samples)
    for graph_id in graphs:
        if ("sequential", 1, graph_id) not in groups:
            raise ValueError(f"Graph {graph_id!r} is missing its sequential baseline")
        counts = {key[1] for key in groups if key[0] == "parallel" and key[2] == graph_id}
        if 1 not in counts:
            raise ValueError(f"Graph {graph_id!r} is missing the one-process MPI result")
        if expected_counts is None:
            expected_counts = counts
        elif counts != expected_counts:
            raise ValueError("All graphs must have the same complete MPI process counts")

    summaries = []
    for key, samples in groups.items():
        first = samples[0]
        times = [row["elapsed_seconds"] for row in samples]
        sequential_mean = means[("sequential", 1, first["graph_id"])]
        result = {name: first[name] for name in GRAPH_COLUMNS}
        result.update(
            mode=first["mode"], processes=first["processes"], repetitions=len(times),
            mean_seconds=means[key], std_seconds=statistics.stdev(times),
            min_seconds=min(times), max_seconds=max(times),
            sequential_mean_seconds=sequential_mean,
            speedup=sequential_mean / means[key],
        )
        summaries.append(result)
    return sorted(summaries, key=lambda row: (
        row["density"], row["vertices"], row["graph_id"],
        row["mode"] == "parallel", row["processes"],
    ))


def write_summary(rows: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _parameters(rows: list[dict[str, object]]) -> str:
    directed = {row["directed"] for row in rows}
    graph_type = ("Directed" if directed == {True} else "Undirected"
                  if directed == {False} else "Directed and undirected")
    weights = ", ".join(str(value) for value in sorted({row["max_weight"] for row in rows}))
    seeds = ", ".join(str(value) for value in sorted({row["seed"] for row in rows}))
    return f"{graph_type}; maximum weight {weights}; seed {seeds}"


def create_plots(rows: list[dict[str, object]], directory: str | Path) -> list[Path]:
    """Write independent PNG and SVG time/speedup figures for every density."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for density in sorted({row["density"] for row in rows}):
        selected = [row for row in rows if row["density"] == density]
        sizes = sorted({row["vertices"] for row in selected})
        counts = sorted({row["processes"] for row in selected if row["mode"] == "parallel"})
        series = [("sequential", 1)] + [("parallel", count) for count in counts]
        stem = "density_" + format(density, "g").replace(".", "p")
        colors = plt.get_cmap("tab10")
        for metric in ("execution_time", "speedup"):
            figure, axes = plt.subplots(figsize=(8.8, 5.7), layout="constrained")
            for index, (mode, count) in enumerate(series):
                if metric == "speedup" and mode == "sequential":
                    continue
                points = sorted((row for row in selected if row["mode"] == mode
                                 and row["processes"] == count),
                                key=lambda row: row["vertices"])
                label = "Sequential" if mode == "sequential" else f"MPI, {count} process" + ("es" if count != 1 else "")
                style = dict(label=label, marker="o", markersize=4,
                             color="black" if mode == "sequential" else colors(index - 1),
                             linestyle="--" if mode == "sequential" else "-", linewidth=1.5)
                xs = [row["vertices"] for row in points]
                if metric == "execution_time":
                    axes.errorbar(xs, [row["mean_seconds"] for row in points],
                                  yerr=[row["std_seconds"] for row in points],
                                  capsize=3, **style)
                else:
                    axes.plot(xs, [row["speedup"] for row in points], **style)
            axes.set_xticks(sizes)
            axes.set_xlabel("Number of vertices")
            axes.grid(True, alpha=0.25)
            if metric == "execution_time":
                axes.set_title(f"Dijkstra execution time — density {density:g}\n{_parameters(selected)}")
                axes.set_ylabel("Elapsed time (seconds; mean ± sample SD)")
                axes.set_ylim(bottom=0)
                caption = ("Complete public dijkstra() calls; MPI uses the maximum elapsed rank time.\n"
                           "Graph generation, reference checks, output, and MPI startup are excluded.")
            else:
                axes.axhline(1, color="0.4", linestyle=":", linewidth=1.2,
                             label="Sequential baseline (1×)")
                axes.set_title(f"Dijkstra speedup — density {density:g}\n{_parameters(selected)}")
                axes.set_ylabel("Speedup (sequential mean / MPI mean)")
                axes.set_ylim(bottom=0)
                caption = ("Matching graph hashes and measured means; values below 1 indicate a slowdown.\n"
                           "Complete public dijkstra() calls, including MPI distribution and reconstruction.")
            axes.legend(fontsize=9)
            figure.supxlabel(caption, fontsize=8)
            for extension in ("png", "svg"):
                path = directory / f"{stem}_{metric}.{extension}"
                metadata = {"Creator": "benchmark_plots.py", "Date": None} if extension == "svg" else {"Software": "benchmark_plots.py"}
                figure.savefig(path, dpi=200, metadata=metadata)
                written.append(path)
            plt.close(figure)
    return written


def write_findings(rows: list[dict[str, object]], path: str | Path) -> None:
    """Describe measured outcomes without inferring unmeasured causes."""
    parallel = [row for row in rows if row["mode"] == "parallel"]
    best = max(parallel, key=lambda row: row["speedup"])
    worst = min(parallel, key=lambda row: row["speedup"])
    slowdowns = sum(row["speedup"] < 1 for row in parallel)
    sizes = sorted({row["vertices"] for row in rows})
    densities = sorted({row["density"] for row in rows})
    counts = sorted({row["processes"] for row in parallel})
    lines = [
        "# Measured benchmark findings", "",
        f"- Vertex counts: {', '.join(map(str, sizes))}.",
        f"- Graph densities: {', '.join(format(d, 'g') for d in densities)}.",
        f"- MPI process counts: {', '.join(map(str, counts))}, including one-process MPI.",
        f"- Verified main-phase measurements: {sum(row['repetitions'] for row in rows)}; "
        f"{len({row['graph_id'] for row in rows})} distinct graph hashes.",
        f"- MPI configurations slower than the matching sequential mean: {slowdowns}/{len(parallel)}.",
        f"- Highest measured speedup: {best['speedup']:.4f}× at {best['vertices']} vertices, "
        f"density {best['density']:g}, {best['processes']} process(es).",
        f"- Lowest measured speedup: {worst['speedup']:.4f}× at {worst['vertices']} vertices, "
        f"density {worst['density']:g}, {worst['processes']} process(es).", "",
        "Times cover complete public dijkstra() calls. MPI times are the maximum of "
        "rank-local elapsed times, collected after timing. Sequential baselines were "
        "measured separately. Pilot and discarded warm-up samples are excluded. "
        "Standard deviations are sample standard deviations (ddof=1).", "",
        f"## Largest measured graph: {max(sizes)} vertices", "",
        "| Density | Implementation | Processes | Mean seconds | Sample SD | Speedup |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["vertices"] == max(sizes):
            lines.append(f"| {row['density']:g} | {row['mode']} | {row['processes']} | "
                         f"{row['mean_seconds']:.6f} | {row['std_seconds']:.6f} | {row['speedup']:.4f}× |")
    lines.extend(("", "These results describe this hardware, runtime, graph set, and timing scope. "
                  "The measurements alone do not establish a cause for observed slowdowns.", ""))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="Raw measured CSV")
    parser.add_argument("--summary", required=True, type=Path, help="Output summary CSV")
    parser.add_argument("--plots-dir", required=True, type=Path, help="Output PNG/SVG directory")
    parser.add_argument("--findings", type=Path, help="Optional measured findings Markdown")
    args = parser.parse_args()
    try:
        rows = summarize(load_measurements(args.raw))
        write_summary(rows, args.summary)
        paths = create_plots(rows, args.plots_dir)
        if args.findings is not None:
            write_findings(rows, args.findings)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(rows)} summary rows to {args.summary} and {len(paths)} plot files.")


if __name__ == "__main__":
    main()
