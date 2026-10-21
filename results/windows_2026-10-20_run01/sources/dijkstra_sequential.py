"""Sequential adjacency-matrix Dijkstra, with assignment source vertex 0."""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from graph_utils import (
    generate_graph,
    load_graph,
    save_graph,
    validate_adjacency_matrix,
)


def dijkstra(matrix: ArrayLike) -> NDArray[np.float64]:
    """Return shortest distances from vertex 0 without changing the input.

    Positive infinity represents a missing edge or an unreachable distance.
    Finite weights must be nonnegative, including valid zero-weight edges.
    A linear minimum search and row relaxation give O(V**2) running time.
    Raise OverflowError if a reachable shortest distance cannot fit in float64.
    """
    graph = validate_adjacency_matrix(matrix)
    num_vertices = graph.shape[0]
    distances = np.full(num_vertices, np.inf, dtype=np.float64)
    visited = np.zeros(num_vertices, dtype=bool)
    distances[0] = 0.0
    overflow_targets = set()

    for _ in range(num_vertices):
        # Choose the closest unsettled vertex. Strict < breaks ties by index.
        current = None
        closest_distance = np.inf
        for vertex in range(num_vertices):
            if not visited[vertex] and distances[vertex] < closest_distance:
                current = vertex
                closest_distance = distances[vertex]

        # No reachable unsettled vertices remain; their distances stay inf.
        if current is None:
            break
        visited[current] = True

        for neighbor in range(num_vertices):
            weight = graph[current, neighbor]
            if visited[neighbor] or not np.isfinite(weight):
                continue
            candidate = float(closest_distance) + float(weight)
            if not np.isfinite(candidate):
                overflow_targets.add(neighbor)
                continue
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate

    # An overflowing route may have a finite alternative discovered later.
    if any(np.isinf(distances[vertex]) for vertex in overflow_targets):
        raise OverflowError("A reachable shortest distance exceeds the float64 range.")
    return distances


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--graph", type=Path, help="Load a .npy or whitespace .txt matrix."
    )
    inputs.add_argument(
        "--vertices", type=int, default=5, help="Generated vertex count (default: 5)."
    )
    parser.add_argument(
        "--density", type=float, default=0.5, help="Edge probability (default: 0.5)."
    )
    parser.add_argument(
        "--max-weight", type=int, default=10, help="Largest generated weight (default: 10)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument("--directed", action="store_true", help="Generate a directed graph.")
    parser.add_argument("--save-graph", type=Path, help="Save the input matrix as .npy or .txt.")
    args = parser.parse_args()

    try:
        if args.graph is not None:
            graph = load_graph(args.graph)
        else:
            graph = generate_graph(
                args.vertices,
                density=args.density,
                max_weight=args.max_weight,
                seed=args.seed,
                directed=args.directed,
            )
        distances = dijkstra(graph)
        if args.save_graph is not None:
            save_graph(args.save_graph, graph)
    except (OSError, ValueError, OverflowError) as error:
        parser.error(str(error))

    print("Shortest distances from source 0:")
    for vertex, distance in enumerate(distances):
        value = "inf (unreachable)" if np.isinf(distance) else f"{distance:.17g}"
        print(f"0 -> {vertex}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
