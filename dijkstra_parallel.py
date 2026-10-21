"""MPI adjacency-matrix Dijkstra from source 0, with column ownership."""

import argparse
from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI
from numpy.typing import ArrayLike, NDArray

from graph_utils import (
    generate_graph,
    load_graph,
    save_graph,
    validate_adjacency_matrix,
)


def _scatter_graph(matrix: ArrayLike | None, comm: MPI.Comm):
    """Pack equal column blocks on root 0 and Scatter actual matrix weights."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0
    send_buffer = None
    num_vertices = 0
    error = None

    if rank == root:
        try:
            graph = validate_adjacency_matrix(matrix)
            num_vertices = graph.shape[0]
            block = (num_vertices + size - 1) // size
            # Each rank receives a contiguous (V, block) matrix. The last
            # column block(s) may include dummy vertices with no incident edges.
            send_buffer = np.full(
                (size, num_vertices, block), np.inf, dtype=np.float64
            )
            for owner in range(size):
                first = owner * block
                count = max(0, min(block, num_vertices - first))
                send_buffer[owner, :, :count] = graph[:, first:first + count]
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            error = str(exc)

    # Root failures must reach every rank before any rank enters Scatter.
    num_vertices, error = comm.bcast((num_vertices, error), root=root)
    if error is not None:
        raise ValueError(f"Cannot distribute graph: {error}")

    block = (num_vertices + size - 1) // size
    local_weights = np.empty((num_vertices, block), dtype=np.float64)
    comm.Scatter(send_buffer, local_weights, root=root)
    return local_weights, num_vertices, rank * block


def _solve_local(
    local_weights: NDArray[np.float64],
    num_vertices: int,
    first_vertex: int,
    comm: MPI.Comm,
) -> NDArray[np.float64]:
    """Run distributed Dijkstra; return this rank's padded distance block.

    This kernel excludes input preparation, final gathering, and verification.
    Local minimum scans and scalar relaxations match the sequential baseline.
    """
    block = local_weights.shape[1]
    count = max(0, min(block, num_vertices - first_vertex))
    distances = np.full(block, np.inf, dtype=np.float64)
    visited = np.zeros(block, dtype=bool)
    overflow_targets = set()
    if first_vertex == 0:
        distances[0] = 0.0

    for _ in range(num_vertices):
        closest_distance = np.inf
        closest_vertex = num_vertices  # Sentinel for no reachable local vertex.
        for neighbor in range(count):
            if not visited[neighbor] and distances[neighbor] < closest_distance:
                closest_distance = distances[neighbor]
                closest_vertex = first_vertex + neighbor

        # MINLOC chooses distance first, then the lowest global vertex ID.
        closest_distance, current = comm.allreduce(
            (float(closest_distance), closest_vertex), op=MPI.MINLOC
        )
        # All ranks see the same choice, including the unreachable sentinel.
        if not np.isfinite(closest_distance):
            break
        if first_vertex <= current < first_vertex + count:
            visited[current - first_vertex] = True

        for neighbor in range(count):
            weight = local_weights[current, neighbor]
            if visited[neighbor] or not np.isfinite(weight):
                continue
            candidate = float(closest_distance) + float(weight)
            if not np.isfinite(candidate):
                overflow_targets.add(neighbor)
                continue
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate

    # Match the reference's deferred overflow check. A later finite route can
    # supersede an overflowing one; every rank must agree whether to raise.
    overflow = any(np.isinf(distances[vertex]) for vertex in overflow_targets)
    if comm.allreduce(overflow, op=MPI.LOR):
        raise OverflowError("A reachable shortest distance exceeds the float64 range.")
    return distances


def dijkstra(
    matrix: ArrayLike | None, comm: MPI.Comm = MPI.COMM_WORLD
) -> NDArray[np.float64] | None:
    """Collectively compute distances from vertex 0; return them only on root 0.

    Pass the complete matrix on rank 0 and None on other ranks. Rank r owns
    vertices [r*ceil(V/P), (r+1)*ceil(V/P)), intersected with [0, V).
    Its contiguous local matrix contains all incoming weights for those vertices:
    local_weights[u, j] = matrix[u, first_vertex + j]. Padding is never visited.

    Root validates before Scatter. Invalid matrices raise ValueError on all
    ranks; unrepresentable reachable distances raise OverflowError on all ranks.
    All ranks must call this function in the same order. The input is unchanged.
    """
    local_weights, num_vertices, first_vertex = _scatter_graph(matrix, comm)
    local_distances = _solve_local(
        local_weights, num_vertices, first_vertex, comm
    )
    # Lowercase gather collects Python objects in rank order, including padding.
    blocks = comm.gather(local_distances, root=0)
    if comm.Get_rank() == 0:
        return np.concatenate(blocks)[:num_vertices].copy()
    return None


def _parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--verify", action="store_true", help="Compare every distance with sequential Dijkstra."
    )
    return parser


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0
    args = None
    graph = None
    run = False
    exit_code = 0
    error = None

    # Parse, load/generate, and optionally save on root. Broadcast the outcome,
    # including --help and argument errors, before workers enter the algorithm.
    if rank == root:
        try:
            args = _parser().parse_args()
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
            if args.save_graph is not None:
                save_graph(args.save_graph, graph)
            run = True
        except SystemExit as exc:
            exit_code = exc.code
        except (OSError, TypeError, ValueError, OverflowError) as exc:
            exit_code, error = 2, str(exc)

    run, exit_code, error = comm.bcast((run, exit_code, error), root=root)
    if not run:
        if rank == root and error is not None:
            print(f"Error: {error}", file=sys.stderr, flush=True)
        return exit_code

    try:
        distances = dijkstra(graph, comm)
    except (ValueError, OverflowError) as exc:
        # These errors are already synchronized by dijkstra().
        if rank == root:
            print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 2

    if rank == root:
        try:
            if args.verify:
                from dijkstra_sequential import dijkstra as sequential_dijkstra

                np.testing.assert_array_equal(distances, sequential_dijkstra(graph))
        except (AssertionError, ValueError, OverflowError) as exc:
            exit_code, error = 1, f"Verification failed: {exc}"

    # A reference mismatch is communicated before any worker exits successfully.
    exit_code, error = comm.bcast((exit_code, error), root=root)
    if rank == root:
        if error is not None:
            print(f"Error: {error}", file=sys.stderr, flush=True)
        else:
            if args.verify:
                print(f"Verification: PASS ({len(distances)} distances, {size} ranks)")
            print("Shortest distances from source 0:")
            for vertex, distance in enumerate(distances):
                value = "inf (unreachable)" if np.isinf(distance) else f"{distance:.17g}"
                print(f"0 -> {vertex}: {value}")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Unexpected rank-local failures must not strand peers in collectives.
        print(
            f"Rank {MPI.COMM_WORLD.Get_rank()}: unexpected failure: {exc}",
            file=sys.stderr,
            flush=True,
        )
        MPI.COMM_WORLD.Abort(1)
