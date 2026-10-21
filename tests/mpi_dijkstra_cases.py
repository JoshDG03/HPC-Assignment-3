"""Collective correctness checks; run with ``mpiexec -n P python -m tests.mpi_dijkstra_cases``.

Only rank 0 supplies a graph to the parallel API. Assertions are exchanged by
all ranks before the next case, so a failed assertion cannot strand a peer in
the next Dijkstra call. Unexpected exceptions abort the communicator.
"""

import json
import traceback

import numpy as np
from mpi4py import MPI

from dijkstra_parallel import dijkstra as parallel_dijkstra
from dijkstra_sequential import dijkstra as sequential_dijkstra
from graph_utils import generate_graph


def known_cases():
    """Graphs with independently calculated distances, including Phase 1 cases."""
    inf = np.inf
    yield "connected_undirected", [
        [0, 4, 1, inf, inf],
        [4, 0, 2, 1, inf],
        [1, 2, 0, 5, 8],
        [inf, 1, 5, 0, 3],
        [inf, inf, 8, 3, 0],
    ], [0, 3, 1, 4, 7]
    yield "disconnected_component", [
        [0, 2, inf, inf, inf],
        [2, 0, 3, inf, inf],
        [inf, 3, 0, inf, inf],
        [inf, inf, inf, 0, 1],
        [inf, inf, inf, 1, 0],
    ], [0, 2, 5, inf, inf]
    yield "zero_weight_directed", [
        [0, 0, 5, inf],
        [inf, 0, 2, 7],
        [inf, inf, 0, 1],
        [inf, inf, inf, 0],
    ], [0, 0, 2, 3]
    yield "directed_edges_not_reversed", [
        [0, inf, 8], [1, 0, inf], [inf, 2, 0],
    ], [0, 10, 8]
    yield "equal_cost_paths", [
        [0, 1, 1, 5], [1, 0, inf, 2], [1, inf, 0, 2], [5, 2, 2, 0],
    ], [0, 1, 1, 3]
    yield "decimal_weights", [
        [0, 0.1, 0.8, inf],
        [inf, 0, 0.2, 0.7],
        [inf, inf, 0, 0.4],
        [inf, inf, inf, 0],
    ], [0, 0.1, 0.3, 0.7]
    yield "singleton", [[0]], [0]
    yield "no_edges", [
        [0, inf, inf], [inf, 0, inf], [inf, inf, 0],
    ], [0, inf, inf]
    yield "input_not_modified", [
        [0, 7, 2], [7, 0, 1], [2, 1, 0],
    ], [0, 3, 2]
    yield "overflow_with_direct_alternative", [
        [0, 1e308, 1e308], [inf, 0, 1e308], [inf, inf, 0],
    ], [0, 1e308, 1e308]
    yield "overflow_with_later_finite_alternative", [
        [0, 1e308, 1.1e308, inf],
        [inf, 0, inf, 1e308],
        [inf, inf, 0, 1e307],
        [inf, inf, inf, 0],
    ], [0, 1e308, 1.1e308, 1.2e308]
    yield "directed_zero_cycle", [
        [0, 0, inf, 8, inf],
        [inf, 0, 0, inf, inf],
        [0, inf, 0, 2, inf],
        [inf, inf, inf, 0, 0],
        [inf, inf, inf, inf, 0],
    ], [0, 0, 0, 2, 2]
    # At four processes this case leaves two ranks with only padding.
    yield "two_vertices", [[0, 7], [inf, 0]], [0, 7]
    chain = np.full((7, 7), inf)
    np.fill_diagonal(chain, 0)
    chain[np.arange(6), np.arange(1, 7)] = 1
    yield "seven_vertex_directed_chain", chain, [0, 1, 2, 3, 4, 5, 6]
    # A transpose checks that the API validates/copies noncontiguous input.
    transposed = np.array([[0, 2, 9], [8, 0, 3], [4, 1, 0]]).T
    yield "noncontiguous_input", transposed, [0, 7, 4]


def invalid_cases():
    """Validation must fail collectively, including unreachable bad edges."""
    inf = np.inf
    yield "empty", np.empty((0, 0))
    yield "nonsquare", np.zeros((2, 3))
    yield "nan", [[0, np.nan], [inf, 0]]
    yield "negative_infinity", [[0, -inf], [inf, 0]]
    yield "complex", [[0, 1j], [0, 0]]
    yield "negative_unreachable_edge", [
        [0, inf, inf], [inf, 0, -1], [inf, 2, 0],
    ]
    yield "nonzero_diagonal", [[1, 2], [inf, 0]]
    yield "nonnumeric", [["0", "2"], ["inf", "0"]]
    yield "ragged", [[0, 1], [0]]


class CollectiveChecks:
    """Keep ranks synchronized around both successful and rejected inputs."""

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.counts = {"known": 0, "random": 0, "recovery": 0, "invalid": 0, "overflow": 0}

    def check_every_rank(self, label, assertion):
        failure = None
        try:
            assertion()
        except Exception as error:
            failure = f"rank {self.rank}: {type(error).__name__}: {error}"
        failures = self.comm.allgather(failure)
        if any(item is not None for item in failures):
            details = "\n".join(item for item in failures if item is not None)
            raise AssertionError(f"{label}\n{details}")

    def good_graph(self, label, graph, category, known_distances=None):
        """Check every distance exactly against the same sequential graph."""
        supplied_graph = graph if self.rank == 0 else None
        before = np.array(graph, copy=True) if self.rank == 0 else None
        distances = parallel_dijkstra(supplied_graph, comm=self.comm)

        def verify():
            if self.rank != 0:
                if distances is not None:
                    raise AssertionError("Only root 0 should return final distances.")
                return
            if not isinstance(distances, np.ndarray):
                raise AssertionError("Root result must be a NumPy array.")
            if distances.dtype != np.dtype(np.float64):
                raise AssertionError("Distances must use float64.")
            reference = sequential_dijkstra(graph)
            # Equality includes every finite and infinite entry and the shape;
            # graph weights and additions are identical in both algorithms.
            np.testing.assert_array_equal(distances, reference)
            if known_distances is not None:
                np.testing.assert_allclose(
                    reference, known_distances, rtol=1e-14, atol=1e-14
                )
                np.testing.assert_allclose(
                    distances, known_distances, rtol=1e-14, atol=1e-14
                )
            np.testing.assert_array_equal(np.asarray(graph), before)

        self.check_every_rank(label, verify)
        self.counts[category] += 1

    def rejected_graph(self, label, graph, error_type, category):
        """An expected error must reach every rank and allow another solve."""
        received = None
        try:
            parallel_dijkstra(graph if self.rank == 0 else None, comm=self.comm)
        except Exception as error:
            received = error

        def verify():
            if not isinstance(received, error_type):
                raise AssertionError(
                    f"Expected {error_type.__name__}; received {received!r}."
                )
            if self.rank == 0:
                try:
                    sequential_dijkstra(graph)
                except error_type:
                    pass
                else:
                    raise AssertionError("Sequential reference accepted rejected graph.")

        self.check_every_rank(label, verify)
        self.counts[category] += 1
        self.good_graph(
            f"recovery_after_{label}", [[0, 2], [np.inf, 0]], "recovery", [0, 2]
        )

    def run(self):
        for label, graph, expected in known_cases():
            self.good_graph(label, graph, "known", expected)

        # 8 sizes x 3 seeds x 2 orientations x 3 densities = 144 graphs.
        # Seed 0 uses exclusively zero weights; the other seeds mix 0..9.
        for vertices in (1, 2, 3, 5, 7, 8, 11, 17):
            for seed in (0, 7, 42):
                for directed in (False, True):
                    for density in (0.0, 0.25, 1.0):
                        graph = None
                        if self.rank == 0:
                            graph = generate_graph(
                                vertices, density=density,
                                max_weight=0 if seed == 0 else 9,
                                seed=seed, directed=directed,
                            )
                        label = (
                            f"random_n{vertices}_seed{seed}_"
                            f"directed{directed}_density{density}"
                        )
                        self.good_graph(label, graph, "random")

        for label, graph in invalid_cases():
            self.rejected_graph(label, graph, ValueError, "invalid")

        self.rejected_graph(
            "unrepresentable_reachable_distance",
            [[0, 1e308, np.inf], [np.inf, 0, 1e308], [np.inf, np.inf, 0]],
            OverflowError, "overflow",
        )
        per_rank_counts = self.comm.gather(self.counts, root=0)
        if self.rank == 0:
            if any(counts != self.counts for counts in per_rank_counts):
                raise AssertionError("Ranks completed different numbers of cases.")
            print(json.dumps({
                "status": "PASS",
                "processes": self.comm.Get_size(),
                "successful_graphs": sum(self.counts[key] for key in ("known", "random", "recovery")),
                "cases": self.counts,
                "per_rank_cases": per_rank_counts,
            }, sort_keys=True), flush=True)


def main():
    comm = MPI.COMM_WORLD
    try:
        CollectiveChecks(comm).run()
    except Exception:
        print(f"MPI correctness failure on rank {comm.Get_rank()}:", flush=True)
        traceback.print_exc()
        comm.Abort(1)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
