"""Shared adjacency-matrix helpers for sequential and future MPI algorithms.

Matrices use ``numpy.inf`` for missing edges and zero on the diagonal. An
off-diagonal zero is a real, zero-weight edge. Matrices may be directed; an
undirected graph is represented by a symmetric matrix. Disconnected graphs are
valid, and shortest-path algorithms should retain infinity for unreachable nodes.
"""

from numbers import Integral, Real
from pathlib import Path

import numpy as np


def validate_adjacency_matrix(matrix) -> np.ndarray:
    """Return an independent, C-contiguous float64 copy of a valid matrix.

    Require a nonempty square matrix containing real numeric, nonnegative edge
    weights or positive infinity, with a zero diagonal. Reject NaN, negative
    infinity, negative weights, and complex or nonnumeric values. Never modify
    the caller's input.
    """
    try:
        array = np.asarray(matrix)
    except (TypeError, ValueError) as exc:
        raise ValueError("Adjacency matrix must be a rectangular numeric array.") from exc

    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise ValueError("Adjacency matrix must be nonempty and square.")
    if array.dtype.kind not in "iuf":
        raise ValueError("Adjacency matrix must contain real numeric values.")

    if np.isnan(array).any():
        raise ValueError("Adjacency matrix must not contain NaN.")
    if np.isneginf(array).any() or (array < 0).any():
        raise ValueError("Adjacency matrix must not contain negative weights or -inf.")
    if not np.all(np.diag(array) == 0):
        raise ValueError("Adjacency matrix must have a zero diagonal.")
    with np.errstate(over="ignore"):
        result = np.array(array, dtype=np.float64, order="C", copy=True)
    if np.any(np.isfinite(array) & ~np.isfinite(result)):
        raise ValueError("Finite edge weights must fit in float64.")
    return result


def generate_graph(
    num_vertices: int,
    density: float = 0.5,
    max_weight: int = 10,
    seed: int | None = 0,
    directed: bool = False,
) -> np.ndarray:
    """Generate a graph without forcing connectivity.

    Each possible edge is present with probability ``density``. Present edges
    have integer weights drawn uniformly from 0 through ``max_weight`` inclusive.
    The same seed and arguments reproduce the same matrix. For undirected graphs,
    each unordered pair is sampled once and mirrored across the diagonal.
    """
    if isinstance(num_vertices, (bool, np.bool_)) or not isinstance(num_vertices, Integral):
        raise ValueError("num_vertices must be a positive integer.")
    if num_vertices <= 0:
        raise ValueError("num_vertices must be a positive integer.")
    if not isinstance(density, Real) or not np.isfinite(density) or not 0 <= density <= 1:
        raise ValueError("density must be a finite number between 0 and 1.")
    if isinstance(max_weight, (bool, np.bool_)) or not isinstance(max_weight, Integral):
        raise ValueError("max_weight must be a nonnegative integer.")
    if max_weight < 0:
        raise ValueError("max_weight must be a nonnegative integer.")
    if not isinstance(directed, (bool, np.bool_)):
        raise ValueError("directed must be a boolean.")

    try:
        rng = np.random.default_rng(seed)
        if directed:
            rows, columns = np.nonzero(~np.eye(num_vertices, dtype=bool))
        else:
            rows, columns = np.triu_indices(num_vertices, k=1)
        present = rng.random(rows.size) < density
        rows, columns = rows[present], columns[present]
        weights = rng.integers(0, max_weight, endpoint=True, size=rows.size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("seed and max_weight must be supported by NumPy's random generator.") from exc

    matrix = np.full((num_vertices, num_vertices), np.inf, dtype=np.float64)
    np.fill_diagonal(matrix, 0)
    matrix[rows, columns] = weights
    if not directed:
        matrix[columns, rows] = weights
    return matrix


def _graph_path(path) -> Path:
    """Require a supported, explicit file extension."""
    try:
        graph_path = Path(path)
    except TypeError as exc:
        raise ValueError("Graph path must name a .npy or .txt file.") from exc
    if graph_path.suffix.lower() not in {".npy", ".txt"}:
        raise ValueError("Graph files must use the .npy or .txt extension.")
    return graph_path


def load_graph(path) -> np.ndarray:
    """Load a validated .npy or whitespace-delimited .txt adjacency matrix.

    Text files use ``inf`` for missing edges. NumPy object/pickle arrays are not
    accepted. A one-vertex text matrix is supported, as are normal square files.
    """
    graph_path = _graph_path(path)
    try:
        if graph_path.suffix.lower() == ".npy":
            with graph_path.open("rb") as graph_file:
                matrix = np.load(graph_file, allow_pickle=False)
        else:
            matrix = np.loadtxt(graph_path, ndmin=2)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not load graph from {graph_path}: {exc}") from exc
    return validate_adjacency_matrix(matrix)


def save_graph(path, matrix) -> None:
    """Save a validated adjacency matrix, preserving zeros and missing edges.

    Text uses 17 significant digits so float64 weights round-trip accurately.
    """
    graph_path = _graph_path(path)
    array = validate_adjacency_matrix(matrix)
    try:
        if graph_path.suffix.lower() == ".npy":
            # A file handle preserves the requested name, including .NPY casing.
            with graph_path.open("wb") as graph_file:
                np.save(graph_file, array, allow_pickle=False)
        else:
            np.savetxt(graph_path, array, fmt="%.17g")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not save graph to {graph_path}: {exc}") from exc
