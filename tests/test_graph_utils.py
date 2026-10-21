"""Validation, reproducible generation and graph file compatibility."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from graph_utils import generate_graph, load_graph, save_graph, validate_adjacency_matrix


class GraphValidationTests(unittest.TestCase):
    def test_preserves_zero_edges_missing_edges_and_direction(self):
        graph = [[0, 0, np.inf], [np.inf, 0, 4], [2, np.inf, 0]]
        result = validate_adjacency_matrix(graph)
        np.testing.assert_array_equal(result, graph)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertTrue(result.flags.c_contiguous)

    def test_normalizes_integer_noncontiguous_input(self):
        graph = np.array([[0, 2, 3], [2, 0, 5], [3, 5, 0]], dtype=np.int32).T
        self.assertFalse(graph.flags.c_contiguous)
        before = graph.copy()
        result = validate_adjacency_matrix(graph)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, before)
        np.testing.assert_array_equal(graph, before)

    def test_rejects_invalid_shapes(self):
        invalid = [[], np.empty((0, 0)), [0, 1], [[0, 1, 2], [1, 0, 3]], [[[0]]]]
        for graph in invalid:
            with self.subTest(graph=graph):
                with self.assertRaises((ValueError, TypeError)):
                    validate_adjacency_matrix(graph)

    def test_rejects_negative_nonfinite_and_nonnumeric_weights(self):
        invalid = [
            [[0, -1], [2, 0]],
            [[0, -np.inf], [2, 0]],
            [[0, np.nan], [2, 0]],
            [[0, 1 + 2j], [2, 0]],
            [[0, "missing"], [2, 0]],
        ]
        for graph in invalid:
            with self.subTest(graph=graph):
                with self.assertRaises((ValueError, TypeError)):
                    validate_adjacency_matrix(graph)

    def test_requires_zero_diagonal(self):
        for diagonal in (1, -1, np.inf, np.nan):
            with self.subTest(diagonal=diagonal):
                with self.assertRaises((ValueError, TypeError)):
                    validate_adjacency_matrix([[diagonal, 2], [2, 0]])


class GraphGenerationTests(unittest.TestCase):
    def test_same_seed_reproduces_graph(self):
        first = generate_graph(12, density=0.4, max_weight=20, seed=812)
        second = generate_graph(12, density=0.4, max_weight=20, seed=812)
        np.testing.assert_array_equal(first, second)

    def test_undirected_graph_is_symmetric(self):
        graph = generate_graph(12, density=0.5, max_weight=7, seed=9)
        np.testing.assert_array_equal(graph, graph.T)
        np.testing.assert_array_equal(np.diag(graph), np.zeros(12))
        self.assertEqual(graph.dtype, np.dtype(np.float64))
        self.assertTrue(graph.flags.c_contiguous)
        finite = graph[np.isfinite(graph)]
        self.assertTrue(np.all((finite >= 0) & (finite <= 7)))
        np.testing.assert_array_equal(finite, np.floor(finite))

    def test_directed_generation_can_produce_asymmetric_edges(self):
        graph = generate_graph(30, density=0.5, max_weight=10, seed=4, directed=True)
        self.assertTrue(np.any(graph != graph.T))
        np.testing.assert_array_equal(np.diag(graph), np.zeros(30))

    def test_density_zero_has_no_edges(self):
        graph = generate_graph(5, density=0, seed=42)
        expected = np.full((5, 5), np.inf)
        np.fill_diagonal(expected, 0)
        np.testing.assert_array_equal(graph, expected)

    def test_density_one_is_complete(self):
        graph = generate_graph(5, density=1, max_weight=4, seed=42)
        self.assertTrue(np.all(np.isfinite(graph)))
        self.assertTrue(np.all((graph >= 0) & (graph <= 4)))
        np.testing.assert_array_equal(np.diag(graph), np.zeros(5))

    def test_maximum_weight_zero_allows_zero_cost_edges(self):
        graph = generate_graph(4, density=1, max_weight=0, seed=42)
        np.testing.assert_array_equal(graph, np.zeros((4, 4)))

    def test_single_vertex(self):
        np.testing.assert_array_equal(generate_graph(1), [[0]])

    def test_rejects_invalid_vertex_counts(self):
        for count in (0, -1, 1.5, True, "3"):
            with self.subTest(count=count):
                with self.assertRaises((ValueError, TypeError)):
                    generate_graph(count)

    def test_rejects_invalid_densities(self):
        for density in (-0.1, 1.1, np.nan, np.inf, -np.inf):
            with self.subTest(density=density):
                with self.assertRaises((ValueError, TypeError)):
                    generate_graph(3, density=density)

    def test_rejects_invalid_maximum_weights(self):
        for weight in (-1, 1.5, np.inf, "4"):
            with self.subTest(weight=weight):
                with self.assertRaises((ValueError, TypeError)):
                    generate_graph(3, max_weight=weight)


class GraphFileTests(unittest.TestCase):
    def setUp(self):
        self.graph = np.array([[0, 0, np.inf], [np.inf, 0, 1.25], [2, np.inf, 0]])

    def test_roundtrip_numpy_and_text(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".npy", ".txt"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"graph{suffix}"
                    save_graph(path, self.graph)
                    loaded = load_graph(path)
                    np.testing.assert_array_equal(loaded, self.graph)
                    self.assertEqual(loaded.dtype, np.dtype(np.float64))
                    self.assertTrue(loaded.flags.c_contiguous)

    def test_loads_handwritten_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.txt"
            path.write_text("0 0 inf\ninf 0 1.25\n2 inf 0\n", encoding="utf-8")
            np.testing.assert_array_equal(load_graph(path), self.graph)

    def test_single_vertex_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".npy", ".txt"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"single{suffix}"
                    save_graph(path, [[0]])
                    np.testing.assert_array_equal(load_graph(path), [[0]])

    def test_loading_invalid_graph_revalidates_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.npy"
            np.save(path, [[0, -3], [1, 0]])
            with self.assertRaises((ValueError, TypeError)):
                load_graph(path)

    def test_saving_invalid_graph_rejects_negative_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.txt"
            with self.assertRaises((ValueError, TypeError)):
                save_graph(path, [[0, -3], [1, 0]])


if __name__ == "__main__":
    unittest.main()
