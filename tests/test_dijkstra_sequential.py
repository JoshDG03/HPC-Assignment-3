"""Known shortest-path results, calculated independently of the algorithm."""

import unittest

import numpy as np

from dijkstra_sequential import dijkstra


class DijkstraTests(unittest.TestCase):
    def test_connected_undirected_graph(self):
        graph = np.array(
            [
                [0, 4, 1, np.inf, np.inf],
                [4, 0, 2, 1, np.inf],
                [1, 2, 0, 5, 8],
                [np.inf, 1, 5, 0, 3],
                [np.inf, np.inf, 8, 3, 0],
            ]
        )
        # Shortest paths to 1, 3 and 4 go through 2, then 1, then 3.
        np.testing.assert_array_equal(dijkstra(graph), [0, 3, 1, 4, 7])

    def test_disconnected_component_stays_unreachable(self):
        graph = np.array(
            [
                [0, 2, np.inf, np.inf, np.inf],
                [2, 0, 3, np.inf, np.inf],
                [np.inf, 3, 0, np.inf, np.inf],
                [np.inf, np.inf, np.inf, 0, 1],
                [np.inf, np.inf, np.inf, 1, 0],
            ]
        )
        np.testing.assert_array_equal(dijkstra(graph), [0, 2, 5, np.inf, np.inf])

    def test_zero_weight_directed_edges_are_real_edges(self):
        graph = np.array(
            [
                [0, 0, 5, np.inf],
                [np.inf, 0, 2, 7],
                [np.inf, np.inf, 0, 1],
                [np.inf, np.inf, np.inf, 0],
            ]
        )
        np.testing.assert_array_equal(dijkstra(graph), [0, 0, 2, 3])

    def test_directed_edges_are_not_reversed(self):
        graph = np.array([[0, np.inf, 8], [1, 0, np.inf], [np.inf, 2, 0]])
        np.testing.assert_array_equal(dijkstra(graph), [0, 10, 8])

    def test_equal_cost_paths(self):
        graph = np.array(
            [[0, 1, 1, 5], [1, 0, np.inf, 2], [1, np.inf, 0, 2], [5, 2, 2, 0]]
        )
        # Both 0 -> 1 -> 3 and 0 -> 2 -> 3 have total weight 3.
        np.testing.assert_array_equal(dijkstra(graph), [0, 1, 1, 3])

    def test_decimal_weights(self):
        graph = np.array(
            [
                [0, 0.1, 0.8, np.inf],
                [np.inf, 0, 0.2, 0.7],
                [np.inf, np.inf, 0, 0.4],
                [np.inf, np.inf, np.inf, 0],
            ]
        )
        np.testing.assert_allclose(dijkstra(graph), [0, 0.1, 0.3, 0.7], rtol=1e-12, atol=1e-12)

    def test_single_vertex(self):
        result = dijkstra([[0]])
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [0])

    def test_graph_with_no_edges(self):
        graph = np.full((3, 3), np.inf)
        np.fill_diagonal(graph, 0)
        np.testing.assert_array_equal(dijkstra(graph), [0, np.inf, np.inf])

    def test_does_not_mutate_input(self):
        graph = np.array([[0, 7, 2], [7, 0, 1], [2, 1, 0]], dtype=np.float64)
        before = graph.copy()
        np.testing.assert_array_equal(dijkstra(graph), [0, 3, 2])
        np.testing.assert_array_equal(graph, before)

    def test_rejects_negative_edge_in_unreachable_component(self):
        graph = [[0, np.inf, np.inf], [np.inf, 0, -1], [np.inf, 2, 0]]
        with self.assertRaises((ValueError, TypeError)):
            dijkstra(graph)

    def test_discards_overflow_when_direct_shortest_path_is_finite(self):
        graph = [
            [0, 1e308, 1e308],
            [np.inf, 0, 1e308],
            [np.inf, np.inf, 0],
        ]
        # The walk 0 -> 1 -> 2 overflows, but direct paths are shortest.
        np.testing.assert_array_equal(dijkstra(graph), [0, 1e308, 1e308])

    def test_overflow_does_not_hide_a_later_finite_alternative(self):
        graph = [
            [0, 1e308, 1.1e308, np.inf],
            [np.inf, 0, np.inf, 1e308],
            [np.inf, np.inf, 0, 1e307],
            [np.inf, np.inf, np.inf, 0],
        ]
        # Vertex 1 is settled before 2, but 0 -> 2 -> 3 remains finite.
        np.testing.assert_allclose(dijkstra(graph), [0, 1e308, 1.1e308, 1.2e308], rtol=1e-14)

    def test_reports_unrepresentable_reachable_shortest_distance(self):
        graph = [
            [0, 1e308, np.inf],
            [np.inf, 0, 1e308],
            [np.inf, np.inf, 0],
        ]
        # Vertex 2 is reachable, but its only path exceeds float64's range.
        with self.assertRaises(OverflowError):
            dijkstra(graph)


if __name__ == "__main__":
    unittest.main()
