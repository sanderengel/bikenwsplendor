import numpy as np
import networkx as nx

from splendor.loops.attributes import loop_sum, loop_max, loop_poi_diversity, node_max, edge_max
from splendor.loops.census import get_loops

def test_loop_sum_equals_length(k4_loops):
    """loop_sum of lengths should equal loops.lengths_m."""
    np.testing.assert_array_equal(loop_sum(k4_loops, np.ones(6)), k4_loops.lengths_m)


def test_loop_max_one_steep_edge(k4_loops):
    """loop_max of max_slope should give the correct slope with a single steep edge."""
    slopes = np.ones(6)
    slopes[0] = 5.0
    max_slope = loop_max(k4_loops, slopes)
    assert (max_slope == 5.0).sum() == 4
    assert (max_slope == 1.0).sum() == 3


def test_poi_diversity(k4_loops):
    """Facility only on one edge, attraction everywhere, no services."""
    has_facility = np.zeros(6)
    has_facility[0] = 1
    has_service = np.zeros(6)
    has_attraction = np.ones(6)
    poi_diversity = loop_poi_diversity(k4_loops, has_facility, has_service, has_attraction)
    # Four loops contain edge 0 (facility + attraction), three don't (only attraction)
    assert sorted(poi_diversity.tolist()) == [1, 1, 1, 2, 2, 2, 2]


def test_node_and_edge_max_match_incidence(k4_loops):
    """The maximum per node and edge should equal the maximum over the loops in its incidence column."""
    values = np.random.default_rng(0).random(k4_loops.n_loops)

    N = k4_loops.N.toarray().astype(bool) # Loop x node
    M = k4_loops.M.toarray().astype(bool) # Loop x edge
    expected_nodes = [values[N[:, j]].max() for j in range(k4_loops.n_network_nodes)]
    expected_edges = [values[M[:, j]].max() for j in range(k4_loops.n_network_edges)]

    np.testing.assert_array_equal(node_max(k4_loops, values), expected_nodes)
    np.testing.assert_array_equal(edge_max(k4_loops, values), expected_edges)


def test_node_and_edge_max_nan_off_loops():
    """A node and edge on no loop (a pendant edge on k4) should get NaN."""
    G = nx.complete_graph(4)
    G.add_edge(0, 4) # Pendant edge: node 4 and this edge lie on no loop
    for eid, (u, v) in enumerate(G.edges()):
        G[u][v]["edge_id"] = eid
        G[u][v]["length_m"] = 1.
    loops = get_loops(G, max_length_m=float("inf"))
    values = np.arange(loops.n_loops, dtype=float)

    node_values = node_max(loops, values)
    edge_values = edge_max(loops, values)

    assert np.isnan(node_values[4])
    assert not np.isnan(node_values[:4]).any()
    pendant = G[0][4]["edge_id"]
    assert np.isnan(edge_values[pendant])
    assert np.isnan(edge_values).sum() == 1
