import numpy as np

from splendor.loops.selection import select_distinct_loops


def test_distinct_loops_k4(k4_loops):
    """
    In k4 with unit lengths, triangles share one edge (overlap 1/3), while every square shares
    two edges with some triangle (overlap 2/3). With triangles first and max overlap 0.5,
    exactly the four triangles are selected.
    """
    triangles = np.flatnonzero(k4_loops.sizes == 3)
    squares = np.flatnonzero(k4_loops.sizes == 4)
    order = np.concatenate([triangles, squares])

    selected = select_distinct_loops(k4_loops, np.ones(6), order, n=7, max_overlap=.5)
    np.testing.assert_array_equal(selected, triangles)


def test_distinct_loops_no_limit(k4_loops):
    """With max overlap 1, every loop is distinct enough, so the first n in order are selected."""
    order = np.arange(k4_loops.n_loops)[::-1]
    selected = select_distinct_loops(k4_loops, np.ones(6), order, n=5, max_overlap=1.)
    np.testing.assert_array_equal(selected, order[:5])
