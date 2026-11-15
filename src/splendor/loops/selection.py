"""Selecting loops."""
import numpy as np

from splendor.loops.census import LoopData


def select_distinct_loops(
    loops: LoopData,
    edge_lengths: np.ndarray,
    order: np.ndarray,
    n: int,
    max_overlap: float = .5,
) -> np.ndarray:
    """
    The first n loops in 'order' (e.g. by rank) that each share at most 'max_overlap' of their length
    with every loop selected before. The overlap of two loops is the length of their shared links
    divided by the shorter loop's length. 'edge_lengths' is in edge_id order.
    Returns fewer than n loops if 'order' runs out.
    """
    selected = []
    for i in order:
        edges_i = loops.loop_edges(i)
        is_distinct = all(
            edge_lengths[np.intersect1d(edges_i, edges_j)].sum()
            / min(loops.lengths_m[i], loops.lengths_m[j]) <= max_overlap
            for j, edges_j in selected
        )
        if is_distinct:
            selected.append((i, edges_i))
            if len(selected) == n:
                break
    return np.array([i for i, _ in selected], dtype=int)
