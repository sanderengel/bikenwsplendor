"""
Compute census attributes for each loop (length, POI diversity, number of nodes)
and save them by loop id, for joining with the loop components.
"""
import numpy as np
import pandas as pd

from splendor.paths import ensure_parent, loop_census_path, network_path, loop_attributes_path
from splendor.cli import get_study_area
from splendor.io import load_network
from splendor.loops.census import LoopData
from splendor.loops.attributes import loop_poi_diversity

STUDY_AREA = get_study_area()
NETWORK = network_path(STUDY_AREA)

_, edges = load_network(NETWORK)
loops = LoopData.load(loop_census_path(STUDY_AREA))
assert len(edges) == loops.n_network_edges, "edge count differs from the loop data"
assert (edges["edge_id"].to_numpy() == np.arange(len(edges))).all(), "edges not in edge_id order"

attributes = pd.DataFrame(
    {
        "length_m": loops.lengths_m,
        "poi_diversity": loop_poi_diversity(
            loops,
            edges["has_facility"].to_numpy(),
            edges["has_service"].to_numpy(),
            edges["has_attraction"].to_numpy(),
        ),
        "n_nodes": loops.sizes
    },
    index=pd.RangeIndex(loops.n_loops, name="loop_id")
)

attributes.to_csv(ensure_parent(loop_attributes_path(STUDY_AREA)))
