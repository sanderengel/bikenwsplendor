"""
Compute the land use shares of each loop's corridor: the union of its links' full buffers,
minus the road pixels in the union of its links' strips.
"""
import numpy as np
import pandas as pd

from splendor.io import load_network, load_raster
from splendor.cli import get_study_area
from splendor.paths import ensure_parent, network_path, basemap05_main_types_path, loop_census_path, landuse_path
from splendor.config import (
    CORRIDOR_RADIUS,
    ROAD_STRIP_RADIUS,
    EXCLUDED_TYPES
)
from splendor.landuse.types import N_TYPES, ROAD
from splendor.landuse.corridors import get_link_strips
from splendor.landuse.raster import get_pixel_matrix
from splendor.landuse.shares import get_unit_shares
from splendor.loops.census import LoopData

STUDY_AREA = get_study_area()
LEVEL = "loops"

_, edges = load_network(network_path(STUDY_AREA))
raster, transform, crs = load_raster(basemap05_main_types_path(STUDY_AREA))
assert crs == edges.crs, f"CRS mismatch: {crs} vs {edges.crs}"
loops = LoopData.load(loop_census_path(STUDY_AREA))

# Rows of the pixel matrices must match the columns of the loop x link matrix M
assert len(edges) == loops.n_network_edges, "edge count differs from the loop data"
assert (edges["edge_id"].to_numpy() == np.arange(len(edges))).all(), "edges not in edge_id order"

buffers = get_pixel_matrix(edges.geometry.buffer(CORRIDOR_RADIUS), raster.shape, transform) # Full buffers without end cuts
road_strips = get_pixel_matrix(get_link_strips(edges, ROAD_STRIP_RADIUS), raster.shape, transform)

loop_shares = get_unit_shares(
    loops.M, pd.RangeIndex(loops.n_loops), buffers, raster, N_TYPES,
    strips=road_strips, strip_excluded_type=ROAD[0], # Ignore the loop's own road
    excluded_types=EXCLUDED_TYPES
)
loop_shares.insert(0, "loop_id", np.arange(loops.n_loops))

loop_shares.to_csv(
    ensure_parent(landuse_path(STUDY_AREA, LEVEL, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS)),
    index=False
)
