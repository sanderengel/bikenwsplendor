"""Score every link under every combination of the grid, and summarize how sensitive its rank is to the choices."""
import pandas as pd

from splendor.io import load_network
from splendor.cli import get_study_area
from splendor.paths import (
    ensure_parent,
    network_path,
    components_path,
    score_grid_path,
    rank_summary_path,
    rank_correlations_path,
    splendor_path
)
from splendor.config import (
    CORRIDOR_RADIUS,
    ROAD_STRIP_RADIUS,
    EXCLUDED_TYPES,
    DIVERSITY_MEASURES,
    RICHNESS_MIN_SHARE,
    NORMALIZATION_METHODS,
    NORMALIZATION_PERCENTILE,
    POI_BOUNDS,
    GEOMETRIC_FLOOR,
    AGGREGATION_METHODS,
    COMPONENT_WEIGHTS,
    REFERENCE_WEIGHTS,
    ATTRACTIVENESS,
    MAIN_SPECIFICATION
)
from splendor.landuse.types import N_TYPES
from splendor.components.diversity import get_bounds
from splendor.composite.sensitivity import get_score_grid, summarize_ranks, get_rank_correlations

STUDY_AREA = get_study_area()
LEVEL = "links"
PATH_ARGS = (STUDY_AREA, LEVEL, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS)

# Components are land use diversity per measure and attractiveness, plus POI diversity from the network
_, edges = load_network(network_path(STUDY_AREA))
components = pd.read_csv(components_path(*PATH_ARGS))

score_components = components.merge(
    edges[["edge_id", "poi_diversity"]], on="edge_id", how="inner", validate="one_to_one"
)
assert len(score_components) == len(edges) == len(components), "edge ids of network and components file don't match"

# Each component's theoretical bounds
n_types = N_TYPES - len(EXCLUDED_TYPES) # Types the measures see
bounds = {f"diversity_{m}": get_bounds(m, n_types, RICHNESS_MIN_SHARE) for m in DIVERSITY_MEASURES}
bounds["poi_diversity"] = POI_BOUNDS
bounds["attractiveness"] = (min(ATTRACTIVENESS.values()), max(ATTRACTIVENESS.values()))

# Every combination including the single-component reference weights
grid = get_score_grid(
    score_components,
    DIVERSITY_MEASURES,
    NORMALIZATION_METHODS,
    AGGREGATION_METHODS,
    weights=(*COMPONENT_WEIGHTS, *REFERENCE_WEIGHTS),
    bounds=bounds,
    p=NORMALIZATION_PERCENTILE,
    geometric_floor=GEOMETRIC_FLOOR,
)

# Rank summary over the combined scores only, but correlations over all combinations
weight_columns = ["land_use_weight", "poi_weight", "attractiveness_weight"]
is_combined = pd.MultiIndex.from_frame(grid[weight_columns]).isin(COMPONENT_WEIGHTS)
summary = summarize_ranks(grid[is_combined])
correlations = get_rank_correlations(grid)

# Filter to main specification to get final splendor scores
splendor = grid.copy()
for column, choice in MAIN_SPECIFICATION.items():
    splendor = splendor[splendor[column] == choice]
splendor = splendor[["edge_id", "score", "rank"]]
assert splendor["edge_id"].is_unique
assert set(splendor["edge_id"]) == set(edges["edge_id"]), "main specification must match exactly one combination in the grid"

grid.to_csv(ensure_parent(score_grid_path(*PATH_ARGS)), index=False)
summary.to_csv(ensure_parent(rank_summary_path(*PATH_ARGS)), index=False)
correlations.to_csv(ensure_parent(rank_correlations_path(*PATH_ARGS)))
splendor.to_csv(ensure_parent(splendor_path(*PATH_ARGS)), index=False)
