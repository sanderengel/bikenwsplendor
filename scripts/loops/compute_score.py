"""Score every loop under every combination of the grid, and summarize how sensitive its rank is to the choices."""
import pandas as pd

from splendor.cli import get_study_area
from splendor.paths import (
    ensure_parent,
    loop_attributes_path,
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
from splendor.components.combine import get_component_bounds
from splendor.composite.sensitivity import (
    get_main_scores,
    get_score_grid,
    select_weights,
    summarize_ranks,
    get_rank_correlations
)

STUDY_AREA = get_study_area()
LEVEL = "loops"
PATH_ARGS = (STUDY_AREA, LEVEL, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS)
ID_COLUMN = "loop_id"

# Components are land use diversity per measure and attractiveness, plus POI diversity from the loop attributes
components = pd.read_csv(components_path(*PATH_ARGS), index_col=ID_COLUMN)
attributes = pd.read_csv(loop_attributes_path(STUDY_AREA), index_col=ID_COLUMN)
assert components.index.equals(attributes.index), "loop ids of components and attributes files don't match"

score_components = components.join(attributes["poi_diversity"])

# Each component's theoretical bounds
bounds = get_component_bounds(
    DIVERSITY_MEASURES, N_TYPES, EXCLUDED_TYPES, RICHNESS_MIN_SHARE, POI_BOUNDS, ATTRACTIVENESS
)

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

splendor = get_main_scores(grid, MAIN_SPECIFICATION, ID_COLUMN)

# Rank summary over the combined scores only, but correlations over all combinations
summary = summarize_ranks(select_weights(grid, COMPONENT_WEIGHTS), ID_COLUMN)
correlations = get_rank_correlations(grid, ID_COLUMN)

grid.to_csv(ensure_parent(score_grid_path(*PATH_ARGS)), index=False)
summary.to_csv(ensure_parent(rank_summary_path(*PATH_ARGS)), index=False)
correlations.to_csv(ensure_parent(rank_correlations_path(*PATH_ARGS)))
splendor.to_csv(ensure_parent(splendor_path(*PATH_ARGS)), index=False)
