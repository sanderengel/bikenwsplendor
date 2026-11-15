import numpy as np
import pandas as pd

from splendor.cli import get_study_area
from splendor.paths import ensure_parent, landuse_path, components_path
from splendor.config import CORRIDOR_RADIUS, ROAD_STRIP_RADIUS, DIVERSITY_MEASURES, ATTRACTIVENESS
from splendor.components.combine import get_landuse_components

STUDY_AREA = get_study_area()
LEVEL = "loops"
PATH_ARGS = (STUDY_AREA, LEVEL, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS)

corridor_shares = pd.read_csv(landuse_path(*PATH_ARGS), index_col="loop_id")
assert (corridor_shares.index.to_numpy() == np.arange(len(corridor_shares))).all(), "loop ids not 0..n-1"

components = get_landuse_components(corridor_shares, DIVERSITY_MEASURES, ATTRACTIVENESS)

components.to_csv(ensure_parent(components_path(*PATH_ARGS)))
