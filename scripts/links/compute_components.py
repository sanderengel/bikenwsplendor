import pandas as pd

from splendor.io import load_network
from splendor.cli import get_study_area
from splendor.paths import ensure_parent, network_path, landuse_path, components_path
from splendor.config import CORRIDOR_RADIUS, ROAD_STRIP_RADIUS, DIVERSITY_MEASURES, ATTRACTIVENESS
from splendor.components.combine import get_landuse_components

STUDY_AREA = get_study_area()
LEVEL = "links"
PATH_ARGS = (STUDY_AREA, LEVEL, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS)
NETWORK = network_path(STUDY_AREA)

_, edges = load_network(NETWORK)
corridor_shares = pd.read_csv(landuse_path(*PATH_ARGS), index_col="edge_id")
assert (edges["edge_id"].to_numpy() == corridor_shares.index.to_numpy()).all()

components = get_landuse_components(corridor_shares, DIVERSITY_MEASURES, ATTRACTIVENESS)

components.to_csv(ensure_parent(components_path(*PATH_ARGS)))
