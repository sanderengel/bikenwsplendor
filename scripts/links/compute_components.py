import pandas as pd

from splendor.io import load_network
from splendor.cli import get_study_area
from splendor.paths import ensure_parent, network_path, landuse_path, components_path
from splendor.landuse.shares import get_shares_array
from splendor.config import CORRIDOR_RADIUS, ROAD_STRIP_RADIUS, DIVERSITY_MEASURES, ATTRACTIVENESS
from splendor.components.diversity import get_diversity
from splendor.components.attractiveness import get_attractiveness

STUDY_AREA = get_study_area()
LEVEL = "links"
PATH_ARGS = (STUDY_AREA, LEVEL, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS)
NETWORK = network_path(STUDY_AREA)

_, edges = load_network(NETWORK)
corridor_shares = pd.read_csv(landuse_path(*PATH_ARGS))
assert edges["edge_id"].equals(corridor_shares["edge_id"])

shares_array = get_shares_array(corridor_shares)

components = pd.DataFrame({
    "edge_id": corridor_shares["edge_id"],
    **{f"diversity_{m}": get_diversity(shares_array, m) for m in DIVERSITY_MEASURES},
    "attractiveness": get_attractiveness(corridor_shares, ATTRACTIVENESS)
})

components.to_csv(
    ensure_parent(components_path(*PATH_ARGS)),
    index=False,
)
