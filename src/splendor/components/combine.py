import pandas as pd
from typing import Sequence

from splendor.landuse.shares import check_excluded_types, get_shares_array
from splendor.components.diversity import get_bounds, get_diversity
from splendor.components.attractiveness import get_attractiveness


def get_landuse_components(
    shares: pd.DataFrame,
    measures: Sequence[str],
    attractiveness: dict[int, float]
) -> pd.DataFrame:
    """
    Return one 'diversity_<measure>' column per measure and an 'attractiveness' column,
    one row per row of 'shares'.
    """
    shares_array = get_shares_array(shares)
    return pd.DataFrame({
        **{f"diversity_{m}": get_diversity(shares_array, m) for m in measures},
        "attractiveness": get_attractiveness(shares, attractiveness)
    }, index=shares.index)


def get_component_bounds(
    measures: Sequence[str],
    n_types: int,
    excluded_types: Sequence[int],
    richness_min_share: float,
    poi_bounds: tuple[float, float],
    attractiveness: dict[int, float]
) -> dict[str, tuple[float, float]]:
    """Theoretical (lo, hi) of each score component, for the land use types the measures see."""
    n_kept = n_types - len(check_excluded_types(excluded_types, n_types)) # Types the measures see
    bounds = {f"diversity_{m}": get_bounds(m, n_kept, richness_min_share) for m in measures}
    bounds["poi_diversity"] = poi_bounds
    bounds["attractiveness"] = (min(attractiveness.values()), max(attractiveness.values()))
    return bounds
