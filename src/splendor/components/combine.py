import pandas as pd
from typing import Sequence

from splendor.landuse.shares import get_shares_array
from splendor.components.diversity import get_diversity
from splendor.components.attractiveness import get_attractiveness


def get_landuse_components(
    shares: pd.DataFrame,
    diversity_measures: Sequence[str],
    attractiveness: dict[int, float]
) -> pd.DataFrame:
    """
    Return one 'diversity_<measure>' column per measure and an 'attractiveness' column,
    one row per row of 'shares'.
    """
    shares_array = get_shares_array(shares)
    return pd.DataFrame({
        **{f"diversity_{m}": get_diversity(shares_array, m) for m in diversity_measures},
        "attractiveness": get_attractiveness(shares, attractiveness)
    }, index=shares.index)
