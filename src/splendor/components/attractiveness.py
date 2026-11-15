"""Landscape attractiveness of links from their land use shares."""
import numpy as np
import pandas as pd

from splendor.landuse.shares import SHARE_PREFIX, get_share_type_id


def get_attractiveness(shares: pd.DataFrame, values: dict[int, float]) -> np.ndarray:
    """
    Share-weighted mean attractiveness per row, sum_k p_k * a_k over the share columns.
    'values' maps land use type ids to an attractiveness in [0, 1].
    """
    share_columns = shares.filter(regex=f"^{SHARE_PREFIX}").columns
    type_ids = [get_share_type_id(c) for c in share_columns]

    missing = [k for k in type_ids if k not in values]
    if missing:
        raise ValueError(f"no attractiveness value for land use types {missing}")
    
    weights = np.array([values[k] for k in type_ids], dtype=float)
    if ((weights < 0) | (weights > 1)).any():
        raise ValueError("attractiveness values must be in [0, 1]")

    p = shares[share_columns].to_numpy(dtype=float)
    if not np.allclose(p.sum(axis=1), 1):
        raise ValueError("shares must sum to 1 per row")
    return p @ weights
