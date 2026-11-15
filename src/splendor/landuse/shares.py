import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Sequence
from affine import Affine
from scipy.sparse import csr_array, eye_array
from splendor.landuse.raster import get_pixel_matrix

SHARE_PREFIX = "share_"


def get_share_column_name(k: int) -> str:
    return SHARE_PREFIX + str(k)


def get_share_type_id(column_name: str) -> int:
    return int(column_name.removeprefix(SHARE_PREFIX))


def check_excluded_types(excluded_types: Sequence[int], n_types: int) -> np.ndarray:
    """Return the excluded types as a sorted array of unique ids, check that they are valid type ids."""
    excluded_types = np.unique(np.asarray(excluded_types, dtype=int))
    if ((excluded_types < 1) | (excluded_types > n_types)).any():
        raise ValueError(f"excluded types must be in 1..{n_types}")
    return excluded_types


def get_shares_from_counts(
    counts: np.ndarray,
    index: pd.Index,
    excluded_types: Sequence[int] = ()
) -> pd.DataFrame:
    """
    Return the shares of each row of pixel counts per type, where column k counts type k
    and column 0 counts no data. Leave no data and 'excluded_types' out of the shares.
    Excluded pixels are counted in the column 'n_excludedø.
    """
    n_types = counts.shape[1] - 1
    excluded_types = check_excluded_types(excluded_types, n_types)

    kept = np.ones(n_types + 1, dtype=bool)
    kept[0] = False        # No data
    kept[excluded_types] = False # Excluded

    n_pixels = counts[:, kept].sum(axis=1)
    assert (n_pixels > 0).all(), "corridor with no land use pixels"
    shares = counts[:, kept] / n_pixels[:, None] # Normalize each row

    kept_types = np.flatnonzero(kept)
    corridor_shares = pd.DataFrame(
        shares,
        columns = [get_share_column_name(k) for k in kept_types],
        index = index
    )
    corridor_shares.insert(0, "n_pixels", n_pixels)
    corridor_shares.insert(1, "n_excluded", counts[:, excluded_types].sum(axis=1))
    return corridor_shares


def get_shares_array(shares: pd.DataFrame | pd.Series) -> np.ndarray:
    """Return array of only share columns, stripping identifiers and aggregate columns."""
    return shares.filter(regex=f"^{SHARE_PREFIX}").to_numpy()


def _get_type_matrix(values: np.ndarray, n_types: int) -> csr_array:
    """One-hot pixel x type matrix, with column 0 for no data."""
    n = len(values)
    return csr_array(
        (np.ones(n, dtype=np.int64), values.astype(np.int64), np.arange(n + 1)),
        shape=(n, n_types + 1)
    )


def _prepare_unit_matrices(
    buffers: csr_array,
    raster: np.ndarray,
    n_types: int,
    strips: csr_array | None,
    strip_excluded_type: int | None,
) -> tuple[csr_array, csr_array, csr_array]:
    """
    Restrict the link x pixel matrices to the pixels in some buffer, and return the buffers,
    the pixels of 'strip_excluded_type' in each link's strip (within its own buffer),
    and a one-hot pixel x type matrix. Without strips, no pixels are excluded.
    """
    used = np.unique(buffers.indices)
    values = raster.ravel()[used]
    used_buffers = buffers[:, used].tocsr()

    if strips is None:
        strip_excluded = csr_array(used_buffers.shape, dtype=np.int8)
    else:
        strip_excluded = strips[:, used].multiply(used_buffers).tocsr() # Each link's strip within its own buffer
        strip_excluded.data = strip_excluded.data * (values[strip_excluded.indices] == strip_excluded_type)
        strip_excluded.eliminate_zeros()

    return used_buffers, strip_excluded, _get_type_matrix(values, n_types)


def _count_unit_pixels(
    membership: csr_array,
    buffers: csr_array,
    strip_excluded: csr_array,
    pixel_types: csr_array,
    chunk_size: int,
) -> np.ndarray:
    """
    Return unit x type pixel counts of the union of each unit's link buffers,
    minus the union of its links' excluded strip pixels.
    """
    n_units = membership.shape[0]
    counts = np.zeros((n_units, pixel_types.shape[1]), dtype=np.int64)
    for start in range(0, n_units, chunk_size):
        stop = min(start + chunk_size, n_units)
        units = membership[start:stop].astype(np.int32) # int32 so the products can't overflow
        in_buffers = (units @ buffers) > 0              # Union of the units' link buffers
        in_excluded = (units @ strip_excluded) > 0      # Union of the units' excluded strip pixels
        kept = in_buffers.astype(np.int64) - in_excluded.astype(np.int64) # Excluded pixels lie inside the buffers
        counts[start:stop] = (kept @ pixel_types).toarray()
    return counts


def get_unit_shares(
    membership: csr_array,
    index: pd.Index,
    buffers: csr_array,
    raster: np.ndarray,
    n_types: int,
    strips: csr_array | None = None,
    strip_excluded_type: int | None = None,
    excluded_types: Sequence[int] = (),
    chunk_size: int = 500,
) -> pd.DataFrame:
    """
    Return the pixel value shares of units made of links, given as a unit x link 'membership' matrix
    (the identity for links themselves, or the loop x link incidence matrix for loops).

    A unit's corridor is the union of its links' buffer pixels. With 'strips', the pixels of
    'strip_excluded_type' in the union of its links' strips (each within its own buffer) are left out.
    Each pixel counts once, so the result doesn't depend on how a unit is split into links.
    Treat types in 'excluded_types' as no data by leaving them out of the shares,
    though their pixels are counted in the column 'n_excluded'.

    'buffers' and 'strips' are link x pixel matrices (columns flat raster pixel ids), see get_pixel_matrix.
    """
    excluded_types = check_excluded_types(excluded_types, n_types)
    if len(index) != membership.shape[0]:
        raise ValueError("index must have one entry per unit")
    if buffers.shape != (membership.shape[1], raster.size):
        raise ValueError("buffers must have one row per link and one column per raster pixel")
    if strips is not None:
        if strip_excluded_type is None:
            raise ValueError("strips given without a type to exclude")
        if strips.shape != buffers.shape:
            raise ValueError("strips must have the same shape as the buffers")

    buffers, strip_excluded, pixel_types = _prepare_unit_matrices(
        buffers, raster, n_types, strips, strip_excluded_type
    )
    counts = _count_unit_pixels(membership, buffers, strip_excluded, pixel_types, chunk_size)
    return get_shares_from_counts(counts, index, excluded_types)


def get_corridor_shares(
    corridors: gpd.GeoSeries,
    raster: np.ndarray,
    transform: Affine,
    n_types: int,
    strips: gpd.GeoSeries | None = None,
    strip_excluded_type: int | None = None,
    excluded_types: Sequence[int] = ()
) -> pd.DataFrame:
    """
    Return the pixel value shares for each corridor from the raster data.
    With 'strips', the pixels of 'strip_excluded_type' inside a corridor's strip are left out.
    Treat types in 'excluded_types' as no data by leaving them out of the shares,
    though their pixels are counted in the column 'n_excluded'.
    """
    if strips is not None and not strips.index.equals(corridors.index):
        raise ValueError("strips must have the same index as the corridors")

    corridor_pixels = get_pixel_matrix(corridors, raster.shape, transform)
    strip_pixels = None if strips is None else get_pixel_matrix(strips, raster.shape, transform)
    membership = eye_array(len(corridors), dtype=np.int8, format="csr") # Each corridor is its own unit

    return get_unit_shares(
        membership, corridors.index, corridor_pixels, raster, n_types,
        strips=strip_pixels, strip_excluded_type=strip_excluded_type, excluded_types=excluded_types
    )
