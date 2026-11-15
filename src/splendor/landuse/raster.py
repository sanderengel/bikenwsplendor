import math
import numpy as np
from typing import Sequence
from rasterio.features import geometry_mask
from rasterio import windows
from shapely import Geometry
from affine import Affine
from scipy.sparse import csr_array


def get_bounds_window(
    bounds: tuple[float, float, float, float],
    transform: Affine,
    raster_shape: tuple[int, int],
    clip: bool = False,
) -> windows.Window:
    """
    Return the pixel window covering the bounds, snapped outward to whole pixels so that
    every pixel touched by the bounds is included. With 'clip', the window is cut to the
    raster instead of raising an error when the bounds extend beyond it.
    """
    bounds_window = windows.from_bounds(*bounds, transform=transform)
    r_start = math.floor(bounds_window.row_off)
    c_start = math.floor(bounds_window.col_off)
    r_stop = math.ceil(bounds_window.row_off + bounds_window.height)
    c_stop = math.ceil(bounds_window.col_off + bounds_window.width)

    if clip:
        r_start, c_start = max(r_start, 0), max(c_start, 0)
        r_stop, c_stop = min(r_stop, raster_shape[0]), min(c_stop, raster_shape[1])
    else:
        assert_error_msg = "geom extends beyond the raster"
        assert r_start >= 0 and c_start >= 0, assert_error_msg
        assert r_stop <= raster_shape[0] and c_stop <= raster_shape[1], assert_error_msg

    return windows.Window.from_slices((r_start, r_stop), (c_start, c_stop))


def _get_window_mask(
    geom: Geometry,
    transform: Affine,
    raster_shape: tuple[int, int]
) -> tuple[windows.Window, np.ndarray]:
    """
    Return the pixel window covering the geom, 
    and the mask of window pixels whose center lies inside it.
    """
    window = get_bounds_window(geom.bounds, transform, raster_shape)

    # Geotransform of the window (same pixel size with origin shifted to the window's corner)
    window_transform = windows.transform(window, transform)

    mask = geometry_mask(
        [geom], # List of geometries, here is a single geom
        out_shape=(int(window.height), int(window.width)),
        transform=window_transform,
        invert=True # True for pixels whose center is in the geom
    )
    return window, mask


def get_pixel_values(geom: Geometry, raster: np.ndarray, transform: Affine) -> np.ndarray:
    """Return the raster values of all pixels whose center lies inside the geom."""
    window, mask = _get_window_mask(geom, transform, raster.shape)
    return raster[window.toslices()][mask]


def get_pixel_ids(geom: Geometry, raster_shape: tuple[int, int], transform: Affine) -> np.ndarray:
    """
    Return flat ids (row * n_cols + col) of all pixels with centers inside the geom
    in increasing order (the same order as get_pixel_values).
    """
    window, mask = _get_window_mask(geom, transform, raster_shape)
    rows, cols = np.nonzero(mask)
    rows = rows.astype(np.int64) + int(window.row_off) # int64: a full Denmark raster exceeds int32
    cols = cols.astype(np.int64) + int(window.col_off)
    return rows * raster_shape[1] + cols


def get_pixel_matrix(
    geoms: Sequence[Geometry],
    raster_shape: tuple[int, int],
    transform: Affine,
) -> csr_array:
    """
    Return a sparse binary matrix with one row per geom (in the given order) and one column
    per raster pixel (flat id), with 1 where the pixel's center lies inside the geom.
    """
    ids = [get_pixel_ids(geom, raster_shape, transform) for geom in geoms]
    indptr = np.zeros(len(ids) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(i) for i in ids])
    indices = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)
    return csr_array(
        (np.ones(len(indices), dtype=np.int8), indices, indptr),
        shape=(len(ids), raster_shape[0] * raster_shape[1])
    )
