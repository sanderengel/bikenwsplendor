import numpy as np
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely import LineString

from splendor.landuse.corridors import get_link_corridor
from splendor.landuse.raster import get_pixel_values, get_pixel_ids, get_pixel_matrix


def test_pixel_values_match_full_mask(random_raster):
    """
    Pixels found through the window should be exactly those whose center lies in the geometry,
    also when the geometry's bounds are not on pixel edges.
    """
    x, y = 123.4, 217.9
    xL, yL = 437.9, 470.3
    r = 37
    line = LineString([(x, y), (x + xL, y + yL)]) # Off the pixel grid
    corridor = get_link_corridor(line, r)

    raster, transform = random_raster
    values = get_pixel_values(corridor, raster, transform)

    # Brute force, mask over the whole raster
    full_mask = geometry_mask([corridor], out_shape=raster.shape, transform=transform, invert=True)
    np.testing.assert_array_equal(np.sort(values), np.sort(raster[full_mask]))


def test_pixel_ids_match_full_mask(random_raster):
    """
    Pixel ids should be exactly the flat ids of the pixels whose center lies in the geometry,
    in increasing order, also when the geometry's bounds are not on pixel edges.
    """
    x, y = 200.5, 200.5
    xL, yL = 400, 300
    r = 40
    line = LineString([(x, y), (x + xL, y + yL)]) # Off the pixel grid
    buffer = line.buffer(r)

    raster, transform = random_raster
    ids = get_pixel_ids(buffer, raster.shape, transform)

    # Brute force, mask over the whole raster
    full_mask = geometry_mask([buffer], out_shape=raster.shape, transform=transform, invert=True)
    np.testing.assert_array_equal(ids, np.flatnonzero(full_mask))


def test_pixel_ids_match_values(random_raster):
    """The raster values at the pixel ids should equal get_pixel_values, in the same order."""
    x, y = 200.5, 200.5
    xL, yL = 400, 300
    r = 40
    line = LineString([(x, y), (x + xL, y + yL)]) # Off the pixel grid
    buffer = line.buffer(r)

    raster, transform = random_raster
    ids = get_pixel_ids(buffer, raster.shape, transform)
    values = get_pixel_values(buffer, raster, transform)

    np.testing.assert_array_equal(raster.ravel()[ids], values)


def test_pixel_matrix_rows(random_raster):
    """Each row of the pixel matrix should hold the pixel ids of its geometry, in the given order."""
    x, y = 200.5, 200.5
    L = 400
    r = 40
    lines = [
        LineString([(x, y), (x + L, y)]),            # Horizontal
        LineString([(x, y + 300), (x + L, y + 300)]) # Horizontal, further up
    ]
    buffers = gpd.GeoSeries([line.buffer(r) for line in lines])

    raster, transform = random_raster
    B = get_pixel_matrix(buffers, raster.shape, transform)

    assert B.shape == (len(buffers), raster.size)
    for i, buffer in enumerate(buffers):
        row_ids = B.indices[B.indptr[i]:B.indptr[i + 1]]
        np.testing.assert_array_equal(row_ids, get_pixel_ids(buffer, raster.shape, transform))


def test_buffer_union_invariant_to_split(random_raster):
    """
    The union of the buffer pixels of the two parts of a line with a sharp turn
    should equal the buffer pixels of the whole line: no pixel counted twice,
    and none missed on the outer side of the turn.
    """
    x, y = 200.5, 200.5
    L = 300
    r = 40
    a, b, c = (x, y), (x + L, y), (x + L, y + L) # 90 degree turn at b, off the pixel grid

    raster, transform = random_raster
    whole = get_pixel_ids(LineString([a, b, c]).buffer(r), raster.shape, transform)
    parts = np.union1d(
        get_pixel_ids(LineString([a, b]).buffer(r), raster.shape, transform),
        get_pixel_ids(LineString([b, c]).buffer(r), raster.shape, transform),
    )

    np.testing.assert_array_equal(parts, whole)
