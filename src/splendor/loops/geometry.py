"""Geometries of loops from the link geometries."""
import numpy as np
import geopandas as gpd
from shapely import MultiLineString, line_merge, union_all

from splendor.loops.census import LoopData


def _loop_edge_ids(loops: LoopData, loop_id: int) -> np.ndarray:
    return loops.edges[loops.ptr[loop_id]:loops.ptr[loop_id + 1]]


def get_loop_lines(loops: LoopData, edges: gpd.GeoDataFrame, loop_ids: np.ndarray) -> gpd.GeoSeries:
    """Each loop's line, merged from its links ('edges' in edge_id order). Indexed by loop id."""
    lines = [
        line_merge(MultiLineString(list(edges.geometry.iloc[_loop_edge_ids(loops, i)])))
        for i in loop_ids
    ]
    return gpd.GeoSeries(lines, index=loop_ids, crs=edges.crs)


def get_loop_corridors(loops: LoopData, edges: gpd.GeoDataFrame, loop_ids: np.ndarray, r: float) -> gpd.GeoSeries:
    """Each loop's corridor: the union of its links' full buffers (no end cuts). Indexed by loop id."""
    corridors = [union_all(edges.geometry.iloc[_loop_edge_ids(loops, i)].buffer(r)) for i in loop_ids]
    return gpd.GeoSeries(corridors, index=loop_ids, crs=edges.crs)
