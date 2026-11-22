import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rasterio import windows
from rasterio.transform import array_bounds
from shapely import Geometry
from shapely.geometry import box
from affine import Affine
from typing import Sequence

from splendor.landuse.types import ALL_TYPES, N_TYPES, SHORT_NAMES
from splendor.landuse.raster import get_bounds_window
from splendor.landuse.shares import get_share_column_name
from splendor.landuse.corridors import get_link_corridor


def _landuse_colormap() -> tuple[list[tuple[float, ...]], ListedColormap, BoundaryNorm]:
    """
    Colors, colormap and norm for land use type ids.
    Index k = type id k (no data is transparent).
    """
    colors = [(1, 1, 1, 0)] + [tuple(c / 255 for c in rgb) for _, _, _, rgb in ALL_TYPES]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5, N_TYPES + 1), cmap.N)
    return colors, cmap, norm


def _panel_title(
    unit: str,
    unit_id: int,
    length_km: float,
    n_pixels: int,
    rank: int | None = None,
    score: float | None = None,
    detail: str | None = None,
    symbol: str | None = None,
) -> str:
    """
    Panel title: "Unit id (length, pixels)" on one line, or with a rank, score or detail
    "#rank: Unit id (score s)" and "length, pixels[, detail]" on two lines.
    With a symbol (e.g. "e" or "L"), length and pixels are written in notation: \ell(symbol) and |K(symbol)|.
    """
    if symbol:
        parts = [rf"$\ell({symbol})$ = {length_km:.1f} km", rf"$|K({symbol})|$ = {n_pixels:,}"]
    else:
        parts = [f"{length_km:.1f} km", f"{n_pixels:,} pixels"]
    parts += [detail] if detail else []
    details_text = ", ".join(parts)
    if rank is None and score is None and detail is None:
        return f"{unit} {unit_id} ({details_text})"
    first = f"#{rank}: {unit} {unit_id}" if rank is not None else f"{unit} {unit_id}"
    if score is not None:
        first += f" (score {score:.2f})"
    return f"{first}\n{details_text}"


def _fit_extent(
    extent: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Shift an extent so it lies within the bounds where possible, keeping its size."""
    left, bottom, right, top = extent
    b_left, b_bottom, b_right, b_top = bounds
    dx = max(b_left - left, 0) - max(right - b_right, 0)
    dy = max(b_bottom - bottom, 0) - max(top - b_top, 0)
    return left + dx, bottom + dy, right + dx, top + dy


def get_map_extent(geom: Geometry, margin: float) -> tuple[float, float, float, float]:
    """Square extent around a geometry padded by 'margin' of its larger side."""
    minx, miny, maxx, maxy = geom.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny) * (.5 + margin)
    return cx - half, cy - half, cx + half, cy + half


def plot_landuse_map(
    ax: Axes,
    raster: np.ndarray,
    transform: Affine,
    line: Geometry,
    corridor: Geometry,
    margin: float = .15,
    min_corridor_fraction: float = .02,
) -> None:
    """
    Plot land use around one link or loop on the given ax. The corridor is shown (dimmed outside, dashed outline)
    only if its width is at least 'min_corridor_fraction' of the map's side; otherwise only the line is drawn.
    """
    _, cmap, norm = _landuse_colormap()

    # Square raster window around the corridor, shifted inward at the raster edge
    raster_bounds = array_bounds(*raster.shape, transform) # (west, south, east, north)
    map_extent = _fit_extent(get_map_extent(corridor, margin), raster_bounds)
    map_left, map_bottom, map_right, map_top = map_extent

    window = get_bounds_window(map_extent, transform, raster.shape, clip=True)
    left, bottom, right, top = windows.bounds(window, transform)
    
    ax.imshow(
        raster[window.toslices()], cmap=cmap, norm=norm, 
        extent=(left, right, bottom, top), interpolation="nearest"
    )

    outside = box(*map_extent)

    # Show corridor only if the corridor is wide enough to read
    corridor_width = corridor.area / line.length # About 2r
    if corridor_width / (map_right - map_left) >= min_corridor_fraction:
        gpd.GeoSeries([corridor]).boundary.plot(ax=ax, color="black", linewidth=.6, linestyle="--")
        # Dim everything outside the corridor
        outside = outside.difference(corridor)

    gpd.GeoSeries([outside]).plot(ax=ax, color="white", alpha=.65, linewidth=0)
    gpd.GeoSeries([line]).plot(ax=ax, color="black", linewidth=1.5)

    ax.set_xlim(map_left, map_right)
    ax.set_ylim(map_bottom, map_top)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def _present_shares(row: pd.Series) -> pd.Series:
    """Shares above zero by type id, in fixed type order, skipping excluded types."""
    s = pd.Series({
        k: row[get_share_column_name(k)]
        for k in range(1, N_TYPES + 1)
        if get_share_column_name(k) in row.index
    })
    return s[s > 0]


def plot_share_bar(ax: Axes, row: pd.Series) -> None:
    """One horizontal 100% bar of the unit's land use shares, in fixed type order."""
    colors, _, _ = _landuse_colormap()
    left = 0.0
    for k, share in _present_shares(row).items():
        ax.barh(0, share, left=left, color=colors[k], height=1)
        left += share
    ax.set_xlim(0, 1)
    ax.set_ylim(-.5, .5)
    ax.set_axis_off()


def _add_landuse_legend(fig: Figure, type_ids: Sequence[int]) -> None:
    """One row of color patches for the given land use types."""
    colors, _, _ = _landuse_colormap()
    handles = [Patch(color=colors[k], label=SHORT_NAMES[k]) for k in type_ids]
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=9)


def plot_landuse_grid(
    ids: Sequence[int],
    lines: gpd.GeoSeries,     # Indexed by id
    corridors: gpd.GeoSeries, # Indexed by id
    shares: pd.DataFrame,     # Indexed by id
    raster: np.ndarray,
    transform: Affine,
    ranks: pd.Series | None = None,   # Indexed by id
    scores: pd.Series | None = None,  # Indexed by id
    details: pd.Series | None = None, # Indexed by id
    unit: str = "Link",
    n_cols: int = 4,
    title: str | None = None,
    margin: float = .15,
    bar_height: float = .06, # Fraction of the map's height
    bar_gap: float = .02,    # Fraction of the map's height
    col_width: float = 3.4,
    row_height: float = 3.9,
    title_height: float = .3,
    legend_height: float = .3,
    min_corridor_fraction: float = .02,
    symbol: str | None = None # Notation for length and pixels in panel titles, e.g. "e" or "L"
) -> Figure:
    """
    Grid of panels, one per unit (link or loop): land use map with corridor, and a 100% bar of its shares
    below it. One legend for all panels lists the land use types present.
    """
    n_rows = -(-len(ids) // n_cols)
    has_title = title is not None
    heights = ([title_height] if has_title else []) + [row_height * n_rows, legend_height]
    fig = plt.figure(
        figsize=(col_width * n_cols, sum(heights)),
        layout="constrained",
    )
    subfigures = fig.subfigures(len(heights), 1, height_ratios=heights)
    title_fig = subfigures[0] if has_title else None
    grid_fig, legend_fig = subfigures[-2:]
    subfigs = grid_fig.subfigures(n_rows, n_cols, wspace=.03, hspace=0, squeeze=False)

    present = set()
    for subfig, unit_id in zip(subfigs.flat, ids):
        unit_id = int(unit_id)
        line = lines.loc[unit_id]
        row = shares.loc[unit_id]
        ax_map = subfig.subplots()
        ax_bar = ax_map.inset_axes([0, -(bar_gap + bar_height), 1, bar_height]) # Below the map, same width

        plot_landuse_map(ax_map, raster, transform, line, corridors.loc[unit_id], margin, min_corridor_fraction)
        panel_title = _panel_title(
            unit,
            unit_id,
            line.length / 1000, # m to km
            int(row["n_pixels"]),
            rank=int(ranks[unit_id]) if ranks is not None else None,
            score=float(scores[unit_id]) if scores is not None else None,
            detail=details[unit_id] if details is not None else None,
            symbol=symbol
        )
        ax_map.set_title(panel_title, fontsize=10, loc="left")

        plot_share_bar(ax_bar, row)
        present |= set(_present_shares(row).index)

    _add_landuse_legend(legend_fig, sorted(present))
    if has_title:
        title_fig.suptitle(title, fontsize=14)
    return fig


def plot_link_landuse_grid(
    edge_ids: Sequence[int],
    edges: gpd.GeoDataFrame,
    corridors: gpd.GeoSeries, # Same order as edges
    shares: pd.DataFrame,
    raster: np.ndarray,
    transform: Affine,
    **kwargs,
) -> Figure:
    """Land use grid for links: plot_landuse_grid with lines and corridors indexed by edge_id."""
    ids = edges["edge_id"].to_numpy()
    return plot_landuse_grid(
        edge_ids, edges.geometry.set_axis(ids), corridors.set_axis(ids),
        shares, raster, transform, unit="Link", **kwargs
    )


def plot_buffer_and_corridor(
    line: Geometry,
    r: float,
    raster: np.ndarray,
    transform: Affine,
    panel_size: float = 3.4,
    margin: float = .15,
) -> Figure:
    """Land use map of one link with its uncut buffer (left) and its corridor with end cuts (right)."""
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * panel_size, panel_size), layout="constrained"
    )
    plot_landuse_map(axes[0], raster, transform, line, line.buffer(r), margin)
    plot_landuse_map(axes[1], raster, transform, line, get_link_corridor(line, r), margin)
    axes[0].set_title(r"Buffer $B(e)$", fontsize=10)
    axes[1].set_title(r"Corridor $C(e)$", fontsize=10)
    return fig
