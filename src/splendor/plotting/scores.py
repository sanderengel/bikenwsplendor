"""Plots of link scores and how sensitive they are to the methodological choices."""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from typing import Sequence

from splendor.plotting.style import sequential_cmap


def plot_link_values(
    ax: Axes,
    edges: gpd.GeoDataFrame,
    values: pd.Series,
    title: str,
    cmap: Colormap,
    norm: Normalize,
    low_is_best: bool = False,
    linewidth: float = 1.5,
) -> None:
    """Links as lines colored by a value per link ('values' indexed by edge_id). The best links are drawn on top."""
    data = edges[["edge_id", "geometry"]].merge(
        values.rename("value"), left_on="edge_id", right_index=True, how="left", validate="one_to_one"
    )
    if data["value"].isna().any():
        raise ValueError(f"{data['value'].isna().sum()} links have no value")

    data = data.sort_values("value", ascending=not low_is_best) # Best last, so on top
    data.plot(ax=ax, column="value", cmap=cmap, norm=norm, linewidth=linewidth)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_link_maps(
    edges: gpd.GeoDataFrame,
    data: pd.DataFrame,
    columns: Sequence[str],
    titles: Sequence[str],
    labels: Sequence[str],
    low_is_best: Sequence[bool] | None = None,
    panel_size: tuple[float, float] = (5, 5),
) -> Figure:
    """One map per column of 'data' (which holds edge_id), side by side, each with its own colorbar."""
    low_is_best = low_is_best or [False] * len(columns)
    fig, axes = plt.subplots(
        1, len(columns), figsize=(panel_size[0] * len(columns), panel_size[1]),
        squeeze=False, layout="constrained",
    )
    for ax, column, title, label, low in zip(axes.flat, columns, titles, labels, low_is_best):
        values = data.set_index("edge_id")[column]
        norm = Normalize(values.min(), values.max())
        cmap = sequential_cmap(reverse=low) # Dark = best
        plot_link_values(ax, edges, values, title, cmap, norm, low)
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), 
            ax=ax, label=label,
            location="bottom", shrink=.6, aspect=30, pad=.02
        )
    return fig


def _blocks(keys: Sequence[tuple]) -> list[tuple[int, int, tuple]]:
    """Runs of equal consecutive keys, as (start, stop, key)."""
    blocks, start = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            blocks.append((start, i, keys[start]))
            start = i
    return blocks


def _format_level_value(value) -> str:
    """Floats (weights) with two significant digits, e.g. 1/3 -> '0.33'."""
    return f"{value:.2g}" if isinstance(value, float) else str(value)


def plot_rank_correlations(
    correlations: pd.DataFrame,
    level_order: Sequence[str] = ("measure", "normalization", "aggregation"),
    n_label_levels: int = 1,
    title: str = "Rank correlations between combinations",
    size: float = 9,
) -> Figure:
    """
    Heatmap of the Spearman correlations between combinations, sorted by 'level_order',
    so blocks show which choice the correlation depends on most (outermost level = largest blocks).
    """
    corr = correlations.reorder_levels(level_order, axis=0).reorder_levels(level_order, axis=1)
    corr = corr.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(size, size), layout="constrained")
    image = ax.imshow(corr.to_numpy(), cmap=sequential_cmap(), vmin=corr.to_numpy().min(), vmax=1)

    blocks = _blocks([key[:n_label_levels] for key in corr.index])
    positions = [(start + stop - 1) / 2 for start, stop, _ in blocks]
    labels = [" · ".join(_format_level_value(v) for v in key) for _, _, key in blocks]
    ax.set_xticks(positions, labels, rotation=90, fontsize=8)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.tick_params(length=0)
    for start, _, _ in blocks[1:]:
        ax.axhline(start - .5, color="white", linewidth=1)
        ax.axvline(start - .5, color="white", linewidth=1)

    inner = " · ".join(level_order[n_label_levels:])
    ax.set_xlabel(f"Within blocks, sorted by: {inner}", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Spearman correlation", shrink=.7)
    return fig
