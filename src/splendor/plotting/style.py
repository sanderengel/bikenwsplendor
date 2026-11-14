import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap

SERIES_COLOR = "#4a6fa5" # Single-series color for distributions and scatter plots


def clean_axes(ax: Axes) -> None:
    """Hide the top and right spines."""
    ax.spines[["top", "right"]].set_visible(False)


def sequential_cmap(name: str = "inferno", start: float = 0.0, end: float = .85, reverse: bool = False) -> Colormap:
    """Single-hue sequential colormap."""
    colors = plt.get_cmap(name)(np.linspace(start, end, 256))
    return ListedColormap(colors[::-1] if reverse else colors)
