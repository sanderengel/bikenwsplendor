"""Sensitivity of unit (links or loops) scores to the methodological choices."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from itertools import product
from typing import Sequence, Any

from splendor.composite.normalize import normalize_all
from splendor.composite.aggregate import aggregate

WEIGHT_COLUMNS = ["land_use_weight", "poi_weight", "attractiveness_weight"]
COMBINATION_COLUMNS = ["measure", "normalization", "aggregation"] + WEIGHT_COLUMNS


def _normalize_components(
    components: pd.DataFrame,
    columns: Sequence[str],
    normalizations: Sequence[str],
    bounds: dict[str, tuple[float, float]],
    p: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Normalize each component column with every method: {column: {method: values}}."""
    return {
        column: normalize_all(components[column].to_numpy(dtype=float), normalizations, *bounds[column], p)
        for column in columns
    }


def _score_frame(ids: pd.Index, combination: dict, score: np.ndarray) -> pd.DataFrame:
    """Long-format scores of one combination, with ranks (1 = best, near-ties share a rank)."""
    return pd.DataFrame({
        ids.name: ids.to_numpy(),
        **combination,
        "score": score,
        "rank": rankdata(-np.round(score, 12), method="min"),
    })


def _rescale_to_unit(score: np.ndarray) -> np.ndarray:
    """Min-max rescaling over the units to [0, 1], which keeps the ranks. Constant scores become 0."""
    score = np.round(score, 12) # Float near-ties stay as ties
    span = score.max() - score.min()
    return np.zeros_like(score) if span == 0 else (score - score.min()) / span


def get_score_grid(
    components: pd.DataFrame,
    measures: Sequence[str],
    normalizations: Sequence[str],
    aggregations: Sequence[str],
    weights: Sequence[tuple[float, float, float]],
    bounds: dict[str, tuple[float, float]],
    p: float,
    geometric_floor: float,
    poi_column: str = "poi_diversity",
    attractiveness_column: str = "attractiveness",
) -> pd.DataFrame:
    """
    Score every unit under every combination of land use measure, normalization, aggregation and weights.

    'components' is indexed by the unit id (e.g. edge_id or loop_id) and holds one 'diversity_<measure>'
    column per measure, the POI and the attractiveness column.
    'bounds' maps each component column to its (lo, hi). Each combination normalizes all components with the same method.
    'weights' holds (land use, POI, attractiveness) weight tuples.

    Scores are in [0, 1]. With z-score normalization,
    the aggregated scores are rescaled per combination (min-max over the units).

    Returns long format: the id column (named after the index), the combination columns, score, rank (1 = best).
    """
    if components.index.name is None:
        raise ValueError("components must be indexed by a named unit id")
    ids = components.index
    columns = [*(f"diversity_{m}" for m in measures), poi_column, attractiveness_column]
    normalized = _normalize_components(components, columns, normalizations, bounds, p)

    frames = []
    for measure, normalization, aggregation, component_weights in product(measures, normalizations, aggregations, weights):
        component_columns = (f"diversity_{measure}", poi_column, attractiveness_column)
        matrix = np.column_stack([normalized[c][normalization] for c in component_columns])
        floor = geometric_floor if aggregation == "geometric" else 0.0

        score = aggregate(matrix, aggregation, component_weights, floor)
        if normalization == "zscore":
            score = _rescale_to_unit(score)
        
        combination = dict(zip(COMBINATION_COLUMNS, (measure, normalization, aggregation, *component_weights)))
        frames.append(_score_frame(ids, combination, score))

    return pd.concat(frames, ignore_index=True)


def summarize_ranks(grid: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Per unit: median rank and rank spread over all combinations in the grid (1 = best)."""
    ranks = grid.groupby(id_column)["rank"]
    summary = pd.DataFrame({
        "median_rank": ranks.median(),
        "q25_rank": ranks.quantile(0.25),
        "q75_rank": ranks.quantile(0.75),
        "min_rank": ranks.min(),
        "max_rank": ranks.max(),
    })
    summary["iqr_rank"] = summary["q75_rank"] - summary["q25_rank"]
    summary["range_rank"] = summary["max_rank"] - summary["min_rank"]
    return summary.reset_index()


def get_rank_correlations(grid: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Spearman correlation between the unit scores of every pair of combinations."""
    wide = grid.pivot(index=id_column, columns=COMBINATION_COLUMNS, values="score")
    return wide.corr(method="spearman")


def select_weights(grid: pd.DataFrame, weights: Sequence[tuple[float, float, float]]) -> pd.DataFrame:
    """Rows of the grid whose weight tuple is one of 'weights'."""
    is_selected = pd.MultiIndex.from_frame(grid[WEIGHT_COLUMNS]).isin(weights)
    return grid[is_selected]


def get_main_scores(grid: pd.DataFrame, specification: dict[str, Any], id_column: str) -> pd.DataFrame:
    """Scores and ranks of the main specification: one row per unit (id, score, rank)."""
    main = grid
    for column, choice in specification.items():
        main = main[main[column] == choice]
    main = main[[id_column, "score", "rank"]]
    if not (main[id_column].is_unique and main[id_column].nunique() == grid[id_column].nunique()):
        raise ValueError("main specification must match exactly one combination in the grid")
    return main.reset_index(drop=True)
