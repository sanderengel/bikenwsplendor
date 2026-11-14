"""Sensitivity of link scores to the methodological choices."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from itertools import product
from typing import Sequence

from splendor.composite.normalize import normalize_all
from splendor.composite.aggregate import aggregate

COMBINATION_COLUMNS = [
    "measure", "normalization", "aggregation", 
    "land_use_weight", "poi_weight", "attractiveness_weight"
]


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


def _score_frame(edge_ids: np.ndarray, combination: dict, score: np.ndarray) -> pd.DataFrame:
    """Long-format scores of one combination, with ranks (1 = best, near-ties share a rank)."""
    return pd.DataFrame({
        "edge_id": edge_ids,
        **combination,
        "score": score,
        "rank": rankdata(-np.round(score, 12), method="min"),
    })


def _rescale_to_unit(score: np.ndarray) -> np.ndarray:
    """Min-max rescaling over the links to [0, 1], which keeps the ranks. Constant scores become 0."""
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
    Score every link under every combination of land use measure, normalization, aggregation and weights.

    'components' holds edge_id, one 'diversity_<measure>' column per measure, the POI and the attractiveness column.
    'bounds' maps each component column to its (lo, hi). Each combination normalizes all components with the same method.
    'weights' holds (land use, POI, attractiveness) weight tuples.
    
    Scores are in [0, 1]. With z-score normalization, 
    the aggregated scores are rescaled per combination (min-max over the links).

    Returns long format: edge_id, the combination columns, score, rank (1 = best).
    """
    edge_ids = components["edge_id"].to_numpy()
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
        frames.append(_score_frame(edge_ids, combination, score))

    return pd.concat(frames, ignore_index=True)


def summarize_ranks(grid: pd.DataFrame) -> pd.DataFrame:
    """Per link: median rank and rank spread over all combinations in the grid (1 = best)."""
    ranks = grid.groupby("edge_id")["rank"]
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


def get_rank_correlations(grid: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between the link scores of every pair of combinations."""
    wide = grid.pivot(index="edge_id", columns=COMBINATION_COLUMNS, values="score")
    return wide.corr(method="spearman")
