import math
import pytest
import numpy as np
import pandas as pd

from splendor.composite.sensitivity import (
    get_score_grid,
    summarize_ranks,
    get_rank_correlations,
    select_weights,
    get_main_scores,
    COMBINATION_COLUMNS,
    WEIGHT_COLUMNS,
)

MEASURES = ("shannon", "richness")
NORMALIZATIONS = ("bounds", "percentile", "rank")
AGGREGATIONS = ("mean", "geometric")
P = 99

WEIGHTS = ((0.5, 0.5, 0.0), (1.0, 0.0, 0.0), (0.5, 0.25, 0.25), (0.0, 0.0, 1.0))
N_COMBINATIONS = len(MEASURES) * len(NORMALIZATIONS) * len(AGGREGATIONS) * len(WEIGHTS)

BOUNDS = {
    "diversity_shannon": (0, math.log(13)),
    "diversity_richness": (1, 10),
    "poi_diversity": (0, 3),
    "attractiveness": (0, 1)
}

MAIN = { # Shannon, bounds, mean, land use only: scores are Shannon / ln 13
    "measure": "shannon", "normalization": "bounds", "aggregation": "mean",
    "land_use_weight": 1.0, "poi_weight": 0.0, "attractiveness_weight": 0.0,
}


@pytest.fixture
def components() -> pd.DataFrame:
    """Four links with non-consecutive edge ids as the index."""
    return pd.DataFrame({
        "diversity_shannon": [0.2, 0.8, 1.2, 0.5],
        "diversity_richness": [1, 2, 3, 2],
        "poi_diversity": [0, 1, 3, 2],
        "attractiveness": [0.0, 0.25, 1.0, 0.5]
    }, index=pd.Index([10, 11, 12, 13], name="edge_id"))


@pytest.fixture
def grid(components) -> pd.DataFrame:
    return get_score_grid(components, MEASURES, NORMALIZATIONS, AGGREGATIONS, WEIGHTS, BOUNDS, P, 0.25)


def test_grid_covers_every_combination_for_every_link(grid, components):
    sizes = grid.groupby(COMBINATION_COLUMNS).size()
    assert len(sizes) == N_COMBINATIONS
    assert (sizes == len(components)).all()


def test_grid_keeps_ids(grid, components):
    """Every combination has all units, in the input order, in a column named after the index."""
    for _, group in grid.groupby(COMBINATION_COLUMNS):
        assert group["edge_id"].to_list() == components.index.to_list()


def _combination(grid: pd.DataFrame, **choices) -> pd.DataFrame:
    """Rows of the grid with the given value in each combination column."""
    mask = (grid[list(choices)] == pd.Series(choices)).all(axis=1)
    return grid[mask]


def test_grid_known_score(grid, components):
    """Shannon, bounds, mean, weight 1: the score is Shannon / ln 13, and the highest Shannon gets rank 1."""
    combination = _combination(
        grid, measure="shannon", normalization="bounds", aggregation="mean",
        land_use_weight=1.0, poi_weight=0.0, attractiveness_weight=0.0,
    )
    expected = components["diversity_shannon"].to_numpy() / math.log(13)
    np.testing.assert_allclose(combination["score"], expected)
    assert combination.set_index("edge_id").loc[12, "rank"] == 1


def test_summary_one_row_per_link(grid, components):
    summary = summarize_ranks(grid, "edge_id")
    assert summary["edge_id"].to_list() == sorted(components.index)


def test_rank_correlations_shape_and_diagonal(grid):
    """One row and column per combination, and every combination correlates perfectly with itself."""
    correlations = get_rank_correlations(grid, "edge_id")
    assert correlations.shape == (N_COMBINATIONS, N_COMBINATIONS)
    np.testing.assert_allclose(np.diag(correlations.to_numpy()), 1)


def test_grid_near_tied_scores_share_rank():
    """Scores that are equal up to float rounding share a rank."""
    components = pd.DataFrame({ # 0.5 * (0.1 + 0.2) and 0.5 * 0.3 differ only by float rounding
        "diversity_shannon": [0.1, 0.3],
        "poi_diversity": [0.2, 0.0],
        "attractiveness": [0.0, 0.0]
    }, index=pd.Index([1, 2], name="edge_id"))
    bounds = {"diversity_shannon": (0, 1), "poi_diversity": (0, 1), "attractiveness": (0, 1)}
    grid = get_score_grid(
        components, ["shannon"], ["bounds"], ["mean"], [(0.5, 0.5, 0.0)], 
        bounds, p=99, geometric_floor=0.25
    )
    assert grid["rank"].to_list() == [1, 1]


def test_grid_attractiveness_only(grid, components):
    """Bounds, mean, a = 1: the score is the attractiveness itself."""
    combination = _combination(
        grid, measure="shannon", normalization="bounds", aggregation="mean",
        land_use_weight=0.0, poi_weight=0.0, attractiveness_weight=1.0,
    )
    np.testing.assert_allclose(combination["score"], components["attractiveness"])


def test_grid_zscore_scores_span_unit_interval(components):
    """Z-score combinations are rescaled so their scores span exactly [0, 1]."""
    grid = get_score_grid(components, ["shannon"], ["zscore"], ["mean"], [(0.5, 0.25, 0.25)], BOUNDS, P, 0.25)
    assert grid["score"].min() == 0
    assert grid["score"].max() == 1


def test_grid_unnamed_index_error(components):
    """Components without a named id index should raise an error."""
    with pytest.raises(ValueError):
        get_score_grid(components.reset_index(drop=True), MEASURES, NORMALIZATIONS, AGGREGATIONS, WEIGHTS, BOUNDS, P, 0.25)


def test_select_weights_keeps_only_given_tuples(grid, components):
    """Only rows with one of the given weight tuples are kept, with all their combinations and units."""
    selected = select_weights(grid, [(0.5, 0.5, 0.0), (1.0, 0.0, 0.0)])

    kept = set(selected[WEIGHT_COLUMNS].itertuples(index=False, name=None))
    assert kept == {(0.5, 0.5, 0.0), (1.0, 0.0, 0.0)}
    assert len(selected) == 2 * len(grid) // len(WEIGHTS)


def test_main_scores_one_row_per_unit(grid, components):
    """The main specification gives one row per unit, with its score and rank."""
    main = get_main_scores(grid, MAIN, "edge_id")

    assert list(main.columns) == ["edge_id", "score", "rank"]
    assert main["edge_id"].to_list() == components.index.to_list()
    np.testing.assert_allclose(main["score"], components["diversity_shannon"].to_numpy() / math.log(13))


def test_main_scores_several_combinations_error(grid):
    """A specification that matches several combinations should raise an error."""
    incomplete = {k: v for k, v in MAIN.items() if k != "aggregation"} # Matches mean and geometric
    with pytest.raises(ValueError):
        get_main_scores(grid, incomplete, "edge_id")


def test_main_scores_no_combination_error(grid):
    """A specification that matches no combination should raise an error."""
    with pytest.raises(ValueError):
        get_main_scores(grid, MAIN | {"land_use_weight": 0.9}, "edge_id")
