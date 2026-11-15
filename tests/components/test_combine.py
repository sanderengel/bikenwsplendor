import math
import pytest

from splendor.components.combine import get_component_bounds

MEASURES = ("shannon", "richness")
POI_BOUNDS = (0, 3)
ATTRACTIVENESS = {1: 0.0, 2: 0.25, 12: 1.0}


def test_component_bounds_leave_out_excluded_types():
    """With 14 types and type 13 excluded, the measures see 13 types: Shannon (0, ln 13), richness (1, 10)."""
    bounds = get_component_bounds(MEASURES, 14, (13,), 0.1, POI_BOUNDS, ATTRACTIVENESS)

    assert bounds["diversity_shannon"] == pytest.approx((0, math.log(13)))
    assert bounds["diversity_richness"] == (1, 10)
    assert bounds["poi_diversity"] == POI_BOUNDS
    assert bounds["attractiveness"] == (0.0, 1.0) # Min and max of the mapping


def test_component_bounds_count_excluded_types_once():
    """A duplicated excluded type should be counted once."""
    once = get_component_bounds(MEASURES, 14, (13,), 0.1, POI_BOUNDS, ATTRACTIVENESS)
    twice = get_component_bounds(MEASURES, 14, (13, 13), 0.1, POI_BOUNDS, ATTRACTIVENESS)
    assert once == twice


def test_component_bounds_invalid_excluded_type_error():
    """An excluded type outside 1..n_types should raise an error."""
    with pytest.raises(ValueError):
        get_component_bounds(MEASURES, 14, (15,), 0.1, POI_BOUNDS, ATTRACTIVENESS)
