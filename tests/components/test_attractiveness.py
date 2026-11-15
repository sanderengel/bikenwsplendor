import pytest
import numpy as np
import pandas as pd

from splendor.components.attractiveness import get_attractiveness

VALUES = {1: 0.0, 4: 0.5, 12: 1.0}


def test_attractiveness_known_values():
    """Pure types give their own value, a mix gives the share-weighted mean."""
    shares = pd.DataFrame({
        "share_1": [1.0, 0.0, 0.0],
        "share_4": [0.0, 0.0, 0.5],
        "share_12": [0.0, 1.0, 0.5],
    })
    np.testing.assert_allclose(get_attractiveness(shares, VALUES), [0.0, 1.0, 0.75])


def test_attractiveness_ignores_other_columns():
    shares = pd.DataFrame({"edge_id": [7], "n_pixels": [100], "share_4": [1.0]})
    np.testing.assert_allclose(get_attractiveness(shares, VALUES), [0.5])


def test_attractiveness_missing_type_error():
    shares = pd.DataFrame({"share_2": [1.0]})
    with pytest.raises(ValueError, match="no attractiveness value"):
        get_attractiveness(shares, VALUES)
