"""Tests for ``NoOpCorrector``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from maldibatchkit import NoOpCorrector


def test_noop_returns_input_dataframe_untouched(tiny_dataset):
    X = tiny_dataset["X"]
    out = NoOpCorrector(batch=tiny_dataset["batch"]).fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_frame_equal(out, X)


def test_noop_returns_ndarray_for_ndarray_input(tiny_dataset):
    X = tiny_dataset["X"].to_numpy()
    out = NoOpCorrector(batch=tiny_dataset["batch"].to_numpy()).fit_transform(X)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, X)


def test_noop_is_check_is_fitted_compatible(tiny_dataset):
    c = NoOpCorrector(batch=tiny_dataset["batch"])
    c.fit(tiny_dataset["X"])
    # check_is_fitted picks up feature_names_in_ from the base class
    check_is_fitted(c)


def test_noop_records_feature_names(tiny_dataset):
    c = NoOpCorrector(batch=tiny_dataset["batch"])
    c.fit(tiny_dataset["X"])
    np.testing.assert_array_equal(
        c.feature_names_in_, np.asarray(tiny_dataset["X"].columns, dtype=object)
    )
    assert c.n_features_in_ == tiny_dataset["X"].shape[1]
