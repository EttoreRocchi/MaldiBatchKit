"""Tests for the BaseBatchCorrector scaffolding."""

from __future__ import annotations

import pandas as pd
import pytest

from maldibatchkit._base import BaseBatchCorrector
from maldibatchkit.corrections.baselines import MedianCentering


class _Counter(BaseBatchCorrector):
    """Minimal concrete subclass that tracks fit / transform calls."""

    def _fit_impl(self, X_df, batch):
        self.seen_columns_ = X_df.columns.tolist()

    def _transform_impl(self, X_df, batch):
        return X_df.copy()


def test_base_corrector_rejects_nan_batch(tiny_dataset):
    X = tiny_dataset["X"]
    batch = pd.Series([None] * len(X), index=X.index)
    with pytest.raises(ValueError, match="NaN"):
        _Counter(batch=batch).fit(X)


def test_base_corrector_rejects_missing_batch(tiny_dataset):
    X = tiny_dataset["X"]
    with pytest.raises(ValueError, match="must not be None"):
        _Counter(batch=None).fit(X)


def test_base_corrector_requires_1d_batch(tiny_dataset):
    X = tiny_dataset["X"]
    batch = pd.DataFrame(
        {"a": tiny_dataset["batch"], "b": tiny_dataset["batch"]}, index=X.index
    )
    with pytest.raises(ValueError, match="must be 1-D"):
        _Counter(batch=batch).fit(X)


def test_base_corrector_ndarray_X_yields_default_feature_names(tiny_dataset):
    X = tiny_dataset["X"].to_numpy()
    batch = tiny_dataset["batch"].to_numpy()
    corrector = MedianCentering(batch=batch).fit(X)
    assert corrector.feature_names_in_[0] == "x0"
