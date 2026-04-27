"""Tests for the simple baseline corrections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from maldibatchkit import MedianCentering, ReferenceScaling, ZScorePerBatch


def test_median_centering_zeroes_per_batch_medians(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = MedianCentering(batch=batch).fit(X)
    out = corrector.transform(X)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == X.shape
    # Per-batch medians should be near zero after centering
    for lvl in batch.unique():
        mask = (batch == lvl).to_numpy()
        per_batch_median = out.loc[mask].median(axis=0).to_numpy()
        assert np.max(np.abs(per_batch_median)) < 1e-10


def test_median_centering_ndarray_input_returns_ndarray(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    # Pass batch as a bare ndarray so the corrector does positional alignment
    # against the default RangeIndex of the ndarray input.
    corrector = MedianCentering(batch=batch.to_numpy()).fit(X.to_numpy())
    out = corrector.transform(X.to_numpy())
    assert isinstance(out, np.ndarray)
    assert out.shape == X.shape


def test_median_centering_falls_back_on_unknown_batch(tiny_dataset, rng):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = MedianCentering(batch=batch).fit(X)
    # Pretend new samples arrive with a never-seen batch label
    X_new = X.iloc[:5].copy()
    batch_new = pd.Series(["new_batch"] * 5, index=X_new.index)
    corrector.batch = batch_new
    out = corrector.transform(X_new)
    expected = X_new.to_numpy() - corrector.grand_median_.to_numpy()
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_zscore_per_batch_standardises(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = ZScorePerBatch(batch=batch).fit(X)
    out = corrector.transform(X)
    for lvl in batch.unique():
        mask = (batch == lvl).to_numpy()
        m = out.loc[mask].mean(axis=0).to_numpy()
        s = out.loc[mask].std(axis=0, ddof=0).to_numpy()
        np.testing.assert_allclose(m, 0.0, atol=1e-10)
        np.testing.assert_allclose(s, 1.0, atol=1e-6)


def test_zscore_handles_unknown_batch(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = ZScorePerBatch(batch=batch).fit(X)
    X_new = X.iloc[:5].copy()
    batch_new = pd.Series(["new"] * 5, index=X_new.index)
    corrector.batch = batch_new
    out = corrector.transform(X_new)
    # Output should use grand stats and have shape matching X_new
    assert out.shape == X_new.shape


def test_reference_scaling_chooses_largest_batch_as_reference(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = ReferenceScaling(batch=batch).fit(X)
    # Both tiny batches are same size -> first alphabetical level wins.
    assert corrector.reference_batch_ in set(batch.unique())
    out = corrector.transform(X)
    assert out.shape == X.shape


def test_reference_scaling_explicit_reference(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = ReferenceScaling(batch=batch, reference_batch="b0").fit(X)
    assert corrector.reference_batch_ == "b0"
    out = corrector.transform(X)
    mask = (batch == "b0").to_numpy()
    np.testing.assert_allclose(out.loc[mask].to_numpy(), X.loc[mask].to_numpy())


def test_reference_scaling_invalid_reference_raises(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    with pytest.raises(ValueError, match="reference_batch="):
        ReferenceScaling(batch=batch, reference_batch="nope").fit(X)


def test_transform_without_fit_raises(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    with pytest.raises(NotFittedError):
        MedianCentering(batch=batch).transform(X)


def test_zscore_eps_floor(rng):
    # Constant feature -> std is 0 pre-floor; eps must kick in without /0 crash
    idx = [f"s{i}" for i in range(10)]
    X = pd.DataFrame(np.column_stack([np.ones(10), rng.standard_normal(10)]), index=idx)
    batch = pd.Series(["a"] * 5 + ["b"] * 5, index=idx)
    corrector = ZScorePerBatch(batch=batch).fit(X)
    out = corrector.transform(X)
    assert np.isfinite(out.to_numpy()).all()
