"""Tests for MALDI-specific corrections."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import BatchAwareWarping

maldiamrkit_available = importlib.util.find_spec("maldiamrkit") is not None


def _peaky_dataset(rng, n=40, n_bins=200):
    """Seven Gaussian peaks per spectrum; the second batch is shifted by 2 bins."""
    bins = np.arange(n_bins)
    centres = np.array([20, 45, 70, 95, 120, 145, 170])
    shifts = np.r_[np.zeros(n // 2), np.full(n // 2, 2.0)]
    rows = [
        np.exp(-((bins[:, None] - centres - s) ** 2) / 8.0).sum(axis=1)
        + rng.uniform(0, 0.01, n_bins)
        for s in shifts
    ]
    idx = [f"s{i}" for i in range(n)]
    batch = pd.Series(["a"] * (n // 2) + ["b"] * (n // 2), index=idx)
    return pd.DataFrame(np.vstack(rows), index=idx), batch


@pytest.mark.skipif(not maldiamrkit_available, reason="maldiamrkit not installed")
def test_batch_aware_warping_fit_transform(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    warper = BatchAwareWarping(batch=batch, method="shift", max_shift=5)
    out = warper.fit_transform(X)
    assert out.shape == X.shape
    assert set(warper.warpers_).issuperset(set(batch.unique()))
    assert "__global__" in warper.warpers_


@pytest.mark.skipif(not maldiamrkit_available, reason="maldiamrkit not installed")
def test_batch_aware_warping_unknown_reference_raises(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    with pytest.raises(ValueError, match="Unknown reference"):
        BatchAwareWarping(batch=batch, reference="???").fit(X)


@pytest.mark.skipif(not maldiamrkit_available, reason="maldiamrkit not installed")
def test_batch_aware_warping_batch_mean_reference(rng):
    X, batch = _peaky_dataset(rng)
    warper = BatchAwareWarping(batch=batch, reference="batch_mean", method="shift").fit(
        X
    )
    assert warper.reference_.shape == (X.shape[1],)
