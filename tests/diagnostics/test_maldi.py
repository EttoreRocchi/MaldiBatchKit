"""Tests for MALDI-specific diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldibatchkit.diagnostics import (
    peak_position_drift,
    per_batch_spectrum_count,
    tic_cov_per_batch,
)
from maldibatchkit.diagnostics.maldi import _local_maxima


def test_per_batch_spectrum_count_counts(tiny_dataset):
    batch = tiny_dataset["batch"]
    counts = per_batch_spectrum_count(batch)
    assert counts.sum() == len(batch)
    for lvl in batch.unique():
        assert counts.loc[lvl] == (batch == lvl).sum()


def test_tic_cov_per_batch_matches_manual(rng):
    n_per_batch = 30
    idx = [f"s{i}" for i in range(2 * n_per_batch)]
    X = pd.DataFrame(rng.uniform(1, 10, size=(2 * n_per_batch, 8)), index=idx)
    batch = pd.Series([0] * n_per_batch + [1] * n_per_batch, index=idx)
    out = tic_cov_per_batch(X, batch)
    tic = X.sum(axis=1)
    for lvl in [0, 1]:
        vals = tic[batch == lvl]
        expected = vals.std(ddof=0) / vals.mean()
        assert np.isclose(out.loc[lvl], expected)


def test_tic_cov_handles_ndarray(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    out = tic_cov_per_batch(X.to_numpy(), batch.to_numpy())
    assert isinstance(out, pd.Series)


def test_peak_position_drift_zero_on_single_batch(big_dataset):
    X = big_dataset["X"]
    batch = pd.Series(["only"] * len(X), index=X.index)
    out = peak_position_drift(X, batch, top_k=5)
    assert "only" in out.index
    # The per-batch median equals the global median -> deltas are 0
    assert np.isclose(out.loc["only", "mean_delta_mz"], 0.0)


def test_peak_position_drift_with_mz_values(rng):
    n, p = 40, 50
    idx = [f"s{i}" for i in range(n)]
    X = pd.DataFrame(rng.standard_normal((n, p)) + 1.0, index=idx)
    # Insert two shared peaks at columns 10 and 30
    X.iloc[:, 10] += 10
    X.iloc[:, 30] += 10
    batch = pd.Series([0] * (n // 2) + [1] * (n // 2), index=idx)
    mz = np.linspace(2000, 20000, p)
    out = peak_position_drift(X, batch, mz_values=mz, top_k=2)
    assert "mean_delta_mz" in out.columns
    assert out.shape[0] == 2


def test_peak_position_drift_mz_length_mismatch(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    with pytest.raises(ValueError, match="mz_values has length"):
        peak_position_drift(X, batch, mz_values=np.arange(3))


def test_peak_position_drift_flat_spectrum_returns_empty():
    X = pd.DataFrame(np.ones((10, 8)))
    batch = pd.Series(["a"] * 5 + ["b"] * 5)
    out = peak_position_drift(X, batch, top_k=5)
    assert out.empty


def test_local_maxima_empty_on_short_array():
    assert _local_maxima(np.array([1.0, 2.0])).size == 0


def test_local_maxima_finds_interior_peaks():
    idx = _local_maxima(np.array([0, 2, 1, 3, 0]))
    assert list(idx) == [1, 3]
