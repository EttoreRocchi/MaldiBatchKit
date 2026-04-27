"""Tests for generic batch-mixing metrics."""

from __future__ import annotations

import pandas as pd

from maldibatchkit.diagnostics import kbet, lisi, silhouette_batch


def test_silhouette_single_batch_returns_zero(tiny_dataset):
    X = tiny_dataset["X"]
    batch = pd.Series(["only"] * len(X), index=X.index)
    assert silhouette_batch(X, batch) == 0.0


def test_silhouette_detects_strong_effect(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    s = silhouette_batch(X, batch)
    assert -1.0 <= s <= 1.0
    assert s > 0.0  # synthetic shift should show batch clustering


def test_silhouette_ndarray(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    s = silhouette_batch(X.to_numpy(), batch.to_numpy())
    assert -1.0 <= s <= 1.0


def test_kbet_single_batch_edge_case(tiny_dataset):
    X = tiny_dataset["X"]
    batch = pd.Series(["only"] * len(X), index=X.index)
    result = kbet(X, batch)
    assert result["acceptance_rate"] == 1.0


def test_kbet_returns_keys_and_ranges(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    result = kbet(X, batch, k=15)
    assert set(result) == {"acceptance_rate", "mean_chi2", "k"}
    assert 0.0 <= result["acceptance_rate"] <= 1.0
    assert result["mean_chi2"] >= 0.0
    assert result["k"] == 15


def test_lisi_single_batch(tiny_dataset):
    X = tiny_dataset["X"]
    batch = pd.Series(["only"] * len(X), index=X.index)
    assert lisi(X, batch) == 1.0


def test_lisi_within_expected_range(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    value = lisi(X, batch)
    assert 1.0 <= value <= len(batch.unique())
