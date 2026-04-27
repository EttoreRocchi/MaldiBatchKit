"""Tests for MALDI-specific corrections."""

from __future__ import annotations

import importlib.util

import pytest

from maldibatchkit import BatchAwareWarping

maldiamrkit_available = importlib.util.find_spec("maldiamrkit") is not None


@pytest.mark.skipif(not maldiamrkit_available, reason="maldiamrkit not installed")
def test_batch_aware_warping_fit_transform(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    warper = BatchAwareWarping(batch=batch, method="shift", max_shift=5)
    out = warper.fit_transform(X)
    assert out.shape == X.shape
    # One warper per batch plus the global fallback
    assert set(warper.warpers_).issuperset(set(batch.unique()))
    assert "__global__" in warper.warpers_


@pytest.mark.skipif(not maldiamrkit_available, reason="maldiamrkit not installed")
def test_batch_aware_warping_unknown_reference_raises(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    with pytest.raises(ValueError, match="Unknown reference"):
        BatchAwareWarping(batch=batch, reference="???").fit(X)


@pytest.mark.skipif(not maldiamrkit_available, reason="maldiamrkit not installed")
def test_batch_aware_warping_batch_mean_reference(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    warper = BatchAwareWarping(batch=batch, reference="batch_mean", method="shift").fit(
        X
    )
    assert warper.reference_.shape == (X.shape[1],)
