"""Tests for limma-style removeBatchEffect."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import Limma
from maldibatchkit.diagnostics import silhouette_batch


def test_limma_reduces_batch_effect(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    before = silhouette_batch(X, batch)
    corrector = Limma(batch=batch).fit(X)
    after = silhouette_batch(corrector.transform(X), batch)
    assert after < before


def test_limma_with_design_preserves_covariate(tiny_dataset):
    X, batch, species = (
        tiny_dataset["X"],
        tiny_dataset["batch"],
        tiny_dataset["species"],
    )
    species_dummy = pd.get_dummies(species, drop_first=True).astype(float)
    corrector = Limma(batch=batch, design=species_dummy).fit(X)
    out = corrector.transform(X)
    assert out.shape == X.shape
    # The gamma_ should have shape (n_contrasts, n_features) with n_batches-1 rows
    assert corrector.gamma_.shape == (len(batch.unique()) - 1, X.shape[1])


def test_limma_unknown_batch_at_transform_raises(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = Limma(batch=batch).fit(X)
    X_new = X.iloc[:5].copy()
    corrector.batch = pd.Series(["new"] * 5, index=X_new.index)
    with pytest.raises(ValueError, match="Unseen batch level"):
        corrector.transform(X_new)


def test_limma_array_design(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    design = np.random.default_rng(0).standard_normal((len(X), 2))
    corrector = Limma(batch=batch, design=design).fit(X)
    out = corrector.transform(X)
    assert out.shape == X.shape
