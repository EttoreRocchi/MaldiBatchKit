"""Tests for the Harmony wrapper with closed-form transform."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import Harmony

harmonypy_available = importlib.util.find_spec("harmonypy") is not None


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_fit_transform_returns_dataframe(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    corrector = Harmony(batch=batch, max_iter=3, random_state=0, nclust=3)
    out = corrector.fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == X.shape


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_same_matrix_fast_path_matches_upstream(big_dataset):
    """Calling transform on the exact training matrix returns harmonypy's output."""
    X, batch = big_dataset["X"], big_dataset["batch"]
    corrector = Harmony(batch=batch, max_iter=3, random_state=0, nclust=3).fit(X)
    out1 = corrector.transform(X)
    out2 = corrector.transform(X)
    np.testing.assert_allclose(out1.to_numpy(), out2.to_numpy())
    # Cached result equals the stored harmonypy Z_corr
    np.testing.assert_allclose(out1.to_numpy(), corrector._corrected_cache_)


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_transforms_unseen_rows(big_dataset):
    """Closed-form transform works on rows not seen at fit time."""
    X, batch = big_dataset["X"], big_dataset["batch"]
    # Stratified split so every batch is in both folds
    train_idx, test_idx = [], []
    for lvl in batch.unique():
        lvl_ids = batch[batch == lvl].index.tolist()
        train_idx.extend(lvl_ids[: len(lvl_ids) // 2])
        test_idx.extend(lvl_ids[len(lvl_ids) // 2 :])
    X_tr, X_te = X.loc[train_idx], X.loc[test_idx]

    corrector = Harmony(batch=batch, max_iter=3, random_state=0, nclust=3).fit(X_tr)
    out = corrector.transform(X_te)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == X_te.shape
    assert np.isfinite(out.to_numpy()).all()


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_rejects_unseen_batch(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    # Use only two of the three batches for fit
    seen = batch.unique()[:2]
    train_mask = batch.isin(seen)
    X_tr = X.loc[train_mask]
    corrector = Harmony(batch=batch, max_iter=3, random_state=0, nclust=3).fit(X_tr)

    # Transform on a subset containing the third batch -> must raise
    X_te = X.loc[~train_mask]
    corrector.batch = batch.loc[X_te.index]
    with pytest.raises(ValueError, match="Unseen batch level"):
        corrector.transform(X_te)


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_accepts_extra_covariates(big_dataset):
    """Pass a ``covariates`` Series through to harmonypy's vars_use."""
    X, batch, species = (
        big_dataset["X"],
        big_dataset["batch"],
        big_dataset["species"],
    )
    corrector = Harmony(
        batch=batch,
        covariates=species,
        random_state=0,
        max_iter=3,
        nclust=3,
    )
    out = corrector.fit_transform(X)
    assert out.shape == X.shape
    assert corrector.W_batch_.shape[1] == len(corrector.batch_levels_)


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_covariates_from_dataframe(big_dataset):
    """A multi-column DataFrame covariate is renamed cov_0 / cov_1 / ..."""
    import numpy as np
    import pandas as pd

    X, batch, species = (
        big_dataset["X"],
        big_dataset["batch"],
        big_dataset["species"],
    )
    extra = pd.DataFrame(
        {
            "site": ["A", "B"] * (len(X) // 2),
            "sp": species.values,
        },
        index=X.index,
    )
    corrector = Harmony(
        batch=batch,
        covariates=extra,
        random_state=0,
        max_iter=3,
        nclust=3,
    ).fit(X)
    assert corrector.Y_.shape[1] == corrector.n_clusters_
    # harmonypy's batch indicator Phi encodes batch|site|sp, so
    # W_batch_'s B dimension equals the number of *batch* levels only.
    assert corrector.W_batch_.shape[1] == len(corrector.batch_levels_)
    # Correction applied, result finite
    assert np.isfinite(corrector.transform(X).to_numpy()).all()


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_exposes_fitted_attributes(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    corrector = Harmony(batch=batch, max_iter=3, random_state=0, nclust=4).fit(X)
    # Shapes per the docstring
    assert corrector.Y_.shape[1] == corrector.n_clusters_
    assert corrector.sigma_.shape == (corrector.n_clusters_,)
    K, B, d = corrector.W_batch_.shape
    assert K == corrector.n_clusters_
    assert B == len(corrector.batch_levels_)
    assert d == X.shape[1]
