"""Tests for the Harmony wrapper with closed-form transform."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import Harmony

harmonypy_available = importlib.util.find_spec("harmonypy") is not None


def _make_shared_species_dataset(rng, *, n_per_batch=60, n_features=50):
    """Two batches (unit shift per feature) sharing three species clusters."""
    n = 2 * n_per_batch
    idx = [f"s{i:03d}" for i in range(n)]
    batch = pd.Series(["b0"] * n_per_batch + ["b1"] * n_per_batch, index=idx)
    species = np.tile(np.arange(n_per_batch) % 3, 2)
    centers = 2.0 * rng.standard_normal((3, n_features))
    X = rng.standard_normal((n, n_features)) + centers[species]
    X[batch.to_numpy() == "b1"] += 1.0
    return pd.DataFrame(X, index=idx), batch


def _batch_mean_gap(X, batch):
    b = batch.loc[X.index]
    return float(np.linalg.norm(X[b == "b1"].mean() - X[b == "b0"].mean()))


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
    np.testing.assert_allclose(out1.to_numpy(), corrector._corrected_cache_)


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_transforms_unseen_rows(big_dataset):
    """Closed-form transform works on rows not seen at fit time."""
    X, batch = big_dataset["X"], big_dataset["batch"]
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
    seen = batch.unique()[:2]
    train_mask = batch.isin(seen)
    X_tr = X.loc[train_mask]
    corrector = Harmony(batch=batch, max_iter=3, random_state=0, nclust=3).fit(X_tr)

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
    assert corrector.W_batch_.shape[1] == len(corrector.batch_levels_)
    assert np.isfinite(corrector.transform(X).to_numpy()).all()


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_exposes_fitted_attributes(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    corrector = Harmony(batch=batch, max_iter=3, random_state=0, nclust=4).fit(X)
    assert corrector.Y_.shape[1] == corrector.n_clusters_
    assert corrector.sigma_.shape == (corrector.n_clusters_,)
    K, B, d = corrector.W_batch_.shape
    assert K == corrector.n_clusters_
    assert B == len(corrector.batch_levels_)
    assert d == X.shape[1]


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
@pytest.mark.parametrize("order", ["sorted", "reversed"])
def test_harmony_transform_independent_of_training_row_order(rng, order):
    """Closed-form transform applies each batch's own correction to new rows."""
    X, batch = _make_shared_species_dataset(rng)
    train, test = X.index[::2], X.index[1::2]
    if order == "reversed":
        train = train[::-1]
    corrector = Harmony(batch=batch, random_state=0, verbose=False).fit(X.loc[train])
    before = _batch_mean_gap(X.loc[test], batch)
    after = _batch_mean_gap(corrector.transform(X.loc[test]), batch)
    assert after < 0.5 * before


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_small_sample_clamps_default_nclust(rng):
    X, batch = _make_shared_species_dataset(rng, n_per_batch=10)
    corrector = Harmony(batch=batch, random_state=0, verbose=False).fit(X)
    assert corrector.n_clusters_ == 2
    assert np.isfinite(corrector.transform(X).to_numpy()).all()


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_rejects_nclust_below_two(rng):
    X, batch = _make_shared_species_dataset(rng, n_per_batch=10)
    with pytest.raises(ValueError, match="nclust"):
        Harmony(batch=batch, nclust=1, verbose=False).fit(X)


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_forwards_random_state(monkeypatch, big_dataset):
    import harmonypy

    seeds = []
    run_harmony = harmonypy.run_harmony

    def spy(*args, **kwargs):
        seeds.append(kwargs.get("random_state"))
        return run_harmony(*args, **kwargs)

    monkeypatch.setattr(harmonypy, "run_harmony", spy)
    X, batch = big_dataset["X"], big_dataset["batch"]
    for random_state in (7, None, None):
        Harmony(
            batch=batch, random_state=random_state, max_iter=3, nclust=3, verbose=False
        ).fit(X)
    assert seeds[0] == 7
    assert seeds[1] != seeds[2]


@pytest.mark.skipif(not harmonypy_available, reason="harmonypy not installed")
def test_harmony_cache_not_stale_after_in_place_edit(big_dataset):
    X, batch = big_dataset["X"].copy(), big_dataset["batch"]
    corrector = Harmony(
        batch=batch, max_iter=3, random_state=0, nclust=3, verbose=False
    ).fit(X)
    X.iloc[0, 0] += 100.0
    out = corrector.transform(X)
    closed_form = corrector.transform(X.iloc[:-1])
    np.testing.assert_allclose(out.iloc[0].to_numpy(), closed_form.iloc[0].to_numpy())
