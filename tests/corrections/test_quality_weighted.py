"""Tests for the QualityWeightedComBat transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import ComBat, QualityWeightedComBat


def _make_weighted_dataset(rng, *, n_per_batch=30, n_features=10, batch_shift=1.0):
    n = 2 * n_per_batch
    idx = [f"s{i}" for i in range(n)]
    batch = pd.Series([0] * n_per_batch + [1] * n_per_batch, index=idx)
    shift = np.where(batch.values[:, None] == 0, batch_shift, -batch_shift)
    X = pd.DataFrame(rng.standard_normal((n, n_features)) + shift, index=idx)
    return X, batch


def test_equal_weights_match_unweighted_estimates(rng):
    X, batch = _make_weighted_dataset(rng)
    quality = pd.Series(np.ones(len(X)), index=X.index)
    eq = QualityWeightedComBat(batch=batch, quality=quality).fit(X)
    np.testing.assert_allclose(eq.grand_mean_, X.mean(axis=0).to_numpy(), atol=1e-10)


def test_reduces_batch_effect(rng):
    X, batch = _make_weighted_dataset(rng)
    quality = pd.Series(np.ones(len(X)), index=X.index)
    from maldibatchkit.diagnostics import silhouette_batch

    before = silhouette_batch(X, batch)
    corrector = QualityWeightedComBat(batch=batch, quality=quality).fit(X)
    after = silhouette_batch(corrector.transform(X), batch)
    assert after < before


def test_high_weight_samples_dominate_moments(rng):
    X, batch = _make_weighted_dataset(rng, batch_shift=2.0)
    weights = np.ones(len(X))
    weights[batch == 0] = 1e-6
    weights[batch == 0] = np.clip(weights[batch == 0], 1e-6, None)
    quality = pd.Series(weights, index=X.index)
    with pytest.warns(UserWarning, match="effective batch size"):
        corrector = QualityWeightedComBat(batch=batch, quality=quality).fit(X)
    batch1_mean = X[batch == 1].mean(axis=0).to_numpy()
    np.testing.assert_allclose(corrector.grand_mean_, batch1_mean, atol=0.1)


def test_non_parametric_mode(rng):
    X, batch = _make_weighted_dataset(rng)
    q = pd.Series(np.ones(len(X)), index=X.index)
    corrector = QualityWeightedComBat(batch=batch, quality=q, parametric=False).fit(X)
    assert corrector.n_iter_ == 0


def test_rejects_negative_weights(rng):
    X, batch = _make_weighted_dataset(rng)
    bad = pd.Series([-1.0] * len(X), index=X.index)
    with pytest.raises(ValueError, match="non-negative"):
        QualityWeightedComBat(batch=batch, quality=bad).fit(X)


def test_rejects_all_zero_weights(rng):
    X, batch = _make_weighted_dataset(rng)
    zero = pd.Series(np.zeros(len(X)), index=X.index)
    with pytest.raises(ValueError, match="at least one"):
        QualityWeightedComBat(batch=batch, quality=zero).fit(X)


def test_rejects_reference_batch_not_present(rng):
    X, batch = _make_weighted_dataset(rng)
    q = pd.Series(np.ones(len(X)), index=X.index)
    with pytest.raises(ValueError, match="reference_batch="):
        QualityWeightedComBat(batch=batch, quality=q, reference_batch=99).fit(X)


def test_reference_batch_is_left_unchanged(rng):
    X, batch = _make_weighted_dataset(rng)
    q = pd.Series(np.ones(len(X)), index=X.index)
    corrector = QualityWeightedComBat(batch=batch, quality=q, reference_batch=0).fit(X)
    out = corrector.transform(X)
    mask = (batch == 0).to_numpy()
    np.testing.assert_allclose(out.loc[mask].to_numpy(), X.loc[mask].to_numpy())


def test_unknown_batch_at_transform_uses_identity_correction(rng):
    X, batch = _make_weighted_dataset(rng)
    q = pd.Series(np.ones(len(X)), index=X.index)
    corrector = QualityWeightedComBat(batch=batch, quality=q).fit(X)
    X_new = X.iloc[:5].copy()
    corrector.batch = pd.Series(["new"] * 5, index=X_new.index)
    out = corrector.transform(X_new)
    assert np.isfinite(out.to_numpy()).all()


def test_all_zero_weights_in_a_batch_raises(rng):
    X, batch = _make_weighted_dataset(rng)
    w = np.ones(len(X))
    w[batch == 0] = 0.0
    q = pd.Series(w, index=X.index)
    with pytest.raises(ValueError, match="Effective batch size"):
        QualityWeightedComBat(batch=batch, quality=q).fit(X)


def test_rejects_non_finite_quality(rng):
    X, batch = _make_weighted_dataset(rng)
    q = pd.Series(np.ones(len(X)), index=X.index)
    q.iloc[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        QualityWeightedComBat(batch=batch, quality=q).fit(X)


def test_single_feature_skips_empirical_bayes(rng):
    X, batch = _make_weighted_dataset(rng, n_features=1, batch_shift=2.0)
    q = pd.Series(np.ones(len(X)), index=X.index)
    out = QualityWeightedComBat(batch=batch, quality=q).fit_transform(X)
    assert np.isfinite(out.to_numpy()).all()
    np.testing.assert_allclose(
        out.groupby(batch).mean().to_numpy().ravel(),
        np.full(2, X.to_numpy().mean()),
        atol=1e-8,
    )


def test_equal_weights_match_johnson_combat(rng):
    X, batch = _make_weighted_dataset(rng, batch_shift=5.0)
    q = pd.Series(np.ones(len(X)), index=X.index)
    out = QualityWeightedComBat(batch=batch, quality=q).fit_transform(X)
    assert out.groupby(batch).std().to_numpy().mean() == pytest.approx(1.0, abs=0.1)
    ref = ComBat(batch=batch, method="johnson").fit_transform(X)
    np.testing.assert_allclose(out.to_numpy(), ref.to_numpy(), atol=1e-4)


def test_reference_batch_moves_other_batches_onto_reference(rng):
    X, batch = _make_weighted_dataset(rng, batch_shift=3.0)
    q = pd.Series(np.ones(len(X)), index=X.index)
    out = QualityWeightedComBat(
        batch=batch, quality=q, reference_batch=0
    ).fit_transform(X)
    ref = ComBat(batch=batch, method="johnson", reference_batch=0).fit_transform(X)
    np.testing.assert_allclose(out.to_numpy(), ref.to_numpy(), atol=1e-4)
    means = out.groupby(batch).mean()
    np.testing.assert_allclose(means.loc[1], means.loc[0], atol=0.2)


def test_rejects_one_sample_batch(rng):
    X, batch = _make_weighted_dataset(rng)
    batch = batch.copy()
    batch.iloc[-1] = 2
    q = pd.Series(np.ones(len(X)), index=X.index)
    with pytest.raises(ValueError, match="requires >= 2 samples per batch"):
        QualityWeightedComBat(batch=batch, quality=q).fit(X)


def test_warns_on_effective_batch_size_at_most_one(rng):
    X, batch = _make_weighted_dataset(rng)
    w = np.ones(len(X))
    w[(batch == 0).to_numpy()] = 1e-3
    q = pd.Series(w, index=X.index)
    with pytest.warns(UserWarning, match="effective batch size"):
        QualityWeightedComBat(batch=batch, quality=q).fit(X)


def test_near_zero_scale_is_left_unscaled(rng):
    X, batch = _make_weighted_dataset(rng)
    X.loc[(batch == 1).to_numpy(), 0] = 0.0
    X_new = pd.DataFrame([np.full(X.shape[1], 0.01)], index=["new"], columns=X.columns)
    batch_all = pd.concat([batch, pd.Series([1], index=["new"])])
    q = pd.Series(np.ones(len(X)), index=X.index)
    with pytest.warns(UserWarning, match="left unscaled"):
        corrector = QualityWeightedComBat(
            batch=batch_all, quality=q, parametric=False
        ).fit(X)
    out = corrector.transform(X_new)
    assert out.iloc[0, 0] == pytest.approx(0.01 + X[0].mean())


def test_rejects_reference_batch_with_tiny_effective_size(rng):
    idx = [f"s{i}" for i in range(15)]
    batch = pd.Series(np.repeat(["a", "b", "c"], 5), index=idx)
    X = pd.DataFrame(rng.standard_normal((15, 20)), index=idx)
    q = pd.Series(np.where(batch == "c", 0.01, 1.0), index=idx)
    corrector = QualityWeightedComBat(batch=batch, quality=q, reference_batch="c")
    with pytest.raises(ValueError, match="reference_batch"):
        corrector.fit(X)
