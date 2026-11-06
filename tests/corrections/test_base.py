"""Tests for the BaseBatchCorrector scaffolding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from maldibatchkit import AutoCorrector, Harmony, Limma, QualityWeightedComBat
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


def _range_indexed_dataset(rng, n=40):
    """Full cohort on a default RangeIndex, two batches of n/2 rows."""
    X = pd.DataFrame(rng.standard_normal((n, 3)))
    batch = pd.Series(np.repeat(["a", "b"], n // 2))
    train = np.sort(rng.choice(n, n // 2, replace=False))
    return X, batch, train


def test_pipeline_numpy_step_with_pandas_batch_raises(rng):
    X, batch, train = _range_indexed_dataset(rng)
    pipe = make_pipeline(FunctionTransformer(np.asarray), MedianCentering(batch=batch))
    with pytest.raises(ValueError, match="set_output"):
        pipe.fit(X.iloc[train])


def test_transform_ndarray_with_pandas_batch_raises(rng):
    X, batch, train = _range_indexed_dataset(rng)
    corrector = MedianCentering(batch=batch).fit(X)
    with pytest.raises(ValueError, match="set_output"):
        corrector.transform(X.iloc[train].to_numpy())


@pytest.mark.parametrize(
    "make",
    [
        lambda b, aux: Limma(batch=b, design=aux),
        lambda b, aux: QualityWeightedComBat(batch=b, quality=aux + 1.0),
        lambda b, aux: Harmony(batch=b, covariates=aux.astype(str), nclust=2),
        lambda b, aux: AutoCorrector(
            batch=b, method="combat-fortin", discrete_covariates=aux
        ),
    ],
    ids=["limma-design", "qw-quality", "harmony-covariates", "auto-covariates"],
)
def test_ndarray_X_with_pandas_auxiliary_input_raises(rng, make):
    X, batch, train = _range_indexed_dataset(rng)
    aux = pd.Series(rng.integers(0, 2, len(X)).astype(float))
    corrector = make(batch.iloc[train].to_numpy(), aux)
    with pytest.raises(ValueError, match="set_output"):
        corrector.fit(X.iloc[train].to_numpy())


def test_ndarray_X_with_range_indexed_pandas_batch_still_works(rng):
    X, batch, _ = _range_indexed_dataset(rng)
    out = MedianCentering(batch=batch).fit_transform(X.to_numpy())
    expected = MedianCentering(batch=batch.to_numpy()).fit_transform(X.to_numpy())
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("change", ["permuted", "renamed"])
def test_transform_rejects_columns_not_seen_at_fit(tiny_dataset, change):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    X = X.set_axis([f"mz{i}" for i in range(X.shape[1])], axis=1)
    corrector = MedianCentering(batch=batch).fit(X)
    if change == "permuted":
        X_new = X[X.columns[::-1]]
    else:
        X_new = X.set_axis([f"other{i}" for i in range(X.shape[1])], axis=1)
    with pytest.raises(ValueError, match="feature names"):
        corrector.transform(X_new)


def test_transform_rejects_wrong_number_of_features(tiny_dataset):
    X, batch = tiny_dataset["X"].to_numpy(), tiny_dataset["batch"].to_numpy()
    corrector = MedianCentering(batch=batch).fit(X)
    with pytest.raises(ValueError, match=f"expecting {X.shape[1]} features"):
        corrector.transform(X[:, :-1])


@pytest.mark.parametrize(
    "make",
    [
        lambda b: MedianCentering(batch=b),
        lambda b: AutoCorrector(batch=b, method="median"),
    ],
    ids=["median", "auto"],
)
def test_fit_ndarray_then_transform_dataframe_still_works(tiny_dataset, make):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"].to_numpy()
    corrector = make(batch).fit(X.to_numpy())
    X_named = X.set_axis([f"mz{i}" for i in range(X.shape[1])], axis=1)
    out = corrector.transform(X_named)
    np.testing.assert_allclose(out.to_numpy(), corrector.transform(X.to_numpy()))


def test_transform_rejects_nan_batch(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = MedianCentering(batch=batch).fit(X)
    corrector.batch = batch.where(batch.index != batch.index[0])
    with pytest.raises(ValueError, match="NaN"):
        corrector.transform(X)


def test_fit_rejects_duplicated_sample_ids(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    X = X.set_axis([X.index[0], *X.index[:-1]], axis=0)
    with pytest.raises(ValueError, match="duplicated sample ID"):
        MedianCentering(batch=batch).fit(X)


def test_fit_rejects_batch_with_duplicated_ids(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    batch = pd.concat([batch, batch.iloc[:1]])
    with pytest.raises(ValueError, match="duplicated sample ID"):
        MedianCentering(batch=batch).fit(X)


@pytest.mark.parametrize("stage", ["fit", "transform"])
def test_rejects_mixed_label_types(tiny_dataset, stage):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    mixed = batch.astype(object).where(batch == "b1", 0)
    corrector = MedianCentering(batch=mixed if stage == "fit" else batch)
    if stage == "transform":
        corrector.fit(X)
        corrector.batch = mixed
    with pytest.raises(ValueError, match="mixes"):
        getattr(corrector, stage)(X)


def test_subclass_with_own_fit_can_transform(tiny_dataset):
    class _OwnFit(BaseBatchCorrector):
        def fit(self, X, y=None):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = X.shape[1]
            return self

        def _transform_impl(self, X_df, batch):
            return X_df.copy()

    X = tiny_dataset["X"]
    out = _OwnFit(batch=tiny_dataset["batch"]).fit(X).transform(X)
    pd.testing.assert_frame_equal(out, X)
