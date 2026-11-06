"""Tests for :class:`MaldiSetAdapter`.

We stub out ``MaldiSet`` with a tiny duck-typed object so we don't need
to load real spectra.
"""

from __future__ import annotations

import copy
import importlib.util
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import (
    ComBat,
    Harmony,
    Limma,
    MedianCentering,
    QualityWeightedComBat,
    SpeciesAwareComBat,
)
from maldibatchkit.integrations import MaldiSetAdapter

maldiamrkit_available = importlib.util.find_spec("maldiamrkit") is not None


class _FakeMaldiSet:
    """Duck-typed subset of ``maldiamrkit.MaldiSet`` sufficient for the adapter.

    ``filter`` rebuilds the set from the raw data, as ``MaldiSet.filter``
    rebuilds it from the spectra.
    """

    def __init__(self, X: pd.DataFrame, meta: pd.DataFrame):
        self._X_cache = X
        self._raw = X
        self.meta = meta
        self.spectra = []

    @property
    def X(self) -> pd.DataFrame:
        return self._X_cache

    def filter(self, *filters):
        keep = [
            sid for sid, row in self.meta.iterrows() if all(f(row) for f in filters)
        ]
        return _FakeMaldiSet(self._raw.loc[keep], self.meta.loc[keep])


@pytest.fixture
def fake_ds(tiny_dataset, monkeypatch):
    X = tiny_dataset["X"]
    meta = pd.DataFrame(
        {
            "Batch": tiny_dataset["batch"].values,
            "Species": tiny_dataset["species"].values,
            "SNR": np.linspace(0.5, 5.0, len(X)),
        },
        index=X.index,
    )
    ds = _FakeMaldiSet(X, meta)
    import maldibatchkit.integrations.maldiset as mod

    def _fake_require():
        return _FakeMaldiSet

    monkeypatch.setattr(mod, "_require_maldiset", _fake_require)
    return ds


def test_extract_returns_aligned_series(fake_ds):
    adapter = MaldiSetAdapter(
        batch_column="Batch", species_column="Species", quality_column="SNR"
    )
    extracted = adapter.extract(fake_ds)
    assert set(extracted) == {"X", "batch", "species", "quality"}
    assert len(extracted["batch"]) == len(fake_ds.X)


def test_extract_unknown_batch_column_raises(fake_ds):
    adapter = MaldiSetAdapter(batch_column="missing")
    with pytest.raises(KeyError):
        adapter.extract(fake_ds)


def test_correct_with_median_centering_returns_new_set(fake_ds):
    adapter = MaldiSetAdapter(batch_column="Batch")
    new_ds = adapter.correct(fake_ds, MedianCentering)
    assert new_ds is not fake_ds
    assert new_ds._X_cache is not fake_ds._X_cache
    assert not np.allclose(new_ds.X.to_numpy(), fake_ds.X.to_numpy())


def test_correct_routes_species_into_species_kwarg(fake_ds):
    adapter = MaldiSetAdapter(batch_column="Batch", species_column="Species")
    new_ds = adapter.correct(fake_ds, SpeciesAwareComBat)
    assert new_ds.X.shape == fake_ds.X.shape


def test_correct_routes_quality_into_quality_kwarg(fake_ds):
    adapter = MaldiSetAdapter(batch_column="Batch", quality_column="SNR")
    new_ds = adapter.correct(fake_ds, QualityWeightedComBat)
    assert new_ds.X.shape == fake_ds.X.shape


def test_correct_forwards_extra_kwargs(fake_ds):
    adapter = MaldiSetAdapter(batch_column="Batch")
    new_ds = adapter.correct(fake_ds, MedianCentering, transformer_kwargs={})
    assert isinstance(new_ds._X_cache, pd.DataFrame)


def test_correct_routes_species_into_combat_discrete_covariates(fake_ds):
    """ComBat uses ``discrete_covariates=`` for protected covariates."""
    adapter = MaldiSetAdapter(batch_column="Batch", species_column="Species")
    new_ds = adapter.correct(fake_ds, ComBat, transformer_kwargs={"method": "fortin"})
    assert new_ds.X.shape == fake_ds.X.shape


def test_correct_does_not_auto_route_species_into_limma_design(fake_ds):
    """Limma's ``design=`` expects a numeric matrix, so the adapter
    must NOT silently feed it a string species Series. Users who want
    to protect species in Limma should pass ``design=`` explicitly via
    ``transformer_kwargs``.
    """
    adapter = MaldiSetAdapter(batch_column="Batch", species_column="Species")
    with pytest.warns(UserWarning, match="species_column"):
        new_ds = adapter.correct(fake_ds, Limma)
    assert new_ds.X.shape == fake_ds.X.shape


def test_correct_limma_with_explicit_design(fake_ds):
    """Users can still route species to Limma by supplying a numeric
    design matrix through ``transformer_kwargs``."""
    adapter = MaldiSetAdapter(batch_column="Batch", species_column="Species")
    species_dummies = pd.get_dummies(fake_ds.meta["Species"], drop_first=True).astype(
        float
    )
    new_ds = adapter.correct(
        fake_ds, Limma, transformer_kwargs={"design": species_dummies}
    )
    assert new_ds.X.shape == fake_ds.X.shape


class _HarmonySpy(Harmony):
    seen_covariates = "unset"

    def __init__(self, batch, *, covariates=None, **kwargs):
        type(self).seen_covariates = covariates
        super().__init__(batch, covariates=covariates, **kwargs)


def test_correct_does_not_route_species_into_harmony_covariates(fake_ds):
    """harmonypy removes its ``covariates`` like the batch: species is not one."""
    adapter = MaldiSetAdapter(batch_column="Batch", species_column="Species")
    with pytest.warns(UserWarning, match="species_column"):
        new_ds = adapter.correct(
            fake_ds,
            _HarmonySpy,
            transformer_kwargs={"random_state": 0, "max_iter": 3, "nclust": 3},
        )
    assert _HarmonySpy.seen_covariates is None
    assert new_ds.X.shape == fake_ds.X.shape


@pytest.mark.parametrize("column", ["species_column", "quality_column"])
def test_extract_unknown_optional_column_raises(fake_ds, column):
    adapter = MaldiSetAdapter(batch_column="Batch", **{column: "missing"})
    with pytest.raises(KeyError, match="missing"):
        adapter.extract(fake_ds)


def test_correct_keeps_meta_index_name(fake_ds):
    fake_ds.meta = fake_ds.meta.rename_axis("ID")
    new_ds = MaldiSetAdapter(batch_column="Batch").correct(fake_ds, MedianCentering)
    assert new_ds.meta.index.name == "ID"


def test_filter_on_corrected_set_keeps_corrected_data(fake_ds):
    corrected = MaldiSetAdapter(batch_column="Batch").correct(fake_ds, MedianCentering)
    sub = corrected.filter(lambda row: row["Batch"] == "b0")
    np.testing.assert_allclose(
        sub.X.to_numpy(), corrected.X.loc[sub.X.index].to_numpy()
    )
    subsub = sub.filter(lambda row: row["Species"] == "sp0")
    np.testing.assert_allclose(
        subsub.X.to_numpy(), corrected.X.loc[subsub.X.index].to_numpy()
    )


def test_no_species_warning_when_design_is_passed(fake_ds):
    adapter = MaldiSetAdapter(batch_column="Batch", species_column="Species")
    design = pd.get_dummies(fake_ds.meta["Species"], drop_first=True).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapter.correct(fake_ds, Limma, transformer_kwargs={"design": design})


def test_filter_on_copy_uses_the_copy(fake_ds):
    corrected = MaldiSetAdapter(batch_column="Batch").correct(fake_ds, MedianCentering)
    clone = copy.copy(corrected)
    clone._X_cache = corrected._X_cache * 0.0
    sub = clone.filter(lambda row: row["Batch"] == "b0")
    assert (sub.X.to_numpy() == 0.0).all()
    restored = pickle.loads(pickle.dumps(corrected)).filter(
        lambda r: r["Batch"] == "b0"
    )
    np.testing.assert_allclose(
        restored.X.to_numpy(), corrected.X.loc[restored.X.index].to_numpy()
    )
