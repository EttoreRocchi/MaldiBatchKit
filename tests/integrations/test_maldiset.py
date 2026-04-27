"""Tests for :class:`MaldiSetAdapter`.

We stub out ``MaldiSet`` with a tiny duck-typed object so we don't need
to load real spectra.
"""

from __future__ import annotations

import importlib.util

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
    """Duck-typed subset of ``maldiamrkit.MaldiSet`` sufficient for the adapter."""

    def __init__(self, X: pd.DataFrame, meta: pd.DataFrame):
        self._X_cache = X
        self.meta = meta
        self.spectra = []

    @property
    def X(self) -> pd.DataFrame:
        return self._X_cache


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
    # Patch the real maldiamrkit.MaldiSet import inside the adapter to the fake
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
    # Original dataset unchanged
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
    # Adapter leaves design=None; Limma runs without covariate protection.
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


def test_correct_routes_species_into_harmony_covariates(fake_ds):
    """Harmony's extra-covariate slot is ``covariates=``."""
    adapter = MaldiSetAdapter(batch_column="Batch", species_column="Species")
    new_ds = adapter.correct(
        fake_ds,
        Harmony,
        transformer_kwargs={"random_state": 0, "max_iter": 3, "nclust": 3},
    )
    assert new_ds.X.shape == fake_ds.X.shape
