"""Tests for ComBat re-export and SpeciesAwareComBat wrapper."""

from __future__ import annotations

import pandas as pd

from maldibatchkit import ComBat, SpeciesAwareComBat


def test_combat_reexport_is_combatlearn_class():
    import combatlearn

    assert ComBat is combatlearn.ComBat


def test_combat_fit_transform_reduces_batch_effect(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    from maldibatchkit.diagnostics import silhouette_batch

    before = silhouette_batch(X, batch)
    corrector = ComBat(batch=batch, method="johnson")
    after = silhouette_batch(corrector.fit_transform(X), batch)
    assert after < before


def test_species_aware_combat_delegates_to_fortin(tiny_dataset):
    X, batch, species = (
        tiny_dataset["X"],
        tiny_dataset["batch"],
        tiny_dataset["species"],
    )
    corrector = SpeciesAwareComBat(batch=batch, species=species)
    assert corrector.method == "fortin"
    assert corrector.discrete_covariates is species
    out = corrector.fit_transform(X)
    assert out.shape == X.shape
    assert isinstance(out, pd.DataFrame)


def test_species_aware_combat_species_is_stored(tiny_dataset):
    X, batch, species = (
        tiny_dataset["X"],
        tiny_dataset["batch"],
        tiny_dataset["species"],
    )
    corrector = SpeciesAwareComBat(batch=batch, species=species)
    # Ensure both canonical combatlearn slot AND new "species" attr are set
    assert corrector.species is species
    assert corrector.discrete_covariates is species
    corrector.fit(X)
