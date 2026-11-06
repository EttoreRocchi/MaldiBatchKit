"""Tests for ComBat re-export and SpeciesAwareComBat wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import ComBat, SpeciesAwareComBat


def _make_unbalanced_species_dataset(
    rng, *, n_per_batch=60, n_features=30, species_shift=2.0
):
    """Two batches with species prevalence 80% / 20% and a fixed species shift."""
    blocks, batch, species = [], [], []
    for i, frac in enumerate((0.8, 0.2)):
        n1 = int(round(frac * n_per_batch))
        sp = np.array(["sp1"] * n1 + ["sp0"] * (n_per_batch - n1))
        block = rng.standard_normal((n_per_batch, n_features)) * (1.0 + 0.5 * i)
        block += 3.0 * i
        block[sp == "sp1"] += species_shift
        blocks.append(block)
        batch += [f"b{i}"] * n_per_batch
        species += sp.tolist()
    idx = [f"s{i:03d}" for i in range(2 * n_per_batch)]
    return (
        pd.DataFrame(np.vstack(blocks), index=idx),
        pd.Series(batch, index=idx),
        pd.Series(species, index=idx),
    )


def _within_batch_species_gap(X, batch, species):
    gaps = [
        X[(batch == b) & (species == "sp1")].to_numpy().mean()
        - X[(batch == b) & (species == "sp0")].to_numpy().mean()
        for b in batch.unique()
    ]
    return float(np.mean(gaps))


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
    assert corrector.species is species
    assert corrector.discrete_covariates is species
    corrector.fit(X)


def test_chen_with_covariates_preserves_species_gap(rng):
    """CovBat must not remove a protected covariate unbalanced across batches."""
    X, batch, species = _make_unbalanced_species_dataset(rng)
    gaps = {
        method: _within_batch_species_gap(
            ComBat(
                batch=batch, discrete_covariates=species, method=method
            ).fit_transform(X),
            batch,
            species,
        )
        for method in ("fortin", "chen")
    }
    assert gaps["fortin"] == pytest.approx(2.0, abs=0.2)
    assert gaps["chen"] >= 0.9 * gaps["fortin"]


def test_species_aware_combat_row_by_row_matches_block(rng):
    """Transform output does not depend on which rows are transformed together."""
    X, batch, species = _make_unbalanced_species_dataset(rng)
    train, test = X.index[::2], X.index[1::2]
    corrector = SpeciesAwareComBat(batch=batch, species=species).fit(X.loc[train])
    block = corrector.transform(X.loc[test])
    rows = pd.concat([corrector.transform(X.loc[[i]]) for i in test])
    np.testing.assert_allclose(rows.to_numpy(), block.to_numpy(), atol=1e-10)


def test_species_aware_combat_rejects_one_sample_batch(tiny_dataset):
    X, species = tiny_dataset["X"], tiny_dataset["species"]
    batch = tiny_dataset["batch"].copy()
    batch.iloc[-1] = "b_single"
    with pytest.raises(ValueError, match="requires >= 2 samples per batch"):
        SpeciesAwareComBat(batch=batch, species=species).fit(X)


def test_species_aware_combat_set_params_species_takes_effect(tiny_dataset):
    X, batch, species = (
        tiny_dataset["X"],
        tiny_dataset["batch"],
        tiny_dataset["species"],
    )
    placeholder = pd.Series("sp0", index=X.index)
    corrector = SpeciesAwareComBat(batch=batch, species=placeholder)
    corrector.set_params(species=species)
    expected = SpeciesAwareComBat(batch=batch, species=species).fit_transform(X)
    np.testing.assert_allclose(
        corrector.fit_transform(X).to_numpy(), expected.to_numpy()
    )


def test_species_aware_combat_init_sets_only_parameters(tiny_dataset):
    corrector = SpeciesAwareComBat(
        batch=tiny_dataset["batch"], species=tiny_dataset["species"]
    )
    assert set(vars(corrector)) == set(corrector.get_params())
