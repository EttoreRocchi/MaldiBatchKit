"""Tests for ``AutoCorrector``."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from maldibatchkit import (
    AutoCorrector,
    ComBat,
    Harmony,
    Limma,
    NoOpCorrector,
    QualityWeightedComBat,
    SpeciesAwareComBat,
)
from maldibatchkit.corrections.auto import (
    METHOD_REGISTRY,
    _resolve_method_key,
)


def test_registry_resolution(tiny_dataset):
    for alias in (
        "combat",
        "combat-fortin",
        "combat-johnson",
        "combat-chen",
        "limma",
        "harmony",
        "qw-combat",
        "species-combat",
        "batch-warping",
        "median",
        "zscore",
        "reference",
        "noop",
    ):
        assert alias in METHOD_REGISTRY
        key, builder = _resolve_method_key(alias)
        assert key == alias
        assert callable(builder) or hasattr(builder, "__init__")


def test_class_method_passes_through(tiny_dataset):
    ac = AutoCorrector(batch=tiny_dataset["batch"], method=NoOpCorrector)
    out = ac.fit_transform(tiny_dataset["X"])
    pd.testing.assert_frame_equal(out, tiny_dataset["X"])


def test_unknown_string_alias_raises(tiny_dataset):
    with pytest.raises(ValueError, match="Unknown method alias"):
        AutoCorrector(batch=tiny_dataset["batch"], method="bogus").fit(
            tiny_dataset["X"]
        )


def test_invalid_method_type_raises(tiny_dataset):
    with pytest.raises(TypeError, match="BaseBatchCorrector subclass"):
        AutoCorrector(batch=tiny_dataset["batch"], method=42).fit(tiny_dataset["X"])


def test_quality_kwarg_is_dropped_for_combat_fortin(tiny_dataset):
    q = pd.Series(
        np.linspace(0.5, 1.0, len(tiny_dataset["batch"])),
        index=tiny_dataset["batch"].index,
    )
    ac = AutoCorrector(batch=tiny_dataset["batch"], method="combat-fortin", quality=q)
    ac.fit(tiny_dataset["X"])
    assert ac.inner_.__class__.__name__ == "ComBat"


def test_quality_routed_to_qw_combat(tiny_dataset):
    q = pd.Series(
        np.linspace(0.5, 1.0, len(tiny_dataset["batch"])),
        index=tiny_dataset["batch"].index,
    )
    ac = AutoCorrector(batch=tiny_dataset["batch"], method="qw-combat", quality=q)
    ac.fit(tiny_dataset["X"])
    assert isinstance(ac.inner_, QualityWeightedComBat)
    assert hasattr(ac.inner_, "gamma_star_")
    np.testing.assert_array_equal(ac.gamma_star_, ac.inner_.gamma_star_)


def test_discrete_covariates_routed_to_combat(tiny_dataset):
    ac = AutoCorrector(
        batch=tiny_dataset["batch"],
        method="combat-fortin",
        discrete_covariates=tiny_dataset["species"],
    )
    ac.fit(tiny_dataset["X"])
    assert isinstance(ac.inner_, ComBat)


@pytest.mark.parametrize("method", ["harmony", Harmony], ids=["alias", "class"])
def test_discrete_covariates_not_forwarded_to_harmony(tiny_dataset, method):
    with pytest.warns(UserWarning, match="does not accept discrete"):
        ac = AutoCorrector(
            batch=tiny_dataset["batch"],
            method=method,
            discrete_covariates=tiny_dataset["species"],
            method_kwargs={"verbose": False, "random_state": 0, "max_iter": 5},
        ).fit(tiny_dataset["X"])
    assert ac.inner_.covariates is None


def _three_species_dataset(rng):
    """Species with a non-linear effect (0, 3, 0), unbalanced across batches."""
    sp = np.r_[np.repeat([0, 1, 2], [40, 10, 10]), np.repeat([0, 1, 2], [10, 40, 10])]
    idx = [f"t{i}" for i in range(len(sp))]
    species = pd.Series(sp, index=idx)
    batch = pd.Series(["a"] * 60 + ["b"] * 60, index=idx)
    shift = np.array([0.0, 3.0, 0.0])[sp] + np.where(batch.to_numpy() == "b", 1.0, 0.0)
    X = pd.DataFrame(rng.standard_normal((len(sp), 5)) + shift[:, None], index=idx)
    return X, batch, species


@pytest.mark.parametrize("labels", ["int", "str"])
@pytest.mark.parametrize("method", ["limma", Limma], ids=["alias", "class"])
def test_discrete_covariates_one_hot_encoded_for_limma(rng, method, labels):
    X, batch, species = _three_species_dataset(rng)
    if labels == "str":
        species = species.map({0: "x", 1: "y", 2: "z"})
    design = pd.get_dummies(species.astype(str), drop_first=True).astype(float)
    expected = Limma(batch=batch, design=design).fit_transform(X)
    out = AutoCorrector(
        batch=batch, method=method, discrete_covariates=species
    ).fit_transform(X)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-8)


def test_discrete_covariates_warns_when_unsupported(tiny_dataset):
    with pytest.warns(UserWarning, match="does not accept discrete"):
        AutoCorrector(
            batch=tiny_dataset["batch"],
            method="median",
            discrete_covariates=tiny_dataset["species"],
        ).fit(tiny_dataset["X"])


def test_method_kwargs_overrides_named_arg(tiny_dataset):
    ac = AutoCorrector(
        batch=tiny_dataset["batch"],
        method="combat-fortin",
        reference_batch="b0",
        method_kwargs={"reference_batch": "b1"},
    )
    ac.fit(tiny_dataset["X"])
    assert ac.inner_.reference_batch == "b1"


def test_noop_is_exact_identity(tiny_dataset):
    ac = AutoCorrector(batch=tiny_dataset["batch"], method="noop")
    out = ac.fit_transform(tiny_dataset["X"])
    pd.testing.assert_frame_equal(out, tiny_dataset["X"])


def test_set_params_rebuilds_inner_on_next_fit(tiny_dataset):
    ac = AutoCorrector(batch=tiny_dataset["batch"], method="median")
    ac.fit(tiny_dataset["X"])
    assert ac.inner_.__class__.__name__ == "MedianCentering"
    ac.set_params(method="zscore")
    ac.fit(tiny_dataset["X"])
    assert ac.inner_.__class__.__name__ == "ZScorePerBatch"


def test_gridsearchcv_pipeline_roundtrip(rng):
    n = 80
    n_features = 6
    idx = [f"s{i:04d}" for i in range(n)]
    y = pd.Series((np.arange(n) % 2), index=idx)
    b = pd.Series(["b0"] * (n // 2) + ["b1"] * (n // 2), index=idx)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features))
        + 0.3 * np.where(y.to_numpy() == 1, 1, -1)[:, None]
        + np.where(b.to_numpy() == "b0", -1.0, 1.0)[:, None],
        index=idx,
    )
    pipe = Pipeline(
        [
            ("correct", AutoCorrector(batch=b, method="combat-fortin")),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    grid = GridSearchCV(
        pipe,
        param_grid={"correct__method": ["noop", "median", "combat-fortin"]},
        scoring="roc_auc",
        cv=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid.fit(X, y)
    assert grid.best_params_["correct__method"] in {"noop", "median", "combat-fortin"}
    assert 0.0 <= grid.best_score_ <= 1.0


def test_getattr_does_not_swallow_other_attrs(tiny_dataset):
    ac = AutoCorrector(batch=tiny_dataset["batch"], method="combat-fortin")
    with pytest.raises(AttributeError):
        ac.does_not_exist_  # noqa: B018
    with pytest.raises(AttributeError):
        ac.no_trailing_underscore  # noqa: B018


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "combat-fortin", "discrete_covariates": "species"},
        {"method": "combat-chen", "discrete_covariates": "species"},
        {"method": "species-combat", "species": "species"},
    ],
    ids=["combat-fortin", "combat-chen", "species-combat"],
)
def test_auto_combat_rejects_one_sample_batch(tiny_dataset, kwargs):
    X = tiny_dataset["X"]
    batch = tiny_dataset["batch"].copy()
    batch.iloc[-1] = "b_single"
    kwargs = {k: tiny_dataset[v] if k != "method" else v for k, v in kwargs.items()}
    with pytest.raises(ValueError, match="requires >= 2 samples per batch"):
        AutoCorrector(batch=batch, **kwargs).fit(X)


def test_ndarray_input_gets_default_feature_names(tiny_dataset):
    X, batch = tiny_dataset["X"].to_numpy(), tiny_dataset["batch"].to_numpy()
    ac = AutoCorrector(batch=batch, method="median").fit(X)
    assert list(ac.get_feature_names_out()) == [f"x{i}" for i in range(X.shape[1])]
    assert ac.transform(X).shape == X.shape


@pytest.mark.parametrize(
    "cls, kwargs",
    [(ComBat, {}), (SpeciesAwareComBat, {"species": "species"})],
    ids=["ComBat", "SpeciesAwareComBat"],
)
def test_combat_classes_accepted_as_method(tiny_dataset, cls, kwargs):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    kwargs = {k: tiny_dataset[v] for k, v in kwargs.items()}
    expected = cls(batch=batch, **kwargs).fit_transform(X)
    out = AutoCorrector(batch=batch, method=cls, **kwargs).fit_transform(X)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy())


def test_limma_rejects_missing_discrete_covariates(tiny_dataset):
    species = tiny_dataset["species"].where(
        tiny_dataset["species"].index != tiny_dataset["species"].index[0]
    )
    ac = AutoCorrector(
        batch=tiny_dataset["batch"], method="limma", discrete_covariates=species
    )
    with pytest.raises(ValueError, match="missing"):
        ac.fit(tiny_dataset["X"])
