"""Tests for ``AutoCorrector``."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from maldibatchkit import AutoCorrector, ComBat, NoOpCorrector, QualityWeightedComBat
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
    # ComBat-Fortin does not accept `quality`; passing it should not raise.
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
    # Mirror via __getattr__:
    np.testing.assert_array_equal(ac.gamma_star_, ac.inner_.gamma_star_)


def test_discrete_covariates_routed_to_combat(tiny_dataset):
    ac = AutoCorrector(
        batch=tiny_dataset["batch"],
        method="combat-fortin",
        discrete_covariates=tiny_dataset["species"],
    )
    ac.fit(tiny_dataset["X"])
    assert isinstance(ac.inner_, ComBat)


def test_discrete_covariates_routed_to_harmony_as_covariates(tiny_dataset):
    ac = AutoCorrector(
        batch=tiny_dataset["batch"],
        method="harmony",
        discrete_covariates=tiny_dataset["species"],
        method_kwargs={"verbose": False, "random_state": 0, "max_iter": 5},
    )
    ac.fit(tiny_dataset["X"])
    assert ac.inner_.covariates is tiny_dataset["species"]


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
