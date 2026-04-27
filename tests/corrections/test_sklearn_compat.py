"""Sklearn compatibility checks for MaldiBatchKit corrections.

We deliberately stop short of the full ``sklearn.utils.estimator_checks``
round (which wouldn't fit the ``batch=`` stored-at-init API anyway) and
instead verify the parts that *should* work cleanly:

* ``BaseEstimator`` / ``TransformerMixin`` provenance.
* ``get_params`` / ``set_params`` produce a consistent dict that can be
  round-tripped through the constructor.
* ``fit`` returns ``self`` and sets ``feature_names_in_`` /
  ``n_features_in_``.
* ``get_feature_names_out`` works after fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from maldibatchkit import (
    Limma,
    MedianCentering,
    QualityWeightedComBat,
    ReferenceScaling,
    ZScorePerBatch,
)


@pytest.fixture
def quality_kwargs(tiny_dataset):
    return {
        "quality": pd.Series(
            np.ones(len(tiny_dataset["X"])), index=tiny_dataset["X"].index
        )
    }


@pytest.mark.parametrize(
    "cls, extra",
    [
        (MedianCentering, {}),
        (ZScorePerBatch, {}),
        (ReferenceScaling, {}),
        (Limma, {}),
    ],
)
def test_inheritance_and_fit_returns_self(cls, extra, tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = cls(batch=batch, **extra)
    assert isinstance(corrector, BaseEstimator)
    assert isinstance(corrector, TransformerMixin)
    result = corrector.fit(X)
    assert result is corrector
    assert corrector.n_features_in_ == X.shape[1]
    assert len(corrector.feature_names_in_) == X.shape[1]
    assert corrector.get_feature_names_out().tolist() == list(
        corrector.feature_names_in_
    )


def test_quality_weighted_inheritance_and_fit(tiny_dataset, quality_kwargs):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrector = QualityWeightedComBat(batch=batch, **quality_kwargs)
    assert isinstance(corrector, BaseEstimator)
    assert isinstance(corrector, TransformerMixin)
    assert corrector.fit(X) is corrector


def test_get_params_roundtrip(tiny_dataset):
    batch = tiny_dataset["batch"]
    corrector = ReferenceScaling(batch=batch, reference_batch="b0", eps=1e-6)
    params = corrector.get_params()
    assert params["reference_batch"] == "b0"
    assert params["eps"] == 1e-6
    fresh = ReferenceScaling(**params)
    fresh.set_params(eps=1e-3)
    assert fresh.get_params()["eps"] == 1e-3


def test_pipeline_usage(tiny_dataset):
    """Corrector plugs into sklearn Pipeline without runtime errors."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    pipe = Pipeline(
        [
            ("correct", MedianCentering(batch=batch)),
            ("scale", StandardScaler()),
        ]
    )
    out = pipe.fit_transform(X)
    assert out.shape == X.shape
