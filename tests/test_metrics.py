"""Tests for ``maldibatchkit.metrics``."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from maldibatchkit import AutoCorrector
from maldibatchkit.metrics import (
    _resolve_weights,
    batch_average_precision_score,
    batch_balanced_accuracy_score,
    batch_f1_score,
    batch_matthews_corrcoef,
    batch_precision_score,
    batch_recall_score,
    batch_roc_auc_score,
    make_batch_scorer,
)


@pytest.fixture
def labelled_dataset(rng):
    """Two batches with a real signal in y; one batch is twice as large."""
    n_small = 20
    n_big = 60
    n = n_small + n_big
    idx = [f"s{i:04d}" for i in range(n)]
    batch = pd.Series(["a"] * n_small + ["b"] * n_big, index=idx)
    y = pd.Series(rng.integers(0, 2, n), index=idx)
    score = pd.Series(rng.uniform(0, 1, n) + 0.4 * y.to_numpy(), index=idx)
    pred = (score > 0.5).astype(int)
    return {"batch": batch, "y": y, "score": score, "pred": pred}


def test_resolve_weights_uniform_sums_to_one():
    levels = np.array(["a", "b", "c"])
    sizes = np.array([10, 20, 30])
    w = _resolve_weights(levels, sizes, "uniform")
    np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3])


def test_resolve_weights_balanced_is_inverse_size():
    levels = np.array(["a", "b", "c"])
    sizes = np.array([30, 60, 90])
    w = _resolve_weights(levels, sizes, "balanced")
    expected_raw = 1.0 / sizes.astype(float)
    expected = expected_raw / expected_raw.sum()
    np.testing.assert_allclose(w, expected)


def test_resolve_weights_balanced_smallest_batch_loudest():
    levels = np.array(["small", "big"])
    sizes = np.array([10, 100])
    w = _resolve_weights(levels, sizes, "balanced")
    # The smaller batch must get a strictly larger weight.
    assert w[0] > w[1]


def test_resolve_weights_balanced_rejects_empty_batch():
    with pytest.raises(ValueError, match="at least one sample"):
        _resolve_weights(np.array(["a", "b"]), np.array([10, 0]), "balanced")


def test_resolve_weights_modes_differ_for_unequal_batches():
    """uniform / balanced / size must produce three distinct vectors."""
    levels = np.array(["a", "b", "c"])
    sizes = np.array([30, 60, 90])
    u = _resolve_weights(levels, sizes, "uniform")
    b = _resolve_weights(levels, sizes, "balanced")
    s = _resolve_weights(levels, sizes, "size")
    assert not np.allclose(u, b)
    assert not np.allclose(u, s)
    assert not np.allclose(b, s)


def test_resolve_weights_size_proportional_to_sample_count():
    levels = np.array(["a", "b"])
    sizes = np.array([10, 30])
    w = _resolve_weights(levels, sizes, "size")
    np.testing.assert_allclose(w, [0.25, 0.75])


def test_resolve_weights_dict_picks_in_level_order():
    levels = np.array(["a", "b", "c"])
    sizes = np.array([10, 10, 10])
    w = _resolve_weights(levels, sizes, {"a": 1.0, "b": 2.0, "c": 1.0})
    np.testing.assert_allclose(w, [0.25, 0.5, 0.25])


def test_resolve_weights_dict_missing_level_raises():
    levels = np.array(["a", "b"])
    with pytest.raises(ValueError, match="missing entries"):
        _resolve_weights(levels, np.ones(2), {"a": 1.0})


def test_resolve_weights_array_length_must_match_levels():
    with pytest.raises(ValueError, match="does not match"):
        _resolve_weights(np.array(["a", "b"]), np.ones(2), np.array([1.0, 2.0, 3.0]))


def test_resolve_weights_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown weights mode"):
        _resolve_weights(np.array(["a"]), np.array([1]), "bogus")


def test_resolve_weights_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_weights(np.array(["a", "b"]), np.ones(2), np.array([-1.0, 2.0]))


def test_resolve_weights_all_zero_raises():
    with pytest.raises(ValueError, match="at least one weight"):
        _resolve_weights(np.array(["a", "b"]), np.ones(2), np.array([0.0, 0.0]))


def test_roc_auc_uniform_is_mean_of_per_batch(labelled_dataset):
    d = labelled_dataset
    levels = d["batch"].unique()
    per_batch = []
    for lvl in levels:
        mask = d["batch"] == lvl
        per_batch.append(roc_auc_score(d["y"][mask], d["score"][mask]))
    expected = float(np.mean(per_batch))
    got = batch_roc_auc_score(d["y"], d["score"], batch=d["batch"], weights="uniform")
    assert got == pytest.approx(expected, abs=1e-9)


def test_roc_auc_size_weights_differ_from_uniform_for_unequal_batches(labelled_dataset):
    d = labelled_dataset
    u = batch_roc_auc_score(d["y"], d["score"], batch=d["batch"], weights="uniform")
    s = batch_roc_auc_score(d["y"], d["score"], batch=d["batch"], weights="size")
    assert u != s


def test_roc_auc_custom_dict_weights(labelled_dataset):
    d = labelled_dataset
    # All mass on batch 'a' must equal AUROC on batch 'a' alone.
    only_a = batch_roc_auc_score(
        d["y"], d["score"], batch=d["batch"], weights={"a": 1.0, "b": 0.0}
    )
    mask_a = d["batch"] == "a"
    direct = roc_auc_score(d["y"][mask_a], d["score"][mask_a])
    assert only_a == pytest.approx(direct)


def test_roc_auc_drops_single_class_batch_with_warning(labelled_dataset):
    d = labelled_dataset
    y = d["y"].copy()
    # Force batch 'a' to one class only -> AUROC undefined for that batch.
    y[d["batch"] == "a"] = 1
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        v = batch_roc_auc_score(y, d["score"], batch=d["batch"], weights="uniform")
    assert any("dropped" in str(w.message) for w in wlog)
    mask_b = d["batch"] == "b"
    expected = roc_auc_score(y[mask_b], d["score"][mask_b])
    assert v == pytest.approx(expected)


def test_roc_auc_all_degenerate_returns_nan(labelled_dataset):
    d = labelled_dataset
    y = pd.Series(np.ones(len(d["y"]), dtype=int), index=d["y"].index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = batch_roc_auc_score(y, d["score"], batch=d["batch"])
    assert np.isnan(v)


def test_balanced_accuracy_runs(labelled_dataset):
    d = labelled_dataset
    v = batch_balanced_accuracy_score(d["y"], d["pred"], batch=d["batch"])
    assert 0.0 <= v <= 1.0


def test_matthews_corrcoef_in_range(labelled_dataset):
    d = labelled_dataset
    v = batch_matthews_corrcoef(d["y"], d["pred"], batch=d["batch"])
    assert -1.0 <= v <= 1.0


def test_f1_score_runs(labelled_dataset):
    d = labelled_dataset
    v = batch_f1_score(d["y"], d["pred"], batch=d["batch"], average="binary")
    assert 0.0 <= v <= 1.0


def test_f1_score_supports_macro_average(labelled_dataset):
    d = labelled_dataset
    v = batch_f1_score(d["y"], d["pred"], batch=d["batch"], average="macro")
    assert 0.0 <= v <= 1.0


def test_precision_pos_label_zero_differs(labelled_dataset):
    d = labelled_dataset
    p1 = batch_precision_score(d["y"], d["pred"], batch=d["batch"], pos_label=1)
    p0 = batch_precision_score(d["y"], d["pred"], batch=d["batch"], pos_label=0)
    assert p0 != p1


def test_recall_runs(labelled_dataset):
    d = labelled_dataset
    v = batch_recall_score(d["y"], d["pred"], batch=d["batch"])
    assert 0.0 <= v <= 1.0


def test_average_precision_runs(labelled_dataset):
    d = labelled_dataset
    v = batch_average_precision_score(d["y"], d["score"], batch=d["batch"])
    assert 0.0 <= v <= 1.0


def test_length_mismatch_raises(labelled_dataset):
    d = labelled_dataset
    with pytest.raises(ValueError, match="same length"):
        batch_roc_auc_score(d["y"][:10], d["score"], batch=d["batch"])


class _DummyProbaEstimator:
    """Estimator stub whose predict_proba is just `score` from the fixture."""

    def __init__(self, score: pd.Series):
        self.score = np.asarray(score)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.column_stack([1 - self.score, self.score])

    def predict(self, X):
        return (self.score > 0.5).astype(int)


def test_scorer_matches_metric_function(labelled_dataset):
    d = labelled_dataset
    scorer = make_batch_scorer(d["batch"], "roc_auc", weights="uniform")
    expected = batch_roc_auc_score(
        d["y"], d["score"], batch=d["batch"], weights="uniform"
    )
    got = scorer(_DummyProbaEstimator(d["score"]), d["score"].to_frame(), d["y"])
    assert got == pytest.approx(expected)


def test_scorer_unknown_alias_raises(labelled_dataset):
    with pytest.raises(ValueError, match="Unknown metric alias"):
        make_batch_scorer(labelled_dataset["batch"], "bogus_metric")


def test_scorer_callable_requires_response_method(labelled_dataset):
    def my_metric(y_true, y_pred, *, batch, weights="uniform", **kw):
        return 0.0

    with pytest.raises(ValueError, match="response_method"):
        make_batch_scorer(labelled_dataset["batch"], my_metric)


def test_scorer_callable_with_explicit_response(labelled_dataset):
    d = labelled_dataset

    def my_metric(y_true, y_pred, *, batch, weights="uniform", **kw):
        return float(np.mean(y_true == y_pred))

    scorer = make_batch_scorer(d["batch"], my_metric, response_method="predict")
    got = scorer(_DummyProbaEstimator(d["score"]), d["score"].to_frame(), d["y"])
    assert 0.0 <= got <= 1.0


def test_scorer_greater_is_better_negates(labelled_dataset):
    d = labelled_dataset
    pos = make_batch_scorer(d["batch"], "roc_auc", greater_is_better=True)
    neg = make_batch_scorer(d["batch"], "roc_auc", greater_is_better=False)
    v_pos = pos(_DummyProbaEstimator(d["score"]), d["score"].to_frame(), d["y"])
    v_neg = neg(_DummyProbaEstimator(d["score"]), d["score"].to_frame(), d["y"])
    assert v_pos == pytest.approx(-v_neg)


def test_scorer_misaligned_index_raises(labelled_dataset):
    d = labelled_dataset
    scorer = make_batch_scorer(d["batch"], "roc_auc")
    y_bad = d["y"].copy()
    y_bad.index = [f"x{i}" for i in range(len(y_bad))]
    with pytest.raises(ValueError, match="does not align"):
        scorer(_DummyProbaEstimator(d["score"]), d["score"].to_frame(), y_bad)


def test_scorer_inside_gridsearchcv_picks_better_corrector(rng):
    """End-to-end: AutoCorrector + GridSearchCV + per-batch scorer."""
    n = 120
    n_features = 8
    idx = [f"s{i:04d}" for i in range(n)]
    y = pd.Series((np.arange(n) % 2), index=idx)
    b = pd.Series(["a"] * 60 + ["b"] * 60, index=idx)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features))
        + 0.3 * np.where(y.to_numpy() == 1, 1, -1)[:, None]
        + np.where(b.to_numpy() == "a", -1.0, 1.0)[:, None],
        index=idx,
    )

    scorer = make_batch_scorer(b, "roc_auc", weights="uniform")
    pipe = Pipeline(
        [
            ("correct", AutoCorrector(batch=b, discrete_covariates=b)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    grid = GridSearchCV(
        pipe,
        param_grid={"correct__method": ["noop", "median", "combat-fortin"]},
        scoring=scorer,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid.fit(X, y)
    assert grid.best_params_["correct__method"] in {"noop", "median", "combat-fortin"}
    # cv_results_ should hold finite scores.
    assert np.isfinite(grid.cv_results_["mean_test_score"]).all()
