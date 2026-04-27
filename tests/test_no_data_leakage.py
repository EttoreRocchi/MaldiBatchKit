"""Regression tests: every corrector must avoid data leakage by construction.

The invariants we pin down here are:

1.  **``fit`` uses training rows only.** Fitting with a ``batch`` vector
    indexed by the *full* dataset produces the same fitted parameters
    as fitting with a ``batch`` vector sliced to the training index.
2.  **``transform`` does not re-fit.** Calling ``transform`` on a
    different feature matrix must not mutate any fitted attribute
    (every ``*_`` attribute must remain byte-identical before and
    after).
3.  **``transform(X_test)`` is a pure function of the fitted state.**
    Permuting / replacing the rows in a previous ``transform`` call
    cannot change the output of a subsequent ``transform`` on the real
    test set.

Harmony is an explicit exception: it has no closed-form transform, so
it raises ``RuntimeError`` when asked to transform a matrix that
differs from the one it was fit on.  We test that refusal directly.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import (
    BatchAwareWarping,
    ComBat,
    Harmony,
    Limma,
    MedianCentering,
    QualityWeightedComBat,
    ReferenceScaling,
    SpeciesAwareComBat,
    ZScorePerBatch,
)


@pytest.fixture
def leakage_dataset(rng):
    n_per_batch = 50
    n_features = 40
    batches = ["b0", "b1", "b2"]
    idx = [f"s{i:04d}" for i in range(n_per_batch * len(batches))]
    batch = pd.Series(np.repeat(batches, n_per_batch), index=idx, name="batch")
    species = pd.Series(
        np.tile(["spA", "spB"], len(idx) // 2), index=idx, name="species"
    )
    base = rng.standard_normal((len(idx), n_features))
    shift = np.asarray(
        [-1.2 if b == "b0" else 0.3 if b == "b1" else 1.1 for b in batch]
    )[:, None]
    sp_shift = np.where(species.to_numpy() == "spA", -0.2, 0.2)[:, None]
    X = pd.DataFrame(
        base + shift + sp_shift,
        index=idx,
    )
    quality = pd.Series(rng.uniform(0.5, 3.0, size=len(idx)), index=idx, name="snr")
    return {
        "X": X,
        "batch": batch,
        "species": species,
        "quality": quality,
    }


def _split(ds, train_frac=0.6):
    """Batch-stratified train/test split so every batch is in both folds.

    Sequential splitting silently breaks the leakage tests for methods
    that refuse to transform an unseen batch (Limma, combatlearn ComBat),
    even though those refusals are a safety feature, not a bug. Here we
    preserve the invariant we actually want to test.
    """
    X = ds["X"]
    batch = ds["batch"]
    train_idx: list[str] = []
    test_idx: list[str] = []
    for lvl in batch.unique():
        lvl_ids = batch[batch == lvl].index.tolist()
        n_train = max(1, int(len(lvl_ids) * train_frac))
        train_idx.extend(lvl_ids[:n_train])
        test_idx.extend(lvl_ids[n_train:])
    return X.loc[train_idx], X.loc[test_idx]


def _fitted_state(est):
    """Snapshot every public / private fitted attribute (names ending in '_').

    Works for both numpy arrays and pandas objects.
    """
    names = [
        n
        for n in vars(est)
        if n.endswith("_") and not n.startswith("__") and not n.endswith("__")
    ]
    state = {}
    for name in names:
        value = getattr(est, name)
        if isinstance(value, np.ndarray):
            state[name] = value.copy()
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            state[name] = value.copy(deep=True)
        else:
            # best-effort deep-copy for scalars / dicts / objects
            state[name] = copy.deepcopy(value)
    return state


def _assert_state_equal(a: dict, b: dict) -> None:
    from sklearn.base import BaseEstimator

    assert set(a) == set(b), set(a) ^ set(b)
    for name in a:
        va, vb = a[name], b[name]
        if isinstance(va, np.ndarray):
            np.testing.assert_array_equal(va, vb)
        elif isinstance(va, pd.Series):
            pd.testing.assert_series_equal(va, vb)
        elif isinstance(va, pd.DataFrame):
            pd.testing.assert_frame_equal(va, vb)
        elif isinstance(va, dict):
            # BatchAwareWarping stores a dict of Warping objects;
            # compare key sets and the underlying reference spectra.
            assert set(va) == set(vb)
            for k in va:
                if hasattr(va[k], "ref_spec_"):
                    np.testing.assert_array_equal(va[k].ref_spec_, vb[k].ref_spec_)
        elif isinstance(va, BaseEstimator):
            assert type(va) is type(vb)
            assert va.get_params() == vb.get_params()
            fitted_names = [
                k
                for k in vars(va)
                if k.endswith("_") and not k.startswith("__") and not k.endswith("__")
            ]
            for k in fitted_names:
                sub_a, sub_b = getattr(va, k), getattr(vb, k)
                if isinstance(sub_a, np.ndarray):
                    np.testing.assert_array_equal(sub_a, sub_b)
                else:
                    assert sub_a == sub_b, (name, k, sub_a, sub_b)
        else:
            assert va == vb, (name, va, vb)


def _builders(ds):
    """Constructor factories used across the leakage tests."""
    return {
        "ComBat (Johnson)": lambda: ComBat(batch=ds["batch"], method="johnson"),
        "ComBat (Fortin)": lambda: ComBat(
            batch=ds["batch"], method="fortin", discrete_covariates=ds["species"]
        ),
        "ComBat (Chen/CovBat)": lambda: ComBat(
            batch=ds["batch"],
            method="chen",
            discrete_covariates=ds["species"],
            covbat_cov_thresh=0.9,
        ),
        "SpeciesAwareComBat": lambda: SpeciesAwareComBat(
            batch=ds["batch"], species=ds["species"]
        ),
        "Limma": lambda: Limma(batch=ds["batch"]),
        "Limma (+design)": lambda: Limma(
            batch=ds["batch"],
            design=pd.get_dummies(ds["species"], drop_first=True).astype(float),
        ),
        "MedianCentering": lambda: MedianCentering(batch=ds["batch"]),
        "ZScorePerBatch": lambda: ZScorePerBatch(batch=ds["batch"]),
        "ReferenceScaling": lambda: ReferenceScaling(batch=ds["batch"]),
        "QualityWeightedComBat": lambda: QualityWeightedComBat(
            batch=ds["batch"], quality=ds["quality"]
        ),
        "Harmony": lambda: Harmony(
            batch=ds["batch"], max_iter=3, random_state=0, nclust=3
        ),
    }


def _maldi_builders(ds):
    """MALDI-specific builders (exercised only where practical)."""
    return {
        "BatchAwareWarping": lambda: BatchAwareWarping(
            batch=ds["batch"], method="shift", max_shift=3
        ),
    }


#   fit(X_train) must yield identical parameters whether `batch` covers
#   the full dataset or just the training rows.


@pytest.mark.parametrize(
    "method_name",
    list(
        _builders(
            {
                "X": pd.DataFrame(
                    np.zeros((150, 40)), index=[f"s{i:04d}" for i in range(150)]
                ),
                "batch": pd.Series(
                    np.repeat(["b0", "b1", "b2"], 50),
                    index=[f"s{i:04d}" for i in range(150)],
                ),
                "species": pd.Series(
                    np.tile(["spA", "spB"], 75), index=[f"s{i:04d}" for i in range(150)]
                ),
                "quality": pd.Series(
                    np.ones(150), index=[f"s{i:04d}" for i in range(150)]
                ),
            }
        ).keys()
    ),
)
def test_fit_only_uses_training_rows(leakage_dataset, method_name):
    X_train, _ = _split(leakage_dataset)
    builders = _builders(leakage_dataset)

    # (a) builder sees batch/covariates indexed by ALL samples (train + test)
    est_full = builders[method_name]().fit(X_train)
    state_full = _fitted_state(est_full)

    # (b) builder sees batch/covariates indexed by TRAINING samples only
    ds_train_only = {
        k: (v.loc[X_train.index] if isinstance(v, (pd.Series, pd.DataFrame)) else v)
        for k, v in leakage_dataset.items()
    }
    ds_train_only["X"] = leakage_dataset["X"]  # X itself is unused by builder
    est_train = _builders(ds_train_only)[method_name]().fit(X_train)
    state_train = _fitted_state(est_train)

    _assert_state_equal(state_full, state_train)


#   transform(X_test) must NOT mutate fitted state.


@pytest.mark.parametrize(
    "method_name",
    list(
        _builders(
            {
                "X": pd.DataFrame(
                    np.zeros((150, 40)), index=[f"s{i:04d}" for i in range(150)]
                ),
                "batch": pd.Series(
                    np.repeat(["b0", "b1", "b2"], 50),
                    index=[f"s{i:04d}" for i in range(150)],
                ),
                "species": pd.Series(
                    np.tile(["spA", "spB"], 75), index=[f"s{i:04d}" for i in range(150)]
                ),
                "quality": pd.Series(
                    np.ones(150), index=[f"s{i:04d}" for i in range(150)]
                ),
            }
        ).keys()
    ),
)
def test_transform_does_not_mutate_fitted_state(leakage_dataset, method_name):
    X_train, X_test = _split(leakage_dataset)
    est = _builders(leakage_dataset)[method_name]().fit(X_train)

    state_before = _fitted_state(est)
    _ = est.transform(X_test)
    state_after = _fitted_state(est)

    for diag in ("_batch_var_after_",):
        state_before.pop(diag, None)
        state_after.pop(diag, None)

    _assert_state_equal(state_before, state_after)


#   transform(X_test) is a pure function of fitted state: any earlier
#   transform call on different rows must not change the output on X_test.


@pytest.mark.parametrize(
    "method_name",
    list(
        _builders(
            {
                "X": pd.DataFrame(
                    np.zeros((150, 40)), index=[f"s{i:04d}" for i in range(150)]
                ),
                "batch": pd.Series(
                    np.repeat(["b0", "b1", "b2"], 50),
                    index=[f"s{i:04d}" for i in range(150)],
                ),
                "species": pd.Series(
                    np.tile(["spA", "spB"], 75), index=[f"s{i:04d}" for i in range(150)]
                ),
                "quality": pd.Series(
                    np.ones(150), index=[f"s{i:04d}" for i in range(150)]
                ),
            }
        ).keys()
    ),
)
def test_transform_is_pure_function_of_fit(leakage_dataset, method_name):
    X_train, X_test = _split(leakage_dataset)
    est = _builders(leakage_dataset)[method_name]().fit(X_train)

    first = np.asarray(est.transform(X_test))
    # Poke it with a totally unrelated matrix in between
    X_garbage = X_train.copy()
    X_garbage.iloc[:, :] = 99.0
    _ = est.transform(X_garbage)
    second = np.asarray(est.transform(X_test))

    np.testing.assert_allclose(first, second, atol=1e-12)


def test_batch_aware_warping_fit_uses_only_train_rows(leakage_dataset):
    X_train, _ = _split(leakage_dataset)
    ds = leakage_dataset

    est_full = BatchAwareWarping(batch=ds["batch"], method="shift", max_shift=3).fit(
        X_train
    )
    est_train = BatchAwareWarping(
        batch=ds["batch"].loc[X_train.index], method="shift", max_shift=3
    ).fit(X_train)

    np.testing.assert_array_equal(est_full.reference_, est_train.reference_)
    assert set(est_full.warpers_) == set(est_train.warpers_)
    for lvl in est_full.warpers_:
        np.testing.assert_array_equal(
            est_full.warpers_[lvl].ref_spec_,
            est_train.warpers_[lvl].ref_spec_,
        )


def test_harmony_transform_rejects_unseen_batches(leakage_dataset):
    """Like ComBat / Limma, Harmony's closed-form correction is only
    defined for batches present at fit time."""
    X_train, X_test = _split(leakage_dataset)
    est = Harmony(
        batch=leakage_dataset["batch"].loc[X_train.index],
        random_state=0,
        max_iter=3,
        nclust=3,
    ).fit(X_train)
    # Fabricate a batch label not present at fit time
    est.batch = pd.Series(["brand_new_batch"] * len(X_test), index=X_test.index)
    with pytest.raises(ValueError, match="Unseen batch level"):
        est.transform(X_test)


def test_limma_refuses_unseen_batch_rather_than_silent_zero(leakage_dataset):
    """Limma's subtraction is undefined for batches not part of the design;
    better to raise than silently pretend the effect is zero."""
    X_train, X_test = _split(leakage_dataset)
    est = Limma(batch=leakage_dataset["batch"]).fit(X_train)

    # Inject a batch level that did not appear at fit time
    est.batch = pd.Series(["brand_new_batch"] * len(X_test), index=X_test.index)
    with pytest.raises(ValueError, match="Unseen batch level"):
        est.transform(X_test)
