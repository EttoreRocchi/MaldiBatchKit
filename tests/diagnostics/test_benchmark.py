"""Tests for ``BatchCorrectionBenchmark``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from maldibatchkit import ComBat, NoOpCorrector
from maldibatchkit._base import BaseBatchCorrector
from maldibatchkit.diagnostics import BatchCorrectionBenchmark


def _correctors(b, sp):
    return {
        "none": NoOpCorrector(batch=b),
        "fortin": ComBat(batch=b, method="fortin", discrete_covariates=sp),
    }


def test_full_data_protocol_populates_results(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors=_correctors(b, sp),
        metrics=("silhouette_batch", "species_preservation"),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)

    assert set(bench.results_.columns) >= {
        "method",
        "metric",
        "value",
        "ci_lo",
        "ci_hi",
        "std",
        "n",
        "better",
    }
    assert set(bench.results_["method"]) == {"none", "fortin"}
    assert set(bench.results_["metric"]) == {"silhouette_batch", "species_preservation"}
    assert set(bench.corrected_) == {"none", "fortin"}
    # Baseline is computed from uncorrected X exactly once
    assert len(bench.baseline_) == 2
    assert (bench.baseline_["method"] == "__baseline__").all()


def test_baseline_matches_metric_on_uncorrected_X(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("silhouette_batch",),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    # NoOp result equals baseline (no bootstrap, point estimate)
    none_value = bench.results_long_.query("method=='none' and bootstrap==-1")[
        "value"
    ].iloc[0]
    base_value = bench.baseline_["value"].iloc[0]
    assert none_value == pytest.approx(base_value)


def test_bootstrap_resample_metric_produces_ci(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors=_correctors(b, sp),
        metrics=("silhouette_batch",),
        n_bootstrap=15,
        bootstrap_mode="resample_metric",
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    row = bench.results_.iloc[0]
    assert np.isfinite(row["ci_lo"]) and np.isfinite(row["ci_hi"])
    assert row["ci_lo"] <= row["value"] <= row["ci_hi"]
    # Long-form has point + bootstrap rows.
    sub = bench.results_long_.query("method=='none'")
    assert (sub["bootstrap"] == -1).sum() == 1
    assert (sub["bootstrap"] >= 0).sum() == 15


def test_bootstrap_refit_mode_adds_extra_rows(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("silhouette_batch",),
        n_bootstrap=5,
        bootstrap_mode="refit",
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    # 1 point estimate + 5 refit bootstraps for NoOp
    assert len(bench.results_long_) == 6


def test_stratified_split_protocol_runs_with_repeats(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors=_correctors(b, sp),
        metrics=("silhouette_batch",),
        protocol="stratified_split",
        n_repeats=3,
        test_size=0.3,
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    # 2 methods x 3 repeats x 1 metric, with bootstrap == -1
    assert len(bench.results_long_) == 6
    # ci_lo / ci_hi populated for n_repeats >= 3
    assert bench.results_["ci_lo"].notna().all()
    assert bench.results_["ci_hi"].notna().all()


def test_string_and_callable_metrics_mix(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]

    def my_metric(X, batch, **_kwargs):
        return float(np.var(np.asarray(X).mean(axis=0)))

    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("silhouette_batch", my_metric),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    metric_names = set(bench.results_["metric"])
    assert "silhouette_batch" in metric_names
    assert "my_metric" in metric_names


def test_unknown_metric_raises(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("does_not_exist",),
    )
    with pytest.raises(ValueError, match="Unknown metric"):
        bench.fit(X, batch=b, species=sp)


def test_invalid_protocol_raises(big_dataset):
    b = big_dataset["batch"]
    with pytest.raises(ValueError, match="protocol must be"):
        BatchCorrectionBenchmark(
            correctors={"none": NoOpCorrector(batch=b)},
            protocol="cv",
        )


def test_invalid_bootstrap_mode_raises(big_dataset):
    b = big_dataset["batch"]
    with pytest.raises(ValueError, match="bootstrap_mode must be"):
        BatchCorrectionBenchmark(
            correctors={"none": NoOpCorrector(batch=b)},
            bootstrap_mode="rolling",
        )


def test_empty_correctors_raises():
    with pytest.raises(ValueError, match="at least one entry"):
        BatchCorrectionBenchmark(correctors={})


def test_rank_uses_metric_direction(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors=_correctors(b, sp),
        metrics=("silhouette_batch",),
        n_bootstrap=10,
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    ranked = bench.rank("silhouette_batch")
    # "better": "zero" → ascending sort on |value|, since both
    # positive and negative silhouettes are bad (only ~0 is well-mixed).
    assert ranked["value"].abs().is_monotonic_increasing


def test_rank_raises_before_fit(big_dataset):
    b = big_dataset["batch"]
    bench = BatchCorrectionBenchmark(correctors={"none": NoOpCorrector(batch=b)})
    with pytest.raises(RuntimeError, match="Call .fit"):
        bench.rank("silhouette_batch")


def test_to_dataframe_alias(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("silhouette_batch",),
    )
    bench.fit(X, batch=b, species=sp)
    assert bench.to_dataframe() is bench.results_


def test_user_correctors_remain_unfitted(big_dataset):
    """Cloning means the user's instances are never mutated by .fit()."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    correctors = _correctors(b, sp)
    bench = BatchCorrectionBenchmark(
        correctors=correctors,
        metrics=("silhouette_batch",),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    for _name, c in correctors.items():
        with pytest.raises(NotFittedError):
            check_is_fitted(c)


def test_maldi_metric_wrappers_score_drift_and_tic(big_dataset):
    """``peak_position_drift`` and ``tic_cov_per_batch`` are exposed by name."""
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    mz = np.linspace(2000.0, 12000.0, X.shape[1])
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("peak_position_drift", "tic_cov_per_batch"),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp, mz_values=mz, top_k_peaks=10)
    drift = bench.results_.query("metric == 'peak_position_drift'")["value"].iloc[0]
    tic = bench.results_.query("metric == 'tic_cov_per_batch'")["value"].iloc[0]
    assert np.isfinite(drift)
    assert np.isfinite(tic)
    # Directions are looked up from the registry, not user-defined.
    assert (
        bench.results_.query("metric == 'peak_position_drift'")["better"].iloc[0]
        == "lower"
    )
    assert (
        bench.results_.query("metric == 'tic_cov_per_batch'")["better"].iloc[0]
        == "lower"
    )


def test_kbet_string_metric_runs(big_dataset):
    """The ``kbet`` alias resolves to the scalar acceptance-rate wrapper."""
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("kbet",),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    val = bench.results_["value"].iloc[0]
    assert 0.0 <= val <= 1.0
    assert bench.results_["better"].iloc[0] == "higher"


def test_non_string_non_callable_metric_raises(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=(123,),
    )
    with pytest.raises(TypeError, match="must be a string or callable"):
        bench.fit(X, batch=b, species=sp)


def test_metric_with_unintrospectable_signature(big_dataset):
    """A callable whose ``inspect.signature`` fails still scores correctly."""

    class _NoSigMetric:
        __name__ = "opaque_metric"

        @property
        def __signature__(self):
            raise ValueError("intentional")

        def __call__(self, X, batch, **kwargs):
            return float(np.var(np.asarray(X).mean(axis=0)))

    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=(_NoSigMetric(),),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    assert "opaque_metric" in set(bench.results_["metric"])
    assert np.isfinite(bench.results_["value"].iloc[0])


def test_extra_kwargs_propagate_to_callable_metric(big_dataset):
    """Keyword args forwarded via ``fit(**extra)`` reach matching metrics only."""
    seen: dict[str, object] = {}

    def metric_with_kw(X, batch, *, scale=1.0, **_):
        seen["scale"] = scale
        return float(scale * np.var(np.asarray(X).mean(axis=0)))

    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=(metric_with_kw,),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp, scale=2.5)
    assert seen["scale"] == 2.5


def test_random_state_generator_is_used_as_is(big_dataset):
    """A pre-built Generator is passed through without re-seeding."""
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    gen = np.random.default_rng(42)
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("silhouette_batch",),
        n_bootstrap=5,
        random_state=gen,
    )
    bench.fit(X, batch=b, species=sp)
    # The Generator's state was advanced by the bootstrap draws.
    assert gen.bit_generator.state != np.random.default_rng(42).bit_generator.state


def test_numpy_array_inputs_for_X_and_species(big_dataset):
    """X as ndarray and species as ndarray both round-trip through fit."""
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    X_np = X.to_numpy()
    sp_np = sp.to_numpy()
    b_np = b.to_numpy()
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b_np)},
        metrics=("silhouette_batch", "species_preservation"),
        random_state=0,
    )
    bench.fit(X_np, batch=b_np, species=sp_np)
    assert set(bench.results_["metric"]) == {"silhouette_batch", "species_preservation"}
    assert len(bench.baseline_) == 2


def test_fit_without_species_uses_batch_as_fallback(big_dataset):
    """``species=None`` is allowed; ``species_preservation`` falls back to batch."""
    b, _sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("silhouette_batch",),
        random_state=0,
    )
    bench.fit(X, batch=b)  # no species kwarg
    assert "silhouette_batch" in set(bench.results_["metric"])


def test_corrector_returning_ndarray_is_wrapped(big_dataset):
    """A corrector whose ``transform`` returns ndarray must still score."""

    class _ArrayCorrector(BaseBatchCorrector):
        def __init__(self, batch):
            super().__init__(batch=batch)

        def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray) -> None:
            return None

        def _transform_impl(
            self, X_df: pd.DataFrame, batch: npt.NDArray
        ) -> pd.DataFrame:
            return X_df.copy()

        def transform(self, X):  # bypass base-class DataFrame wrapping
            X_df, batch_arr, _idx, _was_df = self._prepare_transform(X)
            return self._transform_impl(X_df, batch_arr).to_numpy()

    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"arr": _ArrayCorrector(batch=b)},
        metrics=("silhouette_batch",),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    # The corrected matrix is always materialised back into a DataFrame.
    assert isinstance(bench.corrected_["arr"], pd.DataFrame)
    assert bench.corrected_["arr"].shape == X.shape


def test_refit_bootstrap_warns_and_skips_failed_iterations(big_dataset):
    """If a refit raises mid-bootstrap, the iteration is dropped with a warning."""

    class _FlakyCorrector(BaseBatchCorrector):
        # Class-level counter is shared across clones (sklearn ``clone`` only
        # copies init params, not class state). The point-estimate fit
        # succeeds; every subsequent (bootstrap) refit raises.
        _calls = 0

        def __init__(self, batch):
            super().__init__(batch=batch)

        def _fit_impl(self, X_df, batch):
            type(self)._calls += 1
            if type(self)._calls > 1:
                raise RuntimeError("intentional refit failure")
            return None

        def _transform_impl(self, X_df, batch):
            return X_df.copy()

    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    _FlakyCorrector._calls = 0
    bench = BatchCorrectionBenchmark(
        correctors={"flaky": _FlakyCorrector(batch=b)},
        metrics=("silhouette_batch",),
        n_bootstrap=3,
        bootstrap_mode="refit",
        random_state=0,
    )
    with pytest.warns(UserWarning, match="Bootstrap refit failed"):
        bench.fit(X, batch=b, species=sp)
    # Only the point-estimate row survives; the 3 bootstrap refits were dropped.
    sub = bench.results_long_.query("method == 'flaky'")
    assert len(sub) == 1
    assert (sub["bootstrap"] == -1).all()


def test_rank_unknown_metric_raises(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("silhouette_batch",),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    with pytest.raises(ValueError, match="No metric named"):
        bench.rank("nope_not_here")


def test_rank_default_directions_for_higher_and_lower(big_dataset):
    """``higher`` → descending sort; ``lower`` → ascending."""
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    mz = np.linspace(2000.0, 12000.0, X.shape[1])
    bench = BatchCorrectionBenchmark(
        correctors=_correctors(b, sp),
        metrics=("kbet", "peak_position_drift"),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp, mz_values=mz, top_k_peaks=10)

    ranked_higher = bench.rank("kbet")  # better = "higher"
    assert ranked_higher["value"].is_monotonic_decreasing

    ranked_lower = bench.rank("peak_position_drift")  # better = "lower"
    assert ranked_lower["value"].is_monotonic_increasing


def test_to_dataframe_raises_before_fit(big_dataset):
    b = big_dataset["batch"]
    bench = BatchCorrectionBenchmark(correctors={"none": NoOpCorrector(batch=b)})
    with pytest.raises(RuntimeError, match="Call .fit"):
        bench.to_dataframe()


def test_plot_single_metric_returns_axes(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors=_correctors(b, sp),
        metrics=("silhouette_batch",),
        n_bootstrap=10,
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    ax = bench.plot()
    assert isinstance(ax, plt.Axes)
    # CIs were computed → error bars should have been drawn (one container per bar).
    assert len(ax.containers) >= 1
    plt.close("all")


def test_plot_accepts_explicit_ax(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("silhouette_batch",),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    fig, ax = plt.subplots()
    out = bench.plot(ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_multiple_metrics_returns_figure(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors=_correctors(b, sp),
        metrics=("silhouette_batch", "species_preservation"),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    fig = bench.plot()
    assert isinstance(fig, plt.Figure)
    # One subplot per metric.
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_with_ax_and_multiple_metrics_raises(big_dataset):
    b, sp, X = big_dataset["batch"], big_dataset["species"], big_dataset["X"]
    bench = BatchCorrectionBenchmark(
        correctors={"none": NoOpCorrector(batch=b)},
        metrics=("silhouette_batch", "species_preservation"),
        random_state=0,
    )
    bench.fit(X, batch=b, species=sp)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="single metric"):
        bench.plot(ax=ax)
    plt.close(fig)


def test_plot_raises_before_fit(big_dataset):
    b = big_dataset["batch"]
    bench = BatchCorrectionBenchmark(correctors={"none": NoOpCorrector(batch=b)})
    with pytest.raises(RuntimeError, match="Call .fit"):
        bench.plot()
