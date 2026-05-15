"""Class-based diagnostic comparison of batch correctors.

:class:`BatchCorrectionBenchmark` mirrors the
``GridSearchCV``-style configure-then-``fit``-then-inspect shape: build
it with a dict of unfitted correctors and a list of metric names (or
callables), then call ``.fit(X, batch=..., species=...)`` to compute
every metric on every corrector.

The benchmark is **diagnostic-only by design**: it does not score a
downstream classifier. Use
:class:`~maldibatchkit.corrections.auto.AutoCorrector` together with
``sklearn.model_selection.GridSearchCV`` if you want the downstream
metric (AUROC for AMR) to pick the winner.

Aggregation
-----------
``fit`` always populates two tables:

* ``results_long_`` - one row per (method, metric, repeat, bootstrap).
  Use it when you want to compute your own statistics or feed a
  raincloud / strip plot.
* ``results_`` - one row per (method, metric) with the central tendency
  and a confidence interval. The convention is:

  - When ``n_bootstrap >= 2``, ``ci_lo`` / ``ci_hi`` are the 2.5 / 97.5
    **percentiles** of the bootstrap distribution and ``value`` is the
    bootstrap mean (this is the standard non-parametric bootstrap
    percentile interval, e.g. Efron & Tibshirani 1993).
  - With ``n_bootstrap < 2`` but ``n_repeats >= 3`` (only meaningful for
    ``protocol='stratified_split'``), ``ci_lo`` / ``ci_hi`` are the
    2.5 / 97.5 percentiles **across repeats**, so repeat-to-repeat
    instability shows up as a wider interval. ``value`` is the mean
    across repeats.
  - Below those thresholds the interval columns hold ``NaN`` and
    ``value`` is the single observation. Don't read a CI from one
    or two points.

  ``std`` and ``n`` are always populated so you can post-hoc compute
  a different summary if you prefer.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit

from .._base import BaseBatchCorrector
from .._utils import ArrayLike
from .generic import (
    kbet,
    lisi,
    lisi_normalized,
    silhouette_batch,
    species_preservation,
)
from .maldi import peak_position_drift, tic_cov_per_batch

__all__ = ["BatchCorrectionBenchmark"]


def _kbet_acceptance(X: ArrayLike, batch: ArrayLike, **kw: Any) -> float:
    """KBET acceptance rate (scalar wrapper for the dict-returning kbet)."""
    return float(
        kbet(X, batch, **{k: v for k, v in kw.items() if k in ("k", "alpha")})[
            "acceptance_rate"
        ]
    )


def _peak_drift_mean(X: ArrayLike, batch: ArrayLike, **kw: Any) -> float:
    """Per-batch mean |delta m/z|, averaged across batches."""
    mz_values = kw.get("mz_values")
    top_k = kw.get("top_k_peaks", 50)
    out = peak_position_drift(X, batch, mz_values=mz_values, top_k=top_k)
    if out.empty:
        return float("nan")
    return float(out["mean_delta_mz"].mean())


def _tic_cov_mean(X: ArrayLike, batch: ArrayLike, **kw: Any) -> float:
    """Per-batch TIC coefficient of variation, averaged across batches."""
    out = tic_cov_per_batch(X, batch)
    if out.empty:
        return float("nan")
    return float(out.mean())


_METRIC_REGISTRY: dict[str, Callable[..., float]] = {
    "kbet": _kbet_acceptance,
    "kbet_acceptance_rate": _kbet_acceptance,
    "lisi": lambda X, batch, **kw: float(
        lisi(X, batch, perplexity=kw.get("perplexity", 30.0))
    ),
    "lisi_normalized": lambda X, batch, **kw: float(
        lisi_normalized(X, batch, perplexity=kw.get("perplexity", 30.0))
    ),
    "silhouette_batch": lambda X, batch, **kw: silhouette_batch(X, batch),
    "species_preservation": lambda X, batch, *, species=None, **kw: float(
        species_preservation(
            X,
            species if species is not None else batch,
            perplexity=kw.get("perplexity", 30.0),
        )
    ),
    "peak_position_drift": _peak_drift_mean,
    "tic_cov_per_batch": _tic_cov_mean,
}

_METRIC_DIRECTION: dict[str, str] = {
    "kbet": "higher",
    "kbet_acceptance_rate": "higher",
    "lisi": "higher",
    "lisi_normalized": "higher",
    "silhouette_batch": "zero",
    "species_preservation": "higher",
    "peak_position_drift": "lower",
    "tic_cov_per_batch": "lower",
}


def _resolve_metric(metric: Any) -> tuple[str, Callable[..., float]]:
    """Return ``(name, fn)`` for a string alias or callable."""
    if isinstance(metric, str):
        if metric not in _METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric {metric!r}. Registered: {sorted(_METRIC_REGISTRY)} "
                f"(or pass a callable)."
            )
        return metric, _METRIC_REGISTRY[metric]
    if callable(metric):
        name = getattr(metric, "__name__", repr(metric))
        return name, metric
    raise TypeError(
        f"metric must be a string or callable; got {type(metric).__name__}."
    )


def _call_metric(
    fn: Callable[..., float],
    X: pd.DataFrame,
    batch: np.ndarray,
    *,
    species: Any,
    extra: Mapping[str, Any],
) -> float:
    """Invoke ``fn`` with whichever supported kwargs it accepts."""
    try:
        sig = inspect.signature(fn)
        accepted = {
            name
            for name, p in sig.parameters.items()
            if p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        accepted, has_var_kw = set(), True

    kw: dict[str, Any] = {}
    if "species" in accepted or has_var_kw:
        kw["species"] = species
    for k, v in extra.items():
        if k in accepted or has_var_kw:
            kw[k] = v
    return float(fn(X, batch, **kw))


def _bootstrap_indices(
    rng: np.random.Generator,
    batch: np.ndarray,
    n_bootstrap: int,
) -> list[np.ndarray]:
    """Stratified bootstrap row indices, sampling with replacement within each batch."""
    levels = np.unique(batch)
    by_level = {lvl: np.flatnonzero(batch == lvl) for lvl in levels}
    out = []
    for _ in range(n_bootstrap):
        chunks = [
            rng.choice(idxs, size=len(idxs), replace=True) for idxs in by_level.values()
        ]
        out.append(np.concatenate(chunks))
    return out


def _summarise(group: pd.DataFrame, has_bootstrap: bool) -> pd.Series:
    """Reduce a long-form group to value / ci_lo / ci_hi / std / n."""
    vals = group["value"].to_numpy(dtype=float)
    n = int(len(vals))
    if n == 0:
        return pd.Series(
            {"value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "std": np.nan, "n": 0}
        )
    if has_bootstrap and n >= 2:
        ci_lo, ci_hi = np.nanpercentile(vals, [2.5, 97.5])
        value = float(np.nanmean(vals))
    elif n >= 3:
        ci_lo, ci_hi = np.nanpercentile(vals, [2.5, 97.5])
        value = float(np.nanmean(vals))
    else:
        ci_lo, ci_hi = np.nan, np.nan
        value = float(np.nanmean(vals))
    std = float(np.nanstd(vals, ddof=1)) if n >= 2 else float("nan")
    return pd.Series(
        {"value": value, "ci_lo": ci_lo, "ci_hi": ci_hi, "std": std, "n": n}
    )


class BatchCorrectionBenchmark:
    """Diagnostic comparison of multiple batch correctors.

    Parameters
    ----------
    correctors : dict[str, BaseBatchCorrector]
        Mapping from a display name to an *unfitted*
        :class:`BaseBatchCorrector`. Each will be cloned per protocol
        iteration (so calling ``fit`` on the benchmark does not leave
        the input correctors fitted).
    metrics : sequence of str or callable, default=('kbet', 'lisi_normalized', 'species_preservation')
        Metric specifications. Strings are resolved against the registry
        in :mod:`maldibatchkit.diagnostics`; callables are invoked as
        ``metric(X_corrected, batch, species=species, **extra)`` and
        only receive the kwargs they actually accept.
    protocol : {'full_data', 'stratified_split'}, default='full_data'
        ``'full_data'`` fits each corrector on all rows and scores on
        the same rows (Büttner-2019 convention). ``'stratified_split'``
        fits on a stratified train split and scores on the held-out
        test split; every batch is forced into both folds.
    test_size : float, default=0.2
        Test fraction for ``'stratified_split'``.
    n_repeats : int, default=1
        Number of repeated splits for ``'stratified_split'``.
    n_bootstrap : int, default=0
        If non-zero, every metric is recomputed on ``n_bootstrap``
        stratified row-resamples of the (corrected) matrix to give
        confidence intervals. ``0`` disables bootstrapping.
    bootstrap_mode : {'resample_metric', 'refit'}, default='resample_metric'
        ``'resample_metric'`` (the fast, default mode) fits each
        corrector once and resamples rows of the corrected matrix to
        score the metric repeatedly - CIs reflect the metric's sampling
        noise only. ``'refit'`` resamples rows of ``X`` and refits the
        corrector for every bootstrap iteration; this is slower
        (``n_correctors × n_bootstrap`` extra fits per repeat) but the
        CI also captures corrector stability.
    random_state : int or np.random.Generator, optional
        Seed / generator for splits and bootstrap.

    Attributes
    ----------
    results_long_ : pd.DataFrame
        Tidy raw observations: columns ``method``, ``metric``, ``repeat``,
        ``bootstrap``, ``value``. ``bootstrap == -1`` marks the
        point-estimate row (no bootstrap resampling).
    results_ : pd.DataFrame
        Per-(method, metric) summary: ``value`` (mean), ``ci_lo`` /
        ``ci_hi`` (2.5 / 97.5 percentile), ``std`` (sample std), ``n``
        (number of observations), ``better`` (``'higher'``, ``'lower'``,
        ``'zero'`` for metrics where both signs are bad like
        ``silhouette_batch``, or ``'n/a'`` for user callables - annotate
        via :attr:`_METRIC_DIRECTION` for registered names).
    corrected_ : dict[str, pd.DataFrame]
        For ``protocol='full_data'``, the fitted-and-transformed matrix
        from each corrector. For ``protocol='stratified_split'``, the
        corrected *test* matrix from the **last** repeat (provided for
        downstream inspection / plotting; use ``results_long_`` for
        per-repeat statistics).
    baseline_ : pd.DataFrame
        One-row-per-metric report on the **uncorrected** ``X``, mirroring
        the ``results_`` schema.

    Examples
    --------
    >>> from maldibatchkit import ComBat, NoOpCorrector
    >>> from maldibatchkit.diagnostics import BatchCorrectionBenchmark
    >>> bench = BatchCorrectionBenchmark(
    ...     correctors={
    ...         "none": NoOpCorrector(batch=b),
    ...         "combat-fortin": ComBat(batch=b, method="fortin"),
    ...     },
    ...     metrics=("kbet", "species_preservation"),
    ...     n_bootstrap=200,
    ...     random_state=0,
    ... )
    >>> bench.fit(X, batch=b, species=s)  # doctest: +SKIP
    >>> bench.rank(by="species_preservation")  # doctest: +SKIP
    """

    def __init__(
        self,
        correctors: Mapping[str, BaseBatchCorrector],
        *,
        metrics: Sequence[Any] = ("kbet", "lisi_normalized", "species_preservation"),
        protocol: str = "full_data",
        test_size: float = 0.2,
        n_repeats: int = 1,
        n_bootstrap: int = 0,
        bootstrap_mode: str = "resample_metric",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if protocol not in ("full_data", "stratified_split"):
            raise ValueError(
                f"protocol must be 'full_data' or 'stratified_split'; got {protocol!r}."
            )
        if bootstrap_mode not in ("resample_metric", "refit"):
            raise ValueError(
                "bootstrap_mode must be 'resample_metric' or 'refit'; "
                f"got {bootstrap_mode!r}."
            )
        if not correctors:
            raise ValueError("`correctors` must contain at least one entry.")
        self.correctors = dict(correctors)
        self.metrics = tuple(metrics)
        self.protocol = protocol
        self.test_size = test_size
        self.n_repeats = int(n_repeats)
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_mode = bootstrap_mode
        self.random_state = random_state

    def _rng(self) -> np.random.Generator:
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state
        return np.random.default_rng(self.random_state)

    def _resolve_metrics(self) -> list[tuple[str, Callable[..., float], str]]:
        resolved = []
        for m in self.metrics:
            name, fn = _resolve_metric(m)
            direction = _METRIC_DIRECTION.get(name, "n/a")
            resolved.append((name, fn, direction))
        return resolved

    def fit(
        self,
        X: ArrayLike,
        *,
        batch: ArrayLike,
        species: ArrayLike | None = None,
        y: ArrayLike | None = None,
        **extra: Any,
    ) -> BatchCorrectionBenchmark:
        """Run every corrector under the chosen protocol and score each metric.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        batch : array-like of shape (n_samples,)
            Batch labels.
        species : array-like, optional
            Forwarded to metrics that need it
            (e.g. ``species_preservation``).
        y : array-like, optional
            Ignored at the benchmark level (no classifier scoring).
            Kept on the signature so the call site mirrors sklearn.
        **extra : Any
            Forwarded to every metric callable that accepts the given
            keyword (e.g. ``mz_values=`` for ``peak_position_drift``).

        Returns
        -------
        self : BatchCorrectionBenchmark
            Fitted benchmark with ``results_long_``, ``results_``,
            ``corrected_``, ``baseline_`` populated.
        """
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(np.asarray(X))
        idx = X_df.index

        batch_arr = (
            batch.loc[idx].to_numpy()
            if isinstance(batch, pd.Series | pd.DataFrame)
            else np.asarray(batch)
        )
        species_aligned: Any
        if species is None:
            species_aligned = None
        elif isinstance(species, pd.Series | pd.DataFrame):
            species_aligned = species.loc[idx]
        else:
            species_aligned = pd.Series(np.asarray(species), index=idx)

        metrics_resolved = self._resolve_metrics()
        rng = self._rng()

        if self.protocol == "full_data":
            long_rows, corrected, baseline_rows = self._run_full_data(
                X_df, batch_arr, species_aligned, metrics_resolved, extra, rng
            )
        else:
            long_rows, corrected, baseline_rows = self._run_stratified(
                X_df, batch_arr, species_aligned, metrics_resolved, extra, rng
            )

        results_long = pd.DataFrame(long_rows)
        self.results_long_ = results_long

        direction_lookup = {name: d for name, _, d in metrics_resolved}
        has_bootstrap = self.n_bootstrap >= 2
        summary = (
            results_long.groupby(["method", "metric"], sort=False)
            .apply(
                lambda g: _summarise(g, has_bootstrap=has_bootstrap),
                include_groups=False,
            )
            .reset_index()
        )
        summary["better"] = summary["metric"].map(direction_lookup).fillna("n/a")
        self.results_ = summary
        self.corrected_ = corrected
        self.baseline_ = pd.DataFrame(baseline_rows)
        return self

    def _score_corrected(
        self,
        X_corrected: pd.DataFrame,
        batch: np.ndarray,
        species_aligned: Any,
        metrics_resolved: list[tuple[str, Callable[..., float], str]],
        extra: Mapping[str, Any],
        method_name: str,
        repeat: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """Score a single corrected matrix; bootstrap rows when requested.

        Always emits the point estimate (``bootstrap=-1``) plus
        ``n_bootstrap`` resampled scores when ``bootstrap_mode`` is
        ``'resample_metric'``.
        """
        rows: list[dict[str, Any]] = []
        species_arr = (
            species_aligned.to_numpy()
            if isinstance(species_aligned, pd.Series)
            else (None if species_aligned is None else np.asarray(species_aligned))
        )

        for name, fn, _direction in metrics_resolved:
            value = _call_metric(
                fn, X_corrected, batch, species=species_arr, extra=extra
            )
            rows.append(
                {
                    "method": method_name,
                    "metric": name,
                    "repeat": repeat,
                    "bootstrap": -1,
                    "value": value,
                }
            )

        if self.n_bootstrap > 0 and self.bootstrap_mode == "resample_metric":
            boot_idx_sets = _bootstrap_indices(rng, batch, self.n_bootstrap)
            for b, sel in enumerate(boot_idx_sets):
                Xb = X_corrected.iloc[sel]
                bb = batch[sel]
                sb = species_arr[sel] if species_arr is not None else None
                for name, fn, _direction in metrics_resolved:
                    val = _call_metric(fn, Xb, bb, species=sb, extra=extra)
                    rows.append(
                        {
                            "method": method_name,
                            "metric": name,
                            "repeat": repeat,
                            "bootstrap": b,
                            "value": val,
                        }
                    )
        return rows

    def _fit_transform(
        self, corrector: BaseBatchCorrector, X_train: pd.DataFrame, X_eval: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit on ``X_train`` and transform ``X_eval`` (may be the same).

        ``clone`` is used so the user's original corrector instances stay
        unfitted, and so per-protocol-iteration fits are independent.
        """
        c = clone(corrector)
        c.fit(X_train)
        out = c.transform(X_eval)
        if isinstance(out, pd.DataFrame):
            return out
        return pd.DataFrame(np.asarray(out), index=X_eval.index, columns=X_eval.columns)

    def _run_full_data(
        self,
        X_df: pd.DataFrame,
        batch: np.ndarray,
        species_aligned: Any,
        metrics_resolved: list[tuple[str, Callable[..., float], str]],
        extra: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], list[dict[str, Any]]]:
        long_rows: list[dict[str, Any]] = []
        corrected: dict[str, pd.DataFrame] = {}

        baseline_rows = self._score_baseline(
            X_df, batch, species_aligned, metrics_resolved, extra
        )

        for name, corrector in self.correctors.items():
            X_corr = self._fit_transform(corrector, X_df, X_df)
            corrected[name] = X_corr
            long_rows.extend(
                self._score_corrected(
                    X_corr,
                    batch,
                    species_aligned,
                    metrics_resolved,
                    extra,
                    method_name=name,
                    repeat=0,
                    rng=rng,
                )
            )

            if self.n_bootstrap > 0 and self.bootstrap_mode == "refit":
                boot_idx_sets = _bootstrap_indices(rng, batch, self.n_bootstrap)
                for b, sel in enumerate(boot_idx_sets):
                    X_boot = X_df.iloc[sel]
                    b_boot = batch[sel]
                    sp_boot = (
                        species_aligned.iloc[sel]
                        if isinstance(species_aligned, pd.Series)
                        else None
                    )
                    try:
                        X_corr_b = self._fit_transform(corrector, X_boot, X_boot)
                    except Exception as exc:  # noqa: BLE001
                        warnings.warn(
                            f"Bootstrap refit failed for method {name!r} on "
                            f"iteration {b}: {exc}. Skipping.",
                            stacklevel=2,
                        )
                        continue
                    sp_arr = sp_boot.to_numpy() if sp_boot is not None else None
                    for mname, fn, _ in metrics_resolved:
                        val = _call_metric(
                            fn, X_corr_b, b_boot, species=sp_arr, extra=extra
                        )
                        long_rows.append(
                            {
                                "method": name,
                                "metric": mname,
                                "repeat": 0,
                                "bootstrap": b,
                                "value": val,
                            }
                        )

        return long_rows, corrected, baseline_rows

    def _run_stratified(
        self,
        X_df: pd.DataFrame,
        batch: np.ndarray,
        species_aligned: Any,
        metrics_resolved: list[tuple[str, Callable[..., float], str]],
        extra: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], list[dict[str, Any]]]:
        seed = int(rng.integers(0, 2**31 - 1))
        sss = StratifiedShuffleSplit(
            n_splits=self.n_repeats, test_size=self.test_size, random_state=seed
        )
        long_rows: list[dict[str, Any]] = []
        corrected: dict[str, pd.DataFrame] = {}

        baseline_rows = self._score_baseline(
            X_df, batch, species_aligned, metrics_resolved, extra
        )

        for repeat, (train_idx, test_idx) in enumerate(sss.split(X_df, batch)):
            X_train = X_df.iloc[train_idx]
            X_test = X_df.iloc[test_idx]
            batch_test = batch[test_idx]
            species_test = (
                species_aligned.iloc[test_idx]
                if isinstance(species_aligned, pd.Series)
                else None
            )

            for name, corrector in self.correctors.items():
                X_corr = self._fit_transform(corrector, X_train, X_test)
                if repeat == self.n_repeats - 1:
                    corrected[name] = X_corr
                long_rows.extend(
                    self._score_corrected(
                        X_corr,
                        batch_test,
                        species_test,
                        metrics_resolved,
                        extra,
                        method_name=name,
                        repeat=repeat,
                        rng=rng,
                    )
                )

        return long_rows, corrected, baseline_rows

    def _score_baseline(
        self,
        X_df: pd.DataFrame,
        batch: np.ndarray,
        species_aligned: Any,
        metrics_resolved: list[tuple[str, Callable[..., float], str]],
        extra: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        species_arr = (
            species_aligned.to_numpy()
            if isinstance(species_aligned, pd.Series)
            else (None if species_aligned is None else np.asarray(species_aligned))
        )
        rows = []
        for name, fn, direction in metrics_resolved:
            value = _call_metric(fn, X_df, batch, species=species_arr, extra=extra)
            rows.append(
                {
                    "method": "__baseline__",
                    "metric": name,
                    "value": value,
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "std": float("nan"),
                    "n": 1,
                    "better": direction,
                }
            )
        return rows

    def rank(self, by: str, *, ascending: bool | None = None) -> pd.DataFrame:
        """Return ``results_`` sorted by one metric's mean value.

        Parameters
        ----------
        by : str
            Metric name to rank on.
        ascending : bool, optional
            Sort direction. If omitted, the metric's registered
            "better" direction is used: ``'higher'`` → descending,
            ``'lower'`` → ascending, ``'zero'`` → ascending by ``|value|``
            (e.g. ``silhouette_batch``, where both positive and negative
            extremes are bad - only 0 is well-mixed).
        """
        if not hasattr(self, "results_"):
            raise RuntimeError("Call .fit(...) before .rank().")
        sub = self.results_[self.results_["metric"] == by]
        if sub.empty:
            raise ValueError(f"No metric named {by!r} in results_.")
        direction = sub["better"].iloc[0]
        if direction == "zero":
            asc = True if ascending is None else ascending
            ordered = sub.assign(_abs=sub["value"].abs()).sort_values(
                "_abs", ascending=asc
            )
            return ordered.drop(columns="_abs").reset_index(drop=True)
        if ascending is None:
            ascending = direction != "higher"
        return sub.sort_values("value", ascending=ascending).reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        """Alias for :attr:`results_`."""
        if not hasattr(self, "results_"):
            raise RuntimeError("Call .fit(...) before .to_dataframe().")
        return self.results_

    def plot(self, ax: Any = None) -> Any:
        """Bar-plot the summary results, faceted by metric.

        Requires the ``viz`` extra (seaborn / matplotlib). Returns the
        matplotlib ``Axes`` (single metric) or ``Figure`` (multiple).
        """
        if not hasattr(self, "results_"):
            raise RuntimeError("Call .fit(...) before .plot().")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "BatchCorrectionBenchmark.plot() needs matplotlib + seaborn; "
                "install the `viz` extra (pip install 'maldibatchkit[viz]')."
            ) from exc

        df = self.results_
        metrics = list(df["metric"].unique())
        if ax is not None and len(metrics) > 1:
            raise ValueError(
                "Pass `ax` only when results_ contains a single metric; "
                f"got {len(metrics)} metrics."
            )
        if len(metrics) == 1:
            ax = ax or plt.gca()
            sns.barplot(data=df, x="method", y="value", ax=ax, color="#4C72B0")
            if df[["ci_lo", "ci_hi"]].notna().all().all():
                ax.errorbar(
                    x=range(len(df)),
                    y=df["value"],
                    yerr=[df["value"] - df["ci_lo"], df["ci_hi"] - df["value"]],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                )
            ax.set_title(metrics[0])
            # Anchor-rotated, right-aligned labels stay under their bars.
            plt.setp(
                ax.get_xticklabels(),
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            return ax
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        for ax_i, metric in zip(axes, metrics, strict=True):
            sub = df[df["metric"] == metric]
            sns.barplot(data=sub, x="method", y="value", ax=ax_i, color="#4C72B0")
            if sub[["ci_lo", "ci_hi"]].notna().all().all():
                ax_i.errorbar(
                    x=range(len(sub)),
                    y=sub["value"],
                    yerr=[sub["value"] - sub["ci_lo"], sub["ci_hi"] - sub["value"]],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                )
            ax_i.set_title(metric)
            plt.setp(
                ax_i.get_xticklabels(),
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
        fig.tight_layout()
        return fig
