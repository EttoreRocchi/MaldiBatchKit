"""Batch-aware downstream classifier metrics.

The functions in this module compute a sklearn metric **per batch** on a
held-out fold, then aggregate the per-batch scalars into a single number
according to a chosen weighting scheme. They are designed for selecting
batch correctors with :class:`~maldibatchkit.AutoCorrector` and
:class:`sklearn.model_selection.GridSearchCV` when the goal is a model
that performs well *on every site*, not one that just maximises the
overall pooled score.

Weighting modes
---------------
All ``batch_*_score`` functions accept ``weights=``:

* ``'uniform'`` - each batch contributes one entry to the aggregate
  with equal weight regardless of its sample count
  (``w_i = 1 / n_batches``). The right default when you want a
  corrector that generalises across sites rather than one that
  optimises the dominant site.
* ``'balanced'`` - weights inversely proportional to per-batch size
  (``w_i ∝ 1 / n_i``). The **smallest site gets the loudest voice**,
  mirroring sklearn's ``class_weight='balanced'`` formula applied at
  the per-batch level. Use this when minority sites are the hardest
  distributions to learn and the corrector must not crush them.
* ``'size'`` - weights proportional to per-batch sample counts.
  The relationship to the standard pooled value depends on the metric
  family:

  - **Additive `correct / total` metrics** (accuracy, error rate):
    ``'size'`` exactly recovers the pooled value.
  - **Class-conditional rates** (sensitivity, specificity): the pooled
    value is recovered by *class-conditional* weights (positives-
    per-batch for sensitivity, negatives-per-batch for specificity),
    not by total-sample weights. Equal only when class prevalence is
    constant across batches.
  - **Non-linear metrics** (F1, precision, balanced accuracy, MCC):
    no per-batch weighting recovers the pooled value exactly -
    they are ratios of sums or non-linear functions of confusion-
    matrix counts.
  - **Pairwise / rank-based metrics** (AUROC, average precision):
    the pooled value also counts *cross-batch* positive-vs-negative
    pairs, which any per-batch reducer cannot see by construction;
    those pairs are gone, not approximated.

  In every non-additive case, ``'size'`` still gives the dominant
  site the loudest voice - useful when you want to score the
  classifier as a single-site practitioner would.
* ``Mapping[label, float]`` or array - explicit per-batch weights. The
  array must be aligned with ``np.unique(batch)``.

Within each batch, the wrapped sklearn metric's own averaging parameter
(``average='binary'`` / ``'macro'`` / ``'weighted'`` etc.) is preserved
and untouched. ``weights=`` only controls the *across-batch* reducer.

Degenerate folds
----------------
When a batch contains only one class in the held-out fold, AUROC and
average precision are undefined; F1 / precision / recall fall back to
zero. These batches are **silently dropped from the aggregate with a
warning** and the remaining weights are renormalised. Use ``weights=``
plus a careful split design if you need a different policy.

``make_batch_scorer`` factory
-----------------------------
:func:`make_batch_scorer` wraps the per-batch metric functions into a
:mod:`sklearn` ``scorer(estimator, X, y)`` callable suitable for
``GridSearchCV(scoring=...)``. It captures the full ``batch`` Series at
factory time and slices it by ``y.index`` for every fold.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ._utils import ArrayLike

WeightSpec = (
    Literal["uniform", "balanced", "size"] | Mapping[Any, float] | np.ndarray | list
)


__all__ = [
    "batch_average_precision_score",
    "batch_balanced_accuracy_score",
    "batch_f1_score",
    "batch_matthews_corrcoef",
    "batch_precision_score",
    "batch_recall_score",
    "batch_roc_auc_score",
    "make_batch_scorer",
]


def _resolve_weights(
    batch_levels: np.ndarray,
    sizes: np.ndarray,
    spec: WeightSpec,
) -> np.ndarray:
    """Resolve a weight specification to a non-negative vector summing to 1."""
    if isinstance(spec, str):
        if spec == "uniform":
            w = np.ones_like(sizes, dtype=float)
        elif spec == "balanced":
            sizes_f = sizes.astype(float)
            if np.any(sizes_f <= 0):
                raise ValueError(
                    "weights='balanced' requires every batch to have at "
                    "least one sample; got sizes "
                    f"{sizes_f.tolist()}."
                )
            w = 1.0 / sizes_f
        elif spec == "size":
            w = sizes.astype(float)
        else:
            raise ValueError(
                f"Unknown weights mode {spec!r}. "
                f"Expected one of 'uniform', 'balanced', 'size', or a dict / array."
            )
    elif isinstance(spec, Mapping):
        missing = [lvl for lvl in batch_levels if lvl not in spec]
        if missing:
            raise ValueError(
                f"weights dict is missing entries for batch levels: {missing}."
            )
        w = np.asarray([float(spec[lvl]) for lvl in batch_levels], dtype=float)
    else:
        w = np.asarray(spec, dtype=float)
        if w.shape != batch_levels.shape:
            raise ValueError(
                f"weights array length {w.shape[0]} does not match the number "
                f"of batch levels {batch_levels.shape[0]}."
            )

    if (w < 0).any():
        raise ValueError("weights must be non-negative.")
    total = w.sum()
    if total <= 0:
        raise ValueError("at least one weight must be positive.")
    return w / total


def _to_1d_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series | pd.DataFrame):
        return np.asarray(x).reshape(-1)
    return np.asarray(x)


def _per_batch_metric(
    y_true: ArrayLike,
    y_pred_or_score: ArrayLike,
    batch: ArrayLike,
    metric_fn: Callable[..., float],
    *,
    weights: WeightSpec = "uniform",
    requires_two_classes: bool = False,
    **metric_kwargs: Any,
) -> float:
    """Apply ``metric_fn`` per batch and aggregate.

    Parameters
    ----------
    y_true, y_pred_or_score, batch
        1-D arrays of equal length. Either pandas Series or numpy arrays.
    metric_fn
        Sklearn-style ``(y_true, y_pred_or_score, **kwargs)`` -> scalar.
    weights
        See :data:`WeightSpec`.
    requires_two_classes
        If True, batches that contain only one class in ``y_true`` are
        skipped (the metric is undefined). Skipped batches are logged via
        a ``UserWarning`` and the weights are renormalised.
    metric_kwargs
        Forwarded verbatim to ``metric_fn`` (e.g. ``average='macro'``,
        ``pos_label=0``).
    """
    y_true_arr = _to_1d_array(y_true)
    y_pred_arr = _to_1d_array(y_pred_or_score)
    batch_arr = _to_1d_array(batch)

    if not (len(y_true_arr) == len(y_pred_arr) == len(batch_arr)):
        raise ValueError(
            f"y_true, predictions, and batch must have the same length; got "
            f"{len(y_true_arr)}, {len(y_pred_arr)}, {len(batch_arr)}."
        )

    levels = np.unique(batch_arr)
    n_levels = len(levels)
    scores = np.full(n_levels, np.nan, dtype=float)
    sizes = np.zeros(n_levels, dtype=int)
    for i, lvl in enumerate(levels):
        mask = batch_arr == lvl
        sizes[i] = int(mask.sum())
        if sizes[i] < 2:
            continue
        if requires_two_classes and np.unique(y_true_arr[mask]).size < 2:
            continue
        try:
            scores[i] = float(
                metric_fn(y_true_arr[mask], y_pred_arr[mask], **metric_kwargs)
            )
        except ValueError:
            # Metric undefined for this fold (e.g. AUROC with one class).
            scores[i] = np.nan

    valid = ~np.isnan(scores)
    if not valid.any():
        warnings.warn(
            "All batches were degenerate for this metric (single-class folds "
            "or fewer than two samples). Returning NaN.",
            stacklevel=2,
        )
        return float("nan")
    n_dropped = int((~valid).sum())
    if n_dropped:
        warnings.warn(
            f"{n_dropped} batch(es) dropped from per-batch aggregation "
            f"(metric undefined on that fold). Renormalising weights across "
            f"the remaining {int(valid.sum())} batch(es).",
            stacklevel=2,
        )

    w = _resolve_weights(levels[valid], sizes[valid], weights)
    return float(np.sum(w * scores[valid]))


def batch_roc_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    batch: ArrayLike,
    weights: WeightSpec = "uniform",
    **kwargs: Any,
) -> float:
    """Per-batch AUROC, aggregated with ``weights``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_score : array-like of shape (n_samples,)
        Probability of the positive class (or any monotone score).
    batch : array-like of shape (n_samples,)
        Per-sample batch labels.
    weights : str or mapping or array, default='uniform'
        See module docstring.
    **kwargs
        Forwarded to :func:`sklearn.metrics.roc_auc_score`
        (``average``, ``multi_class``, ...).
    """
    return _per_batch_metric(
        y_true,
        y_score,
        batch,
        roc_auc_score,
        weights=weights,
        requires_two_classes=True,
        **kwargs,
    )


def batch_average_precision_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    batch: ArrayLike,
    weights: WeightSpec = "uniform",
    **kwargs: Any,
) -> float:
    """Per-batch average precision (PR-AUC), aggregated with ``weights``."""
    return _per_batch_metric(
        y_true,
        y_score,
        batch,
        average_precision_score,
        weights=weights,
        requires_two_classes=True,
        **kwargs,
    )


def batch_balanced_accuracy_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    batch: ArrayLike,
    weights: WeightSpec = "uniform",
    **kwargs: Any,
) -> float:
    """Per-batch balanced accuracy, aggregated with ``weights``."""
    return _per_batch_metric(
        y_true,
        y_pred,
        batch,
        balanced_accuracy_score,
        weights=weights,
        requires_two_classes=True,
        **kwargs,
    )


def batch_matthews_corrcoef(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    batch: ArrayLike,
    weights: WeightSpec = "uniform",
) -> float:
    """Per-batch Matthews correlation coefficient, aggregated with ``weights``.

    MCC is in ``[-1, 1]``; 0 means random prediction. A batch where
    ``y_pred`` is constant or only one class is present yields MCC = 0
    and is **not** considered degenerate (sklearn returns 0 with a
    warning) -- this differs from AUROC where the metric is undefined.
    """
    return _per_batch_metric(
        y_true,
        y_pred,
        batch,
        matthews_corrcoef,
        weights=weights,
        requires_two_classes=False,
    )


def batch_f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    batch: ArrayLike,
    weights: WeightSpec = "uniform",
    average: str | None = "binary",
    **kwargs: Any,
) -> float:
    """Per-batch F1 score, aggregated with ``weights``.

    Parameters
    ----------
    average : str, default='binary'
        Forwarded to :func:`sklearn.metrics.f1_score`. Controls how the
        F1 is averaged *within* each batch's class set (``'binary'``,
        ``'macro'``, ``'weighted'``, ``'micro'``). The across-batch
        reduction is controlled by ``weights=``.
    """
    return _per_batch_metric(
        y_true,
        y_pred,
        batch,
        f1_score,
        weights=weights,
        average=average,
        **kwargs,
    )


def batch_precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    batch: ArrayLike,
    weights: WeightSpec = "uniform",
    average: str | None = "binary",
    pos_label: int | str = 1,
    **kwargs: Any,
) -> float:
    """Per-batch precision, aggregated with ``weights``.

    Set ``pos_label=0`` (or ``average='macro'`` / ``'micro'`` /
    ``'weighted'``) for negative-class or aggregated precision; see
    :func:`sklearn.metrics.precision_score`.
    """
    return _per_batch_metric(
        y_true,
        y_pred,
        batch,
        precision_score,
        weights=weights,
        average=average,
        pos_label=pos_label,
        **kwargs,
    )


def batch_recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    batch: ArrayLike,
    weights: WeightSpec = "uniform",
    average: str | None = "binary",
    pos_label: int | str = 1,
    **kwargs: Any,
) -> float:
    """Per-batch recall, aggregated with ``weights``."""
    return _per_batch_metric(
        y_true,
        y_pred,
        batch,
        recall_score,
        weights=weights,
        average=average,
        pos_label=pos_label,
        **kwargs,
    )


# (metric_fn, default response_method)
_SCORER_REGISTRY: dict[str, tuple[Callable[..., float], str]] = {
    "roc_auc": (batch_roc_auc_score, "predict_proba"),
    "average_precision": (batch_average_precision_score, "predict_proba"),
    "balanced_accuracy": (batch_balanced_accuracy_score, "predict"),
    "matthews_corrcoef": (batch_matthews_corrcoef, "predict"),
    "mcc": (batch_matthews_corrcoef, "predict"),
    "f1": (batch_f1_score, "predict"),
    "precision": (batch_precision_score, "predict"),
    "recall": (batch_recall_score, "predict"),
}


def _resolve_metric(
    metric: str | Callable[..., float],
    response_method: str | None,
) -> tuple[Callable[..., float], str]:
    if callable(metric):
        if response_method is None:
            raise ValueError(
                "When passing a callable metric, `response_method` must be set "
                "to 'predict_proba', 'predict', or 'decision_function'."
            )
        return metric, response_method
    if metric not in _SCORER_REGISTRY:
        raise ValueError(
            f"Unknown metric alias {metric!r}. Registered: {sorted(_SCORER_REGISTRY)}."
        )
    fn, default_response = _SCORER_REGISTRY[metric]
    return fn, response_method or default_response


def make_batch_scorer(
    batch: ArrayLike,
    metric: str | Callable[..., float] = "roc_auc",
    *,
    weights: WeightSpec = "uniform",
    response_method: str | None = None,
    greater_is_better: bool = True,
    pos_class_index: int = 1,
    **metric_kwargs: Any,
) -> Callable[..., float]:
    """Return a sklearn-compatible ``scorer(estimator, X, y)``.

    Parameters
    ----------
    batch : array-like or pandas.Series
        Per-sample batch labels for the **full** dataset. The scorer
        slices this by ``y.index`` on each fold; pass a
        :class:`pandas.Series` indexed by the sample IDs used by ``X``
        and ``y`` for safe alignment.
    metric : str or callable, default='roc_auc'
        Registered alias (``'roc_auc'``, ``'average_precision'``,
        ``'balanced_accuracy'``, ``'mcc'`` / ``'matthews_corrcoef'``,
        ``'f1'``, ``'precision'``, ``'recall'``), or a callable with the
        same signature as the ``batch_*_score`` functions
        (``(y_true, y_pred_or_score, *, batch, weights, **kw)``).
    weights : str or mapping or array, default='uniform'
        Per-batch aggregation weighting (see module docstring).
    response_method : {'predict_proba', 'predict', 'decision_function'}, optional
        How the scorer should produce predictions from the estimator.
        Defaults to the metric's natural choice (``'predict_proba'`` for
        AUROC / AP, ``'predict'`` for the rest). Required when ``metric``
        is a callable.
    greater_is_better : bool, default=True
        If False, the returned scorer negates the metric so larger is
        always better (sklearn convention).
    pos_class_index : int, default=1
        Column index in ``predict_proba(...)`` to use as the positive
        score when ``response_method='predict_proba'``.
    **metric_kwargs
        Forwarded verbatim to the underlying metric (e.g.
        ``average='macro'``, ``pos_label=0``).

    Returns
    -------
    scorer : callable
        ``scorer(estimator, X, y)`` -> float, signed by
        ``greater_is_better``.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> from maldibatchkit import AutoCorrector
    >>> from maldibatchkit.metrics import make_batch_scorer
    >>>
    >>> scorer = make_batch_scorer(batch=batch, metric='roc_auc',
    ...                            weights='uniform')
    >>> grid = GridSearchCV(
    ...     Pipeline([
    ...         ('correct', AutoCorrector(batch=batch,
    ...                                   discrete_covariates=species)),
    ...         ('clf', LogisticRegression()),
    ...     ]),
    ...     param_grid={'correct__method': ['noop', 'combat-fortin', 'harmony']},
    ...     scoring=scorer,
    ... )  # doctest: +SKIP
    """
    metric_fn, response = _resolve_metric(metric, response_method)
    if response not in {"predict_proba", "predict", "decision_function"}:
        raise ValueError(
            f"response_method must be 'predict_proba', 'predict', or "
            f"'decision_function'; got {response!r}."
        )

    if isinstance(batch, pd.Series):
        batch_lookup = batch
    else:
        batch_lookup = pd.Series(np.asarray(batch))

    def _align_batch(y: Any, X: Any) -> np.ndarray:
        if isinstance(y, pd.Series | pd.DataFrame):
            idx = y.index
        elif isinstance(X, pd.DataFrame | pd.Series):
            idx = X.index
        else:
            n = len(np.asarray(y))
            warnings.warn(
                "`y` is not a pandas object; aligning `batch` positionally on "
                f"the first {n} entries of the captured batch Series. Pass `y` "
                "as a pandas Series indexed by sample IDs for safe CV alignment.",
                stacklevel=2,
            )
            return np.asarray(batch_lookup)[:n]
        try:
            return batch_lookup.loc[idx].to_numpy()
        except KeyError as exc:
            raise ValueError(
                "y's index does not align with `batch`'s index. Index both by "
                "the same sample IDs."
            ) from exc

    def _scorer(estimator: Any, X: Any, y: Any, sample_weight: Any = None) -> float:
        batch_fold = _align_batch(y, X)
        if response == "predict_proba":
            proba = estimator.predict_proba(X)
            proba_arr = np.asarray(proba)
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                pred = proba_arr[:, pos_class_index]
            else:
                pred = proba_arr.reshape(-1)
        elif response == "predict":
            pred = estimator.predict(X)
        else:
            pred = estimator.decision_function(X)
        score = metric_fn(
            np.asarray(y),
            np.asarray(pred),
            batch=batch_fold,
            weights=weights,
            **metric_kwargs,
        )
        return float(score) if greater_is_better else -float(score)

    _scorer._batch = batch_lookup  # type: ignore[attr-defined]
    _scorer._metric = metric_fn  # type: ignore[attr-defined]
    _scorer._weights = weights  # type: ignore[attr-defined]
    return _scorer
