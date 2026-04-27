"""Simple batch-correction baselines.

These are the "obvious" things to try before reaching for ComBat /
Harmony. They are useful as sanity-check baselines in benchmarks, and
they remain well-defined even when per-batch sample counts are tiny.

None of these methods models biological covariates. If you need to
preserve species structure while correcting batch, use
:class:`maldibatchkit.ComBat` (Fortin) or
:class:`maldibatchkit.SpeciesAwareComBat`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .._base import BaseBatchCorrector
from .._utils import ArrayLike

__all__ = ["MedianCentering", "ReferenceScaling", "ZScorePerBatch"]


def _per_batch_reduce(
    X: pd.DataFrame,
    batch: npt.NDArray[Any],
    reducer,
) -> dict[Any, pd.Series]:
    """Apply ``reducer(X_b)`` for each distinct batch label."""
    out: dict[Any, pd.Series] = {}
    for lvl in np.unique(batch):
        mask = batch == lvl
        out[lvl] = reducer(X.loc[mask])
    return out


class MedianCentering(BaseBatchCorrector):
    """Subtract per-batch medians from each feature.

    The fitted parameter is a ``(n_batches, n_features)`` table of
    medians. At ``transform`` time, the median for the batch each sample
    belongs to is subtracted. Batches that were absent at ``fit`` time
    fall back to the grand median learned from training data (with a
    warning); this keeps the transformer defined on unseen folds without
    leaking information from the test set.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels.

    Attributes
    ----------
    batch_medians_ : pd.DataFrame
        Per-batch feature medians learned from the training data
        (index: batch level, columns: feature names).
    grand_median_ : pd.Series
        Feature-wise median across all training rows. Used as a
        fallback for unseen batch levels at transform time.

    Examples
    --------
    >>> from maldibatchkit import MedianCentering
    >>> corrector = MedianCentering(batch=batches)
    >>> X_corrected = corrector.fit_transform(X)
    """

    def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray[Any]) -> None:
        meds = _per_batch_reduce(X_df, batch, lambda df: df.median(axis=0))
        self.batch_medians_ = pd.DataFrame(meds).T
        self.batch_medians_.index.name = "batch"
        self.grand_median_ = X_df.median(axis=0)

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:
        out = X_df.copy().astype(float)
        known = set(self.batch_medians_.index)
        for lvl in np.unique(batch):
            mask = batch == lvl
            offset = (
                self.batch_medians_.loc[lvl].to_numpy()
                if lvl in known
                else self.grand_median_.to_numpy()
            )
            out.loc[mask] = out.loc[mask].to_numpy() - offset
        return out


class ZScorePerBatch(BaseBatchCorrector):
    """Per-batch z-score normalisation.

    For each training batch we learn a mean and standard deviation per
    feature. At ``transform`` time we subtract the mean and divide by
    the standard deviation of the sample's batch (again falling back to
    grand statistics for unseen batches).

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels.
    eps : float, default=1e-8
        Floor applied to the standard deviation to avoid division by
        zero on constant features.

    Attributes
    ----------
    batch_means_ : pd.DataFrame
        Per-batch feature means.
    batch_stds_ : pd.DataFrame
        Per-batch feature standard deviations (floored at ``eps``).
    grand_mean_ : pd.Series
        Feature-wise mean across all training rows.
    grand_std_ : pd.Series
        Feature-wise standard deviation across all training rows.

    Examples
    --------
    >>> from maldibatchkit import ZScorePerBatch
    >>> corrector = ZScorePerBatch(batch=batches)
    >>> X_corrected = corrector.fit_transform(X)
    """

    def __init__(self, batch: ArrayLike, *, eps: float = 1e-8) -> None:
        super().__init__(batch=batch)
        self.eps = eps

    def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray[Any]) -> None:
        means = _per_batch_reduce(X_df, batch, lambda df: df.mean(axis=0))
        stds = _per_batch_reduce(
            X_df, batch, lambda df: df.std(axis=0, ddof=0).clip(lower=self.eps)
        )
        self.batch_means_ = pd.DataFrame(means).T
        self.batch_stds_ = pd.DataFrame(stds).T
        self.batch_means_.index.name = "batch"
        self.batch_stds_.index.name = "batch"
        self.grand_mean_ = X_df.mean(axis=0)
        self.grand_std_ = X_df.std(axis=0, ddof=0).clip(lower=self.eps)

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:
        out = X_df.copy().astype(float)
        known = set(self.batch_means_.index)
        for lvl in np.unique(batch):
            mask = batch == lvl
            if lvl in known:
                mean = self.batch_means_.loc[lvl].to_numpy()
                std = self.batch_stds_.loc[lvl].to_numpy()
            else:
                mean = self.grand_mean_.to_numpy()
                std = self.grand_std_.to_numpy()
            out.loc[mask] = (out.loc[mask].to_numpy() - mean) / std
        return out


class ReferenceScaling(BaseBatchCorrector):
    """Rescale each batch so its per-feature mean matches a reference batch.

    Every non-reference batch is multiplied by
    ``reference_mean / batch_mean`` (feature-wise), which is a simple
    multiplicative drift correction useful for mass-spectrometry
    intensities where batch-specific gain terms are plausible. The
    reference batch itself is left unchanged.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels.
    reference_batch : str, optional
        Batch level to use as the reference. If ``None``, the batch with
        the most training samples is chosen.
    eps : float, default=1e-8
        Floor on the denominator mean to avoid division by zero.

    Attributes
    ----------
    reference_batch_ : Any
        The batch level actually used as reference at ``fit`` time.
    scale_factors_ : pd.DataFrame
        Per-batch feature scale factors
        (``reference_mean / batch_mean``).
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        reference_batch: Any | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(batch=batch)
        self.reference_batch = reference_batch
        self.eps = eps

    def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray[Any]) -> None:
        levels, counts = np.unique(batch, return_counts=True)
        if self.reference_batch is None:
            ref = levels[int(np.argmax(counts))]
        else:
            ref = self.reference_batch
            if ref not in levels:
                raise ValueError(
                    f"reference_batch={ref!r} not found among training "
                    f"batch levels {list(levels)}."
                )
        self.reference_batch_ = ref
        ref_mean = X_df.loc[batch == ref].mean(axis=0)
        scales: dict[Any, pd.Series] = {}
        for lvl in levels:
            if lvl == ref:
                scales[lvl] = pd.Series(np.ones(X_df.shape[1]), index=X_df.columns)
                continue
            lvl_mean = X_df.loc[batch == lvl].mean(axis=0)
            denom = np.where(np.abs(lvl_mean) < self.eps, self.eps, lvl_mean)
            scales[lvl] = pd.Series(ref_mean.to_numpy() / denom, index=X_df.columns)
        self.scale_factors_ = pd.DataFrame(scales).T
        self.scale_factors_.index.name = "batch"

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:
        out = X_df.copy().astype(float)
        known = set(self.scale_factors_.index)
        for lvl in np.unique(batch):
            mask = batch == lvl
            if lvl in known:
                factor = self.scale_factors_.loc[lvl].to_numpy()
            else:
                factor = np.ones(X_df.shape[1])
            out.loc[mask] = out.loc[mask].to_numpy() * factor
        return out
