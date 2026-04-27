r"""Quality-weighted ComBat.

This module implements a *weighted* empirical-Bayes variant of the
classical Johnson-et-al. (2007) ComBat algorithm. Samples with higher
acquisition quality (typically a signal-to-noise ratio, or any
user-supplied scalar score) contribute proportionally more to the
batch-level moment estimates, and the shrinkage prior on each batch is
scaled by the effective (weighted) batch size rather than the raw
sample count.

Mathematical formulation
------------------------
For feature :math:`g` in batch :math:`i`, let :math:`w_{ij} \\ge 0` be the
per-sample weight (weights are normalised so the weight vector sums to
the number of samples). Write :math:`\\bar w_i = \\sum_j w_{ij}` for the
effective batch size.

*Grand mean and pooled variance* are estimated with weights:

.. math::

    \\hat\\alpha_g &= \\frac{\\sum_{i,j} w_{ij} X_{ij,g}}{\\sum_{i,j} w_{ij}}

    \\hat\\sigma_g^2 &= \\frac{\\sum_{i,j} w_{ij} (X_{ij,g} - \\hat\\alpha_g)^2}
                              {\\sum_{i,j} w_{ij}}

*Per-batch L/S model.* After standardisation
:math:`Z_{ij,g} = (X_{ij,g} - \\hat\\alpha_g) / \\hat\\sigma_g`, the batch-
and-feature effects are estimated with weighted sums:

.. math::

    \\hat\\gamma_{i,g} &= \\frac{\\sum_j w_{ij} Z_{ij,g}}{\\bar w_i}

    \\hat\\delta_{i,g}^2 &= \\frac{\\sum_j w_{ij} (Z_{ij,g} - \\hat\\gamma_{i,g})^2}
                                {\\bar w_i - 1}

(``max(bar_w_i - 1, eps)`` is used to protect against tiny effective
batch sizes.)

*Empirical-Bayes priors.* The Normal-Inverse-Gamma hyperparameters
are fit by method-of-moments on the per-feature
:math:`\\hat\\gamma_{i,g}` (mean, variance) and per-feature
:math:`\\hat\\delta_{i,g}^2` (inverse-gamma :math:`\\alpha, \\beta`),
exactly as in classical ComBat.

*Posteriors.* The posterior mean and variance are then the usual
weighted combination of prior and data, but with the **effective** batch
size :math:`\\bar w_i` replacing :math:`n_i`:

.. math::

    \\gamma^*_{i,g} &= \\frac{\\bar w_i \\bar\\tau_i^2 \\hat\\gamma_{i,g}
                               + \\delta^*_{i,g} \\bar\\gamma_i}
                             {\\bar w_i \\bar\\tau_i^2 + \\delta^*_{i,g}}

    \\delta^{*2}_{i,g} &= \\frac{\\bar\\theta_i + 0.5 \\sum_j w_{ij}
                                 (Z_{ij,g} - \\gamma^*_{i,g})^2}
                               {\\bar w_i / 2 + \\bar\\lambda_i - 1}

with :math:`\\bar\\gamma_i, \\bar\\tau_i^2, \\bar\\lambda_i, \\bar\\theta_i`
the prior hyperparameters fit as in Johnson et al. (2007). The
iterative scheme follows Johnson's original fixed-point updates; only
the summation weights differ.

Rationale
---------
In MALDI-TOF AMR studies, not every spectrum is equally informative:
low-SNR acquisitions or spectra flagged by the lab QC are often kept in
the analysis to avoid discarding hard-won samples, but they should not
drive the per-batch correction. Down-weighting low-quality spectra in
the empirical-Bayes shrinkage keeps the posterior pulled towards the
estimate obtained from the high-quality spectra.

This corrector is **not** a replacement for more general covariate
modelling: like classical Johnson ComBat it does not preserve
biological signal encoded in a covariate. If you need that, either
residualise your covariate first or use
:class:`maldibatchkit.ComBat` with ``method='fortin'``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .._base import BaseBatchCorrector
from .._utils import ArrayLike, _subset

__all__ = ["QualityWeightedComBat"]


def _normalise_weights(
    w: npt.NDArray[np.float64], *, eps: float
) -> npt.NDArray[np.float64]:
    w = np.asarray(w, dtype=float)
    if np.any(w < 0):
        raise ValueError("quality weights must be non-negative.")
    if not np.any(w > 0):
        raise ValueError("at least one quality weight must be positive.")
    # Normalise so weights sum to the number of samples: equal weights
    # should recover the unweighted estimates exactly.
    w = w * (w.size / max(w.sum(), eps))
    return w


def _aprior(gamma_hat: np.ndarray) -> np.ndarray:
    """Method-of-moments hyperparameter for the gamma prior (per batch)."""
    m = gamma_hat.mean(axis=1)
    s2 = gamma_hat.var(axis=1, ddof=1)
    return (2 * s2 + m**2) / np.clip(s2, 1e-12, None)


def _bprior(gamma_hat: np.ndarray) -> np.ndarray:
    m = gamma_hat.mean(axis=1)
    s2 = gamma_hat.var(axis=1, ddof=1)
    return (m * s2 + m**3) / np.clip(s2, 1e-12, None)


def _postmean(
    g_hat: np.ndarray,
    g_bar: float,
    n_bar: float,
    t2: float,
    d_star: np.ndarray,
) -> np.ndarray:
    return (n_bar * t2 * g_hat + d_star * g_bar) / (n_bar * t2 + d_star)


def _postvar(
    sum_sq: np.ndarray,
    n_bar: float,
    a: float,
    b: float,
) -> np.ndarray:
    return (0.5 * sum_sq + b) / (n_bar / 2.0 + a - 1.0)


class QualityWeightedComBat(BaseBatchCorrector):
    r"""Weighted empirical-Bayes extension of Johnson-ComBat.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels.
    quality : array-like of shape (n_samples,)
        Non-negative per-sample quality scores (typically SNR). Higher
        values mean a sample should contribute more to per-batch
        estimation. Internally rescaled to sum to ``n_samples``.
    parametric : bool, default=True
        Use the parametric EB fixed-point iteration (as in Johnson
        2007). ``False`` falls back to the data estimates
        :math:`\\hat\\gamma` / :math:`\\hat\\delta^2` directly - this is
        useful as a sanity check.
    reference_batch : Any, optional
        Batch level to leave unchanged.
    eps : float, default=1e-8
        Numerical jitter.
    max_iter : int, default=50
        Hard cap on the parametric fixed-point iterations.
    tol : float, default=1e-4
        Convergence tolerance on the max absolute change in
        :math:`\\gamma^*, \\delta^*`.

    Attributes
    ----------
    batch_levels_ : np.ndarray
        Batch levels observed at fit time.
    grand_mean_ : np.ndarray
        Weighted feature-wise grand mean (:math:`\\hat\\alpha`).
    pooled_var_ : np.ndarray
        Weighted feature-wise pooled variance (:math:`\\hat\\sigma^2`).
    gamma_star_ : np.ndarray
        Posterior batch means, shape ``(n_batches, n_features)``.
    delta_star_ : np.ndarray
        Posterior batch variances, shape ``(n_batches, n_features)``.
    effective_batch_sizes_ : np.ndarray
        :math:`\\bar w_i` per batch (weighted sample counts).
    n_iter_ : int
        Iterations actually taken by the parametric EB loop.

    Examples
    --------
    >>> from maldibatchkit import QualityWeightedComBat
    >>> corrector = QualityWeightedComBat(batch=batches, quality=snr)
    >>> X_corrected = corrector.fit_transform(X)

    References
    ----------
    Johnson, W.E., Li, C. and Rabinovic, A. (2007). "Adjusting batch
    effects in microarray expression data using empirical Bayes
    methods." *Biostatistics* 8(1): 118-127. - the unweighted
    predecessor of the scheme implemented here.
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        quality: ArrayLike,
        parametric: bool = True,
        reference_batch: Any | None = None,
        eps: float = 1e-8,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(batch=batch)
        self.quality = quality
        self.parametric = parametric
        self.reference_batch = reference_batch
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol

    def _weights_at(self, idx: pd.Index) -> np.ndarray:
        w = _subset(self.quality, idx)
        if isinstance(w, pd.DataFrame):
            if w.shape[1] != 1:
                raise ValueError("quality must be 1-D.")
            w = w.iloc[:, 0]
        if w is None:
            raise ValueError("quality weights must be provided.")
        w = np.asarray(w, dtype=float)
        return _normalise_weights(w, eps=self.eps)

    def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray[Any]) -> None:
        X = X_df.to_numpy(dtype=float)
        n_samples, n_features = X.shape

        w = self._weights_at(X_df.index)  # shape (n,)
        levels = np.unique(batch)
        self.batch_levels_ = levels

        if self.reference_batch is not None and self.reference_batch not in levels:
            raise ValueError(
                f"reference_batch={self.reference_batch!r} not found among "
                f"training batch levels {list(levels)}."
            )
        self.reference_batch_ = self.reference_batch

        W = w.sum()
        self.grand_mean_ = (w[:, None] * X).sum(axis=0) / max(W, self.eps)
        centred = X - self.grand_mean_
        self.pooled_var_ = (w[:, None] * centred**2).sum(axis=0) / max(W, self.eps)
        std = np.sqrt(self.pooled_var_ + self.eps)
        Z = centred / std

        n_batches = len(levels)
        eff = np.zeros(n_batches)
        gamma_hat = np.zeros((n_batches, n_features))
        delta_hat2 = np.zeros((n_batches, n_features))

        for i, lvl in enumerate(levels):
            mask = batch == lvl
            w_i = w[mask]
            w_bar = float(w_i.sum())
            eff[i] = w_bar
            if w_bar < self.eps:
                raise ValueError(
                    f"Effective batch size for {lvl!r} is ~0 "
                    f"(all quality weights are zero in this batch)."
                )
            g_hat = (w_i[:, None] * Z[mask]).sum(axis=0) / w_bar
            gamma_hat[i] = g_hat
            resid = Z[mask] - g_hat
            denom = max(w_bar - 1.0, self.eps)
            delta_hat2[i] = (w_i[:, None] * resid**2).sum(axis=0) / denom

        self.effective_batch_sizes_ = eff

        gamma_bar = gamma_hat.mean(axis=1)
        t2 = gamma_hat.var(axis=1, ddof=1)
        a_prior = _aprior(delta_hat2)
        b_prior = _bprior(delta_hat2)

        if not self.parametric:
            gamma_star = gamma_hat
            delta_star = delta_hat2
            self.n_iter_ = 0
        else:
            gamma_star = gamma_hat.copy()
            delta_star = delta_hat2.copy()
            n_iter = 0
            for _ in range(self.max_iter):
                n_iter += 1
                prev_g = gamma_star.copy()
                prev_d = delta_star.copy()
                for i, lvl in enumerate(levels):
                    mask = batch == lvl
                    w_i = w[mask]
                    w_bar = eff[i]
                    g_new = _postmean(
                        gamma_hat[i], gamma_bar[i], w_bar, t2[i], delta_star[i]
                    )
                    resid = Z[mask] - g_new
                    sum_sq = (w_i[:, None] * resid**2).sum(axis=0)
                    d_new = _postvar(sum_sq, w_bar, a_prior[i], b_prior[i])
                    gamma_star[i] = g_new
                    delta_star[i] = np.clip(d_new, self.eps, None)
                change = max(
                    np.max(np.abs(gamma_star - prev_g)),
                    np.max(np.abs(delta_star - prev_d)),
                )
                if change < self.tol:
                    break
            self.n_iter_ = n_iter

        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:
        X = X_df.to_numpy(dtype=float)
        std = np.sqrt(self.pooled_var_ + self.eps)
        Z = (X - self.grand_mean_) / std
        out = np.empty_like(X)
        level_to_idx = {lvl: i for i, lvl in enumerate(self.batch_levels_)}
        for lvl in np.unique(batch):
            mask = batch == lvl
            if lvl == self.reference_batch_:
                out[mask] = X[mask]
                continue
            if lvl in level_to_idx:
                i = level_to_idx[lvl]
                gamma = self.gamma_star_[i]
                delta = self.delta_star_[i]
            else:
                # Unknown batch at transform time: correct using zero
                # per-batch effect (identity in standardised space).
                gamma = np.zeros(X.shape[1])
                delta = np.ones(X.shape[1])
            z_corr = (Z[mask] - gamma) / np.sqrt(np.clip(delta, self.eps, None))
            out[mask] = z_corr * std + self.grand_mean_
        return pd.DataFrame(out, index=X_df.index, columns=X_df.columns)
