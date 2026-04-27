r"""Limma-style ``removeBatchEffect`` (Ritchie et al. 2015).

This is a faithful port of the linear-model batch subtraction performed by
``limma::removeBatchEffect``. Unlike ComBat, Limma makes no empirical-Bayes
shrinkage; it simply fits the batch indicators (plus any protected
covariates) by ordinary least squares and subtracts the fitted batch
contribution from the data.

Mathematically, for each feature we solve

.. math::

    X = \\alpha + B \\gamma + C \\beta + \\varepsilon

where :math:`B` holds sum-to-zero contrasts on the batch indicator,
:math:`C` holds the protected covariates (design-of-interest), and we return
:math:`X - B \\hat\\gamma`. The intercept and covariate terms are left
untouched so biological signal encoded in ``design`` is preserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import pandas as pd

from .._base import BaseBatchCorrector
from .._utils import ArrayLike, _subset

__all__ = ["Limma"]


def _sum_to_zero_contrasts(levels: np.ndarray) -> np.ndarray:
    """Return the ``(n_levels, n_levels - 1)`` sum-to-zero contrast matrix."""
    n = len(levels)
    if n < 2:
        return np.zeros((n, 0))
    contrasts = np.eye(n)[:, :-1]
    contrasts[-1, :] = -1.0
    return contrasts


def _design_matrix(
    labels: np.ndarray, levels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sum-to-zero batch design matrix aligned to ``labels``."""
    contrasts = _sum_to_zero_contrasts(levels)
    index_of = {lvl: i for i, lvl in enumerate(levels)}
    unknown = [lvl for lvl in np.unique(labels) if lvl not in index_of]
    if unknown:
        raise ValueError(
            f"Unseen batch level(s) at transform: {unknown}. "
            f"Limma's batch subtraction is undefined for batches not "
            f"present at fit time."
        )
    row_idx = np.asarray([index_of[lvl] for lvl in labels])
    design = contrasts[row_idx]
    return design, contrasts


class Limma(BaseBatchCorrector):
    """Linear-model batch subtraction following ``limma::removeBatchEffect``.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels.
    design : array-like of shape (n_samples, n_covariates), optional
        Protected covariates (the "design of interest" in Limma). Any
        effect explained by ``design`` is left in the output.
    eps : float, default=1e-8
        Ridge-like regularisation added to the diagonal of the normal
        equations to keep things numerically stable on rank-deficient
        designs.

    Attributes
    ----------
    batch_levels_ : np.ndarray
        Batch levels observed at ``fit`` time (in sorted order).
    gamma_ : np.ndarray
        Estimated batch coefficients in the sum-to-zero contrast basis,
        of shape ``(n_contrasts, n_features)``.
    contrasts_ : np.ndarray
        Sum-to-zero contrast matrix used at fit time.

    Notes
    -----
    * Unknown batch levels at ``transform`` raise a ``ValueError``. This
      is intentional: Limma's subtraction is not defined for a batch that
      was not part of the design.
    * The design matrix is augmented with a constant column internally so
      the intercept is absorbed into the OLS fit; only the batch
      coefficients are subtracted on output.

    References
    ----------
    Ritchie, M.E. et al. (2015) "limma powers differential expression
    analyses for RNA-sequencing and microarray studies."
    Nucleic Acids Research 43(7): e47.

    Examples
    --------
    >>> from maldibatchkit import Limma
    >>> corrector = Limma(batch=batches, design=species_dummies)
    >>> X_corrected = corrector.fit_transform(X)
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        design: ArrayLike | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(batch=batch)
        self.design = design
        self.eps = eps

    def _design_at(self, idx: pd.Index) -> np.ndarray | None:
        if self.design is None:
            return None
        sub = _subset(self.design, idx)
        arr = (
            sub.to_numpy(dtype=float)
            if hasattr(sub, "to_numpy")
            else np.asarray(sub, dtype=float)
        )
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray[Any]) -> None:
        levels = np.unique(batch)
        self.batch_levels_ = levels
        batch_design, contrasts = _design_matrix(batch, levels)
        self.contrasts_ = contrasts

        intercept = np.ones((X_df.shape[0], 1))
        cov = self._design_at(X_df.index)
        if cov is not None:
            full_design = np.hstack([intercept, cov, batch_design])
            n_before = 1 + cov.shape[1]
        else:
            full_design = np.hstack([intercept, batch_design])
            n_before = 1

        # Tikhonov-regularised normal equations
        gram = full_design.T @ full_design
        gram += self.eps * np.eye(gram.shape[0])
        rhs = full_design.T @ X_df.to_numpy(dtype=float)
        beta = la.solve(gram, rhs)

        # Batch coefficients live in the last n_contrasts rows
        self.gamma_ = beta[n_before:, :]

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:
        batch_design, _ = _design_matrix(batch, self.batch_levels_)
        correction = batch_design @ self.gamma_
        out = X_df.to_numpy(dtype=float) - correction
        return pd.DataFrame(out, index=X_df.index, columns=X_df.columns)
