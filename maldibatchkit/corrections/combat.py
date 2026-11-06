"""ComBat variants and convenience wrappers.

The core ComBat implementation (Johnson 2007, Fortin 2018, Chen 2022 /
CovBat) lives in the :mod:`combatlearn` package. MaldiBatchKit re-exports
the sklearn-compatible ``ComBat`` transformer from ``combatlearn`` and adds
a small MALDI-specific convenience wrapper, :class:`SpeciesAwareComBat`,
which preselects the Fortin variant with ``species`` as a protected
biological covariate.
"""

from __future__ import annotations

import inspect
from typing import Any

from combatlearn import ComBat

from .._utils import ArrayLike, _check_min_batch_size, _ensure_index, _subset

__all__ = ["ComBat", "SpeciesAwareComBat"]

_COMBAT_DEFAULTS = {
    name: p.default
    for name, p in inspect.signature(ComBat.__init__).parameters.items()
    if p.default is not inspect.Parameter.empty
}


class SpeciesAwareComBat(ComBat):
    """ComBat-Fortin preset with species as a protected biological covariate.

    This is a *thin* convenience wrapper: all the work is delegated to
    :class:`combatlearn.ComBat` with ``method='fortin'`` and ``species``
    plugged in as the ``discrete_covariates`` argument. It exists only so
    MALDI users can write ``SpeciesAwareComBat(batch=..., species=...)``
    instead of remembering which keyword corresponds to the covariate slot.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels for each sample.
    species : array-like of shape (n_samples,)
        Species labels for each sample. Passed to ``combatlearn``'s Fortin
        variant as ``discrete_covariates`` so per-species biological
        structure is preserved during correction.
    continuous_covariates : array-like, optional
        Additional continuous covariates to protect.
    parametric : bool, default=True
        Use parametric empirical Bayes.
    mean_only : bool, default=False
        Adjust only the mean (ignore variance).
    reference_batch : str, optional
        Batch level to leave unchanged.
    eps : float, default=1e-8
        Numerical jitter for stability.

    Notes
    -----
    This class is deliberately minimal - it is **not** a new algorithm.
    Calling ``SpeciesAwareComBat(batch=b, species=s).fit_transform(X)`` is
    equivalent to::

        ComBat(batch=b, discrete_covariates=s, method='fortin').fit_transform(X)

    except that a batch with fewer than 2 samples raises ``ValueError``.

    Examples
    --------
    >>> from maldibatchkit import SpeciesAwareComBat
    >>> corrector = SpeciesAwareComBat(batch=batches, species=species)
    >>> X_corrected = corrector.fit_transform(X)
    """

    method = "fortin"

    def __init__(
        self,
        batch: ArrayLike,
        *,
        species: ArrayLike,
        continuous_covariates: ArrayLike | None = None,
        parametric: bool = True,
        mean_only: bool = False,
        reference_batch: str | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.batch = batch
        self.species = species
        self.continuous_covariates = continuous_covariates
        self.parametric = parametric
        self.mean_only = mean_only
        self.reference_batch = reference_batch
        self.eps = eps

    @property
    def discrete_covariates(self) -> ArrayLike:
        """``species``, in the covariate slot combatlearn reads at fit."""
        return self.species

    def __getattr__(self, name: str) -> Any:
        try:
            return _COMBAT_DEFAULTS[name]
        except KeyError:
            raise AttributeError(name) from None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> SpeciesAwareComBat:
        """Fit ComBat-Fortin with ``species`` as the protected covariate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : None
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        self : SpeciesAwareComBat
            Fitted estimator.

        Raises
        ------
        ValueError
            If a batch has fewer than 2 samples.
        """
        _check_min_batch_size(_subset(self.batch, _ensure_index(X)))
        return super().fit(X, y)
