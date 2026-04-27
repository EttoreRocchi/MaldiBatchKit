"""ComBat variants and convenience wrappers.

The core ComBat implementation (Johnson 2007, Fortin 2018, Chen 2022 /
CovBat) lives in the :mod:`combatlearn` package. MaldiBatchKit re-exports
the sklearn-compatible ``ComBat`` transformer from ``combatlearn`` and adds
a small MALDI-specific convenience wrapper, :class:`SpeciesAwareComBat`,
which preselects the Fortin variant with ``species`` as a protected
biological covariate.
"""

from __future__ import annotations

from combatlearn import ComBat

from .._utils import ArrayLike

__all__ = ["ComBat", "SpeciesAwareComBat"]


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
    exactly equivalent to::

        ComBat(batch=b, discrete_covariates=s, method='fortin').fit_transform(X)

    Examples
    --------
    >>> from maldibatchkit import SpeciesAwareComBat
    >>> corrector = SpeciesAwareComBat(batch=batches, species=species)
    >>> X_corrected = corrector.fit_transform(X)
    """

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
        super().__init__(
            batch=batch,
            discrete_covariates=species,
            continuous_covariates=continuous_covariates,
            method="fortin",
            parametric=parametric,
            mean_only=mean_only,
            reference_batch=reference_batch,
            eps=eps,
        )
        self.species = species
