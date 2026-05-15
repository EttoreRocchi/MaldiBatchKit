"""Identity batch corrector.

:class:`NoOpCorrector` is an *honest baseline*: it implements the
:class:`~maldibatchkit._base.BaseBatchCorrector` contract but returns the
input matrix untouched. It exists so users can put "do nothing" on the
candidate list when comparing correctors with
:class:`~maldibatchkit.corrections.auto.AutoCorrector` +
``sklearn.model_selection.GridSearchCV``, or in
:class:`~maldibatchkit.diagnostics.benchmark.BatchCorrectionBenchmark`.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt
import pandas as pd

from .._base import BaseBatchCorrector
from .._utils import ArrayLike

__all__ = ["NoOpCorrector"]


class NoOpCorrector(BaseBatchCorrector):
    """Pass-through corrector that returns ``X`` unchanged.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels. Kept on the signature so ``NoOpCorrector`` is a
        drop-in for any other corrector, but never used for correction.

    Attributes
    ----------
    feature_names_in_ : np.ndarray
        Feature names captured at ``fit`` time.
    n_features_in_ : int
        Number of features captured at ``fit`` time.

    Examples
    --------
    >>> from maldibatchkit import NoOpCorrector
    >>> corrector = NoOpCorrector(batch=batches)
    >>> X_out = corrector.fit_transform(X)
    >>> (X_out == X).all().all()
    True
    """

    def __init__(self, batch: ArrayLike) -> None:
        super().__init__(batch=batch)

    def _fit_impl(self, X_df: pd.DataFrame, batch: npt.NDArray[Any]) -> None:
        return None

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:
        return X_df.copy()
