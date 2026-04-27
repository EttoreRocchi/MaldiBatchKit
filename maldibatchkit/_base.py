"""Shared scaffolding for MaldiBatchKit batch correctors.

Every corrector in this package stores ``batch`` (and any covariates) at
``__init__`` time and aligns the stored arrays to ``X.index`` at
``fit`` / ``transform``. This mirrors the ``combatlearn`` convention and
lets sklearn ``Pipeline`` / cross-validation pass only ``X`` around while
keeping train and test folds correctly labelled.

The contract implemented here avoids the two common data-leakage traps for
batch correction:

1. ``fit`` learns correction parameters **only** from the training rows it
   actually receives; no statistic is derived from rows the user has
   already held back as a test set.
2. ``transform`` never re-fits. Fitted attributes (those ending in ``_``)
   are used verbatim on unseen rows, even when a new batch label appears
   in the test set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ._utils import ArrayLike, _ensure_index, _subset


def _check_array_finite(X: Any) -> None:
    """Run :func:`sklearn.utils.validation.check_array` with finite values.

    The kwarg that controls this check was renamed from
    ``force_all_finite`` (sklearn < 1.6) to ``ensure_all_finite``
    (sklearn >= 1.6). We support both transparently so MaldiBatchKit
    installs cleanly on either.
    """
    try:
        check_array(X, ensure_all_finite=True, dtype="numeric")
    except TypeError:
        check_array(X, force_all_finite=True, dtype="numeric")


class BaseBatchCorrector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Base class for batch correctors that store ``batch`` at construction.

    Subclasses must implement :meth:`_fit_impl` and :meth:`_transform_impl`.
    They should store fitted attributes as ``..._`` members so sklearn's
    ``check_is_fitted`` works out of the box.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels for each sample. Passed once at construction; the
        correct subset is aligned to ``X.index`` on ``fit`` / ``transform``.

    Notes
    -----
    Subclasses typically override ``__init__`` to add method-specific
    hyperparameters but should always call ``super().__init__(batch=batch)``
    and leave ``batch`` stored under ``self.batch`` so
    :meth:`get_params` / :meth:`set_params` work.
    """

    def __init__(self, batch: ArrayLike) -> None:
        self.batch = batch

    def _validate_X(self, X: ArrayLike) -> None:
        """Validate the feature matrix."""
        _check_array_finite(X)

    def _prepare_fit(
        self, X: ArrayLike
    ) -> tuple[pd.DataFrame, npt.NDArray[Any], pd.Index]:
        """Run common ``fit``-time bookkeeping.

        Returns
        -------
        X_df : pd.DataFrame
            ``X`` as a DataFrame with the inferred / preserved index.
        batch_arr : ndarray
            Batch labels aligned to ``X.index``.
        idx : pd.Index
            The shared index used by ``X`` and ``batch``.
        """
        self._validate_X(X)
        idx = _ensure_index(X)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            X_df = X
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object
            )
            X_df = pd.DataFrame(np.asarray(X), index=idx)
        batch_vec = _subset(self.batch, idx)
        if batch_vec is None:
            raise ValueError("`batch` must not be None for a batch corrector.")
        if isinstance(batch_vec, pd.DataFrame):
            if batch_vec.shape[1] != 1:
                raise ValueError(f"`batch` must be 1-D; got shape {batch_vec.shape}.")
            batch_vec = batch_vec.iloc[:, 0]
        batch_ser = pd.Series(batch_vec, index=idx)
        if batch_ser.isna().any():
            raise ValueError(
                f"batch contains {int(batch_ser.isna().sum())} NaN value(s); "
                f"all batch labels must be non-null."
            )
        return X_df, batch_ser.to_numpy(), idx

    def _prepare_transform(
        self, X: ArrayLike
    ) -> tuple[pd.DataFrame, npt.NDArray[Any], pd.Index, bool]:
        """Run common ``transform``-time bookkeeping."""
        check_is_fitted(self)
        self._validate_X(X)
        idx = _ensure_index(X)
        was_df = isinstance(X, pd.DataFrame)
        X_df = X if was_df else pd.DataFrame(np.asarray(X), index=idx)
        batch_vec = _subset(self.batch, idx)
        if isinstance(batch_vec, pd.DataFrame):
            batch_vec = batch_vec.iloc[:, 0]
        batch_ser = pd.Series(batch_vec, index=idx)
        return X_df, batch_ser.to_numpy(), idx, was_df

    def _fit_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _transform_impl(
        self, X_df: pd.DataFrame, batch: npt.NDArray[Any]
    ) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> BaseBatchCorrector:
        """Fit the corrector on the training rows supplied.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : None
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        self : BaseBatchCorrector
            Fitted estimator.
        """
        X_df, batch_arr, _ = self._prepare_fit(X)
        self._fit_impl(X_df, batch_arr)
        self.n_features_in_ = X_df.shape[1]
        return self

    def transform(self, X: ArrayLike) -> Any:
        """Apply the fitted correction to ``X``.

        Returns a :class:`pandas.DataFrame` when the input was a DataFrame,
        or a :class:`numpy.ndarray` otherwise.
        """
        X_df, batch_arr, idx, was_df = self._prepare_transform(X)
        X_corrected = self._transform_impl(X_df, batch_arr)
        X_corrected.index = idx
        X_corrected.columns = X_df.columns
        return X_corrected if was_df else X_corrected.to_numpy()

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> npt.NDArray[Any]:
        """Return the feature names seen during :meth:`fit`."""
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_
