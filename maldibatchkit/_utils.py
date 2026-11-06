"""Internal helpers shared across MaldiBatchKit corrections."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

ArrayLike = pd.DataFrame | pd.Series | npt.NDArray[Any]


def _as_dataframe(X: ArrayLike) -> tuple[pd.DataFrame, bool]:
    """Coerce ``X`` to a ``pd.DataFrame``.

    Returns
    -------
    df : pd.DataFrame
        The input as a DataFrame (copy if necessary so mutation is safe).
    was_dataframe : bool
        ``True`` if the caller passed a DataFrame (so caller can choose to
        return a DataFrame); ``False`` otherwise.
    """
    if isinstance(X, pd.DataFrame):
        return X, True
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr), False


def _subset(obj: ArrayLike | None, idx: pd.Index) -> pd.Series | pd.DataFrame | None:
    """Align a batch/covariate array-like to ``idx``.

    Mirrors ``combatlearn._utils._subset`` so all MaldiBatchKit correctors
    behave the same way: the caller stores ``batch`` at ``__init__`` time
    and we select the rows matching ``X.index`` at ``fit`` / ``transform``.

    When ``obj`` is a pandas object it is sliced with ``.loc[idx]`` (which
    naturally supports train/test splits). A numpy array or list is wrapped
    in a Series/DataFrame using ``idx`` directly - the user is responsible
    for making sure the ordering matches ``X``.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.Series | pd.DataFrame):
        _check_unique_ids(obj.index, "batch/covariate index")
        try:
            return obj.loc[idx]
        except KeyError as exc:
            missing = [k for k in idx if k not in obj.index]
            raise ValueError(
                f"batch/covariate index does not cover all of X.index. "
                f"{len(missing)} missing key(s) (first 3): {missing[:3]}. "
                f"Ensure your batch labels are indexed by the same sample IDs "
                f"used by X, or pass a bare numpy array when X has a default "
                f"RangeIndex."
            ) from exc
    arr = np.asarray(obj)
    if arr.ndim == 1:
        return pd.Series(arr, index=idx)
    return pd.DataFrame(arr, index=idx)


def _check_positional_alignment(obj: ArrayLike | None, n_rows: int, name: str) -> None:
    """Raise if pandas ``obj`` is not indexed by ``RangeIndex(n_rows)``."""
    if isinstance(obj, pd.Series | pd.DataFrame) and not obj.index.equals(
        pd.RangeIndex(n_rows)
    ):
        raise ValueError(
            f"`{name}` is a pandas object but X is an ndarray with no index, "
            f"and `{name}`'s index is not RangeIndex({n_rows}), so its rows "
            f"cannot be matched to X. Pass X as a DataFrame (in a Pipeline, "
            f'call `set_output(transform="pandas")` on the previous steps) or '
            f"pass `{name}` as a numpy array in the same row order as X."
        )


def _warn_unscaled(n_pairs: int, eps: float) -> None:
    """Warn that ``n_pairs`` (batch, feature) scales <= ``eps`` are left unscaled."""
    if n_pairs:
        warnings.warn(
            f"{n_pairs} (batch, feature) pair(s) have a training scale <= "
            f"eps={eps:g} (e.g. a bin that is all zero in a batch); they are "
            f"left unscaled.",
            stacklevel=4,
        )


def _check_labels(labels: ArrayLike, n_rows: int, name: str) -> npt.NDArray[Any]:
    """Return ``labels`` as a 1-D array of ``n_rows`` non-missing values or raise."""
    arr = np.asarray(labels)
    if arr.ndim != 1 or arr.shape[0] != n_rows:
        raise ValueError(
            f"`{name}` must hold one label per row of X ({n_rows}); got shape "
            f"{arr.shape}."
        )
    _check_label_values(arr, name)
    return arr


def _check_label_values(values: ArrayLike, name: str) -> None:
    """Raise on missing labels or labels mixing strings and non-strings."""
    arr = np.asarray(values, dtype=object)
    n_missing = int(pd.isna(arr).sum())
    if n_missing:
        raise ValueError(f"`{name}` contains {n_missing} missing (NaN) label(s).")
    is_str = np.array([isinstance(v, str) for v in arr])
    if is_str.any() and not is_str.all():
        raise ValueError(
            f"`{name}` mixes string and non-string labels (e.g. 1 and '1'); "
            f"use one type for all labels."
        )


def _check_unique_ids(index: pd.Index, what: str) -> None:
    """Raise if ``index`` holds duplicated sample IDs."""
    if index.has_duplicates:
        dup = index[index.duplicated()].unique().tolist()
        raise ValueError(
            f"{what} has duplicated sample ID(s) (first 3: {dup[:3]}); sample "
            f"IDs must be unique."
        )


def _check_min_batch_size(batch: ArrayLike) -> None:
    """Raise if any batch has fewer than 2 samples."""
    counts = pd.Series(np.asarray(batch)).value_counts()
    small = counts[counts < 2]
    if len(small):
        raise ValueError(
            f"Batch {small.index[0]!r} has only {int(small.iloc[0])} sample(s). "
            f"ComBat requires >= 2 samples per batch. Consider merging small "
            f"batches or removing them."
        )


def _ensure_index(X: ArrayLike) -> pd.Index:
    """Return ``X.index`` if ``X`` is a pandas object, else a default RangeIndex."""
    if isinstance(X, pd.DataFrame | pd.Series):
        return X.index
    return pd.RangeIndex(len(X))


def _restore_output(X_corrected: pd.DataFrame, was_dataframe: bool) -> Any:
    """Return a DataFrame if the input was a DataFrame, otherwise a numpy array."""
    if was_dataframe:
        return X_corrected
    return X_corrected.to_numpy()


def _resolve_mz_axis(
    df: pd.DataFrame, mz_values: ArrayLike | None
) -> tuple[npt.NDArray[np.float64], bool]:
    """Resolve the m/z axis for an ``(n_samples, n_features)`` feature matrix.

    Resolution order:

    1. If the caller passed ``mz_values`` explicitly, use it verbatim
       (length is validated against ``df.shape[1]``). Return
       ``(mz, True)``.
    2. Else, try to coerce ``df.columns`` to ``float`` - this is the
       case for matrices coming straight from
       :class:`maldiamrkit.MaldiSet`, whose columns are the binned m/z
       values (string-typed in the current MaldiAMRKit release, but
       numerically convertible, e.g. ``'2000.0'`` ...). Return
       ``(mz, True)`` when it works.
    3. Fall back to ``np.arange(df.shape[1])`` so ndarray inputs still
       produce a plot - but signal to the caller that the axis is
       bin indices rather than m/z, by returning ``(mz, False)``.
    """
    n_features = df.shape[1]
    if mz_values is not None:
        mz = np.asarray(mz_values, dtype=float)
        if mz.ndim != 1:
            raise ValueError(f"mz_values must be 1-D; got shape {mz.shape}.")
        if not np.isfinite(mz).all():
            raise ValueError("mz_values must be finite (no NaN or inf).")
        if mz.shape[0] != n_features:
            raise ValueError(
                f"mz_values has length {mz.shape[0]} but X has {n_features} columns."
            )
        return mz, True
    if isinstance(df.columns, pd.RangeIndex):
        return np.arange(n_features, dtype=float), False
    try:
        mz = np.asarray(df.columns, dtype=float)
    except (TypeError, ValueError):
        return np.arange(n_features, dtype=float), False
    if np.any(np.isnan(mz)):
        return np.arange(n_features, dtype=float), False
    if np.array_equal(mz, np.arange(n_features, dtype=float)):
        return mz, False
    return mz, True
