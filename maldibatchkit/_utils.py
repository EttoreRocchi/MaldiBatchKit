"""Internal helpers shared across MaldiBatchKit corrections."""

from __future__ import annotations

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
        if mz.shape[0] != n_features:
            raise ValueError(
                f"mz_values has length {mz.shape[0]} but X has {n_features} columns."
            )
        return mz, True
    # pandas.DataFrame(ndarray) yields a default RangeIndex(0, n) -
    # that is positional, not an m/z axis. Treat it as bin indices.
    if isinstance(df.columns, pd.RangeIndex):
        return np.arange(n_features, dtype=float), False
    try:
        mz = np.asarray(df.columns, dtype=float)
    except (TypeError, ValueError):
        return np.arange(n_features, dtype=float), False
    if np.any(np.isnan(mz)):
        return np.arange(n_features, dtype=float), False
    # If the coerced axis is exactly ``[0, 1, ..., n-1]`` it is an
    # integer-labelled DataFrame that happens to parse as floats;
    # the caller almost certainly did not mean those as m/z.
    if np.array_equal(mz, np.arange(n_features, dtype=float)):
        return mz, False
    return mz, True
