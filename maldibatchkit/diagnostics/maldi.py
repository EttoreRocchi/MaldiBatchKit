"""MALDI-specific batch diagnostics.

These functions operate directly on binned-intensity matrices (rows =
samples, columns = m/z bins) and produce tidy per-batch summaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .._utils import ArrayLike, _resolve_mz_axis

__all__ = [
    "peak_position_drift",
    "per_batch_spectrum_count",
    "tic_cov_per_batch",
]


def _to_dataframe(X: ArrayLike) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    return pd.DataFrame(arr)


def per_batch_spectrum_count(batch: ArrayLike) -> pd.Series:
    """Return the number of spectra per batch as a sorted Series."""
    s = pd.Series(np.asarray(batch)).value_counts().sort_index()
    s.name = "n_spectra"
    s.index.name = "batch"
    return s


def tic_cov_per_batch(X: ArrayLike, batch: ArrayLike) -> pd.Series:
    """Per-batch coefficient of variation of the Total Ion Count.

    High CoV within a batch suggests substantial within-batch intensity
    fluctuation, which in MALDI-TOF tends to be driven by acquisition
    issues rather than biology.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binned intensities.
    batch : array-like of shape (n_samples,)
        Batch labels.

    Returns
    -------
    pd.Series
        CoV (``std/mean``) of TIC for each batch, indexed by batch level.
    """
    df = _to_dataframe(X)
    tic = df.sum(axis=1).to_numpy()
    b = np.asarray(batch)
    out: dict[Any, float] = {}
    for lvl in np.unique(b):
        vals = tic[b == lvl]
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        out[lvl] = std / mean if mean > 0 else float("nan")
    s = pd.Series(out).sort_index()
    s.name = "tic_cov"
    s.index.name = "batch"
    return s


def peak_position_drift(
    X: ArrayLike,
    batch: ArrayLike,
    *,
    mz_values: ArrayLike | None = None,
    top_k: int = 50,
) -> pd.DataFrame:
    """Per-batch peak-position drift relative to a global reference.

    For each batch we compute its median spectrum, identify the top
    ``top_k`` peaks of the **global** median spectrum, and locate the
    nearest local maximum of the per-batch median to each of those
    global peaks. The returned table summarises the distribution of
    (signed) position shifts per batch.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binned intensities.
    batch : array-like of shape (n_samples,)
        Batch labels.
    mz_values : array-like of shape (n_features,), optional
        m/z values for the columns of ``X``. If None, integer column
        positions are used, and the returned ``mean_delta_mz`` column is
        in bin units.
    top_k : int, default=50
        Number of global peaks to track.

    Returns
    -------
    pd.DataFrame
        One row per batch, with columns
        ``mean_delta_mz``, ``median_delta_mz``, ``max_abs_delta_mz``.
    """
    df = _to_dataframe(X).astype(float)
    b = np.asarray(batch)
    mz, _ = _resolve_mz_axis(df, mz_values)

    global_median = df.median(axis=0).to_numpy()
    # Local maxima of the global median
    peaks_idx = _local_maxima(global_median)
    if peaks_idx.size == 0:
        # Degenerate case: flat / monotonic spectrum
        return pd.DataFrame(
            {
                "mean_delta_mz": [],
                "median_delta_mz": [],
                "max_abs_delta_mz": [],
            },
        )
    peak_intensities = global_median[peaks_idx]
    order = np.argsort(-peak_intensities)[:top_k]
    tracked = peaks_idx[order]

    records = []
    for lvl in np.unique(b):
        batch_median = df.loc[b == lvl].median(axis=0).to_numpy()
        batch_peaks = _local_maxima(batch_median)
        if batch_peaks.size == 0:
            records.append(
                {
                    "batch": lvl,
                    "mean_delta_mz": float("nan"),
                    "median_delta_mz": float("nan"),
                    "max_abs_delta_mz": float("nan"),
                }
            )
            continue
        deltas = []
        for p in tracked:
            nearest = batch_peaks[int(np.argmin(np.abs(batch_peaks - p)))]
            deltas.append(mz[nearest] - mz[p])
        deltas_arr = np.asarray(deltas)
        records.append(
            {
                "batch": lvl,
                "mean_delta_mz": float(deltas_arr.mean()),
                "median_delta_mz": float(np.median(deltas_arr)),
                "max_abs_delta_mz": float(np.max(np.abs(deltas_arr))),
            }
        )
    out = pd.DataFrame(records).set_index("batch")
    out = out.sort_index()
    return out


def _local_maxima(x: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Return indices of strict local maxima of a 1-D array."""
    if x.size < 3:
        return np.asarray([], dtype=np.int64)
    greater_prev = x[1:-1] > x[:-2]
    greater_next = x[1:-1] > x[2:]
    return np.where(greater_prev & greater_next)[0] + 1
