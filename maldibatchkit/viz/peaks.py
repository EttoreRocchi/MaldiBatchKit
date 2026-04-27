"""Visualize per-batch peak position drift."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._utils import ArrayLike, _resolve_mz_axis

__all__ = ["plot_peak_shift"]


def _to_dataframe(X: ArrayLike) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))


def plot_peak_shift(
    batches: ArrayLike,
    X: ArrayLike,
    reference: ArrayLike | None = None,
    *,
    mz_values: ArrayLike | None = None,
    ax: Any | None = None,
    max_batches: int | None = 6,
) -> tuple[Any, Any]:
    """Overlay per-batch median spectra against a reference spectrum.

    Parameters
    ----------
    batches : array-like of shape (n_samples,)
        Batch labels.
    X : array-like of shape (n_samples, n_features)
        Binned intensities.
    reference : array-like of shape (n_features,), optional
        Reference spectrum. If None, the global median across all rows
        is used.
    mz_values : array-like of shape (n_features,), optional
        m/z coordinates for the x-axis. Defaults to column positions.
    ax : matplotlib Axes, optional
        Axis to draw on. If None, a new figure is created.
    max_batches : int or None, default=6
        Maximum number of batches to overlay. Oldest ties broken
        alphabetically. Set to None to draw every batch (slow on
        many-batch studies).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    df = _to_dataframe(X).astype(float)
    b = np.asarray(batches)
    mz, mz_is_real = _resolve_mz_axis(df, mz_values)

    if reference is None:
        ref = df.median(axis=0).to_numpy()
    else:
        ref = np.asarray(reference, dtype=float).ravel()

    levels = sorted(set(b.tolist()))
    if max_batches is not None:
        levels = levels[:max_batches]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(mz, ref, color="black", lw=1.3, label="reference", zorder=5)
    cmap = plt.get_cmap("tab10")
    for i, lvl in enumerate(levels):
        med = df.loc[b == lvl].median(axis=0).to_numpy()
        ax.plot(mz, med, lw=0.9, color=cmap(i % 10), label=str(lvl), alpha=0.85)
    ax.set_xlabel("m/z" if mz_is_real else "bin index")
    ax.set_ylabel("intensity (median per batch)")
    ax.legend(title="batch", fontsize=7, loc="best")
    ax.set_title("Per-batch peak-shape comparison")
    return fig, ax
