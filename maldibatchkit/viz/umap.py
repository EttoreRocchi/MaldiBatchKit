"""Side-by-side UMAP before/after batch correction."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .._utils import ArrayLike

__all__ = ["plot_batch_umap"]


def _require_umap():
    try:
        import umap  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "plot_batch_umap requires `umap-learn`. Install the MaldiBatchKit "
            "viz extras with `pip install maldibatchkit[viz]`."
        ) from exc
    return umap


def _embed(X: np.ndarray, random_state: int) -> np.ndarray:
    umap_mod = _require_umap()
    reducer = umap_mod.UMAP(n_components=2, random_state=random_state)
    return np.asarray(reducer.fit_transform(X))


def _to_ndarray(X: ArrayLike) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        return np.asarray(X.to_numpy())
    return np.asarray(X)


def plot_batch_umap(
    before: ArrayLike,
    after: ArrayLike,
    batch: ArrayLike,
    *,
    color_by: str = "batch",
    species: ArrayLike | None = None,
    random_state: int = 42,
    pca_preprocess: int | None = 50,
    ax: tuple[Any, Any] | None = None,
) -> tuple[Any, Any]:
    """Plot UMAP embeddings of ``X`` before and after batch correction.

    Parameters
    ----------
    before : array-like of shape (n_samples, n_features)
        Feature matrix prior to correction.
    after : array-like of shape (n_samples, n_features)
        Feature matrix after correction. Must share ``before``'s shape.
    batch : array-like of shape (n_samples,)
        Batch labels.
    color_by : {'batch', 'species'}, default='batch'
        Which label drives colouring. ``'species'`` requires passing
        ``species`` explicitly.
    species : array-like of shape (n_samples,), optional
        Species labels. Required when ``color_by='species'``.
    random_state : int, default=42
        Seed for UMAP.
    pca_preprocess : int or None, default=50
        If not None, reduce to this many PCs before UMAP. Gives a large
        speed-up on typical MALDI-TOF matrices without hurting the plot
        qualitatively.
    ax : tuple of matplotlib Axes, optional
        Two axes to draw on. If None, a new 1x2 figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure used (or the parent figure of the provided axes).
    axes : tuple of matplotlib.axes.Axes
        The two axes that were drawn on.
    """
    before_arr = _to_ndarray(before).astype(float)
    after_arr = _to_ndarray(after).astype(float)
    if before_arr.shape != after_arr.shape:
        raise ValueError(
            f"before and after must share shape; got {before_arr.shape} vs {after_arr.shape}."
        )
    if color_by == "species" and species is None:
        raise ValueError("color_by='species' requires the `species` argument.")
    labels = np.asarray(batch if color_by == "batch" else species)

    def _maybe_pca(X: np.ndarray) -> np.ndarray:
        if pca_preprocess is None:
            return X
        n = min(pca_preprocess, X.shape[0], X.shape[1])
        if n < 2:
            return X
        return np.asarray(
            PCA(n_components=n, random_state=random_state).fit_transform(X)
        )

    b_emb = _embed(_maybe_pca(before_arr), random_state)
    a_emb = _embed(_maybe_pca(after_arr), random_state)

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    else:
        axes = ax
        fig = axes[0].figure

    for title, emb, ax_ in (("Before", b_emb, axes[0]), ("After", a_emb, axes[1])):
        for lvl in np.unique(labels):
            mask = labels == lvl
            ax_.scatter(emb[mask, 0], emb[mask, 1], s=8, label=str(lvl), alpha=0.7)
        ax_.set_title(f"UMAP - {title}")
        ax_.set_xlabel("UMAP1")
        ax_.set_ylabel("UMAP2")
    axes[0].legend(title=color_by, loc="best", fontsize=7)
    return fig, tuple(axes) if isinstance(axes, (list, np.ndarray)) else axes
