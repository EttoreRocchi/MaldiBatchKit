"""Generic batch-mixing metrics.

All functions in this module take a feature matrix ``X`` and a vector of
batch labels and return a single scalar (or an array keyed by k for
kBET) summarising how much batch structure remains after correction.

These metrics are intentionally batch-effect-specific: they say nothing
about whether biological signal is preserved. Pair them with supervised
metrics (AMR classifier AUROC, VME, etc.) in any real comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .._utils import ArrayLike

__all__ = [
    "kbet",
    "lisi",
    "lisi_max",
    "lisi_normalized",
    "silhouette_batch",
    "species_preservation",
]


def _to_ndarray(X: ArrayLike) -> npt.NDArray[Any]:
    if hasattr(X, "to_numpy"):
        return np.asarray(X.to_numpy())
    return np.asarray(X)


def silhouette_batch(
    X: ArrayLike, batch: ArrayLike, *, metric: str = "euclidean"
) -> float:
    """Silhouette coefficient using batch labels as clusters.

    Values close to 0 indicate good mixing; values close to 1 indicate
    strong batch separation; values close to -1 indicate batches sitting
    inside each others' clusters (typically an artefact).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    batch : array-like of shape (n_samples,)
        Batch labels.
    metric : str, default='euclidean'
        Distance metric forwarded to ``sklearn.metrics.silhouette_score``.

    Returns
    -------
    float
        Silhouette coefficient, or ``0.0`` if there is fewer than two
        distinct batches (silhouette is undefined in that case).
    """
    Xa = _to_ndarray(X).astype(float)
    batch_arr = np.asarray(batch)
    if len(np.unique(batch_arr)) < 2:
        return 0.0
    return float(silhouette_score(Xa, batch_arr, metric=metric))


def kbet(
    X: ArrayLike,
    batch: ArrayLike,
    *,
    k: int | None = None,
    alpha: float = 0.05,
) -> dict[str, float]:
    """k-nearest-neighbours Batch Effect Test (kBET; Büttner et al. 2019).

    For each sample we compute a chi-square statistic testing whether
    its k-nearest-neighbour batch composition matches the global batch
    frequencies. The reported statistics are the acceptance rate (the
    fraction of samples whose p-value exceeds ``alpha`` - higher is
    better) and the mean chi-square statistic (lower is better).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    batch : array-like of shape (n_samples,)
        Batch labels.
    k : int, optional
        Number of nearest neighbours. Defaults to
        ``max(10, int(0.1 * n_samples))``.
    alpha : float, default=0.05
        Significance threshold used to compute the acceptance rate.

    Returns
    -------
    dict
        ``{"acceptance_rate": float, "mean_chi2": float, "k": int}``.
    """
    Xa = _to_ndarray(X).astype(float)
    batch_arr = np.asarray(batch)
    n_samples = Xa.shape[0]
    levels, counts = np.unique(batch_arr, return_counts=True)
    n_batches = len(levels)
    if n_batches < 2:
        return {"acceptance_rate": 1.0, "mean_chi2": 0.0, "k": int(k or 0)}
    if k is None:
        k = max(10, int(0.1 * n_samples))
    k = min(k, n_samples - 1)
    global_freq = counts / counts.sum()
    level_to_idx = {lvl: i for i, lvl in enumerate(levels)}

    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xa)
    _, neigh_idx = nn.kneighbors(Xa)
    neigh_idx = neigh_idx[:, 1:]

    p_values = np.empty(n_samples)
    chi2_stats = np.empty(n_samples)
    for s in range(n_samples):
        local_counts = np.zeros(n_batches)
        for j in neigh_idx[s]:
            local_counts[level_to_idx[batch_arr[j]]] += 1
        expected = k * global_freq
        chi2_stat = np.sum(
            (local_counts - expected) ** 2 / np.clip(expected, 1e-9, None)
        )
        chi2_stats[s] = chi2_stat
        p_values[s] = float(1.0 - chi2.cdf(chi2_stat, df=n_batches - 1))

    return {
        "acceptance_rate": float(np.mean(p_values > alpha)),
        "mean_chi2": float(chi2_stats.mean()),
        "k": int(k),
    }


def lisi(
    X: ArrayLike,
    batch: ArrayLike,
    *,
    perplexity: float = 30.0,
) -> float:
    """Local Inverse Simpson's Index for batch mixing.

    LISI is the effective number of batches represented in each sample's
    local neighbourhood (Gaussian-kernel weighted to the requested
    perplexity). The returned value is the median LISI across samples.
    It lies in ``[1, n_batches]``; values close to ``n_batches`` indicate
    strong mixing, values close to 1 indicate batch-segregated
    neighbourhoods.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    batch : array-like of shape (n_samples,)
        Batch labels.
    perplexity : float, default=30.0
        Target perplexity of the Gaussian kernel - matches the
        convention used in Korsunsky et al. (2019) and combatlearn.

    Returns
    -------
    float
        Median LISI across samples.
    """
    Xa = _to_ndarray(X).astype(float)
    batch_arr = np.asarray(batch)
    n_samples = Xa.shape[0]
    levels = np.unique(batch_arr)
    n_batches = len(levels)
    if n_batches < 2:
        return 1.0
    level_to_idx = {lvl: i for i, lvl in enumerate(levels)}
    k = int(min(max(3 * perplexity, 10), n_samples - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xa)
    distances, neigh_idx = nn.kneighbors(Xa)
    distances = distances[:, 1:]
    neigh_idx = neigh_idx[:, 1:]

    target = np.log(perplexity)
    lisi_values = np.empty(n_samples)
    for s in range(n_samples):
        d = distances[s]
        # Binary search for beta matching target perplexity
        lo, hi = 1e-10, 1e10
        beta = 1.0
        for _ in range(50):
            weights = np.exp(-d * beta)
            Z = max(weights.sum(), 1e-12)
            h = np.log(Z) + beta * np.sum(d * weights) / Z
            if h > target:
                lo = beta
                beta = beta * 2 if hi == 1e10 else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
            if abs(h - target) < 1e-5:
                break
        weights = np.exp(-d * beta)
        weights /= max(weights.sum(), 1e-12)
        probs = np.zeros(n_batches)
        for w, j in zip(weights, neigh_idx[s], strict=True):
            probs[level_to_idx[batch_arr[j]]] += w
        simpson = float(np.sum(probs**2))
        lisi_values[s] = 1.0 / max(simpson, 1e-12)
    return float(np.median(lisi_values))


def lisi_max(labels: ArrayLike) -> float:
    r"""Theoretical maximum LISI under perfect mixing.

    LISI is the inverse Simpson index of the local label distribution
    in a sample's neighbourhood, so under *perfect* mixing (every
    neighbourhood matches the global label frequencies) it equals the
    inverse Simpson index of the global frequencies:

    .. math::

        \text{lisi}_{\max} = \frac{1}{\sum_i p_i^2}

    where :math:`p_i` is the global fraction of label :math:`i`. For
    perfectly balanced designs this equals the number of unique labels;
    for **imbalanced** designs (e.g. DRIAMS-A dominating in a multi-site
    cohort) the ceiling is strictly less than ``n_levels``, which is
    why raw LISI numbers can mislead readers who expect them to
    approach the level count.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Label vector (e.g. batch or species).

    Returns
    -------
    float
        Inverse Simpson index of the global label frequencies, in the
        range ``[1, n_levels]``. Returns ``1.0`` when the input has
        zero or one unique level (LISI is degenerate in that case).
    """
    arr = np.asarray(labels)
    if arr.size == 0:
        return 1.0
    _, counts = np.unique(arr, return_counts=True)
    if len(counts) < 2:
        return 1.0
    p = counts / counts.sum()
    return float(1.0 / float((p**2).sum()))


def lisi_normalized(
    X: ArrayLike, labels: ArrayLike, *, perplexity: float = 30.0
) -> float:
    """LISI divided by its theoretical maximum.

    Reports how close the local mixing is to *perfect* mixing given
    the global label frequencies, on a ``[0, 1]`` scale where higher
    means better mixed. This is the right way to compare LISI across
    cohorts with different label-imbalance profiles -- raw LISI is
    bounded by ``lisi_max(labels)`` (the inverse Simpson index of the
    global label frequencies), not by the number of unique labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Label vector (typically batch).
    perplexity : float, default=30.0
        Forwarded to :func:`lisi`.

    Returns
    -------
    float
        ``lisi(X, labels) / lisi_max(labels)``, or ``nan`` when there
        are fewer than two unique labels (normalisation is undefined).
    """
    mx = lisi_max(labels)
    if mx <= 1.0:
        return float("nan")
    return float(lisi(X, labels, perplexity=perplexity) / mx)


def species_preservation(
    X: ArrayLike, species: ArrayLike, *, perplexity: float = 30.0
) -> float:
    r"""Score on ``[0, 1]`` where 1 = species perfectly preserved, 0 = blurred.

    The ideal post-correction LISI for *species* (or any biological
    grouping you want to keep apart) is :math:`1` -- each
    neighbourhood is dominated by one species. The *worst* value under
    perfect mixing is :math:`\text{lisi}_{\max}(\text{species})`. We
    rescale the gap to ``[0, 1]``:

    .. math::

        \text{species\_preservation}
            = \frac{\text{lisi}_{\max} - \text{lisi}(X, \text{species})}
                   {\text{lisi}_{\max} - 1}

    so the metric reads naturally as "fraction of biological structure
    retained" and is invariant to species-imbalance (unlike raw LISI).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    species : array-like of shape (n_samples,)
        Species (or other biology) labels.
    perplexity : float, default=30.0
        Forwarded to :func:`lisi`.

    Returns
    -------
    float
        Preservation score in ``[0, 1]`` (``1.0`` = perfectly preserved,
        ``0.0`` = blurred at the imbalance ceiling). Returns ``nan``
        when fewer than two unique species are present.
    """
    mx = lisi_max(species)
    if mx <= 1.0:
        return float("nan")
    raw = lisi(X, species, perplexity=perplexity)
    return float((mx - raw) / (mx - 1.0))
