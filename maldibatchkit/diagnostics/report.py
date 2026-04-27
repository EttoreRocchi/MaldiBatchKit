"""Combined before/after diagnostic report."""

from __future__ import annotations

import pandas as pd

from .._utils import ArrayLike
from .generic import kbet, lisi, silhouette_batch
from .maldi import peak_position_drift, per_batch_spectrum_count, tic_cov_per_batch

__all__ = ["diagnostic_report"]


def diagnostic_report(
    before: ArrayLike,
    after: ArrayLike,
    batch: ArrayLike,
    *,
    mz_values: ArrayLike | None = None,
    k: int | None = None,
    lisi_perplexity: float = 30.0,
    top_k_peaks: int = 50,
) -> pd.DataFrame:
    """Run every diagnostic on a (before, after) pair.

    Parameters
    ----------
    before : array-like of shape (n_samples, n_features)
        Feature matrix prior to batch correction.
    after : array-like of shape (n_samples, n_features)
        Feature matrix after batch correction. Must have the same shape
        as ``before``.
    batch : array-like of shape (n_samples,)
        Batch labels.
    mz_values : array-like, optional
        m/z positions for the feature columns (passed to
        :func:`peak_position_drift`).
    k : int, optional
        Neighbourhood size for kBET.
    lisi_perplexity : float, default=30.0
        Perplexity for LISI.
    top_k_peaks : int, default=50
        Number of peaks tracked for drift.

    Returns
    -------
    pd.DataFrame
        Tidy report with columns ``metric``, ``scope``, ``value_before``,
        ``value_after`` (and ``delta`` where both columns are numeric and
        the metric's improvement direction is well-defined).
    """
    rows: list[dict] = []

    # Generic batch mixing (smaller = better for silhouette / chi2)
    sil_b = silhouette_batch(before, batch)
    sil_a = silhouette_batch(after, batch)
    rows.append(
        {
            "metric": "silhouette_batch",
            "scope": "overall",
            "value_before": sil_b,
            "value_after": sil_a,
            "delta": sil_a - sil_b,
            "better": "lower",
        }
    )

    kbet_b = kbet(before, batch, k=k)
    kbet_a = kbet(after, batch, k=k)
    rows.append(
        {
            "metric": "kbet_acceptance_rate",
            "scope": "overall",
            "value_before": kbet_b["acceptance_rate"],
            "value_after": kbet_a["acceptance_rate"],
            "delta": kbet_a["acceptance_rate"] - kbet_b["acceptance_rate"],
            "better": "higher",
        }
    )
    rows.append(
        {
            "metric": "kbet_mean_chi2",
            "scope": "overall",
            "value_before": kbet_b["mean_chi2"],
            "value_after": kbet_a["mean_chi2"],
            "delta": kbet_a["mean_chi2"] - kbet_b["mean_chi2"],
            "better": "lower",
        }
    )

    lisi_b = lisi(before, batch, perplexity=lisi_perplexity)
    lisi_a = lisi(after, batch, perplexity=lisi_perplexity)
    rows.append(
        {
            "metric": "lisi",
            "scope": "overall",
            "value_before": lisi_b,
            "value_after": lisi_a,
            "delta": lisi_a - lisi_b,
            "better": "higher",
        }
    )

    # TIC CoV per batch
    tic_b = tic_cov_per_batch(before, batch)
    tic_a = tic_cov_per_batch(after, batch)
    for lvl in tic_b.index:
        rows.append(
            {
                "metric": "tic_cov",
                "scope": str(lvl),
                "value_before": float(tic_b.loc[lvl]),
                "value_after": float(tic_a.loc[lvl]),
                "delta": float(tic_a.loc[lvl] - tic_b.loc[lvl]),
                "better": "lower",
            }
        )

    # Peak-position drift per batch (mean |delta_mz|)
    drift_b = peak_position_drift(before, batch, mz_values=mz_values, top_k=top_k_peaks)
    drift_a = peak_position_drift(after, batch, mz_values=mz_values, top_k=top_k_peaks)
    for lvl in drift_b.index:
        rows.append(
            {
                "metric": "peak_drift_mean",
                "scope": str(lvl),
                "value_before": float(drift_b.loc[lvl, "mean_delta_mz"]),
                "value_after": float(drift_a.loc[lvl, "mean_delta_mz"]),
                "delta": float(
                    drift_a.loc[lvl, "mean_delta_mz"]
                    - drift_b.loc[lvl, "mean_delta_mz"]
                ),
                "better": "zero",
            }
        )

    # Spectrum counts (unchanged by correction; still useful in reports)
    counts = per_batch_spectrum_count(batch)
    for lvl, n in counts.items():
        rows.append(
            {
                "metric": "n_spectra",
                "scope": str(lvl),
                "value_before": int(n),
                "value_after": int(n),
                "delta": 0,
                "better": "n/a",
            }
        )

    return pd.DataFrame(rows)
