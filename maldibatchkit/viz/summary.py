"""Grid bar plot of before/after diagnostic metrics.

Different diagnostic metrics live on very different scales
(silhouette_batch ~ 0, LISI ~ 1-n_batches, kBET mean chi-square in the
tens to thousands...), so sharing a single y-axis makes the smaller
metrics unreadable.  We render one subplot per metric, each with its
own y-axis.
"""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot_diagnostic_summary"]


_BAR_BEFORE = "#b0b0b0"
_BAR_AFTER = "#2b8cbe"


def plot_diagnostic_summary(
    report_df: pd.DataFrame,
    *,
    scope: str | Iterable[str] = "overall",
    ncols: int | None = None,
    figsize_per_plot: tuple[float, float] = (3.2, 3.0),
    axes: Any | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot before/after diagnostic values, one subplot per metric.

    Parameters
    ----------
    report_df : pd.DataFrame
        Output of :func:`maldibatchkit.diagnostics.diagnostic_report`.
    scope : str or iterable of str, default='overall'
        Which slice(s) of the report to plot. Pass a single scope name
        (``"overall"``, ``"batch_00"``, ...) to render one pair of bars
        per metric, or a list (``["batch_00", "batch_01"]``) to group
        bars by scope inside each metric's subplot.
    ncols : int, optional
        Number of subplot columns. Defaults to ``min(n_metrics, 4)``.
    figsize_per_plot : (float, float), default=(3.2, 3.0)
        Width, height of each metric's subplot (inches). The returned
        figure has size ``(ncols * w, nrows * h)``.
    axes : array of matplotlib Axes, optional
        Pre-built axes grid. Must have at least ``n_metrics`` entries
        when flattened.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
        Flattened array of the axes actually used. Unused slots from a
        non-rectangular grid are turned off.
    """
    required = {"metric", "scope", "value_before", "value_after"}
    missing = required - set(report_df.columns)
    if missing:
        raise ValueError(f"report_df is missing required columns: {sorted(missing)}.")

    scopes = [scope] if isinstance(scope, str) else list(scope)
    if not scopes:
        raise ValueError("`scope` must name at least one scope.")

    df = report_df[report_df["scope"].isin(scopes)].copy()
    if df.empty:
        raise ValueError(
            f"No rows in report_df with scope in {scopes!r}. "
            f"Available scopes: {sorted(report_df['scope'].unique())}."
        )

    # Preserve the metric order as it appears in the report
    metrics: list[str] = []
    for m in df["metric"].tolist():
        if m not in metrics:
            metrics.append(m)

    n_metrics = len(metrics)
    if ncols is None:
        ncols = min(n_metrics, 4)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n_metrics / ncols))

    if axes is None:
        fig, ax_grid = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * figsize_per_plot[0], nrows * figsize_per_plot[1]),
        )
        ax_flat = np.atleast_1d(ax_grid).ravel()
    else:
        ax_flat = np.atleast_1d(np.asarray(axes)).ravel()
        if ax_flat.size < n_metrics:
            raise ValueError(
                f"Provided axes has {ax_flat.size} slots but {n_metrics} "
                f"metrics need plotting."
            )
        fig = ax_flat[0].figure

    used_axes = []
    for i, metric in enumerate(metrics):
        ax = ax_flat[i]
        sub = df[df["metric"] == metric]
        scope_order = [s for s in scopes if s in sub["scope"].unique()]
        if not scope_order:
            ax.set_axis_off()
            continue
        sub = sub.set_index("scope").loc[scope_order]
        n = len(scope_order)
        xs = np.arange(n)
        width = 0.38
        ax.bar(
            xs - width / 2,
            sub["value_before"].astype(float).to_numpy(),
            width,
            label="before",
            color=_BAR_BEFORE,
        )
        ax.bar(
            xs + width / 2,
            sub["value_after"].astype(float).to_numpy(),
            width,
            label="after",
            color=_BAR_AFTER,
        )
        ax.set_xticks(xs)
        if n > 1:
            ax.set_xticklabels(scope_order, rotation=30, ha="right")
        else:
            ax.set_xticks([])
        ax.set_title(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        # Hairline guide at 0 for signed metrics
        if sub[["value_before", "value_after"]].astype(float).min().min() < 0:
            ax.axhline(0.0, color="k", lw=0.5)
        if i == 0:
            ax.legend(fontsize=8, loc="best")
        used_axes.append(ax)

    # Turn off any leftover subplot slots
    for j in range(n_metrics, ax_flat.size):
        ax_flat[j].set_axis_off()

    title_suffix = scopes[0] if len(scopes) == 1 else f"{len(scopes)} scopes"
    fig.suptitle(
        f"Batch-effect diagnostics ({title_suffix})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig, ax_flat[:n_metrics]
