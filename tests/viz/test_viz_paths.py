"""Extra paths through the visualization helpers."""

from __future__ import annotations

import importlib.util

import matplotlib.pyplot as plt
import pytest

from maldibatchkit import ComBat
from maldibatchkit.diagnostics import diagnostic_report
from maldibatchkit.viz import plot_diagnostic_summary, plot_peak_shift

umap_available = importlib.util.find_spec("umap") is not None


def test_plot_peak_shift_ndarray_input(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    fig, ax = plt.subplots()
    f, a = plot_peak_shift(batch.to_numpy(), X.to_numpy(), ax=ax)
    assert a is ax
    plt.close(f)


def test_plot_diagnostic_summary_with_external_axes(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrected = ComBat(batch=batch).fit_transform(X)
    report = diagnostic_report(X, corrected, batch, top_k_peaks=3)
    n_overall = report[report["scope"] == "overall"]["metric"].nunique()
    fig, axes = plt.subplots(1, n_overall)
    f, used = plot_diagnostic_summary(report, axes=axes)
    assert f is fig
    assert len(used) == n_overall
    plt.close(f)


@pytest.mark.skipif(not umap_available, reason="umap-learn not installed")
def test_plot_batch_umap_with_external_axes(tiny_dataset):
    from maldibatchkit.viz import plot_batch_umap

    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    fig, axes = plt.subplots(1, 2)
    f, a = plot_batch_umap(X, X, batch, ax=axes, random_state=0, pca_preprocess=5)
    plt.close(f)


@pytest.mark.skipif(not umap_available, reason="umap-learn not installed")
def test_plot_batch_umap_no_pca(tiny_dataset):
    from maldibatchkit.viz import plot_batch_umap

    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    fig, _ = plot_batch_umap(X, X, batch, random_state=0, pca_preprocess=None)
    import matplotlib.pyplot as plt

    plt.close(fig)
