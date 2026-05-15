"""Tests for matplotlib-based plotting helpers (run headless)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import ComBat
from maldibatchkit.diagnostics import diagnostic_report
from maldibatchkit.viz import plot_diagnostic_summary, plot_peak_shift

umap_available = importlib.util.find_spec("umap") is not None


def test_plot_peak_shift_returns_axes(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    fig, ax = plot_peak_shift(batch, X)
    assert fig is not None
    assert ax is not None


def test_plot_peak_shift_with_mz_and_explicit_reference(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    mz = np.linspace(2000, 20000, X.shape[1])
    ref = X.median(axis=0)
    fig, ax = plot_peak_shift(batch, X, reference=ref, mz_values=mz, max_batches=None)
    assert ax.get_xlabel() == "m/z"


def test_plot_diagnostic_summary(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrected = ComBat(batch=batch).fit_transform(X)
    report = diagnostic_report(X, corrected, batch, top_k_peaks=3)
    fig, axes = plot_diagnostic_summary(report)
    assert fig is not None
    # Four 'overall' metrics => four subplot axes
    overall_metrics = report[report["scope"] == "overall"]["metric"].nunique()
    assert len(axes) == overall_metrics


def test_plot_diagnostic_summary_missing_scope(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrected = ComBat(batch=batch).fit_transform(X)
    report = diagnostic_report(X, corrected, batch, top_k_peaks=3)
    with pytest.raises(ValueError, match="No rows"):
        plot_diagnostic_summary(report, scope="nope")


def test_plot_diagnostic_summary_list_of_scopes(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrected = ComBat(batch=batch).fit_transform(X)
    report = diagnostic_report(X, corrected, batch, top_k_peaks=3)
    batches = batch.unique().tolist()
    fig, axes = plot_diagnostic_summary(report, scope=batches)
    assert fig is not None
    # per-batch metrics: tic_cov, peak_drift_mean, n_spectra = 3 subplots
    per_batch_metrics = report[report["scope"].isin(batches)]["metric"].nunique()
    assert len(axes) == per_batch_metrics


def test_plot_diagnostic_summary_empty_scope_list(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrected = ComBat(batch=batch).fit_transform(X)
    report = diagnostic_report(X, corrected, batch, top_k_peaks=3)
    with pytest.raises(ValueError, match="at least one"):
        plot_diagnostic_summary(report, scope=[])


def test_plot_diagnostic_summary_with_external_axes_grid(tiny_dataset):
    import matplotlib.pyplot as plt

    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrected = ComBat(batch=batch).fit_transform(X)
    report = diagnostic_report(X, corrected, batch, top_k_peaks=3)
    n_overall = report[report["scope"] == "overall"]["metric"].nunique()
    fig, axes = plt.subplots(1, n_overall)
    f, _ = plot_diagnostic_summary(report, axes=axes)
    assert f is fig
    plt.close(fig)


def test_plot_diagnostic_summary_external_axes_too_few(tiny_dataset):
    import matplotlib.pyplot as plt

    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrected = ComBat(batch=batch).fit_transform(X)
    report = diagnostic_report(X, corrected, batch, top_k_peaks=3)
    fig, ax = plt.subplots(1, 1)
    with pytest.raises(ValueError, match="axes"):
        plot_diagnostic_summary(report, axes=ax)
    plt.close(fig)


def test_plot_diagnostic_summary_missing_columns():
    df = pd.DataFrame({"metric": ["a"], "value_before": [1]})
    with pytest.raises(ValueError, match="missing required columns"):
        plot_diagnostic_summary(df)


@pytest.mark.skipif(not umap_available, reason="umap-learn not installed")
def test_plot_batch_umap(tiny_dataset):
    from maldibatchkit.viz import plot_batch_umap

    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    corrected = ComBat(batch=batch).fit_transform(X)
    fig, axes = plot_batch_umap(X, corrected, batch, random_state=0, pca_preprocess=5)
    assert fig is not None
    assert len(axes) == 2


@pytest.mark.skipif(not umap_available, reason="umap-learn not installed")
def test_plot_batch_umap_color_by_species(tiny_dataset):
    from maldibatchkit.viz import plot_batch_umap

    X, batch, species = (
        tiny_dataset["X"],
        tiny_dataset["batch"],
        tiny_dataset["species"],
    )
    corrected = ComBat(batch=batch).fit_transform(X)
    fig, _ = plot_batch_umap(
        X, corrected, batch, color_by="species", species=species, pca_preprocess=5
    )
    assert fig is not None


@pytest.mark.skipif(not umap_available, reason="umap-learn not installed")
def test_plot_batch_umap_shape_mismatch(tiny_dataset):
    from maldibatchkit.viz import plot_batch_umap

    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    with pytest.raises(ValueError, match="shape"):
        plot_batch_umap(X, X.iloc[:, :3], batch)


@pytest.mark.skipif(not umap_available, reason="umap-learn not installed")
def test_plot_batch_umap_species_without_labels(tiny_dataset):
    from maldibatchkit.viz import plot_batch_umap

    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    with pytest.raises(ValueError, match="color_by='species' requires"):
        plot_batch_umap(X, X, batch, color_by="species")


def test_plot_batch_umap_requires_umap(monkeypatch, tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    import maldibatchkit.viz.umap as umap_mod

    # Force the umap import to fail regardless of whether the extra is present
    def fake_import():
        raise ImportError("fake: umap-learn missing")

    monkeypatch.setattr(umap_mod, "_require_umap", fake_import)
    with pytest.raises(ImportError, match="fake"):
        umap_mod.plot_batch_umap(X, X, batch)
