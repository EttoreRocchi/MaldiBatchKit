"""Tests for the combined diagnostic report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldibatchkit import ComBat
from maldibatchkit.diagnostics import diagnostic_report


def test_report_columns_and_rows(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    corrected = ComBat(batch=batch).fit_transform(X)
    report = diagnostic_report(X, corrected, batch, top_k_peaks=5)
    assert isinstance(report, pd.DataFrame)
    expected_cols = {
        "metric",
        "scope",
        "value_before",
        "value_after",
        "delta",
        "better",
    }
    assert expected_cols.issubset(report.columns)
    overall_metrics = set(report[report["scope"] == "overall"]["metric"])
    assert {
        "silhouette_batch",
        "kbet_acceptance_rate",
        "kbet_mean_chi2",
        "lisi",
    }.issubset(overall_metrics)


def test_report_identity_after_equals_before(big_dataset):
    X, batch = big_dataset["X"], big_dataset["batch"]
    report = diagnostic_report(X, X, batch, top_k_peaks=5)
    identity_rows = report[report["metric"].isin(["silhouette_batch", "lisi"])]
    for _, row in identity_rows.iterrows():
        assert row["value_before"] == row["value_after"]


def test_report_peak_drift_uses_absolute_shifts(shifted_peaks):
    X, batch = shifted_peaks
    report = diagnostic_report(X, X, batch, k=5, lisi_perplexity=5.0)
    row = report[(report["metric"] == "peak_drift_mean") & (report["scope"] == "shift")]
    assert row["value_before"].iloc[0] == pytest.approx(6.0)
    assert row["better"].iloc[0] == "lower"


def test_report_aligns_batch_by_index(tiny_dataset, rng):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    after = X.iloc[rng.permutation(len(X))]
    report = diagnostic_report(X, after, batch, top_k_peaks=5)
    for metric in ("kbet_acceptance_rate", "lisi", "tic_cov"):
        rows = report[report["metric"] == metric]
        np.testing.assert_allclose(rows["value_after"], rows["value_before"])


def test_report_accepts_positional_batch_series(tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    positional = pd.Series(batch.to_numpy())
    got = diagnostic_report(X, X, positional, top_k_peaks=5)
    expected = diagnostic_report(X, X, batch.to_numpy(), top_k_peaks=5)
    pd.testing.assert_frame_equal(got, expected)
