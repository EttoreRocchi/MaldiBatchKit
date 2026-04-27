"""Tests for the combined diagnostic report."""

from __future__ import annotations

import pandas as pd

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
    # Contains the four overall metrics
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
