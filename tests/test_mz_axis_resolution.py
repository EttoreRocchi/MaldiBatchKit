"""Tests for the shared m/z axis auto-resolution.

The helper in :mod:`maldibatchkit._utils` lets
:func:`maldibatchkit.viz.plot_peak_shift` and
:func:`maldibatchkit.diagnostics.peak_position_drift` auto-pick the m/z
axis off a DataFrame's column labels when the caller forgets
``mz_values=``.  These tests pin the resolution order (explicit >
numeric-convertible string columns > plain ndarray / RangeIndex).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from maldibatchkit._utils import _resolve_mz_axis  # noqa: E402
from maldibatchkit.diagnostics import peak_position_drift  # noqa: E402
from maldibatchkit.viz import plot_peak_shift  # noqa: E402


def test_resolve_mz_explicit_values_win():
    df = pd.DataFrame(np.zeros((3, 4)), columns=["2000", "2003", "2006", "2009"])
    mz, is_real = _resolve_mz_axis(df, [10, 20, 30, 40])
    assert is_real
    np.testing.assert_array_equal(mz, [10.0, 20.0, 30.0, 40.0])


def test_resolve_mz_explicit_length_mismatch_raises():
    df = pd.DataFrame(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="length"):
        _resolve_mz_axis(df, [10, 20, 30])


def test_resolve_mz_from_numeric_string_columns():
    df = pd.DataFrame(
        np.zeros((3, 4)),
        columns=["2000.0", "2003.0", "2006.0", "2009.0"],
    )
    mz, is_real = _resolve_mz_axis(df, None)
    assert is_real
    np.testing.assert_array_equal(mz, [2000.0, 2003.0, 2006.0, 2009.0])


def test_resolve_mz_from_float_columns():
    df = pd.DataFrame(np.zeros((3, 4)), columns=[2000.0, 2003.0, 2006.0, 2009.0])
    mz, is_real = _resolve_mz_axis(df, None)
    assert is_real
    np.testing.assert_array_equal(mz, [2000.0, 2003.0, 2006.0, 2009.0])


def test_resolve_mz_rangeindex_is_bin_index():
    df = pd.DataFrame(np.zeros((3, 4)))  # RangeIndex(0, 4)
    mz, is_real = _resolve_mz_axis(df, None)
    assert not is_real
    np.testing.assert_array_equal(mz, [0.0, 1.0, 2.0, 3.0])


def test_resolve_mz_zero_to_n_minus_one_is_bin_index():
    # An explicit Int64Index [0, 1, 2, 3] should also be treated as positional.
    df = pd.DataFrame(np.zeros((3, 4)), columns=pd.Index([0, 1, 2, 3]))
    mz, is_real = _resolve_mz_axis(df, None)
    assert not is_real


def test_resolve_mz_non_numeric_columns_fall_back():
    df = pd.DataFrame(np.zeros((3, 4)), columns=["a", "b", "c", "d"])
    mz, is_real = _resolve_mz_axis(df, None)
    assert not is_real
    np.testing.assert_array_equal(mz, [0.0, 1.0, 2.0, 3.0])


def _tiny_df(rng):
    X = pd.DataFrame(
        rng.standard_normal((20, 6)) + 1.0,
        columns=["2000.0", "2003.0", "2006.0", "2009.0", "2012.0", "2015.0"],
    )
    batch = pd.Series(["a"] * 10 + ["b"] * 10, index=X.index)
    return X, batch


def test_plot_peak_shift_auto_reads_maldiset_columns():
    rng = np.random.default_rng(0)
    X, batch = _tiny_df(rng)
    _, ax = plot_peak_shift(batch, X)
    assert ax.get_xlabel() == "m/z"
    # x-data of the first line (the reference) should match the float-cast columns.
    np.testing.assert_array_equal(
        ax.lines[0].get_xdata(),
        np.asarray([float(c) for c in X.columns]),
    )


def test_plot_peak_shift_falls_back_to_bin_index_on_ndarray():
    rng = np.random.default_rng(0)
    X, batch = _tiny_df(rng)
    _, ax = plot_peak_shift(batch, X.to_numpy())
    assert ax.get_xlabel() == "bin index"


def test_plot_peak_shift_explicit_overrides_auto():
    rng = np.random.default_rng(0)
    X, batch = _tiny_df(rng)
    explicit = np.array([100, 200, 300, 400, 500, 600])
    _, ax = plot_peak_shift(batch, X, mz_values=explicit)
    assert ax.get_xlabel() == "m/z"
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), explicit)


def test_peak_position_drift_auto_reads_maldiset_columns():
    # Inject a known peak in column 3 (m/z 2009) so deltas are computed in Da.
    rng = np.random.default_rng(0)
    X, batch = _tiny_df(rng)
    X.iloc[:, 3] += 10
    out = peak_position_drift(X, batch, top_k=1)
    # Deltas should be in Da (spacing = 3), not in bin counts (spacing = 1).
    max_abs = out["max_abs_delta_mz"].abs().max()
    assert max_abs == 0.0 or max_abs % 3 == 0
