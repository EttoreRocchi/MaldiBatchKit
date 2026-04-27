"""Tests for the internal helpers in maldibatchkit._utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maldibatchkit._utils import (
    _as_dataframe,
    _ensure_index,
    _restore_output,
    _subset,
)


def test_as_dataframe_preserves_dataframe():
    X = pd.DataFrame([[1.0, 2.0]], index=["a"], columns=["x", "y"])
    out, flag = _as_dataframe(X)
    assert flag is True
    assert out is X


def test_as_dataframe_wraps_ndarray():
    arr = np.arange(6).reshape(3, 2)
    out, flag = _as_dataframe(arr)
    assert flag is False
    assert out.shape == (3, 2)


def test_as_dataframe_wraps_1d_array():
    out, flag = _as_dataframe(np.array([1, 2, 3]))
    assert flag is False
    assert out.shape == (3, 1)


def test_subset_none():
    assert _subset(None, pd.Index([0, 1])) is None


def test_subset_series_by_loc():
    ser = pd.Series([10, 20, 30], index=["a", "b", "c"])
    out = _subset(ser, pd.Index(["b", "c"]))
    assert list(out) == [20, 30]


def test_subset_ndarray_1d():
    out = _subset(np.array([5, 6, 7]), pd.Index(["x", "y", "z"]))
    assert isinstance(out, pd.Series)
    assert list(out.values) == [5, 6, 7]


def test_subset_ndarray_2d():
    out = _subset(np.arange(6).reshape(3, 2), pd.Index(["a", "b", "c"]))
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (3, 2)


def test_subset_series_missing_index_errors():
    ser = pd.Series([10, 20], index=["a", "b"])
    with pytest.raises(ValueError, match="batch/covariate index"):
        _subset(ser, pd.Index(["c"]))


def test_ensure_index_uses_frame_index():
    df = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
    assert list(_ensure_index(df)) == ["a", "b"]


def test_ensure_index_uses_rangeindex_for_array():
    arr = np.zeros((3, 2))
    idx = _ensure_index(arr)
    assert isinstance(idx, pd.RangeIndex)
    assert len(idx) == 3


def test_restore_output_roundtrips_dataframe():
    df = pd.DataFrame([[1, 2]])
    assert _restore_output(df, True) is df


def test_restore_output_returns_ndarray_for_non_dataframe():
    df = pd.DataFrame([[1, 2]])
    out = _restore_output(df, False)
    assert isinstance(out, np.ndarray)
