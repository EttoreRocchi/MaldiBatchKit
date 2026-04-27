"""Unit tests for the CLI IO helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import typer

from maldibatchkit._cli_io import (
    _align_frame,
    _align_series,
    load_dataframe_csv,
    load_labels_csv,
    load_matrix,
    load_mz_csv,
    save_matrix,
)


@pytest.fixture
def tiny_df(tmp_path):
    X = pd.DataFrame(np.arange(20).reshape(5, 4), index=[f"s{i}" for i in range(5)])
    batch = pd.Series(["a", "b", "a", "b", "a"], index=X.index, name="batch")
    return X, batch, tmp_path


def test_load_matrix_csv_requires_batch_csv(tiny_df):
    X, _, tmp = tiny_df
    p = tmp / "X.csv"
    X.to_csv(p)
    with pytest.raises(typer.BadParameter):
        load_matrix(p, None)


def test_load_matrix_csv_with_batch_csv(tiny_df):
    X, batch, tmp = tiny_df
    p = tmp / "X.csv"
    bp = tmp / "batch.csv"
    X.to_csv(p)
    batch.to_frame("batch").to_csv(bp)
    X_loaded, batch_loaded, extras = load_matrix(p, bp)
    # CSV round-trip turns integer column labels into strings; check values + shape.
    np.testing.assert_allclose(X_loaded.to_numpy(), X.to_numpy())
    assert list(X_loaded.index) == list(X.index)
    assert list(batch_loaded) == list(batch)
    assert extras == {}


def test_load_matrix_npz_with_bundled_batch(tiny_df):
    X, batch, tmp = tiny_df
    p = tmp / "X.npz"
    np.savez(
        p,
        X=X.to_numpy(),
        columns=np.asarray(X.columns, dtype=object),
        index=np.asarray(X.index, dtype=object),
        batch=batch.to_numpy(),
        extra=np.arange(5),
    )
    X_loaded, batch_loaded, extras = load_matrix(p, None)
    assert X_loaded.shape == X.shape
    assert list(batch_loaded) == list(batch)
    assert "extra" in extras


def test_load_matrix_npz_missing_X_raises(tiny_df):
    _, _, tmp = tiny_df
    p = tmp / "bad.npz"
    np.savez(p, Y=np.zeros(3))
    with pytest.raises(typer.BadParameter, match="'X'"):
        load_matrix(p, None)


def test_load_matrix_npz_without_batch_needs_sidecar(tiny_df):
    X, _, tmp = tiny_df
    p = tmp / "X.npz"
    np.savez(p, X=X.to_numpy())
    with pytest.raises(typer.BadParameter, match="--batch-csv"):
        load_matrix(p, None)


def test_load_matrix_npz_batch_length_mismatch(tiny_df):
    X, _, tmp = tiny_df
    p = tmp / "X.npz"
    np.savez(p, X=X.to_numpy(), batch=np.asarray(["a", "b"], dtype=object))
    with pytest.raises(typer.BadParameter, match="batch length"):
        load_matrix(p, None)


def test_load_labels_csv_rejects_no_data_column(tmp_path):
    p = tmp_path / "labels.csv"
    pd.DataFrame(index=pd.Index(["a", "b"], name="id")).to_csv(p)
    with pytest.raises(typer.BadParameter):
        load_labels_csv(p, pd.Index(["a", "b"]))


def test_load_labels_csv_rejects_multi_column(tmp_path):
    p = tmp_path / "labels.csv"
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=["a", "b"]).to_csv(p)
    with pytest.raises(typer.BadParameter, match="one"):
        load_labels_csv(p, pd.Index(["a", "b"]))


def test_load_labels_csv_positional_fallback(tmp_path):
    p = tmp_path / "labels.csv"
    # Index doesn't match X.index but row count does -> positional align
    pd.DataFrame({"x": [1, 2, 3]}, index=["X", "Y", "Z"]).to_csv(p)
    out = load_labels_csv(p, pd.Index(["a", "b", "c"]))
    assert list(out.index) == ["a", "b", "c"]


def test_load_labels_csv_row_count_mismatch(tmp_path):
    p = tmp_path / "labels.csv"
    pd.DataFrame({"x": [1, 2]}, index=["x", "y"]).to_csv(p)
    with pytest.raises(typer.BadParameter, match="align"):
        load_labels_csv(p, pd.Index(["a", "b", "c"]))


def test_load_dataframe_csv_roundtrip(tmp_path):
    idx = pd.Index([f"s{i}" for i in range(4)])
    df = pd.DataFrame({"age": [20, 30, 40, 50], "site": [0, 1, 0, 1]}, index=idx)
    p = tmp_path / "cov.csv"
    df.to_csv(p)
    out = load_dataframe_csv(p, idx)
    pd.testing.assert_frame_equal(out, df, check_dtype=False)


def test_load_dataframe_csv_rejects_empty(tmp_path):
    p = tmp_path / "empty.csv"
    pd.DataFrame(index=pd.Index(["a"], name="id")).to_csv(p)
    with pytest.raises(typer.BadParameter, match="no data columns"):
        load_dataframe_csv(p, pd.Index(["a"]))


def test_load_mz_csv_length_match(tmp_path):
    p = tmp_path / "mz.csv"
    pd.DataFrame({"mz": np.linspace(2000, 20000, 5)}).to_csv(p, index=False)
    mz = load_mz_csv(p, 5)
    assert mz.shape == (5,)


def test_load_mz_csv_length_mismatch_raises(tmp_path):
    p = tmp_path / "mz.csv"
    pd.DataFrame({"mz": [1.0, 2.0, 3.0]}).to_csv(p, index=False)
    with pytest.raises(typer.BadParameter, match="length"):
        load_mz_csv(p, 10)


def test_save_matrix_csv(tmp_path):
    df = pd.DataFrame(
        np.arange(6).reshape(2, 3), index=["a", "b"], columns=["x", "y", "z"]
    )
    p = tmp_path / "out.csv"
    save_matrix(df, p)
    loaded = pd.read_csv(p, index_col=0)
    pd.testing.assert_frame_equal(loaded, df, check_dtype=False)


def test_save_matrix_npz_with_batch_and_extras(tmp_path):
    df = pd.DataFrame(np.arange(6).reshape(2, 3), index=["a", "b"])
    batch = pd.Series([0, 1], index=df.index)
    p = tmp_path / "out.npz"
    save_matrix(df, p, batch=batch, extras={"extra": np.array([9, 8])})
    with np.load(p, allow_pickle=True) as npz:
        assert "batch" in npz.files
        assert "extra" in npz.files


def test_save_matrix_npz_ignores_conflicting_extras(tmp_path):
    df = pd.DataFrame(np.arange(6).reshape(2, 3), index=["a", "b"])
    p = tmp_path / "out.npz"
    # "X" in extras would clobber the real X; save_matrix must skip it
    save_matrix(df, p, extras={"X": np.zeros(999)})
    with np.load(p, allow_pickle=True) as npz:
        assert np.asarray(npz["X"]).shape == df.shape


def test_align_series_equal_index_fastpath():
    idx = pd.Index(["a", "b"])
    s = pd.Series([1, 2], index=idx)
    out = _align_series(s, idx, source=Path("src.csv"))
    assert out is s


def test_align_frame_equal_index_fastpath():
    idx = pd.Index(["a", "b"])
    df = pd.DataFrame({"x": [1, 2]}, index=idx)
    out = _align_frame(df, idx, source=Path("src.csv"))
    assert out is df


def test_align_series_reindex_superset(tmp_path):
    idx = pd.Index(["a", "b"])
    s = pd.Series([1, 2, 3], index=["a", "b", "c"])
    out = _align_series(s, idx, source=tmp_path / "x.csv")
    assert list(out) == [1, 2]


def test_align_frame_reindex_superset(tmp_path):
    idx = pd.Index(["a", "b"])
    df = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])
    out = _align_frame(df, idx, source=tmp_path / "x.csv")
    assert out.shape == (2, 1)


def test_align_frame_positional_fallback(tmp_path):
    idx = pd.Index(["a", "b"])
    df = pd.DataFrame({"x": [1, 2]}, index=["Z", "Y"])
    out = _align_frame(df, idx, source=tmp_path / "x.csv")
    assert list(out.index) == ["a", "b"]


def test_align_frame_row_count_mismatch_raises(tmp_path):
    idx = pd.Index(["a", "b", "c"])
    df = pd.DataFrame({"x": [1, 2]}, index=["X", "Y"])
    with pytest.raises(typer.BadParameter, match="align"):
        _align_frame(df, idx, source=tmp_path / "x.csv")


def test_load_matrix_npz_without_batch_uses_sidecar(tmp_path, tiny_df):
    X, batch, _ = tiny_df
    npz = tmp_path / "X.npz"
    bp = tmp_path / "batch.csv"
    np.savez(
        npz,
        X=X.to_numpy(),
        columns=np.asarray(X.columns, dtype=object),
        index=np.asarray(X.index, dtype=object),
    )
    batch.to_frame("batch").to_csv(bp)
    X_loaded, batch_loaded, _ = load_matrix(npz, bp)
    assert list(batch_loaded) == list(batch)
