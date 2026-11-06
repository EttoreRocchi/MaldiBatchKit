"""Shared I/O helpers for the MaldiBatchKit CLI.

Every ``correct`` subcommand and ``diagnose`` goes through this module so
the file-format contract is documented once.

Supported input / output containers
-----------------------------------
* **CSV**: first column is treated as the sample index; remaining
  columns are features.  Labels (batch, species, quality, covariates)
  live in sidecar CSVs with a sample-id index column and one or more
  data columns.
* **NPZ**: a ``np.savez`` archive containing at least ``X``; optionally
  ``columns``, ``index``, ``batch``, and arbitrary extra arrays kept as
  passthrough metadata.  This is the MaldiSet-friendly format.

All auxiliary loaders align the loaded object to ``X.index`` by sample
ID and fail, listing the missing IDs, when the sidecar does not cover
``X.index``. Sample IDs and labels are read as strings (``"001"`` stays
``"001"`` and ``"NA"`` is a label, not a missing value).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from ._utils import _check_unique_ids

__all__ = [
    "load_features",
    "load_matrix",
    "load_dataframe_csv",
    "load_labels_csv",
    "load_mz_csv",
    "save_matrix",
]


def _read_csv_indexed(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV whose first column holds the sample IDs, kept as strings."""
    df = pd.read_csv(path, index_col=0, **kwargs)
    ids = pd.read_csv(path, usecols=[0], dtype=str, keep_default_na=False)
    df.index = pd.Index(ids.iloc[:, 0].to_numpy(), name=df.index.name)
    return df


def _load_npz(path: Path) -> tuple[pd.DataFrame, pd.Series | None, dict]:
    """Return ``X``, the bundled batch (or None) and the extra arrays of an NPZ."""
    with np.load(path, allow_pickle=True) as npz:
        if "X" not in npz.files:
            raise typer.BadParameter(
                f"{path} is missing the 'X' array. Expected "
                f"np.savez(..., X=, [columns=, index=, batch=])."
            )
        X_arr = np.asarray(npz["X"])
        columns = list(npz["columns"]) if "columns" in npz.files else None
        index = list(npz["index"]) if "index" in npz.files else None
        npz_batch = (
            pd.Series(np.asarray(npz["batch"])) if "batch" in npz.files else None
        )
        extras = {
            k: np.asarray(npz[k])
            for k in npz.files
            if k not in {"X", "columns", "index", "batch"}
        }
    return pd.DataFrame(X_arr, index=index, columns=columns), npz_batch, extras


def load_features(path: Path) -> pd.DataFrame:
    """Load only the feature matrix of a CSV or NPZ file."""
    if path.suffix.lower() == ".npz":
        return _load_npz(path)[0]
    return _read_csv_indexed(path)


def load_matrix(
    path: Path, batch_csv: Path | None
) -> tuple[pd.DataFrame, pd.Series, dict[str, np.ndarray]]:
    """Load a feature matrix and its aligned batch labels.

    Parameters
    ----------
    path : Path
        CSV or NPZ file. CSV first column is the sample index. NPZ must
        contain an ``X`` array and may also contain ``columns``,
        ``index`` and ``batch`` arrays.
    batch_csv : Path or None
        Required when ``path`` is a CSV or when an NPZ does not include
        ``batch``. A single-column CSV with a sample-id index column.

    Returns
    -------
    X : pd.DataFrame
    batch : pd.Series
        Indexed by ``X.index``.
    extras : dict
        Any additional arrays found in an NPZ (beyond X / index /
        columns / batch).  Empty for CSV inputs.
    """
    if path.suffix.lower() == ".npz":
        X, npz_batch, extras = _load_npz(path)
        if npz_batch is not None:
            if len(npz_batch) != len(X):
                raise typer.BadParameter(
                    f"NPZ batch length {len(npz_batch)} does not match X rows {len(X)}."
                )
            npz_batch.index = X.index
            batch = npz_batch
        else:
            if batch_csv is None:
                raise typer.BadParameter(
                    f"No 'batch' array in {path}; pass --batch-csv."
                )
            batch = load_labels_csv(batch_csv, X.index)
        return X, batch, extras

    X = _read_csv_indexed(path)
    if batch_csv is None:
        raise typer.BadParameter(
            "CSV input requires --batch-csv with a single-column file of "
            "batch labels (first column is the sample index)."
        )
    batch = load_labels_csv(batch_csv, X.index)
    return X, batch, {}


def load_labels_csv(path: Path, index: pd.Index) -> pd.Series:
    """Load a single-column sidecar CSV as a Series aligned to ``index``.

    The CSV's first column is the sample ID and the second the label, both
    read as strings.
    """
    df = _read_csv_indexed(path, dtype=str, keep_default_na=False)
    if df.shape[1] == 0:
        raise typer.BadParameter(
            f"{path} has no data columns. Expected <sample_id>,<label>."
        )
    if df.shape[1] > 1:
        raise typer.BadParameter(
            f"{path} has {df.shape[1]} data columns; expected exactly one. "
            f"For multi-column covariates use a dedicated "
            f"--*-covariates-csv / --design-csv flag."
        )
    series = df.iloc[:, 0]
    return _align_series(series, index, source=path)


def load_dataframe_csv(
    path: Path, index: pd.Index, *, categorical: bool = False
) -> pd.DataFrame:
    """Load a multi-column sidecar CSV and align it to ``index``.

    Parameters
    ----------
    path : Path
        CSV whose first column holds the sample IDs.
    index : pd.Index
        ``X.index`` to align to.
    categorical : bool, default=False
        Read the values as strings (labels such as ``"NA"`` or ``"001"``
        are kept as written); otherwise they are parsed as numbers.
    """
    df = (
        _read_csv_indexed(path, dtype=str, keep_default_na=False)
        if categorical
        else _read_csv_indexed(path)
    )
    if df.shape[1] == 0:
        raise typer.BadParameter(f"{path} has no data columns.")
    return _align_frame(df, index, source=path)


def load_mz_csv(path: Path, n_features: int) -> np.ndarray:
    """Load an m/z axis CSV (single column, length = n_features)."""
    series = pd.read_csv(path).iloc[:, -1]
    values = series.to_numpy(dtype=float)
    if values.shape[0] != n_features:
        raise typer.BadParameter(
            f"--mz-csv length {values.shape[0]} does not match n_features {n_features}."
        )
    return values


def save_matrix(
    X: pd.DataFrame,
    path: Path,
    *,
    batch: pd.Series | None = None,
    extras: dict[str, np.ndarray] | None = None,
) -> None:
    """Write ``X`` to CSV (index + features) or NPZ (bundled arrays)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        payload: dict[str, np.ndarray] = {
            "X": X.to_numpy(),
            "columns": np.asarray(X.columns),
            "index": np.asarray(X.index),
        }
        if batch is not None:
            payload["batch"] = np.asarray(batch.to_numpy())
        if extras:
            for k, v in extras.items():
                if k in payload:
                    continue
                payload[k] = np.asarray(v)
        np.savez(path, **payload)
    else:
        X.to_csv(path)


def _align_by_id(obj, index: pd.Index, *, source: Path):
    """Reindex ``obj`` onto ``index`` by sample ID (compared as strings)."""
    if obj.index.equals(index):
        return obj
    keys = pd.Index(index).astype(str)
    if obj.index.name is not None and str(obj.index.name) in keys:
        raise typer.BadParameter(
            f"{source} has no header row: its first line starts with sample ID "
            f"{obj.index.name!r}. Add a header such as 'id,label'."
        )
    try:
        _check_unique_ids(obj.index, str(source))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    by_id = obj.set_axis(obj.index.astype(str), axis=0)
    missing = keys[~keys.isin(by_id.index)]
    if len(missing):
        raise typer.BadParameter(
            f"{source}: cannot align to X.index; {len(missing)} sample ID(s) "
            f"of X are missing (first 5: {list(missing[:5])})."
        )
    return by_id.reindex(keys).set_axis(index, axis=0)


def _align_series(series: pd.Series, index: pd.Index, *, source: Path) -> pd.Series:
    return _align_by_id(series, index, source=source)


def _align_frame(df: pd.DataFrame, index: pd.Index, *, source: Path) -> pd.DataFrame:
    return _align_by_id(df, index, source=source)
