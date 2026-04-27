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

All auxiliary loaders align the loaded object to ``X.index`` via
``.reindex`` when an index is available in the CSV, or positionally
otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

__all__ = [
    "load_matrix",
    "load_dataframe_csv",
    "load_labels_csv",
    "load_mz_csv",
    "save_matrix",
]


def _read_csv_indexed(path: Path) -> pd.DataFrame:
    """Read a CSV assuming the first column is the sample index."""
    return pd.read_csv(path, index_col=0)


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
        X = pd.DataFrame(X_arr, index=index, columns=columns)
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

    # CSV path
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

    The CSV's first column is interpreted as the sample index; the
    second column is the label value. If the CSV has no explicit sample
    index (i.e. a single data column read with the default
    ``RangeIndex``), the labels are assumed to already be in ``X`` order
    and are aligned positionally.
    """
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] == 0:
        # Only an index column was present -> data is the index itself
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


def load_dataframe_csv(path: Path, index: pd.Index) -> pd.DataFrame:
    """Load a multi-column sidecar CSV and align it to ``index``."""
    df = pd.read_csv(path, index_col=0)
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


def _align_series(series: pd.Series, index: pd.Index, *, source: Path) -> pd.Series:
    if series.index.equals(index):
        return series
    if set(index).issubset(set(series.index)):
        return series.reindex(index)
    if len(series) == len(index):
        # Positional fallback: the CSV did not carry the X sample IDs.
        aligned = series.copy()
        aligned.index = index
        return aligned
    raise typer.BadParameter(
        f"{source}: cannot align labels to X.index. "
        f"Labels have {len(series)} rows, X has {len(index)} rows, "
        f"and label indices do not cover X.index."
    )


def _align_frame(df: pd.DataFrame, index: pd.Index, *, source: Path) -> pd.DataFrame:
    if df.index.equals(index):
        return df
    if set(index).issubset(set(df.index)):
        return df.reindex(index)
    if len(df) == len(index):
        aligned = df.copy()
        aligned.index = index
        return aligned
    raise typer.BadParameter(
        f"{source}: cannot align to X.index. Got {len(df)} rows, X has {len(index)}."
    )
