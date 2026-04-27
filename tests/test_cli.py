"""CLI smoke tests covering every `correct <method>` subcommand plus diagnose."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from maldibatchkit.cli import app

runner = CliRunner()


def _write_matrix_csv(path: Path, X: pd.DataFrame) -> None:
    X.to_csv(path)


def _write_labels_csv(path: Path, labels: pd.Series, name: str) -> None:
    labels.rename(name).to_csv(path, header=[name])


def _write_dataframe_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path)


@pytest.fixture
def csv_inputs(tmp_path, tiny_dataset):
    X = tiny_dataset["X"]
    paths = {
        "X": tmp_path / "X.csv",
        "batch": tmp_path / "batch.csv",
        "species": tmp_path / "species.csv",
        "quality": tmp_path / "quality.csv",
        "discrete": tmp_path / "discrete.csv",
        "continuous": tmp_path / "continuous.csv",
        "design": tmp_path / "design.csv",
        "covariates": tmp_path / "harmony_covariates.csv",
        "out": tmp_path / "corrected.csv",
    }
    _write_matrix_csv(paths["X"], X)
    _write_labels_csv(paths["batch"], tiny_dataset["batch"], "batch")
    _write_labels_csv(paths["species"], tiny_dataset["species"], "species")
    rng = np.random.default_rng(0)
    quality = pd.Series(rng.uniform(0.5, 2.0, len(X)), index=X.index, name="snr")
    _write_labels_csv(paths["quality"], quality, "snr")
    discrete = pd.DataFrame({"site": ["A", "B"] * (len(X) // 2)}, index=X.index)
    _write_dataframe_csv(paths["discrete"], discrete)
    continuous = pd.DataFrame({"age": rng.uniform(20, 80, len(X))}, index=X.index)
    _write_dataframe_csv(paths["continuous"], continuous)
    design = pd.get_dummies(
        tiny_dataset["species"].rename("species"), drop_first=True
    ).astype(float)
    _write_dataframe_csv(paths["design"], design)
    _write_dataframe_csv(paths["covariates"], discrete)
    return paths


def test_root_help_lists_both_verbs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "correct" in result.output
    assert "diagnose" in result.output


def test_correct_help_lists_every_method():
    result = runner.invoke(app, ["correct", "--help"])
    assert result.exit_code == 0
    for name in [
        "combat",
        "combat-fortin",
        "combat-chen",
        "species-combat",
        "quality-combat",
        "limma",
        "harmony",
        "median-center",
        "zscore-per-batch",
        "reference-scaling",
        "warping",
    ]:
        assert name in result.output, name


def test_correct_combat_johnson(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "combat",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
        ],
    )
    assert result.exit_code == 0, result.output
    assert csv_inputs["out"].exists()


def test_correct_combat_fortin_with_discrete(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "combat-fortin",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
            "--discrete-covariates-csv",
            str(csv_inputs["discrete"]),
        ],
    )
    assert result.exit_code == 0, result.output


def test_correct_combat_fortin_without_covariates_fails(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "combat-fortin",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
        ],
    )
    assert result.exit_code != 0
    assert (
        "combat-fortin" in result.output.lower() or "covariate" in result.output.lower()
    )


def test_correct_combat_chen(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "combat-chen",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
            "--continuous-covariates-csv",
            str(csv_inputs["continuous"]),
            "--covbat-cov-thresh",
            "0.95",
        ],
    )
    assert result.exit_code == 0, result.output


def test_correct_species_combat(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "species-combat",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
            "--species-csv",
            str(csv_inputs["species"]),
        ],
    )
    assert result.exit_code == 0, result.output


def test_correct_species_combat_requires_species(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "species-combat",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
        ],
    )
    assert result.exit_code != 0


def test_correct_quality_combat(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "quality-combat",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
            "--quality-csv",
            str(csv_inputs["quality"]),
            "--max-iter",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output


def test_correct_quality_combat_requires_quality(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "quality-combat",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
        ],
    )
    assert result.exit_code != 0


def test_correct_limma(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "limma",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
            "--design-csv",
            str(csv_inputs["design"]),
        ],
    )
    assert result.exit_code == 0, result.output


def test_correct_harmony(csv_inputs):
    result = runner.invoke(
        app,
        [
            "correct",
            "harmony",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(csv_inputs["batch"]),
            "--max-iter",
            "3",
            "--random-state",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output


def test_correct_median_center_and_zscore_and_reference_scaling(csv_inputs, tmp_path):
    for sub in ["median-center", "zscore-per-batch", "reference-scaling"]:
        out = tmp_path / f"{sub}.csv"
        result = runner.invoke(
            app,
            [
                "correct",
                sub,
                "-i",
                str(csv_inputs["X"]),
                "-o",
                str(out),
                "--batch-csv",
                str(csv_inputs["batch"]),
            ],
        )
        assert result.exit_code == 0, f"{sub}: {result.output}"
        assert out.exists()


def test_correct_combat_npz_roundtrip(tmp_path, tiny_dataset):
    X, batch = tiny_dataset["X"], tiny_dataset["batch"]
    npz_in = tmp_path / "X.npz"
    np.savez(
        npz_in,
        X=X.to_numpy(),
        columns=np.asarray(X.columns, dtype=object),
        index=np.asarray(X.index, dtype=object),
        batch=batch.to_numpy(),
    )
    npz_out = tmp_path / "corr.npz"
    result = runner.invoke(
        app,
        [
            "correct",
            "combat",
            "-i",
            str(npz_in),
            "-o",
            str(npz_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert npz_out.exists()
    with np.load(npz_out, allow_pickle=True) as npz:
        assert "X" in npz.files
        assert "batch" in npz.files


def test_diagnose_before_only(csv_inputs, tmp_path):
    report = tmp_path / "report.csv"
    result = runner.invoke(
        app,
        [
            "diagnose",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(report),
            "--batch-csv",
            str(csv_inputs["batch"]),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(report)
    assert "metric" in df.columns


def test_diagnose_with_corrected(csv_inputs, tmp_path):
    # First produce a corrected file, then diagnose
    corrected = tmp_path / "after.csv"
    runner.invoke(
        app,
        [
            "correct",
            "combat",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(corrected),
            "--batch-csv",
            str(csv_inputs["batch"]),
        ],
    )
    report = tmp_path / "diag.csv"
    result = runner.invoke(
        app,
        [
            "diagnose",
            "-i",
            str(csv_inputs["X"]),
            "--corrected",
            str(corrected),
            "-o",
            str(report),
            "--batch-csv",
            str(csv_inputs["batch"]),
        ],
    )
    assert result.exit_code == 0, result.output


def test_diagnose_with_mz(csv_inputs, tmp_path, tiny_dataset):
    mz_path = tmp_path / "mz.csv"
    mz = np.linspace(2000, 20000, tiny_dataset["X"].shape[1])
    pd.DataFrame({"mz": mz}).to_csv(mz_path, index=False)
    report = tmp_path / "diag.csv"
    result = runner.invoke(
        app,
        [
            "diagnose",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(report),
            "--batch-csv",
            str(csv_inputs["batch"]),
            "--mz-csv",
            str(mz_path),
            "--top-k-peaks",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output


def test_correct_without_batch_for_csv_fails(csv_inputs, tmp_path):
    out = tmp_path / "out.csv"
    result = runner.invoke(
        app,
        [
            "correct",
            "combat",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code != 0


def test_labels_csv_rejects_multi_column(tmp_path, csv_inputs):
    bad = tmp_path / "bad_batch.csv"
    # Write a batch CSV with TWO data columns -> should be rejected
    df = pd.DataFrame(
        {"a": [0] * 60, "b": [1] * 60},
        index=pd.Index([f"s{i:04d}" for i in range(60)], name="id"),
    )
    df.to_csv(bad)
    result = runner.invoke(
        app,
        [
            "correct",
            "combat",
            "-i",
            str(csv_inputs["X"]),
            "-o",
            str(csv_inputs["out"]),
            "--batch-csv",
            str(bad),
        ],
    )
    assert result.exit_code != 0
