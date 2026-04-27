"""Command-line interface for MaldiBatchKit.

The CLI is organised as::

    maldibatchkit
    +-- correct
    |   +-- combat               (Johnson 2007)
    |   +-- combat-fortin        (Fortin 2018, covariate-aware)
    |   +-- combat-chen          (Chen 2022, CovBat)
    |   +-- species-combat       (Fortin preset with species)
    |   +-- quality-combat       (weighted EB extension)
    |   +-- limma                (Ritchie 2015)
    |   +-- harmony              (Korsunsky 2019)
    |   +-- median-center
    |   +-- zscore-per-batch
    |   +-- reference-scaling
    |   +-- warping              (BatchAwareWarping)
    +-- diagnose

Every ``correct`` subcommand shares the same ``--input / --output /
--batch-csv`` contract and adds the flags that are meaningful for its
method (covariates, reference batch, parametric toggles, ...). Use
``maldibatchkit correct <method> --help`` to see the exact options for
a given corrector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd

try:
    import typer
except ImportError as _typer_err:  # pragma: no cover - typer is a hard dep
    raise ImportError(
        "The MaldiBatchKit CLI depends on `typer`. "
        "Install it with `pip install typer[all]`."
    ) from _typer_err

from . import (
    BatchAwareWarping,
    ComBat,
    Harmony,
    Limma,
    MedianCentering,
    QualityWeightedComBat,
    ReferenceScaling,
    SpeciesAwareComBat,
    ZScorePerBatch,
)
from ._cli_io import (
    load_dataframe_csv,
    load_labels_csv,
    load_matrix,
    load_mz_csv,
    save_matrix,
)
from .diagnostics import diagnostic_report

app = typer.Typer(
    name="maldibatchkit",
    help="Batch-effect correction for MALDI-TOF AMR feature matrices.",
    add_completion=False,
    no_args_is_help=True,
)

correct_app = typer.Typer(
    name="correct",
    help="Apply a batch-correction method to a feature matrix.",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(correct_app)


InputOpt = Annotated[
    Path, typer.Option("-i", "--input", help="Feature matrix (CSV or NPZ).")
]
OutputOpt = Annotated[
    Path, typer.Option("-o", "--output", help="Output path (CSV or NPZ).")
]
BatchOpt = Annotated[
    Optional[Path],
    typer.Option(
        "--batch-csv",
        help="Single-column CSV of batch labels (index = sample IDs).",
    ),
]


def _run_and_save(
    transformer, X: pd.DataFrame, output: Path, batch: pd.Series, extras: dict
) -> None:
    """Fit-transform + save, preserving the input container shape."""
    X_corrected = transformer.fit_transform(X)
    if not isinstance(X_corrected, pd.DataFrame):
        X_corrected = pd.DataFrame(
            np.asarray(X_corrected), index=X.index, columns=X.columns
        )
    save_matrix(X_corrected, output, batch=batch, extras=extras)
    typer.echo(f"wrote {output} ({X_corrected.shape[0]} x {X_corrected.shape[1]})")


def _load_covariates(
    discrete_csv: Path | None,
    continuous_csv: Path | None,
    index: pd.Index,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    discrete = load_dataframe_csv(discrete_csv, index) if discrete_csv else None
    continuous = load_dataframe_csv(continuous_csv, index) if continuous_csv else None
    return discrete, continuous


@correct_app.command("combat")
def correct_combat(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    parametric: Annotated[
        bool, typer.Option(help="Use parametric empirical Bayes.")
    ] = True,
    mean_only: Annotated[
        bool, typer.Option(help="Adjust only the mean, not the variance.")
    ] = False,
    reference_batch: Annotated[
        Optional[str],
        typer.Option(help="Batch level to leave unchanged."),
    ] = None,
    eps: Annotated[float, typer.Option(help="Numerical jitter.")] = 1e-8,
) -> None:
    """Vanilla Johnson (2007) ComBat. No covariates."""
    X, batch, extras = load_matrix(input, batch_csv)
    transformer = ComBat(
        batch=batch,
        method="johnson",
        parametric=parametric,
        mean_only=mean_only,
        reference_batch=reference_batch,
        eps=eps,
    )
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("combat-fortin")
def correct_combat_fortin(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    discrete_covariates_csv: Annotated[
        Optional[Path],
        typer.Option(
            "--discrete-covariates-csv",
            help="CSV of categorical covariates to protect "
            "(index = sample IDs, any number of columns).",
        ),
    ] = None,
    continuous_covariates_csv: Annotated[
        Optional[Path],
        typer.Option(
            "--continuous-covariates-csv",
            help="CSV of continuous covariates to protect.",
        ),
    ] = None,
    parametric: Annotated[bool, typer.Option()] = True,
    mean_only: Annotated[bool, typer.Option()] = False,
    reference_batch: Annotated[Optional[str], typer.Option()] = None,
    eps: Annotated[float, typer.Option()] = 1e-8,
) -> None:
    """Fortin (2018) covariate-aware ComBat."""
    X, batch, extras = load_matrix(input, batch_csv)
    discrete, continuous = _load_covariates(
        discrete_covariates_csv, continuous_covariates_csv, X.index
    )
    if discrete is None and continuous is None:
        raise typer.BadParameter(
            "combat-fortin requires at least one of "
            "--discrete-covariates-csv / --continuous-covariates-csv. "
            "Without covariates this reduces to Johnson; use "
            "`maldibatchkit correct combat` instead."
        )
    transformer = ComBat(
        batch=batch,
        discrete_covariates=discrete,
        continuous_covariates=continuous,
        method="fortin",
        parametric=parametric,
        mean_only=mean_only,
        reference_batch=reference_batch,
        eps=eps,
    )
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("combat-chen")
def correct_combat_chen(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    discrete_covariates_csv: Annotated[Optional[Path], typer.Option()] = None,
    continuous_covariates_csv: Annotated[Optional[Path], typer.Option()] = None,
    covbat_cov_thresh: Annotated[
        float,
        typer.Option(
            help="Variance threshold in (0, 1] for PCs, or int >= 1 for a "
            "fixed number of components."
        ),
    ] = 0.9,
    parametric: Annotated[bool, typer.Option()] = True,
    mean_only: Annotated[bool, typer.Option()] = False,
    reference_batch: Annotated[Optional[str], typer.Option()] = None,
    eps: Annotated[float, typer.Option()] = 1e-8,
) -> None:
    """Chen (2022) CovBat: ComBat on PCA-decomposed batch covariance."""
    X, batch, extras = load_matrix(input, batch_csv)
    discrete, continuous = _load_covariates(
        discrete_covariates_csv, continuous_covariates_csv, X.index
    )
    transformer = ComBat(
        batch=batch,
        discrete_covariates=discrete,
        continuous_covariates=continuous,
        method="chen",
        parametric=parametric,
        mean_only=mean_only,
        reference_batch=reference_batch,
        eps=eps,
        covbat_cov_thresh=covbat_cov_thresh,
    )
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("species-combat")
def correct_species_combat(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    species_csv: Annotated[
        Optional[Path],
        typer.Option(
            "--species-csv",
            help="Single-column CSV with species labels "
            "(index = sample IDs). Required.",
        ),
    ] = None,
    continuous_covariates_csv: Annotated[Optional[Path], typer.Option()] = None,
    parametric: Annotated[bool, typer.Option()] = True,
    mean_only: Annotated[bool, typer.Option()] = False,
    reference_batch: Annotated[Optional[str], typer.Option()] = None,
    eps: Annotated[float, typer.Option()] = 1e-8,
) -> None:
    """Run ComBat-Fortin with species as the protected covariate."""
    X, batch, extras = load_matrix(input, batch_csv)
    if species_csv is None:
        raise typer.BadParameter("species-combat requires --species-csv.")
    species = load_labels_csv(species_csv, X.index)
    continuous = (
        load_dataframe_csv(continuous_covariates_csv, X.index)
        if continuous_covariates_csv
        else None
    )
    transformer = SpeciesAwareComBat(
        batch=batch,
        species=species,
        continuous_covariates=continuous,
        parametric=parametric,
        mean_only=mean_only,
        reference_batch=reference_batch,
        eps=eps,
    )
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("quality-combat")
def correct_quality_combat(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    quality_csv: Annotated[
        Optional[Path],
        typer.Option(
            "--quality-csv",
            help="Single-column CSV with non-negative per-sample weights "
            "(typically SNR). Required.",
        ),
    ] = None,
    parametric: Annotated[bool, typer.Option()] = True,
    reference_batch: Annotated[Optional[str], typer.Option()] = None,
    max_iter: Annotated[int, typer.Option(help="Parametric EB iteration cap.")] = 50,
    tol: Annotated[float, typer.Option(help="Convergence tolerance.")] = 1e-4,
    eps: Annotated[float, typer.Option()] = 1e-8,
) -> None:
    """Quality-weighted empirical-Bayes ComBat."""
    X, batch, extras = load_matrix(input, batch_csv)
    if quality_csv is None:
        raise typer.BadParameter("quality-combat requires --quality-csv.")
    quality = load_labels_csv(quality_csv, X.index)
    transformer = QualityWeightedComBat(
        batch=batch,
        quality=quality,
        parametric=parametric,
        reference_batch=reference_batch,
        eps=eps,
        max_iter=max_iter,
        tol=tol,
    )
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("limma")
def correct_limma(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    design_csv: Annotated[
        Optional[Path],
        typer.Option(
            "--design-csv",
            help="CSV with covariates to protect (index = sample IDs, "
            "any number of numeric columns).",
        ),
    ] = None,
    eps: Annotated[float, typer.Option(help="Ridge regularisation.")] = 1e-8,
) -> None:
    """limma::removeBatchEffect (Ritchie 2015)."""
    X, batch, extras = load_matrix(input, batch_csv)
    design = load_dataframe_csv(design_csv, X.index) if design_csv else None
    transformer = Limma(batch=batch, design=design, eps=eps)
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("harmony")
def correct_harmony(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    covariates_csv: Annotated[
        Optional[Path],
        typer.Option(
            "--covariates-csv",
            help="Extra categorical covariates forwarded to harmonypy "
            "vars_use (index = sample IDs).",
        ),
    ] = None,
    theta: Annotated[float, typer.Option(help="Diversity clustering penalty.")] = 2.0,
    max_iter: Annotated[int, typer.Option(help="Harmony iterations.")] = 20,
    random_state: Annotated[Optional[int], typer.Option()] = None,
) -> None:
    """Harmony (Korsunsky 2019) via harmonypy. Train/test safe via closed-form transform."""
    X, batch, extras = load_matrix(input, batch_csv)
    covariates = load_dataframe_csv(covariates_csv, X.index) if covariates_csv else None
    transformer = Harmony(
        batch=batch,
        covariates=covariates,
        theta=theta,
        max_iter=max_iter,
        random_state=random_state,
    )
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("median-center")
def correct_median_center(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
) -> None:
    """Per-batch median centering."""
    X, batch, extras = load_matrix(input, batch_csv)
    transformer = MedianCentering(batch=batch)
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("zscore-per-batch")
def correct_zscore_per_batch(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    eps: Annotated[float, typer.Option()] = 1e-8,
) -> None:
    """Per-batch z-score normalisation."""
    X, batch, extras = load_matrix(input, batch_csv)
    transformer = ZScorePerBatch(batch=batch, eps=eps)
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("reference-scaling")
def correct_reference_scaling(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    reference_batch: Annotated[
        Optional[str],
        typer.Option(
            help="Batch level to use as reference. Defaults to the "
            "largest training batch."
        ),
    ] = None,
    eps: Annotated[float, typer.Option()] = 1e-8,
) -> None:
    """Multiplicative per-batch scaling to a reference."""
    X, batch, extras = load_matrix(input, batch_csv)
    transformer = ReferenceScaling(
        batch=batch, reference_batch=reference_batch, eps=eps
    )
    _run_and_save(transformer, X, output, batch, extras)


@correct_app.command("warping")
def correct_warping(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    reference: Annotated[
        str,
        typer.Option(help="Global reference: 'median' or 'batch_mean'."),
    ] = "median",
    method: Annotated[
        str,
        typer.Option(help="Warping strategy: shift / linear / piecewise / dtw."),
    ] = "shift",
    n_segments: Annotated[int, typer.Option()] = 5,
    max_shift: Annotated[int, typer.Option()] = 50,
    dtw_radius: Annotated[int, typer.Option()] = 10,
    smooth_sigma: Annotated[float, typer.Option()] = 2.0,
    n_jobs: Annotated[int, typer.Option()] = 1,
) -> None:
    """Batch-aware m/z warping onto a shared global reference."""
    X, batch, extras = load_matrix(input, batch_csv)
    transformer = BatchAwareWarping(
        batch=batch,
        reference=reference,
        method=method,
        n_segments=n_segments,
        max_shift=max_shift,
        dtw_radius=dtw_radius,
        smooth_sigma=smooth_sigma,
        n_jobs=n_jobs,
    )
    _run_and_save(transformer, X, output, batch, extras)


@app.command("diagnose")
def diagnose(
    input: InputOpt,
    output: OutputOpt,
    batch_csv: BatchOpt = None,
    corrected: Annotated[
        Optional[Path],
        typer.Option(
            "--corrected",
            help="Corrected matrix (CSV or NPZ). If omitted, the "
            "report compares the input to itself.",
        ),
    ] = None,
    mz_csv: Annotated[
        Optional[Path],
        typer.Option(
            "--mz-csv",
            help="m/z values (single-column CSV of length n_features) "
            "used by the peak-drift metric.",
        ),
    ] = None,
    top_k_peaks: Annotated[
        int, typer.Option(help="Number of peaks tracked for drift.")
    ] = 50,
    lisi_perplexity: Annotated[float, typer.Option(help="Perplexity for LISI.")] = 30.0,
    kbet_k: Annotated[
        Optional[int],
        typer.Option("--kbet-k", help="Neighbourhood size for kBET."),
    ] = None,
) -> None:
    """Run the full diagnostic report and write a tidy CSV."""
    before, batch, _ = load_matrix(input, batch_csv)
    if corrected is not None:
        after, _, _ = load_matrix(corrected, batch_csv)
    else:
        after = before.copy()
    mz_values = load_mz_csv(mz_csv, before.shape[1]) if mz_csv else None
    report = diagnostic_report(
        before,
        after,
        batch,
        mz_values=mz_values,
        k=kbet_k,
        lisi_perplexity=lisi_perplexity,
        top_k_peaks=top_k_peaks,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    typer.echo(f"wrote {output} ({len(report)} rows)")


if __name__ == "__main__":  # pragma: no cover
    app()
