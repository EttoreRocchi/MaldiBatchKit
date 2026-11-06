# Changelog

All notable changes to MaldiBatchKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-11-06

### Fixed

- `Harmony`: `transform` applied the wrong batch's correction when training rows were not in sorted batch order. Fits with fewer than 45 samples no longer fail. `random_state` is forwarded to harmonypy, and the training cache is a copy.
- `QualityWeightedComBat`: the pooled variance now matches Johnson ComBat, and `reference_batch` maps the other batches onto the reference. A single feature no longer gives NaN output.
- Near-zero scales (e.g. a bin that is all zero in a batch) are left unscaled with a warning instead of exploding.
- Silent wrong or NaN output now raises in these cases:
  - batches with fewer than 2 samples;
  - non-finite `quality` / `design`;
  - an ndarray `X` with misaligned pandas `batch` / covariates;
  - permuted or renamed columns at `transform`;
  - mixed-type, duplicated or missing labels.

  The CLI exits with 1 on non-finite output.
- `AutoCorrector`: discrete covariates are one-hot encoded for Limma, and class-valued `method` (including ComBat classes) is routed like its alias. `SpeciesAwareComBat.set_params(species=...)` takes effect.
- `BatchCorrectionBenchmark`:
  - resamples are drawn without replacement;
  - `species_preservation` without `species` is NaN;
  - peak drift averages |delta m/z|;
  - rows are no longer mislabelled with ndarray inputs under `stratified_split` / `refit`.

  `make_batch_scorer` no longer misaligns folds. f1 / precision / recall drop single-class batches.
- CLI and `MaldiSetAdapter`:
  - sidecar CSVs are matched by sample ID, with IDs and labels read as strings;
  - `diagnose` aligns the corrected matrix by ID;
  - `.filter()` on a corrected set keeps the corrected data;
  - species is no longer passed to Harmony's `covariates`.

### Changed

- Requires `combatlearn>=2.4.0`. This changes the output of every ComBat variant (EB shrinkage as in `sva::ComBat`, `reference_batch`), and Chen/CovBat now protects covariates.
- `Harmony(random_state=None)` draws a fresh seed at each fit (previously always 0). `AutoCorrector(method='harmony')` drops covariates with a warning.
- `BatchCorrectionBenchmark` has a new `subsample_fraction` parameter (default 0.8), and `value` is the full-data estimate. `peak_position_drift` has a new `mean_abs_delta_mz` column. CLI: new `--nclust`.
- Numerical output changes for QW-ComBat, near-zero scales, AutoCorrector + Limma, and f1 / precision / recall.

## [0.2.0] - 2026-05-15

### Added

- `NoOpCorrector`: identity baseline corrector for use in `AutoCorrector` / benchmark candidate lists.
- `AutoCorrector`: sklearn-compatible meta-corrector with `method` as a tunable hyperparameter, so `GridSearchCV` can sweep across corrector families. Accepts `METHOD_REGISTRY` aliases or a `BaseBatchCorrector` subclass; forwards only kwargs the inner method supports; proxies fitted attributes.
- `BatchCorrectionBenchmark`: configure-then-`.fit` diagnostic comparison with `full_data` / `stratified_split` protocols and optional bootstrap CIs (`resample_metric` or `refit`). Exposes `results_`, `results_long_`, `corrected_`, `baseline_`, `.rank()`, `.to_dataframe()`, `.plot()`; `silhouette_batch` ranked by `|value|`.
- `maldibatchkit.metrics`: per-batch sklearn metrics aggregated with `weights='uniform' | 'balanced' | 'size'` (or custom). Ships `batch_{roc_auc, average_precision, balanced_accuracy, matthews_corrcoef, f1, precision, recall}_score`; drops degenerate single-class folds with a warning.
- `make_batch_scorer(batch, metric=..., weights=...)`: factory yielding a `GridSearchCV`-compatible scorer that slices `batch` per fold; supports all `response_method` values and string/callable metrics.

## [0.1.0] - 2026-04-27

Initial release.

[0.2.1]: https://github.com/EttoreRocchi/MaldiBatchKit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/EttoreRocchi/MaldiBatchKit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/EttoreRocchi/MaldiBatchKit/releases/tag/v0.1.0
