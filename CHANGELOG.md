# Changelog

All notable changes to MaldiBatchKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-15

### Added

- `NoOpCorrector`: identity baseline corrector for use in `AutoCorrector` / benchmark candidate lists.
- `AutoCorrector`: sklearn-compatible meta-corrector with `method` as a tunable hyperparameter, so `GridSearchCV` can sweep across corrector families. Accepts `METHOD_REGISTRY` aliases or a `BaseBatchCorrector` subclass; forwards only kwargs the inner method supports; proxies fitted attributes.
- `BatchCorrectionBenchmark`: configure-then-`.fit` diagnostic comparison with `full_data` / `stratified_split` protocols and optional bootstrap CIs (`resample_metric` or `refit`). Exposes `results_`, `results_long_`, `corrected_`, `baseline_`, `.rank()`, `.to_dataframe()`, `.plot()`; `silhouette_batch` ranked by `|value|`.
- `maldibatchkit.metrics`: per-batch sklearn metrics aggregated with `weights='uniform' | 'balanced' | 'size'` (or custom). Ships `batch_{roc_auc, average_precision, balanced_accuracy, matthews_corrcoef, f1, precision, recall}_score`; drops degenerate single-class folds with a warning.
- `make_batch_scorer(batch, metric=..., weights=...)`: factory yielding a `GridSearchCV`-compatible scorer that slices `batch` per fold; supports all `response_method` values and string/callable metrics.

## [0.1.0] - 2026-04-27

Initial release.
