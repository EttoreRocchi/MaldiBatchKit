# Contributing to MaldiBatchKit

Thanks for considering a contribution!

## Development setup

```bash
git clone https://github.com/EttoreRocchi/MaldiBatchKit.git
cd MaldiBatchKit
pip install -e ".[dev,viz]"
pre-commit install
```

## Testing

```bash
make test        # fast subset (excludes slow tests)
make test-cov    # full run with term coverage report
```

The coverage gate is 95%. Please add tests for any new public API.

## Linting

```bash
make lint        # ruff check
make format      # ruff format
```

Pre-commit hooks run `ruff-check --fix`, `ruff-format`,
`end-of-file-fixer`, and `trailing-whitespace`. Please make sure they
are installed (`pre-commit install`).

## Style

- NumPy-style docstrings for all public API.
- `BaseBatchCorrector` subclasses should implement `_fit_impl` and
  `_transform_impl`. Store fitted attributes with trailing
  underscores (`gamma_star_`, `batch_levels_`, ...).
- No side effects outside of `fit` - `transform` must be idempotent.
- Raise clear `ImportError`s when optional dependencies are missing
  (see `Harmony._require_harmonypy` for the reference style).

## Pull requests

- Branch off `main`, rebase before opening the PR.
- Summarise the change in one sentence in the PR title and use the PR
  body to motivate the design decision.
- Update `CHANGELOG.md` under the next version heading.

## Issues

Before opening a bug report, please include:

- MaldiBatchKit version (`python -c "import maldibatchkit; print(maldibatchkit.__version__)"`)
- scikit-learn version, combatlearn version, pandas version
- A minimal reproducer (feature matrix shape, batch labels, the
  exception / wrong result)
