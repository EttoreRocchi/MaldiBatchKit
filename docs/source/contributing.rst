Contributing
============

Thanks for your interest in contributing! Here's how to get started.

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/MaldiBatchKit.git
   cd MaldiBatchKit
   pip install -e ".[dev,viz]"
   pre-commit install

Running Tests
-------------

.. code-block:: bash

   make test        # fast subset (excludes slow tests)
   make test-cov    # full run with term coverage report

The coverage gate is 95%. Please add tests for any new public API.

Linting
-------

.. code-block:: bash

   make lint        # ruff check --fix
   make format      # ruff format

Pre-commit hooks run ``ruff-check --fix``, ``ruff-format``,
``end-of-file-fixer``, and ``trailing-whitespace``. Make sure they are
installed (``pre-commit install``).

Style
-----

- NumPy-style docstrings for all public API.
- ``BaseBatchCorrector`` subclasses should implement ``_fit_impl`` and
  ``_transform_impl``. Store fitted attributes with trailing
  underscores (``gamma_star_``, ``batch_levels_``, ...).
- No side effects outside of ``fit`` - ``transform`` must be idempotent.
- Raise clear ``ImportError`` when optional dependencies are missing
  (see ``Harmony._require_harmonypy`` for the reference style).

Submitting Changes
------------------

1. Fork the repository and create a feature branch from ``main``.
2. Add tests for any new functionality.
3. Run ``make test`` and ``make lint`` - both must be clean.
4. Update ``CHANGELOG.md`` under the next version heading.
5. Open a pull request with a clear title and a body that motivates the
   design decision (not just what changed).

Reporting Issues
----------------

Open an issue on `GitHub <https://github.com/EttoreRocchi/MaldiBatchKit/issues>`_
with a minimal reproducer:

- MaldiBatchKit version (``python -c "import maldibatchkit; print(maldibatchkit.__version__)"``)
- scikit-learn, combatlearn, and pandas versions
- Feature matrix shape, batch labels, and the exception or wrong result
