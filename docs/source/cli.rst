CLI Reference
=============

MaldiBatchKit ships a command-line interface built on `Typer <https://typer.tiangolo.com/>`_,
organised around two top-level commands: ``correct`` (apply a
batch-correction method to a feature matrix) and ``diagnose`` (run the
full diagnostic suite on a before/after pair).

.. code-block:: text

    maldibatchkit
    ├── correct
    │   ├── combat                (Johnson 2007)
    │   ├── combat-fortin         (Fortin 2018, covariate-aware)
    │   ├── combat-chen           (Chen 2022, CovBat)
    │   ├── species-combat        (Fortin preset with species)
    │   ├── quality-combat        (weighted empirical-Bayes)
    │   ├── limma                 (Ritchie 2015)
    │   ├── harmony               (Korsunsky 2019)
    │   ├── median-center
    │   ├── zscore-per-batch
    │   ├── reference-scaling
    │   └── warping               (BatchAwareWarping)
    └── diagnose

Every ``correct`` subcommand shares the same
``-i / -o / --batch-csv`` contract and adds just the flags that matter
for its method. Run ``maldibatchkit correct <method> --help`` to see the
full option list for any corrector.

Command Reference
-----------------

.. click:: maldibatchkit.cli:typer_click_object
   :prog: maldibatchkit
   :nested: full

Input / Output Formats
----------------------

* **CSV** - first column is the sample index, remaining columns are
  features. A companion ``--batch-csv`` is required (single data column;
  first column is the sample id).
* **NPZ** - a ``np.savez`` archive with at least ``X``, and optionally
  ``columns``, ``index``, and ``batch``. When ``batch`` is bundled in
  the archive, ``--batch-csv`` becomes optional.

Covariate / Auxiliary CSVs
--------------------------

The following sidecar CSVs are used by the methods that need them. Each
has a sample-id column first and one or more data columns after:

+--------------------------------+----------------------------------+
| Flag                           | Accepted columns                 |
+================================+==================================+
| ``--species-csv``              | 1 (species label)                |
+--------------------------------+----------------------------------+
| ``--quality-csv``              | 1 (non-negative scalar, SNR...)  |
+--------------------------------+----------------------------------+
| ``--discrete-covariates-csv``  | >= 1 (categorical covariates)    |
+--------------------------------+----------------------------------+
| ``--continuous-covariates-csv``| >= 1 (numeric covariates)        |
+--------------------------------+----------------------------------+
| ``--design-csv``               | >= 1 (Limma design of interest)  |
+--------------------------------+----------------------------------+
| ``--covariates-csv``           | >= 1 (Harmony vars_use)          |
+--------------------------------+----------------------------------+
| ``--mz-csv``                   | 1 (m/z per feature, for drift)   |
+--------------------------------+----------------------------------+

Indices in all sidecar CSVs should match ``X.index``. If they do not,
MaldiBatchKit falls back to positional alignment when the row counts
match, and fails with a clear error otherwise.

Usage Examples
--------------

Vanilla ComBat
^^^^^^^^^^^^^^

.. code-block:: bash

    maldibatchkit correct combat \
        -i X.csv --batch-csv batch.csv \
        -o X_corrected.csv

Fortin ComBat with species as a protected categorical covariate:

.. code-block:: bash

    maldibatchkit correct combat-fortin \
        -i X.csv --batch-csv batch.csv \
        --discrete-covariates-csv species.csv \
        -o X_corrected.csv

Species-Aware Preset
^^^^^^^^^^^^^^^^^^^^

Exactly the same effect as the previous example, minus the typing
overhead:

.. code-block:: bash

    maldibatchkit correct species-combat \
        -i X.csv --batch-csv batch.csv \
        --species-csv species.csv \
        -o X_corrected.csv

Quality-Weighted ComBat
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    maldibatchkit correct quality-combat \
        -i X.csv --batch-csv batch.csv \
        --quality-csv snr.csv \
        --max-iter 30 \
        -o X_corrected.csv

Limma
^^^^^

.. code-block:: bash

    maldibatchkit correct limma \
        -i X.csv --batch-csv batch.csv \
        --design-csv species_design.csv \
        -o X_corrected.csv

Harmony
^^^^^^^

.. code-block:: bash

    maldibatchkit correct harmony \
        -i X.csv --batch-csv batch.csv \
        --theta 2.0 --max-iter 20 --random-state 0 \
        -o X_corrected.csv

Batch-Aware Warping
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    maldibatchkit correct warping \
        -i X.csv --batch-csv batch.csv \
        --method piecewise --n-segments 8 --max-shift 10 \
        -o X_warped.csv

Diagnostics
^^^^^^^^^^^

.. code-block:: bash

    maldibatchkit diagnose \
        -i X.csv --corrected X_corrected.csv \
        --batch-csv batch.csv \
        --mz-csv mz.csv --top-k-peaks 40 \
        -o report.csv

NPZ End-to-End
^^^^^^^^^^^^^^

Bundle ``X``, ``batch``, ``index``, and ``columns`` in a single archive:

.. code-block:: python

    np.savez("maldiset.npz",
             X=X.to_numpy(),
             columns=X.columns.to_numpy(),
             index=X.index.to_numpy(),
             batch=batch.to_numpy())

Then every ``correct`` subcommand accepts the NPZ directly:

.. code-block:: bash

    maldibatchkit correct combat-fortin \
        -i maldiset.npz \
        --discrete-covariates-csv species.csv \
        -o corrected.npz

Refusal / Error Modes
---------------------

* ``combat-fortin`` / ``combat-chen`` without any covariate CSV refuse
  to run with a hint to use plain ``combat`` instead - Fortin / CovBat
  without covariates reduce to Johnson.
* ``species-combat`` without ``--species-csv`` and ``quality-combat``
  without ``--quality-csv`` refuse to run.
* Index mismatches between ``X`` and a sidecar CSV produce a clear
  error identifying which rows are missing, rather than silently
  realigning.

See the :doc:`Quickstart Guide <quickstart>` for the matching Python API
and the :doc:`API Reference <api/index>` for full class documentation.
