Quickstart Guide
================

This guide walks through the MaldiBatchKit API from a single ComBat call
to full train/test workflows, MaldiSet integration, and diagnostics.

All correctors expect a feature matrix ``X`` (rows = samples, columns =
m/z bins) and a ``batch`` vector aligned to ``X.index``. When ``X`` is a
``pandas.DataFrame``, covariates can be passed as ``Series`` / ``DataFrames``
with matching indices - the corrector realigns them on each ``fit`` and
``transform`` call so the same object is safe inside sklearn pipelines.

Vanilla ComBat
--------------

Apply Johnson (2007) ComBat to a single harmonisation:

.. code-block:: python

   from maldibatchkit import ComBat

   corrector = ComBat(batch=batch, method="johnson")
   X_corrected = corrector.fit_transform(X)

Covariate-Aware ComBat (Fortin)
-------------------------------

Protect biological covariates with Fortin (2018):

.. code-block:: python

   from maldibatchkit import ComBat

   corrector = ComBat(
       batch=batch,
       discrete_covariates=species,
       continuous_covariates=age,
       method="fortin",
   )
   X_corrected = corrector.fit_transform(X)

Species-Aware Preset
--------------------

:class:`~maldibatchkit.SpeciesAwareComBat` is a convenience wrapper for
the common case of "Fortin with species as the protected covariate":

.. code-block:: python

   from maldibatchkit import SpeciesAwareComBat

   corrector = SpeciesAwareComBat(batch=batch, species=species)
   X_corrected = corrector.fit_transform(X)

Quality-Weighted ComBat
-----------------------

Weight samples by a per-sample quality score (typically SNR) so
low-quality spectra contribute less to the shrinkage prior:

.. code-block:: python

   from maldibatchkit import QualityWeightedComBat

   corrector = QualityWeightedComBat(
       batch=batch,
       quality=snr,
       parametric=True,
       max_iter=30,
   )
   X_corrected = corrector.fit_transform(X)

Train/Test Without Leakage
--------------------------

Every corrector follows the standard sklearn ``fit`` / ``transform``
split - call ``fit`` on the training data only, then ``transform`` each
split with the fitted parameters:

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from maldibatchkit import ComBat

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, stratify=batch
   )

   corrector = ComBat(
       batch=batch, method="fortin", discrete_covariates=species,
   )
   corrector.fit(X_train)              # learns on train only
   X_train_c = corrector.transform(X_train)
   X_test_c = corrector.transform(X_test)   # same parameters on test

``batch`` is indexed by the same sample IDs as ``X``, so the corrector
slices the right subset on each call.

Inside an Sklearn Pipeline
--------------------------

All correctors expose ``fit`` / ``transform`` / ``fit_transform`` and are
drop-in pipeline steps:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import cross_val_score
   from maldibatchkit import ComBat

   pipe = Pipeline([
       ("combat", ComBat(batch=batch, method="fortin",
                         discrete_covariates=species)),
       ("scaler", StandardScaler()),
       ("clf", RandomForestClassifier(n_estimators=200)),
   ])

   scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
   print(f"AUROC: {scores.mean():.3f} +/- {scores.std():.3f}")

Linear-Model Subtraction (Limma)
--------------------------------

``Limma`` is a pure-Python implementation of ``limma::removeBatchEffect``
that subtracts an OLS fit of the batch indicators while protecting the
columns of ``design``:

.. code-block:: python

   from maldibatchkit import Limma

   corrector = Limma(batch=batch, design=species_design)
   X_corrected = corrector.fit_transform(X)

Harmony
-------

:class:`~maldibatchkit.Harmony` wraps ``harmonypy`` for iterative
soft-clustering integration. Like the ``harmonypy`` reference, the
corrector is fit-transform only (no clean test-time application):

.. code-block:: python

   from maldibatchkit import Harmony

   corrector = Harmony(
       batch=batch,
       covariates=extra_vars,
       theta=2.0,
       max_iter=20,
       random_state=0,
   )
   X_corrected = corrector.fit_transform(X)

Simple Baselines
----------------

Auditable, no-covariate alternatives useful for sanity checks:

.. code-block:: python

   from maldibatchkit import MedianCentering, ZScorePerBatch, ReferenceScaling

   median = MedianCentering(batch=batch).fit_transform(X)
   zscored = ZScorePerBatch(batch=batch).fit_transform(X)
   ref = ReferenceScaling(batch=batch, reference_batch="site_A").fit_transform(X)

MALDI-Specific: Batch-Aware Warping
-----------------------------------

:class:`~maldibatchkit.BatchAwareWarping` is a MALDI-specific corrector
that warps each batch onto a shared global reference before any
intensity-domain step:

.. code-block:: python

   from maldibatchkit import BatchAwareWarping

   warper = BatchAwareWarping(
       batch=batch,
       reference="median",
       method="piecewise",
       n_segments=8,
       max_shift=10,
       n_jobs=-1,
   )
   X_warped = warper.fit_transform(X)

It is safe to chain with a downstream intensity-domain corrector:

.. code-block:: python

   from sklearn.pipeline import Pipeline

   pipe = Pipeline([
       ("warp", BatchAwareWarping(batch=batch, method="piecewise")),
       ("combat", ComBat(batch=batch, method="fortin",
                         discrete_covariates=species)),
   ])
   X_corrected = pipe.fit_transform(X)

Diagnostics
-----------

The :mod:`~maldibatchkit.diagnostics` subpackage provides generic
batch-mixing metrics (kBET, LISI, silhouette) and MALDI-specific
summaries (per-batch peak drift, TIC CoV, spectrum count):

.. code-block:: python

   from maldibatchkit.diagnostics import (
       silhouette_batch, kbet, lisi,
       peak_position_drift, tic_cov_per_batch,
       diagnostic_report,
   )

   # Individual metrics
   sil = silhouette_batch(X_corrected, batch)
   kbet_stats = kbet(X_corrected, batch)
   lisi_mean = lisi(X_corrected, batch, perplexity=30.0)

   # One-shot before/after summary
   report = diagnostic_report(
       X, X_corrected, batch,
       mz_values=mz, top_k_peaks=40,
   )
   print(report)

The report is a tidy DataFrame with ``metric`` / ``scope`` /
``value_before`` / ``value_after`` / ``delta`` columns - suitable for
further pandas manipulation or for the bundled plotting helpers.

Visualization
-------------

.. code-block:: python

   from maldibatchkit.viz import (
       plot_batch_umap,
       plot_peak_shift,
       plot_diagnostic_summary,
   )

   # Side-by-side UMAP (requires maldibatchkit[viz])
   plot_batch_umap(X, X_corrected, batch, random_state=0)

   # Per-batch peak overlays against a reference spectrum
   plot_peak_shift(batch, X_corrected, mz_values=mz)

   # Before/after bar chart built from a diagnostic_report
   plot_diagnostic_summary(report, scope="overall")

MaldiSet Integration
--------------------

:class:`~maldibatchkit.integrations.MaldiSetAdapter` lets you correct a
:class:`maldiamrkit.MaldiSet` directly, returning a new ``MaldiSet``
whose ``X`` is harmonised and whose AMR labels / metadata are untouched:

.. code-block:: python

   from maldiamrkit import MaldiSet
   from maldibatchkit.integrations import MaldiSetAdapter
   from maldibatchkit import SpeciesAwareComBat

   ds = MaldiSet.from_directory(
       "spectra/", "metadata.csv",
       aggregate_by=dict(antibiotics="Ceftriaxone"),
   )

   adapter = MaldiSetAdapter(
       batch_column="Batch",
       species_column="Species",
       quality_column="SNR",
   )
   corrected_ds = adapter.correct(ds, SpeciesAwareComBat)

   corrected_ds.X      # harmonised feature matrix
   corrected_ds.y      # AMR labels, unchanged

Command-Line Interface
----------------------

MaldiBatchKit ships a Typer-based CLI organised as
``maldibatchkit correct <method>`` plus ``maldibatchkit diagnose``.
Each method has its own subcommand with only the flags it actually uses:

.. code-block:: bash

   # Vanilla Johnson ComBat
   maldibatchkit correct combat \
       -i X.csv --batch-csv batch.csv -o X_corrected.csv

   # Species-aware preset
   maldibatchkit correct species-combat \
       -i X.csv --batch-csv batch.csv --species-csv species.csv \
       -o X_corrected.csv

   # Diagnostic report
   maldibatchkit diagnose \
       -i X.csv --corrected X_corrected.csv --batch-csv batch.csv \
       --mz-csv mz.csv -o report.csv

See the :doc:`CLI Reference <cli>` for the full command tree, NPZ I/O
format, and covariate CSV conventions.
