Choosing a corrector
====================

MaldiBatchKit ships several correctors with overlapping use cases. The
v0.2 ``AutoCorrector`` and ``BatchCorrectionBenchmark`` classes are the
two recommended tools for picking among them, depending on whether the
deciding signal is **downstream task performance** or
**post-correction diagnostics**.

.. contents::
   :local:

When to use which
-----------------

* Use :class:`~maldibatchkit.AutoCorrector` + ``GridSearchCV`` when the
  question is *"which corrector gives the best AMR classifier?"*. The
  scorer is your downstream metric (typically AUROC), and the split
  defines the generalisation goal.

* Use :class:`~maldibatchkit.diagnostics.BatchCorrectionBenchmark` when
  the question is *"how well does each method mix batches and preserve
  species?"*. The output is a tidy table suitable for paper figures and
  side-by-side reporting; it does **not** simulate a downstream
  classifier.

Neither tool simulates a *new clinical site* — both rely on every batch
being present at fit time for the inter-batch correctors we ship.

AutoCorrector with GridSearchCV
-------------------------------

``AutoCorrector`` exposes ``method`` as a settable hyperparameter that
swaps the inner corrector at ``fit`` time. Wrap it in a
:class:`sklearn.pipeline.Pipeline` and let ``GridSearchCV`` sweep the
method together with any classifier hyperparameters:

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import GridSearchCV, StratifiedKFold
   from sklearn.pipeline import Pipeline

   from maldibatchkit import AutoCorrector

   # X: (n_samples, n_features) DataFrame; batch & species aligned to X.index
   pipe = Pipeline([
       ("correct", AutoCorrector(batch=batch, discrete_covariates=species)),
       ("clf", LogisticRegression(max_iter=1000)),
   ])

   param_grid = {
       "correct__method": [
           "noop",            # honest baseline
           "median",
           "combat-fortin",
           "combat-johnson",
           "harmony",
           "qw-combat",       # accepts `quality=`
       ],
       "clf__C": [0.1, 1.0, 10.0],
   }

   grid = GridSearchCV(
       pipe,
       param_grid=param_grid,
       scoring="roc_auc",
       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
   )
   grid.fit(X, y)
   print(grid.best_params_, grid.best_score_)

Covariate routing
^^^^^^^^^^^^^^^^^

``AutoCorrector`` accepts ``discrete_covariates``, ``continuous_covariates``,
``quality``, ``species``, and ``reference_batch`` and forwards each to
the inner method *only when the inner accepts that name*. The mapping
table is:

============================  ==========================================
``AutoCorrector`` argument    Forwarded as ...
============================  ==========================================
``discrete_covariates``       ComBat ``discrete_covariates``,
                              Harmony ``covariates``, Limma ``design``
``continuous_covariates``     ComBat ``continuous_covariates``
``quality``                   ``QualityWeightedComBat.quality``
``species``                   ``SpeciesAwareComBat.species``
``reference_batch``           Any inner that accepts ``reference_batch``
============================  ==========================================

Pass anything else through ``method_kwargs={...}``. Unrecognised entries
are dropped silently (sklearn convention), so a single ``param_grid``
can target several methods.

BatchCorrectionBenchmark
------------------------

``BatchCorrectionBenchmark`` is the diagnostic counterpart: configure
once, then ``.fit(X, batch=..., species=...)`` to run every corrector
under the chosen protocol.

.. code-block:: python

   from maldibatchkit import ComBat, Harmony, Limma, NoOpCorrector
   from maldibatchkit.diagnostics import BatchCorrectionBenchmark

   bench = BatchCorrectionBenchmark(
       correctors={
           "none":     NoOpCorrector(batch=batch),
           "limma":    Limma(batch=batch, design=species_dummies),
           "fortin":   ComBat(batch=batch, method="fortin",
                              discrete_covariates=species),
           "harmony":  Harmony(batch=batch, covariates=species, verbose=False),
       },
       metrics=("kbet", "lisi_normalized", "species_preservation",
                "tic_cov_per_batch"),
       n_bootstrap=500,
       random_state=0,
   )
   bench.fit(X, batch=batch, species=species)

   print(bench.rank(by="species_preservation"))
   bench.plot()  # bar facet per metric, error bars from the bootstrap CIs

Protocols
^^^^^^^^^

``protocol="full_data"`` (default) follows the Büttner-2019 convention:
fit each corrector on all rows, score on the corrected matrix. This is
the right setting for paper figures.

``protocol="stratified_split"`` holds out a stratified test fold per
repeat, fits each corrector on train, transforms train + test, and
scores on the held-out rows. Useful for checking that a method's
benefits survive on unseen samples.

Aggregation
^^^^^^^^^^^

Two tables are always available after ``fit``:

* ``bench.results_long_`` — one row per (method, metric, repeat,
  bootstrap iteration). Use it to compute custom statistics or to
  drive a raincloud / strip plot.
* ``bench.results_`` — one row per (method, metric) with ``value``
  (mean), ``ci_lo`` / ``ci_hi`` (2.5 / 97.5 percentile), ``std``, ``n``,
  and ``better`` (``'higher'``, ``'lower'``, or ``'n/a'`` for user
  callables). The interval columns hold ``NaN`` when fewer than two
  bootstraps and fewer than three repeats are available — don't read a
  CI from one point.

``bench.baseline_`` is a parallel one-row-per-metric table computed on
the **uncorrected** ``X`` and is useful as a reference line in plots.

Bootstrap modes
^^^^^^^^^^^^^^^

``bootstrap_mode="resample_metric"`` (default, fast) fits each corrector
once and resamples rows of the corrected matrix to give the CI. Use it
when you want CIs that reflect metric sampling noise on a fixed fit.

``bootstrap_mode="refit"`` resamples rows of ``X`` (stratified by
batch) and refits the corrector on each bootstrap. Slower
(``n_correctors × n_bootstrap`` extra fits per repeat), but the CI also
captures corrector stability. Use it when you want to advertise a
method's reproducibility.
