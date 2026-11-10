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

Neither tool simulates a *new clinical site*: both rely on every batch
being present at fit time for the inter-batch correctors we ship.

Correctors differ in what they do with a batch not seen at ``fit``:

* :class:`~maldibatchkit.ComBat` (all variants),
  :class:`~maldibatchkit.SpeciesAwareComBat`,
  :class:`~maldibatchkit.Limma` and :class:`~maldibatchkit.Harmony` raise
  ``ValueError``.
* :class:`~maldibatchkit.MedianCentering` and
  :class:`~maldibatchkit.ZScorePerBatch` fall back to the grand statistics
  of the training data, and :class:`~maldibatchkit.BatchAwareWarping` to a
  warper trained on all training rows.
* :class:`~maldibatchkit.QualityWeightedComBat`,
  :class:`~maldibatchkit.ReferenceScaling` and
  :class:`~maldibatchkit.NoOpCorrector` return those rows unchanged.

:class:`~maldibatchkit.AutoCorrector` behaves like its inner method.

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

   # X: (n_samples, n_features) DataFrame; batch, species & snr aligned to X.index
   pipe = Pipeline([
       ("correct", AutoCorrector(batch=batch, discrete_covariates=species, quality=snr)),
       ("clf", LogisticRegression(max_iter=1000)),
   ])

   param_grid = {
       "correct__method": [
           "noop",            # honest baseline
           "median",
           "combat-fortin",
           "combat-johnson",
           "harmony",
           "qw-combat",       # requires `quality=`
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

Method aliases
^^^^^^^^^^^^^^

``method`` takes one of these aliases, or a corrector class (a
:class:`~maldibatchkit.BaseBatchCorrector` subclass or
:class:`~maldibatchkit.ComBat`). Instances raise ``TypeError``.

====================  ==============================================
Alias                 Corrector
====================  ==============================================
``combat``            :class:`~maldibatchkit.ComBat`, ``method="fortin"``
``combat-fortin``     :class:`~maldibatchkit.ComBat`, ``method="fortin"``
``combat-johnson``    :class:`~maldibatchkit.ComBat`, ``method="johnson"``
``combat-chen``       :class:`~maldibatchkit.ComBat`, ``method="chen"``
``species-combat``    :class:`~maldibatchkit.SpeciesAwareComBat`
``qw-combat``         :class:`~maldibatchkit.QualityWeightedComBat`
``limma``             :class:`~maldibatchkit.Limma`
``harmony``           :class:`~maldibatchkit.Harmony`
``batch-warping``     :class:`~maldibatchkit.BatchAwareWarping`
``median``            :class:`~maldibatchkit.MedianCentering`
``zscore``            :class:`~maldibatchkit.ZScorePerBatch`
``reference``         :class:`~maldibatchkit.ReferenceScaling`
``noop``              :class:`~maldibatchkit.NoOpCorrector`
====================  ==============================================

The alias ``combat`` means Fortin, whereas ``maldibatchkit correct combat``
on the command line runs Johnson.

Covariate routing
^^^^^^^^^^^^^^^^^

``AutoCorrector`` accepts ``discrete_covariates``, ``continuous_covariates``,
``quality``, ``species``, and ``reference_batch`` and forwards each to
the inner method *only when the inner accepts that name*. The mapping
table is:

============================  ==========================================
``AutoCorrector`` argument    Forwarded as ...
============================  ==========================================
``discrete_covariates``       ComBat ``discrete_covariates``, Limma
                              ``design`` (one-hot encoded); dropped with
                              a warning for Harmony, whose
                              ``covariates`` are removed like the batch
``continuous_covariates``     ComBat ``continuous_covariates``
``quality``                   ``QualityWeightedComBat.quality``
``species``                   ``SpeciesAwareComBat.species``
``reference_batch``           Any inner that accepts ``reference_batch``
============================  ==========================================

Pass anything else through ``method_kwargs={...}``. Entries the inner
method does not accept are dropped silently, so a single ``param_grid``
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
           "harmony":  Harmony(batch=batch, verbose=False),
       },
       metrics=("kbet", "lisi_normalized", "species_preservation",
                "tic_cov_per_batch"),
       n_bootstrap=500,
       random_state=0,
   )
   bench.fit(X, batch=batch, species=species)

   print(bench.rank(by="species_preservation"))
   bench.plot()  # bar facet per metric, error bars from the subsample CIs

``metrics`` takes callables or these names: ``kbet`` and
``kbet_acceptance_rate`` (the same metric), ``lisi``, ``lisi_normalized``,
``silhouette_batch``, ``species_preservation``, ``peak_position_drift`` and
``tic_cov_per_batch``. ``species_preservation`` needs ``species=`` in
``fit``, otherwise it is ``NaN`` with a warning. ``peak_position_drift``
accepts ``mz_values=`` in ``fit``.

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

* ``bench.results_long_``: one row per (method, metric, repeat,
  bootstrap iteration). Use it to compute custom statistics or to
  drive a raincloud / strip plot.
* ``bench.results_``: one row per (method, metric) with ``value``
  (the full-data estimate when resampling, otherwise the mean across
  repeats), ``ci_lo`` / ``ci_hi`` (2.5 / 97.5 percentile), ``std``, ``n``,
  and ``better`` (``'higher'``, ``'lower'``, ``'zero'``, or ``'n/a'`` for
  user callables). The interval columns hold ``NaN`` when fewer than two
  bootstraps and fewer than three repeats are available, so don't read a
  CI from one point.

``bench.baseline_`` is a parallel one-row-per-metric table computed on
the **uncorrected** ``X`` and is useful as a reference line in plots.

Resampling modes
^^^^^^^^^^^^^^^^

Every resample draws ``subsample_fraction`` (default 0.8) of each batch
**without replacement**: duplicated rows would be distance-0 same-batch
neighbours and bias the kNN metrics. kNN / perplexity-based metrics also
depend on the number of samples, so the interval describes the metric at
the subsample size and may not contain the full-data ``value``.

``bootstrap_mode="resample_metric"`` (default, fast) fits each corrector
once and subsamples rows of the corrected matrix to give the CI. Use it
when you want CIs that reflect metric sampling noise on a fixed fit.

``bootstrap_mode="refit"`` subsamples rows of ``X`` (stratified by
batch) and refits the corrector on each resample. Slower
(``n_correctors × n_bootstrap`` extra fits per repeat), but the CI also
captures corrector stability. Use it when you want to advertise a
method's reproducibility.
