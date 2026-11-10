Diagnostics module
==================

Quantitative checks that a correction step actually removed batch
structure. The subpackage exposes two families of metrics plus a
convenience report helper:

- :mod:`maldibatchkit.diagnostics.generic` - classic batch-mixing
  metrics (silhouette by batch, kBET, LISI).
- :mod:`maldibatchkit.diagnostics.maldi` - MALDI-specific summaries
  (per-batch peak drift, TIC coefficient of variation, spectrum count).
- :func:`maldibatchkit.diagnostics.diagnostic_report` - run every
  metric on a ``(before, after)`` pair and collapse it into a tidy
  DataFrame suitable for downstream tables and plots.

Most metrics take ``(X, batch)``; ``per_batch_spectrum_count`` takes only
``batch`` and ``diagnostic_report`` takes ``(before, after, batch)``.
``silhouette_batch`` and ``lisi`` return a float, ``kbet`` a dict, and the
MALDI-specific metrics and the report ``pandas`` objects.

.. warning::

   Batch-mixing metrics say nothing about whether biological signal is
   preserved. Always pair them with a supervised metric (AMR classifier
   AUROC, VME, ...) in any real comparison of correctors.

Generic batch-mixing metrics
----------------------------

.. autofunction:: maldibatchkit.diagnostics.silhouette_batch

.. autofunction:: maldibatchkit.diagnostics.kbet

.. autofunction:: maldibatchkit.diagnostics.lisi

MALDI-specific metrics
----------------------

.. autofunction:: maldibatchkit.diagnostics.peak_position_drift

.. autofunction:: maldibatchkit.diagnostics.tic_cov_per_batch

.. autofunction:: maldibatchkit.diagnostics.per_batch_spectrum_count

Combined report
---------------

.. autofunction:: maldibatchkit.diagnostics.diagnostic_report

Benchmark
---------

:class:`~maldibatchkit.diagnostics.BatchCorrectionBenchmark` runs a
fixed set of metrics across multiple correctors under a single
protocol, returning tidy per-(method, metric) summaries plus the raw
long-form observations. See :doc:`/choosing` for the recipe.

.. autoclass:: maldibatchkit.diagnostics.BatchCorrectionBenchmark
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from maldibatchkit import SpeciesAwareComBat
   from maldibatchkit.diagnostics import diagnostic_report
   from maldibatchkit.viz import plot_diagnostic_summary

   # Correct and summarise
   corrector = SpeciesAwareComBat(batch=batch, species=species)
   X_corrected = corrector.fit_transform(X)

   report = diagnostic_report(
       X, X_corrected, batch,
       mz_values=mz, top_k_peaks=40,
   )
   print(report[report["scope"] == "overall"])
   #                  metric    scope  value_before  value_after   delta  better
   # 0      silhouette_batch  overall         0.147       -0.011  -0.159   lower
   # 1  kbet_acceptance_rate  overall         0.020        0.262   0.242  higher
   # 2        kbet_mean_chi2  overall        54.016       15.746 -38.270   lower
   # 3                  lisi  overall         1.015        1.236   0.222  higher

   # Quick bar-chart visualisation of the overall metrics
   plot_diagnostic_summary(report, scope="overall")
