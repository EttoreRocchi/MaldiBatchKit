Diagnostics Module
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

All metrics take the same ``(X, batch)`` signature and return scalars
or tidy ``pandas`` objects.

.. warning::

   Batch-mixing metrics say nothing about whether biological signal is
   preserved. Always pair them with a supervised metric (AMR classifier
   AUROC, VME, ...) in any real comparison of correctors.

Generic Batch-Mixing Metrics
----------------------------

.. autofunction:: maldibatchkit.diagnostics.silhouette_batch

.. autofunction:: maldibatchkit.diagnostics.kbet

.. autofunction:: maldibatchkit.diagnostics.lisi

MALDI-Specific Metrics
----------------------

.. autofunction:: maldibatchkit.diagnostics.peak_position_drift

.. autofunction:: maldibatchkit.diagnostics.tic_cov_per_batch

.. autofunction:: maldibatchkit.diagnostics.per_batch_spectrum_count

Combined Report
---------------

.. autofunction:: maldibatchkit.diagnostics.diagnostic_report

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
   print(report.head())
   #              metric    scope  value_before  value_after     delta  better
   # 0  silhouette_batch  overall         0.311        0.042    -0.269   lower
   # 1  kbet_acceptance   overall         0.124        0.561     0.437  higher
   # 2              lisi  overall         1.420        2.310     0.890  higher

   # Quick bar-chart visualisation of the overall metrics
   plot_diagnostic_summary(report, scope="overall")
