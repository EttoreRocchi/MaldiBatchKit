Visualization Module
====================

Plotting helpers for before/after batch-effect inspection. All plotting
functions use lazy ``matplotlib`` imports, so ``matplotlib`` is only
required when a plot function is actually called.

``plot_batch_umap`` additionally requires ``umap-learn`` and can be
installed via ``pip install maldibatchkit[viz]``.

UMAP Before/After
-----------------

.. autofunction:: maldibatchkit.viz.plot_batch_umap

Peak-Shape Overlay
------------------

.. autofunction:: maldibatchkit.viz.plot_peak_shift

Diagnostic Summary
------------------

.. autofunction:: maldibatchkit.viz.plot_diagnostic_summary

Example
-------

.. code-block:: python

   from maldibatchkit import SpeciesAwareComBat
   from maldibatchkit.diagnostics import diagnostic_report
   from maldibatchkit.viz import (
       plot_batch_umap,
       plot_peak_shift,
       plot_diagnostic_summary,
   )

   corrector = SpeciesAwareComBat(batch=batch, species=species)
   X_corrected = corrector.fit_transform(X)

   # Side-by-side UMAP of raw vs. corrected matrices
   plot_batch_umap(X, X_corrected, batch, random_state=0)

   # Per-batch median spectra overlaid on a reference
   plot_peak_shift(batch, X_corrected, mz_values=mz)

   # Before/after bar chart built from a diagnostic_report DataFrame
   report = diagnostic_report(X, X_corrected, batch)
   plot_diagnostic_summary(report, scope="overall")
