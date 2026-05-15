:html_theme.sidebar_secondary.remove: true

.. image:: _static/maldibatchkit_logo.png
   :align: center
   :width: 280px
   :class: only-light

.. image:: _static/maldibatchkit_logo.png
   :align: center
   :width: 280px
   :class: only-dark

.. rst-class:: hero-section

MaldiBatchKit Documentation
===========================

Batch-effect correction methods for MALDI-TOF mass spectrometry in
clinical AMR prediction workflows. Scikit-learn compatible transformers,
unified CLI, and a diagnostics suite for quantifying batch mixing before
and after correction.

.. container:: sd-d-flex-row sd-flex-justify-content-center sd-gap-2 sd-mb-4

   .. button-link:: installation.html
      :color: primary
      :shadow:

      Installation

   .. button-link:: api/index.html
      :color: primary
      :outline:
      :shadow:

      API Reference

   .. button-link:: quickstart.html
      :color: primary
      :outline:
      :shadow:

      Quickstart Guide

----

Key Features
------------

.. grid:: 2 2 3 3
   :gutter: 3
   :class-container: feature-grid

   .. grid-item-card:: ComBat Variants
      :link: api/corrections.html#combat-variants
      :link-type: url

      Johnson (2007), Fortin (2018) covariate-aware, and Chen (2022)
      CovBat re-exported from ``combatlearn`` with a unified sklearn API.

   .. grid-item-card:: Species-Aware ComBat
      :link: api/corrections.html#species-aware-combat
      :link-type: url

      ComBat-Fortin preset with species as the protected biological
      covariate, so inter-species structure is preserved during
      harmonisation.

   .. grid-item-card:: Quality-Weighted ComBat
      :link: api/corrections.html#quality-weighted-combat
      :link-type: url

      Weighted empirical-Bayes extension where low-SNR spectra contribute
      less to the shrinkage prior.

   .. grid-item-card:: Linear-Model Corrections
      :link: api/corrections.html#linear-model-corrections
      :link-type: url

      ``Limma`` ``removeBatchEffect`` (Ritchie 2015) as a pure-Python
      OLS subtraction that protects a user-supplied design.

   .. grid-item-card:: Harmony
      :link: api/corrections.html#single-cell-style-integration
      :link-type: url

      Iterative soft-clustering integration (Korsunsky 2019) via
      ``harmonypy``, suitable for many-batch designs.

   .. grid-item-card:: Baselines & Scaling
      :link: api/corrections.html#baselines
      :link-type: url

      Median centering, z-score per batch, and reference scaling -
      simple, auditable corrections to compare against.

   .. grid-item-card:: Batch-Aware Warping
      :link: api/corrections.html#maldi-specific-corrections
      :link-type: url

      Per-batch m/z warping sharing a global reference, wrapping
      ``maldiamrkit.alignment.Warping`` for MALDI-specific drift.

   .. grid-item-card:: Diagnostics Suite
      :link: api/diagnostics.html
      :link-type: url

      kBET, LISI, silhouette-by-batch, per-batch peak drift, TIC CoV, and
      a tidy ``diagnostic_report`` that summarises before/after deltas.

   .. grid-item-card:: AutoCorrector
      :link: choosing.html
      :link-type: url

      Meta-corrector with a swappable ``method`` hyperparameter — sweep
      across corrector families inside ``GridSearchCV`` and let the
      downstream AUROC pick the winner.

   .. grid-item-card:: Benchmark
      :link: choosing.html
      :link-type: url

      ``BatchCorrectionBenchmark`` scores every corrector on every
      metric with bootstrap CIs, returning a tidy table ready for
      paper-figure comparisons.

   .. grid-item-card:: MaldiSet Integration
      :link: api/integrations.html
      :link-type: url

      ``MaldiSetAdapter`` bridges ``maldiamrkit.MaldiSet`` and any
      MaldiBatchKit corrector in a single call, preserving metadata.

   .. grid-item-card:: CLI
      :link: cli.html
      :link-type: url

      ``maldibatchkit correct <method>`` and ``maldibatchkit diagnose``
      for batch processing, with matched NPZ / CSV I/O.

   .. grid-item-card:: Extensible Base Class
      :link: extending.html
      :link-type: url

      Subclass :class:`~maldibatchkit.BaseBatchCorrector` and implement
      two methods to ship a custom corrector that plugs into sklearn
      pipelines with no leakage.

----

Quick Example
-------------

.. code-block:: python

   from maldibatchkit import SpeciesAwareComBat
   from maldibatchkit.diagnostics import diagnostic_report

   # X: (n_samples, n_bins) DataFrame; batch & species indexed by X.index
   corrector = SpeciesAwareComBat(batch=batch, species=species)
   X_corrected = corrector.fit_transform(X)

   # Summarise the effect of correction
   report = diagnostic_report(X, X_corrected, batch)
   print(report)

Train/test without leakage - the corrector is fit on training data only
and then applied to held-out samples via ``transform``:

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from maldibatchkit import ComBat

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, stratify=batch
   )

   corrector = ComBat(
       batch=batch, method="fortin", discrete_covariates=species,
   )
   corrector.fit(X_train)
   X_train_c = corrector.transform(X_train)
   X_test_c = corrector.transform(X_test)

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Get Started

   installation
   quickstart
   choosing
   extending

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api/index
   cli

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Resources

   tutorials/index
   contributing
   papers
   changelog
