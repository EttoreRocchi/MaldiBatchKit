API Reference
=============

Complete reference for all public classes and functions in MaldiBatchKit,
organised by module.

.. toctree::
   :maxdepth: 2
   :hidden:

   corrections
   diagnostics
   metrics
   visualization
   integrations

Correctors
----------

Every corrector exposes the scikit-learn ``fit`` / ``transform`` /
``fit_transform`` API, accepts ``batch`` and covariates at construction
time, and aligns them to ``X.index`` at each call - so the same object
is safe inside ``Pipeline`` and ``cross_val_score`` without leakage.

Base Class
~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.BaseBatchCorrector

Subclass this to ship a custom corrector. See :doc:`/extending` for a
walkthrough.

ComBat Family
~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.ComBat
   maldibatchkit.SpeciesAwareComBat
   maldibatchkit.QualityWeightedComBat

Linear / Non-Parametric
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.Limma

Single-Cell-Style Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.Harmony

Simple Baselines
~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.MedianCentering
   maldibatchkit.ZScorePerBatch
   maldibatchkit.ReferenceScaling

MALDI-Specific Corrections
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.BatchAwareWarping

Diagnostics
-----------

Generic Batch-Mixing Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.diagnostics.silhouette_batch
   maldibatchkit.diagnostics.kbet
   maldibatchkit.diagnostics.lisi

MALDI-Specific Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.diagnostics.peak_position_drift
   maldibatchkit.diagnostics.tic_cov_per_batch
   maldibatchkit.diagnostics.per_batch_spectrum_count

Combined Report
~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.diagnostics.diagnostic_report

Benchmark
~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.diagnostics.BatchCorrectionBenchmark

Metrics
-------

Batch-aware downstream classifier metrics for model selection that
generalises across sites. See :doc:`metrics` for the full reference.

Per-batch Metric Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.batch_roc_auc_score
   maldibatchkit.batch_average_precision_score
   maldibatchkit.batch_balanced_accuracy_score
   maldibatchkit.batch_matthews_corrcoef
   maldibatchkit.batch_f1_score
   maldibatchkit.batch_precision_score
   maldibatchkit.batch_recall_score

Scorer Factory
~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   maldibatchkit.make_batch_scorer

Visualization
-------------

.. autosummary::
   :nosignatures:

   maldibatchkit.viz.plot_batch_umap
   maldibatchkit.viz.plot_peak_shift
   maldibatchkit.viz.plot_diagnostic_summary

Integrations
------------

.. autosummary::
   :nosignatures:

   maldibatchkit.integrations.MaldiSetAdapter
