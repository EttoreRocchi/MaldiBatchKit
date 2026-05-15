"""MaldiBatchKit - batch-effect correction for MALDI-TOF AMR workflows.

Subpackage guide
----------------
- ``maldibatchkit.corrections``  - ComBat variants, Limma, Harmony,
  simple baselines, MALDI-specific corrections
  (``BatchAwareWarping``, ``QualityWeightedComBat``,
  ``SpeciesAwareComBat``), an identity ``NoOpCorrector``, and the
  ``AutoCorrector`` meta-corrector with a swappable ``method``
  hyperparameter for ``GridSearchCV``.
- ``maldibatchkit.diagnostics``  - generic batch-mixing metrics (kBET, LISI,
  silhouette) and MALDI-specific metrics (peak drift, per-batch TIC CoV,
  spectrum counts), plus a ``diagnostic_report`` helper and the
  ``BatchCorrectionBenchmark`` class for tidy multi-corrector comparisons.
- ``maldibatchkit.metrics``      - batch-aware downstream classifier metrics
  (``batch_roc_auc_score``, ``batch_balanced_accuracy_score``,
  ``batch_f1_score``, ``batch_matthews_corrcoef``, ``batch_precision_score``,
  ``batch_recall_score``, ``batch_average_precision_score``) and a
  ``make_batch_scorer`` factory for plugging them into ``GridSearchCV``.
- ``maldibatchkit.viz``          - ``plot_batch_umap``, ``plot_peak_shift``,
  ``plot_diagnostic_summary``.
- ``maldibatchkit.integrations`` - ``MaldiSetAdapter`` for seamless use with
  ``maldiamrkit.MaldiSet``.

Examples
--------
>>> from maldibatchkit import ComBat, QualityWeightedComBat
>>> from maldibatchkit.diagnostics import diagnostic_report
>>> from maldibatchkit.integrations import MaldiSetAdapter

Extending MaldiBatchKit: subclass :class:`BaseBatchCorrector` and implement
``_fit_impl`` / ``_transform_impl`` to ship a custom corrector that plugs
into sklearn pipelines without re-solving index alignment or leakage.
"""

from ._base import BaseBatchCorrector
from .corrections.auto import AutoCorrector
from .corrections.baselines import (
    MedianCentering,
    ReferenceScaling,
    ZScorePerBatch,
)
from .corrections.combat import ComBat, SpeciesAwareComBat
from .corrections.harmony import Harmony
from .corrections.limma import Limma
from .corrections.maldi import BatchAwareWarping
from .corrections.noop import NoOpCorrector
from .corrections.quality_weighted import QualityWeightedComBat
from .diagnostics.benchmark import BatchCorrectionBenchmark
from .metrics import (
    batch_average_precision_score,
    batch_balanced_accuracy_score,
    batch_f1_score,
    batch_matthews_corrcoef,
    batch_precision_score,
    batch_recall_score,
    batch_roc_auc_score,
    make_batch_scorer,
)

__version__ = "0.2.0"
__author__ = "Ettore Rocchi"

__all__ = [
    "AutoCorrector",
    "BaseBatchCorrector",
    "BatchAwareWarping",
    "BatchCorrectionBenchmark",
    "ComBat",
    "Harmony",
    "Limma",
    "MedianCentering",
    "NoOpCorrector",
    "QualityWeightedComBat",
    "ReferenceScaling",
    "SpeciesAwareComBat",
    "ZScorePerBatch",
    "__author__",
    "__version__",
    "batch_average_precision_score",
    "batch_balanced_accuracy_score",
    "batch_f1_score",
    "batch_matthews_corrcoef",
    "batch_precision_score",
    "batch_recall_score",
    "batch_roc_auc_score",
    "make_batch_scorer",
]
