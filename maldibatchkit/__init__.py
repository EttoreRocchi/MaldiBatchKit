"""MaldiBatchKit - batch-effect correction for MALDI-TOF AMR workflows.

Subpackage guide
----------------
- ``maldibatchkit.corrections``  - ComBat variants, Limma, Harmony,
  simple baselines, and MALDI-specific corrections
  (``BatchAwareWarping``, ``QualityWeightedComBat``,
  ``SpeciesAwareComBat``).
- ``maldibatchkit.diagnostics``  - generic batch-mixing metrics (kBET, LISI,
  silhouette) and MALDI-specific metrics (peak drift, per-batch TIC CoV,
  spectrum counts), plus a ``diagnostic_report`` helper.
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
from .corrections.baselines import (
    MedianCentering,
    ReferenceScaling,
    ZScorePerBatch,
)
from .corrections.combat import ComBat, SpeciesAwareComBat
from .corrections.harmony import Harmony
from .corrections.limma import Limma
from .corrections.maldi import BatchAwareWarping
from .corrections.quality_weighted import QualityWeightedComBat

__version__ = "0.1.0"
__author__ = "Ettore Rocchi"

__all__ = [
    "BaseBatchCorrector",
    "BatchAwareWarping",
    "ComBat",
    "Harmony",
    "Limma",
    "MedianCentering",
    "QualityWeightedComBat",
    "ReferenceScaling",
    "SpeciesAwareComBat",
    "ZScorePerBatch",
    "__author__",
    "__version__",
]
