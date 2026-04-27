"""Batch-effect diagnostics.

This subpackage exposes two families of metrics plus a convenience
"report" helper:

- :mod:`maldibatchkit.diagnostics.generic` - classic batch-mixing
  metrics (k-nearest-neighbour batch test / kBET, local inverse
  Simpson index / LISI, silhouette coefficient by batch).
- :mod:`maldibatchkit.diagnostics.maldi` - MALDI-specific summaries
  (per-batch peak-position drift, per-batch TIC coefficient of
  variation, per-batch spectrum count).
- :func:`maldibatchkit.diagnostics.diagnostic_report` - run every
  metric on a ``(before, after)`` pair and summarise into a tidy
  DataFrame.
"""

from .generic import (
    kbet,
    lisi,
    lisi_max,
    lisi_normalized,
    silhouette_batch,
    species_preservation,
)
from .maldi import peak_position_drift, per_batch_spectrum_count, tic_cov_per_batch
from .report import diagnostic_report

__all__ = [
    "diagnostic_report",
    "kbet",
    "lisi",
    "lisi_max",
    "lisi_normalized",
    "peak_position_drift",
    "per_batch_spectrum_count",
    "silhouette_batch",
    "species_preservation",
    "tic_cov_per_batch",
]
